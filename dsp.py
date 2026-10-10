"""WiSense signal pipeline.

CSI packets (~100/s) -> amplitude per sub-carrier -> 20 Hz bins -> Hampel (7, 3 sigma)
-> Butterworth band-pass 0.1-0.6 Hz -> 20 s sliding FFT -> presence / breathing / fall.

Everything runs causally on the live stream. Thresholds that depend on the room are
learned by the empty-room calibration; the constants below are the ones to tune.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

# ---- pipeline constants (from the project spec) -------------------------------------
N_SUB = 64
FS = 20.0                      # Hz after averaging
HAMPEL_WIN = 7
HAMPEL_K = 3.0
BAND = (0.1, 0.6)              # Butterworth band-pass, Hz
WIN_S = 20                     # FFT window, seconds
BREATH_BAND = (0.2, 0.5)       # Hz  -> 12-30 breaths/min
NFFT = 2048                    # zero-padding, only for a smoother-looking spectrum

# ---- decision tuning ----------------------------------------------------------------
E_WIN_S = 8                    # presence energy window, seconds
PEAK_SNR = 4.0                 # breathing peak must beat the band median by this factor
MOVING_LVL = 2.0               # movement level (x empty-room noise) that counts as "moving"
BURST_LVL = 4.0                # ... and as a fall-sized burst
STILL_LVL = 1.6                # ... and as "still again"
FALL_HOLD_S = 20.0

STATUS_EVERY = 10              # frames between status messages (0.5 s)


class Engine:
    def __init__(self, emit, cal_collect_s: float = 20.0):
        self.emit = emit
        self.cal_collect_s = cal_collect_s
        self.lock = threading.RLock()

        self.sos = butter(2, BAND, btype="band", fs=FS, output="sos")  # 4th order overall
        self.zi0 = sosfilt_zi(self.sos)
        self.zi = None

        # packet -> 20 Hz binning
        self.cur_bin = None
        self.acc = np.zeros(N_SUB)
        self.cnt = 0
        self.last_frame = None
        self.pkt_times = deque(maxlen=300)
        self.rssi = None

        # per-frame state
        self.frame_n = 0
        self.mean_amp = None
        self.raw = deque(maxlen=HAMPEL_WIN)
        self.prev_clean = None
        self.bp_buf = deque(maxlen=int(WIN_S * FS))
        self.mot_recent = deque(maxlen=10)
        self.pow_hist = deque(maxlen=int(E_WIN_S * FS))
        self.lvl_hist = deque(maxlen=int(10 * FS))
        self.E = 0.0
        self.M = 0.0
        self.valid = np.zeros(N_SUB, bool)

        # calibration
        self.cal_state = "none"            # none | running | done
        self.cal_start_n = 0
        self._cal_E, self._cal_M, self._cal_W = [], [], []
        self.E_base = None
        self.thr_E = None
        self.M_ref = None
        self.baseline = None
        self.baseline_dirty = False

        # decisions
        self.present = False
        self._above_n = 0
        self._below_n = 0
        self.bpm_hist = deque(maxlen=5)
        self.bpm = None
        self.peak_hz = None
        self.snr = 0.0
        self.spec = None
        self.fall_n = None                 # frame index when the last fall fired
        self.window_ready = False

    # ------------------------------------------------------------------ public API
    def start_calibration(self):
        with self.lock:
            self.cal_state = "running"
            self.cal_start_n = self.frame_n
            self._cal_E, self._cal_M, self._cal_W = [], [], []
            self.fall_n = None
            self.present = False

    def hello(self):
        with self.lock:
            return {"type": "hello", "n_sub": N_SUB, "fs": FS,
                    "valid": self.valid.astype(int).tolist(),
                    "base": None if self.baseline is None else np.round(self.baseline, 2).tolist(),
                    "cal": self._cal_info()}

    def push_packet(self, t: float, amp: np.ndarray, rssi=None):
        """t = monotonic seconds when the packet arrived; amp = |CSI| for 64 sub-carriers."""
        with self.lock:
            self.pkt_times.append(t)
            if rssi is not None:
                self.rssi = rssi
            b = int(t * FS)
            if self.cur_bin is None:
                self.cur_bin = b
            if b != self.cur_bin:
                if self.cnt:
                    frame = self.acc / self.cnt
                    self.last_frame = frame
                    self._frame(frame)
                    # sensor went quiet for a few bins: hold the last value so time stays uniform
                    for _ in range(min(b - self.cur_bin - 1, 20)):
                        self._frame(frame)
                self.acc[:] = 0
                self.cnt = 0
                self.cur_bin = b
            self.acc += amp
            self.cnt += 1

    # ------------------------------------------------------------------ internals
    def _cal_info(self):
        prog = 0.0
        if self.cal_state == "running":
            total = E_WIN_S + self.cal_collect_s
            prog = min(1.0, (self.frame_n - self.cal_start_n) / FS / total)
        return {"state": self.cal_state, "progress": round(prog, 3),
                "seconds_left": round(max(0.0, (E_WIN_S + self.cal_collect_s) * (1 - prog)), 1)}

    def _frame(self, amp: np.ndarray):
        self.frame_n += 1
        n = self.frame_n

        # running mean amplitude per sub-carrier; sub-carriers stuck at 0 are the null ones
        if self.mean_amp is None:
            self.mean_amp = amp.copy()
        else:
            self.mean_amp += 0.01 * (amp - self.mean_amp)
        valid = self.mean_amp > 0.5
        if not np.array_equal(valid, self.valid):
            self.valid = valid
            self.baseline_dirty = True
        den = np.maximum(self.mean_amp, 1.0)
        vi = np.flatnonzero(self.valid)
        if len(vi) == 0:
            return

        # --- Hampel filter (centred window of 7, 3 sigma), output lags 3 samples ----------
        self.raw.append(amp)
        if len(self.raw) < HAMPEL_WIN:
            return
        w = np.asarray(self.raw)
        med = np.median(w, axis=0)
        mad = 1.4826 * np.median(np.abs(w - med), axis=0)
        c = w[HAMPEL_WIN // 2]
        sigma = np.maximum(mad, 0.005 * np.abs(med) + 1e-6)
        clean = np.where(np.abs(c - med) > HAMPEL_K * sigma, med, c)

        # --- band-pass 0.1-0.6 Hz, streaming with filter state --------------------------
        if self.zi is None:
            self.zi = self.zi0[:, :, None] * clean[None, None, :]
        y, self.zi = sosfilt(self.sos, clean[None, :], axis=0, zi=self.zi)
        bp_rel = 100.0 * y[0] / den                      # % of the sub-carrier's mean amplitude
        bp_rel[~self.valid] = 0.0
        self.bp_buf.append(bp_rel)

        # --- movement (fast changes) and energy (slow band) ------------------------------
        if self.prev_clean is not None:
            d = np.abs(clean - self.prev_clean) / den
            self.mot_recent.append(float(d[vi].mean()))
        self.prev_clean = clean
        self.M = float(np.mean(self.mot_recent)) if self.mot_recent else 0.0
        self.pow_hist.append(float(np.mean(bp_rel[vi] ** 2)))
        self.E = float(np.sqrt(np.mean(self.pow_hist)))

        self._calibration_step(n, clean)
        lvl = None if self.M_ref is None else self.M / self.M_ref
        if lvl is not None:
            self.lvl_hist.append(lvl)
        self._decide(lvl)

        msg = {"type": "frame", "n": n, "amp": np.round(amp, 1).tolist(),
               "bp": np.round(bp_rel, 2).tolist(), "E": round(self.E, 3),
               "M": round(self.M, 5), "lvl": None if lvl is None else round(lvl, 2)}
        if n % STATUS_EVERY == 0:
            msg["status"] = self._status()
        self.emit(msg)

    def _calibration_step(self, n, clean):
        if self.cal_state != "running":
            return
        elapsed = (n - self.cal_start_n) / FS
        if elapsed >= E_WIN_S:                           # first E_WIN_S: let old data flush out
            self._cal_E.append(self.E)
            self._cal_M.append(self.M)
            self._cal_W.append(clean.copy())
        if elapsed >= E_WIN_S + self.cal_collect_s and len(self._cal_E) > 10:
            E, M = np.array(self._cal_E), np.array(self._cal_M)
            self.E_base = float(E.mean())
            self.thr_E = float(E.mean() + max(4 * E.std(), 0.5 * E.mean()))
            self.M_ref = float(M.mean() + max(4 * M.std(), 0.5 * M.mean()))
            self.baseline = np.mean(self._cal_W, axis=0)
            self.baseline_dirty = True
            self.cal_state = "done"
            self.lvl_hist.clear()

    def _decide(self, lvl):
        if self.thr_E is None:
            return
        # presence: energy above the empty-room baseline, with hysteresis
        if self.E > self.thr_E:
            self._above_n += 1
            self._below_n = 0
        else:
            self._above_n = 0
            self._below_n += 1
        if not self.present and self._above_n >= FS:
            self.present = True
        elif self.present and self.E < 0.8 * self.thr_E and self._below_n >= 6 * FS:
            self.present = False

        # fall: a short sharp burst, then stillness, while someone is still there
        if lvl is not None and self.present and len(self.lvl_hist) >= int(8 * FS):
            fell = self.fall_n is not None and (self.frame_n - self.fall_n) / FS < FALL_HOLD_S
            if not fell:
                lv = np.asarray(self.lvl_hist)
                burst = lv[-int(8 * FS):-int(3 * FS)]
                still = lv[-int(3 * FS):]
                burst_s = float((burst > BURST_LVL).sum()) / FS
                if 0.3 <= burst_s <= 4.0 and np.percentile(still, 90) < STILL_LVL:
                    self.fall_n = self.frame_n

        # spectrum every STATUS_EVERY frames once 10 s of data exist
        if self.frame_n % STATUS_EVERY == 0:
            self._spectrum()

    def _spectrum(self):
        N = len(self.bp_buf)
        self.window_ready = N >= int(10 * FS)
        if not self.window_ready:
            return
        X = np.asarray(self.bp_buf)[:, self.valid]
        X = X - X.mean(axis=0)
        F = np.fft.rfft(X * np.hanning(N)[:, None], n=NFFT, axis=0)
        P = (np.abs(F) ** 2).sum(axis=1)
        f = np.fft.rfftfreq(NFFT, 1 / FS)

        inband = (f >= BAND[0]) & (f <= BAND[1])
        breath = (f >= BREATH_BAND[0]) & (f <= BREATH_BAND[1])
        floor = float(np.median(P[inband])) + 1e-12
        idx = np.flatnonzero(breath)[np.argmax(P[breath])]
        self.peak_hz = float(f[idx])
        self.snr = float(P[idx] / floor)

        lvl = self.lvl_hist[-1] if self.lvl_hist else None
        calm = lvl is None or lvl < MOVING_LVL
        ok = self.snr >= PEAK_SNR and calm and (self.present or self.thr_E is None)
        if ok:
            self.bpm_hist.append(60.0 * self.peak_hz)
            self.bpm = float(np.median(self.bpm_hist))
        else:
            self.bpm_hist.clear()
            self.bpm = None

        show = f <= 1.0
        top = float(P[inband].max()) + 1e-12
        self.spec = np.round(np.minimum(P[show] / top, 1.0), 3).tolist()

    def _status(self):
        now = time.monotonic()
        recent = [t for t in self.pkt_times if now - t <= 1.0]
        fall = self.fall_n is not None and (self.frame_n - self.fall_n) / FS < FALL_HOLD_S
        lvl = self.lvl_hist[-1] if self.lvl_hist else None
        s = {
            "pps": len(recent),
            "rssi": self.rssi,
            "cal": self._cal_info(),
            "calibrated": self.thr_E is not None,
            "present": self.present if self.thr_E is not None else None,
            "moving": None if lvl is None else bool(lvl >= MOVING_LVL),
            "E": round(self.E, 3),
            "E_thr": None if self.thr_E is None else round(self.thr_E, 3),
            "E_base": None if self.E_base is None else round(self.E_base, 3),
            "bpm": None if self.bpm is None else round(self.bpm, 1),
            "peak_hz": self.peak_hz,
            "snr": round(self.snr, 1),
            "window_ready": self.window_ready,
            "fall": fall,
            "fall_age": None if not fall else round((self.frame_n - self.fall_n) / FS, 1),
            "valid": self.valid.astype(int).tolist(),
            "spec": self.spec,
        }
        if self.baseline_dirty and self.baseline is not None:
            s["base"] = np.round(self.baseline, 2).tolist()
            self.baseline_dirty = False
        return s
