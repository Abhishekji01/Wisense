"""Where CSI packets come from: the ESP32 receiver over serial, or a built-in simulator."""
from __future__ import annotations

import math
import threading
import time

import numpy as np

N_SUB = 64


def parse_csi_line(line: str):
    """Parse one serial line. Returns (amp[64], rssi|None) or None.

    Handles the ESP-IDF esp-csi example format
        CSI_DATA,seq,mac,rssi,...,len,first_word,"[i,q,i,q,...]"
    and any other line that contains a bracketed list of >=128 integers.
    ESP-IDF stores each sub-carrier as (imag, real) int8 pairs; the amplitude is the same either way.
    """
    lb, rb = line.find("["), line.rfind("]")
    if lb < 0 or rb < lb:
        return None
    try:
        vals = np.array(line[lb + 1:rb].replace(",", " ").split(), dtype=np.float64)
    except ValueError:
        return None
    if vals.size < 2 * N_SUB:
        return None
    vals = vals[:2 * N_SUB]
    amp = np.hypot(vals[1::2], vals[0::2])
    rssi = None
    if line.startswith("CSI_DATA"):
        parts = line[:lb].split(",")
        if len(parts) > 3:
            try:
                rssi = int(parts[3])
            except ValueError:
                pass
    return amp, rssi


def find_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''} {p.hwid}".upper()
        if any(k in text for k in ("CP210", "CH340", "CH910", "USB SERIAL", "UART", "ACM", "FTDI")):
            return p.device
    return ports[0].device if ports else None


class SerialSource(threading.Thread):
    def __init__(self, engine, log, port: str, baud: int = 921600):
        super().__init__(daemon=True)
        self.engine, self.log, self.baud = engine, log, baud
        self.port = port
        self._halt = threading.Event()
        self.name_ = "serial"

    @property
    def label(self):
        return f"serial {self.port or 'auto'} @ {self.baud}"

    def stop(self):
        self._halt.set()

    def run(self):
        import serial
        while not self._halt.is_set():
            port = self.port if self.port not in (None, "auto") else find_port()
            if not port:
                self.log("No serial port found. Plug in the receiver ESP32 and check the USB cable.")
                time.sleep(2)
                continue
            try:
                with serial.Serial(port, self.baud, timeout=0.05) as ser:
                    self.log(f"Reading {port} at {self.baud} baud")
                    buf = b""
                    got = False
                    last_warn = time.monotonic()
                    while not self._halt.is_set():
                        buf += ser.read(max(1, ser.in_waiting))
                        *lines, buf = buf.split(b"\n")
                        if len(buf) > 20000:
                            buf = b""
                        for raw in lines:
                            parsed = parse_csi_line(raw.decode("ascii", "ignore"))
                            if parsed:
                                got = True
                                self.engine.push_packet(time.monotonic(), parsed[0], parsed[1])
                        if not got and time.monotonic() - last_warn > 5:
                            self.log(f"Port {port} is open but no CSI lines yet. Check the baud rate "
                                     f"({self.baud}) and that the transmitter is powered.")
                            last_warn = time.monotonic()
            except Exception as e:  # port busy, unplugged, ...
                self.log(f"Serial error on {port}: {e}. Close other programs using the port (idf.py monitor, "
                         f"Arduino IDE). Retrying...")
                time.sleep(2)


class Simulator(threading.Thread):
    """Synthetic Wi-Fi channel: a few static multipath rays plus one 'body' ray whose length changes."""

    MODES = ("empty", "breathe", "walk", "fall")

    def __init__(self, engine, log, rate: float = 100.0, seed: int = 3):
        super().__init__(daemon=True)
        self.engine, self.log, self.rate = engine, log, rate
        self.rng = np.random.default_rng(seed)
        self.mode = "empty"
        self.t0 = time.monotonic()
        self._halt = threading.Event()

    label = "simulator"

    def stop(self):
        self._halt.set()

    def set_mode(self, mode: str):
        if mode in self.MODES:
            self.mode, self.t0 = mode, time.monotonic()

    def _body(self, t: float):
        """(ray amplitude, path-length change in metres) for the current scenario."""
        breath = 0.006 * math.sin(2 * math.pi * 0.27 * t) + 0.0005 * math.sin(2 * math.pi * 1.2 * t)
        if self.mode == "empty":
            return 0.0, 0.0
        if self.mode == "breathe":
            return 0.2, breath
        if self.mode == "walk":
            return 0.35, (0.8 * math.sin(2 * math.pi * 0.35 * t) + 0.25 * math.sin(2 * math.pi * 1.1 * t + 1)
                          + 0.05 * math.sin(2 * math.pi * 2.3 * t))
        tau = time.monotonic() - self.t0                                  # fall
        if tau < 1.6:
            x = tau / 1.6
            return 0.4, 0.7 * math.sin(2 * math.pi * 1.3 * tau) * (1 - x) + 0.4 * (3 * x * x - 2 * x ** 3)
        return 0.2, 0.4 + breath

    def run(self):
        k = np.arange(-32, 32)
        f = k * 312.5e3
        valid = (np.abs(k) <= 26) & (k != 0)               # LLTF guard bands + DC are null on real hardware
        fc, c = 2.437e9, 3e8
        taus = np.array([0, 18, 35, 60, 90]) * 1e-9
        amps = np.array([1.0, 0.6, 0.5, 0.35, 0.25])
        ph = self.rng.uniform(0, 2 * np.pi, 5)
        Hs = (amps[:, None] * np.exp(1j * ph[:, None]) * np.exp(-2j * np.pi * f[None, :] * taus[:, None])).sum(0)
        scale = 40.0 / np.abs(Hs[valid]).mean()
        tau_h0 = 45e-9
        dyn = 0.0
        dt = 1.0 / self.rate
        t_next = time.monotonic()
        while not self._halt.is_set():
            now = time.monotonic()
            if now < t_next:
                time.sleep(min(t_next - now, 0.004))
                continue
            t_next = now + dt if now - t_next > 0.2 else t_next + dt
            a, L = self._body(now)
            dyn += (a - dyn) * (dt / 0.4)
            H = Hs + dyn * np.exp(-2j * np.pi * (fc + f) * (tau_h0 + L / c))
            H = H * (1 + 0.012 * (self.rng.standard_normal(N_SUB) + 1j * self.rng.standard_normal(N_SUB)))
            H = H + 0.02 * np.abs(Hs).mean() * (self.rng.standard_normal(N_SUB) + 1j * self.rng.standard_normal(N_SUB))
            re, im = np.round(H.real * scale), np.round(H.imag * scale)
            amp = np.hypot(re, im) * valid
            if self.rng.random() < 0.005:                  # isolated glitches for the Hampel filter
                amp[self.rng.integers(0, N_SUB, 6)] *= 3.0
            self.engine.push_packet(now, amp, int(-45 + self.rng.standard_normal() * 1.5))
