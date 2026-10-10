# WiSense live dashboard

Camera-free sensing with Wi-Fi CSI. The ESP32 receiver prints CSI over UART; this program reads it on your
laptop, runs the processing pipeline, and shows the channel live in a browser. It needs no internet once the
Python packages are installed.

## Run

    pip install -r requirements.txt
    python server.py --list-ports            # find the receiver's port
    python server.py --port COM5             # Windows
    python server.py --port /dev/ttyUSB0     # Linux  (macOS: /dev/cu.usbserial-XXXX)
    python server.py --sim                   # no hardware: simulated room, with scenario buttons

Open http://127.0.0.1:8000. To show it on another screen on the same Wi-Fi, add `--host 0.0.0.0`.

Close `idf.py monitor` or any serial monitor first: only one program can hold the port.

## Using it

1. Leave the room, press **Calibrate empty room**, wait about 28 s. This learns the baseline channel, the
   presence threshold and the movement noise level.
2. Walk in. Watch the white line leave its dashed baseline (top), ripples cross the waterfall (middle),
   and presence turn on (right).
3. Stand still and breathe normally for about 20 s: the spectrum grows a peak in the shaded band and a
   breathing rate appears.
4. Fall: a short sharp burst, then no movement while still in the room.

The **Change** toggle shows live minus baseline in percent instead of raw amplitude.
The **Sensitivity** slider only changes the waterfall colours.

## Serial format

Each line must contain a bracketed list of at least 128 integers (I/Q int8 pairs for 64 sub-carriers). The
default ESP-IDF `esp-csi` example output (`CSI_DATA,...,"[...]"`) works as is. The RSSI is read from the 4th
field when present.

## Where to tune (dsp.py)

| Constant | Meaning |
|---|---|
| `PEAK_SNR` | how much the breathing peak must beat the 0.1–0.6 Hz background |
| `MOVING_LVL`, `BURST_LVL`, `STILL_LVL` | movement levels (× empty-room noise) for moving, fall-sized burst, still again |
| `E_WIN_S` | seconds of signal averaged for presence |
| `--cal-seconds` | length of the calibration, after an 8 s settle |

Thresholds were checked against the built-in simulator only. Re-calibrate in the real room and expect to adjust them.
