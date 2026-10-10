"""WiSense server: reads the ESP32 CSI stream (or a simulator), runs the DSP, streams to the browser.

    python server.py --port COM5          # Windows
    python server.py --port /dev/ttyUSB0  # Linux / macOS: /dev/cu.usbserial-xxxx
    python server.py                      # auto-detect the serial port
    python server.py --sim                # no hardware: simulated room
Then open http://127.0.0.1:8000
"""
import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from dsp import Engine
from sources import SerialSource, Simulator

HERE = Path(__file__).parent


def parse_args():
    ap = argparse.ArgumentParser(description="WiSense live CSI dashboard")
    ap.add_argument("--port", default="auto", help="serial port of the receiver ESP32 (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--sim", action="store_true", help="use the built-in simulator instead of hardware")
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--cal-seconds", type=float, default=20.0, help="empty-room calibration length (after an 8 s settle)")
    ap.add_argument("--auto-cal", action="store_true", help="calibrate automatically at start (room must be empty)")
    return ap.parse_args()


ARGS = parse_args() if __name__ == "__main__" else argparse.Namespace(
    port="auto", baud=921600, sim=True, host="127.0.0.1", http_port=8000, cal_seconds=20.0, auto_cal=True,
    list_ports=False)

clients: set = set()
engine: Engine = None
source = None
queue: asyncio.Queue = None


@asynccontextmanager
async def lifespan(app):
    global engine, source, queue
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=600)

    def put(s):
        if not queue.full():
            queue.put_nowait(s)

    def emit(msg):
        loop.call_soon_threadsafe(put, json.dumps(msg, separators=(",", ":")))

    def log(text):
        print("[wisense]", text, flush=True)
        emit({"type": "log", "msg": text})

    engine = Engine(emit, cal_collect_s=ARGS.cal_seconds)
    source = Simulator(engine, log) if ARGS.sim else SerialSource(engine, log, ARGS.port, ARGS.baud)
    source.start()
    if ARGS.auto_cal or ARGS.sim:
        engine.start_calibration()
    task = asyncio.create_task(broadcaster())
    yield
    task.cancel()
    source.stop()


app = FastAPI(lifespan=lifespan)


async def broadcaster():
    while True:
        msg = await queue.get()
        for ws in list(clients):
            try:
                await ws.send_text(msg)
            except Exception:
                clients.discard(ws)


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    hello = engine.hello()
    hello["source"] = "sim" if ARGS.sim else "serial"
    hello["source_label"] = source.label
    await ws.send_text(json.dumps(hello))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "calibrate":
                engine.start_calibration()
            elif cmd == "sim" and ARGS.sim:
                source.set_mode(msg.get("mode", "empty"))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if __name__ == "__main__":
    if ARGS.list_ports:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(p.device, "-", p.description)
        raise SystemExit
    print(f"[wisense] open http://{ARGS.host}:{ARGS.http_port}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.http_port, log_level="warning")
