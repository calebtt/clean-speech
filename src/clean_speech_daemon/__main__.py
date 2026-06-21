from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

import soundfile as sf

from .audio import frames_from_socket_bytes, list_input_devices, list_pulse_sources
from .config import DEFAULT_CONFIG_PATH, default_config_text, load_config
from .daemon import CleanSpeechDaemon


def main() -> int:
    parser = argparse.ArgumentParser(prog="clean-speech-daemon")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run the background audio cleanup service")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    subparsers.add_parser("devices", help="list usable input devices")
    subparsers.add_parser("pulse-sources", help="list Pulse/PipeWire sources and monitor states")

    config_parser = subparsers.add_parser("write-config", help="write a default configuration file")
    config_parser.add_argument("--path", type=Path, default=DEFAULT_CONFIG_PATH)
    config_parser.add_argument("--force", action="store_true")

    record_parser = subparsers.add_parser("record-socket", help="record cleaned socket output to a WAV file")
    record_parser.add_argument("output", type=Path)
    record_parser.add_argument("--socket", default="/tmp/clean-speech-daemon.sock")
    record_parser.add_argument("--seconds", type=float, default=10.0)
    record_parser.add_argument("--sample-rate", type=int, default=48_000)

    status_parser = subparsers.add_parser("status", help="print latest diagnostics status JSON")
    status_parser.add_argument("--path", type=Path, default=Path("/tmp/clean-speech-daemon-status.json"))
    status_parser.add_argument("--watch", action="store_true")

    args = parser.parse_args()
    if args.command in (None, "run"):
        config = load_config(getattr(args, "config", DEFAULT_CONFIG_PATH))
        CleanSpeechDaemon(config).run()
        return 0
    if args.command == "devices":
        for row in list_input_devices():
            print(row)
        return 0
    if args.command == "pulse-sources":
        for row in list_pulse_sources():
            print(row)
        return 0
    if args.command == "write-config":
        if args.path.exists() and not args.force:
            print(f"config already exists: {args.path}", file=sys.stderr)
            return 1
        args.path.parent.mkdir(parents=True, exist_ok=True)
        args.path.write_text(default_config_text(), encoding="utf-8")
        print(args.path)
        return 0
    if args.command == "record-socket":
        record_socket(args.socket, args.output, args.seconds, args.sample_rate)
        return 0
    if args.command == "status":
        print_status(args.path, args.watch)
        return 0
    parser.print_help()
    return 1


def record_socket(socket_path: str, output: Path, seconds: float, sample_rate: int) -> None:
    frame_samples = int(sample_rate * 0.02)
    frame_bytes = frame_samples * 2
    pending = b""
    deadline = time.monotonic() + seconds
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(socket_path)
        metadata = b""
        while not metadata.endswith(b"\n"):
            metadata += client.recv(1)
        with sf.SoundFile(output, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16") as wav:
            while time.monotonic() < deadline:
                chunk = client.recv(frame_bytes)
                if not chunk:
                    break
                pending += chunk
                frames = []
                while len(pending) >= frame_bytes:
                    frames.append(pending[:frame_bytes])
                    pending = pending[frame_bytes:]
                for frame in frames_from_socket_bytes(frames, frame_samples):
                    wav.write(frame)
    print(output)


def print_status(path: Path, watch: bool) -> None:
    while True:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if not watch:
            return
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
