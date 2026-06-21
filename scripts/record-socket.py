#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import time

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--socket", default="/tmp/clean-speech-daemon.sock")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    args = parser.parse_args()

    frame_bytes = int(args.sample_rate * 0.02) * 2
    pending = b""
    deadline = time.monotonic() + args.seconds

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(args.socket)
        metadata = b""
        while not metadata.endswith(b"\n"):
            metadata += client.recv(1)
        print(metadata.decode("utf-8").strip())

        with sf.SoundFile(args.output, mode="w", samplerate=args.sample_rate, channels=1, subtype="PCM_16") as wav:
            while time.monotonic() < deadline:
                chunk = client.recv(frame_bytes)
                if not chunk:
                    break
                pending += chunk
                while len(pending) >= frame_bytes:
                    raw = pending[:frame_bytes]
                    pending = pending[frame_bytes:]
                    wav.write(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
