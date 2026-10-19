"""Start the local demo with `python -m demo`."""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local TM band demo.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    os.environ["TM_DEMO_DEVICE"] = args.device
    os.environ["TM_DEMO_CPU_THREADS"] = str(args.cpu_threads)
    uvicorn.run("demo.app:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
