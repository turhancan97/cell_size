"""CLI entry-point that launches the Gradio demo app."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Cell Size Estimator Gradio demo",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    demo_script = Path(__file__).resolve().parents[2] / "demo" / "app.py"
    if not demo_script.is_file():
        print(f"Error: demo app not found at {demo_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(demo_script)]
    if args.share:
        cmd.append("--share")
    cmd.extend(["--server-name", args.server_name])
    cmd.extend(["--server-port", str(args.server_port)])

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
