"""CLI entry-point to launch the Streamlit embedding explorer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Cell Size Streamlit embedding explorer",
    )
    parser.add_argument("--server-name", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--server-port", type=int, default=8501, help="Server port")
    parser.add_argument("--headless", action="store_true", help="Force headless Streamlit mode")
    parser.add_argument(
        "--file-watcher-type",
        default="none",
        choices=["auto", "watchdog", "poll", "none"],
        help="Streamlit file watcher mode (default: none for stability on large data trees)",
    )
    args = parser.parse_args()

    app_script = Path(__file__).resolve().parents[2] / "demo" / "streamlit_embedding_app.py"
    if not app_script.is_file():
        print(f"Error: Streamlit app not found at {app_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_script),
        "--server.address",
        args.server_name,
        "--server.port",
        str(args.server_port),
        "--server.fileWatcherType",
        args.file_watcher_type,
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
