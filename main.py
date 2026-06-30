"""
main.py
Entry point for the AI Driver Drowsiness Detection System.

Run with:
    python main.py
"""

from __future__ import annotations

import sys
import os


def _ensure_working_directory() -> None:
    """Make relative paths (config.json, alarm.wav, logs/) resolve correctly
    regardless of where the script is invoked from."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)


def main() -> None:
    _ensure_working_directory()
    try:
        from gui import run_app
    except ImportError as exc:
        print("Missing dependency:", exc)
        print("Install requirements with: pip install -r requirements.txt")
        sys.exit(1)

    run_app()


if __name__ == "__main__":
    main()
