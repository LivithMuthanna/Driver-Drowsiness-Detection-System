"""
logger.py
Writes per-frame (throttled) detection metrics to a CSV file for later
analysis: timestamp, EAR, MAR, blink count, fatigue score, status.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Optional

from detector import FrameMetrics


class CSVLogger:
    """
    Appends rows to a CSV log file. To avoid flooding the disk with one row
    per frame (~30/sec), writes are throttled to `min_interval_seconds`.
    """

    HEADERS = ["timestamp", "ear", "mar", "blink_count",
               "fatigue_score", "status"]

    def __init__(self, csv_path: str = "logs/session_log.csv",
                 min_interval_seconds: float = 1.0) -> None:
        self.csv_path = csv_path
        self.min_interval_seconds = min_interval_seconds
        self._last_write_time = 0.0
        self._ensure_file()

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log(self, metrics: FrameMetrics, force: bool = False) -> None:
        """Write a row for the given metrics, respecting the throttle."""
        now = time.time()
        if not force and (now - self._last_write_time) < self.min_interval_seconds:
            return
        self._last_write_time = now

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(metrics.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                f"{metrics.ear:.4f}",
                f"{metrics.mar:.4f}",
                metrics.blink_count,
                f"{metrics.fatigue_score:.2f}",
                metrics.status.value,
            ])

    def log_event(self, event_text: str,
                   events_csv_path: Optional[str] = None) -> None:
        """Optionally log discrete events (e.g. 'Yawning Detected') separately."""
        path = events_csv_path or self.csv_path.replace(".csv", "_events.csv")
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        is_new = not os.path.exists(path)
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "event"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event_text,
            ])
