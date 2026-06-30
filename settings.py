"""
settings.py
Loads and saves application settings (detection thresholds, alarm
preferences, camera index) to/from config.json. Provides a single
AppSettings dataclass used by both detector.py and gui.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class AppSettings:
    ear_threshold: float = 0.21
    mar_threshold: float = 0.6
    ear_consec_frames: int = 18
    alarm_volume: float = 0.8
    alarm_enabled: bool = True
    camera_index: int = 0
    detection_sensitivity: str = "medium"
    head_yaw_threshold: float = 18.0
    head_pitch_threshold: float = 15.0
    log_csv_path: str = "logs/session_log.csv"

    @staticmethod
    def load(path: str = DEFAULT_CONFIG_PATH) -> "AppSettings":
        """Load settings from a JSON file, falling back to defaults for any
        missing keys, and creating the file with defaults if absent."""
        if not os.path.exists(path):
            settings = AppSettings()
            settings.save(path)
            return settings

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return AppSettings()

        defaults = asdict(AppSettings())
        defaults.update({k: v for k, v in data.items() if k in defaults})
        return AppSettings(**defaults)

    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        """Persist current settings to a JSON file."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)
