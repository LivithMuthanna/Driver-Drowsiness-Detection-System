"""
alarm.py
Handles the audible alarm using pygame's mixer. Designed so the alarm can be
started/stopped repeatedly without re-loading the file each time, and so
volume / enable state can be changed at runtime from Settings.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import pygame
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pygame = None


class AlarmManager:
    """
    Wraps a looped alarm sound. Call `start()` while the driver is drowsy
    and `stop()` as soon as the eyes reopen / drowsiness ends.
    """

    def __init__(self, sound_path: str = "alarm.wav",
                 volume: float = 0.8, enabled: bool = True) -> None:
        self.sound_path = sound_path
        self.volume = volume
        self.enabled = enabled
        self._playing = False
        self._sound: Optional["pygame.mixer.Sound"] = None
        self._channel: Optional["pygame.mixer.Channel"] = None
        self._initialized = False

        self._init_mixer()

    def _init_mixer(self) -> None:
        if pygame is None:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            if os.path.exists(self.sound_path):
                self._sound = pygame.mixer.Sound(self.sound_path)
                self._sound.set_volume(self.volume)
            self._initialized = True
        except Exception as exc:  # pragma: no cover
            print(f"[AlarmManager] Failed to initialize audio: {exc}")
            self._initialized = False

    def set_volume(self, volume: float) -> None:
        """Update alarm volume (0.0 - 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self._sound is not None:
            self._sound.set_volume(self.volume)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the alarm. Stops playback if disabled."""
        self.enabled = enabled
        if not enabled:
            self.stop()

    def start(self) -> None:
        """Start looping the alarm sound if not already playing."""
        if not self.enabled or self._playing:
            return
        if self._sound is not None:
            self._channel = self._sound.play(loops=-1)
        else:
            # Fallback: beep via console bell if no sound file is available.
            print("\a[AlarmManager] alarm.wav not found - using fallback beep")
        self._playing = True

    def stop(self) -> None:
        """Stop the alarm sound if it is currently playing."""
        if not self._playing:
            return
        if self._channel is not None:
            self._channel.stop()
            self._channel = None
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def shutdown(self) -> None:
        """Clean up mixer resources on application exit."""
        self.stop()
        if pygame is not None and pygame.mixer.get_init():
            pygame.mixer.quit()
