"""
detector.py
Core detection logic: takes per-frame face-mesh landmarks and turns them
into EAR / MAR values, blink counts, drowsiness state, fatigue score,
head-pose direction, and an overall DriverStatus.

This module has NO GUI or camera dependencies, which keeps it independently
testable. It is driven by `mediapipe_detector.FaceMeshResult` objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

import utils
from mediapipe_detector import FaceMeshResult


class DriverStatus(Enum):
    NO_FACE = "No Face Detected"
    AWAKE = "Awake"
    BLINKING = "Blinking"
    SLEEPY = "Sleepy"
    YAWNING = "Yawning"
    DISTRACTED = "Distracted"
    DROWSY = "Drowsy"


@dataclass
class DetectionSettings:
    """Mutable, user-configurable detection thresholds."""
    ear_threshold: float = 0.21
    mar_threshold: float = 0.6
    ear_consec_frames: int = 18           # frames eyes must stay closed
    head_yaw_threshold: float = 18.0
    head_pitch_threshold: float = 15.0
    sensitivity: str = "medium"           # low / medium / high

    def apply_sensitivity(self) -> None:
        """Adjust thresholds based on a simple sensitivity preset."""
        if self.sensitivity == "low":
            self.ear_consec_frames = 25
        elif self.sensitivity == "high":
            self.ear_consec_frames = 12
        else:
            self.ear_consec_frames = 18


@dataclass
class FrameMetrics:
    """All derived values for a single processed frame."""
    timestamp: float
    face_found: bool
    ear: float = 0.0
    mar: float = 0.0
    blink_count: int = 0
    drowsy_count: int = 0
    fatigue_score: float = 0.0
    status: DriverStatus = DriverStatus.NO_FACE
    head_direction: str = "Center"
    confidence: float = 0.0
    eyes_closed_frames: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    left_eye_pts: List[Tuple[float, float]] = field(default_factory=list)
    right_eye_pts: List[Tuple[float, float]] = field(default_factory=list)
    mouth_pts: List[Tuple[float, float]] = field(default_factory=list)


class DrowsinessDetector:
    """
    Stateful per-session detector. Call `process(face_result)` once per
    camera frame; it returns a FrameMetrics snapshot describing the
    driver's current state.
    """

    def __init__(self, settings: DetectionSettings) -> None:
        self.settings = settings

        # Rolling state
        self._closed_frames = 0
        self._blink_count = 0
        self._drowsy_events = 0
        self._was_eye_closed = False
        self._yawn_in_progress = False

        self._ear_history: List[float] = []
        self._fatigue_score = 0.0

        self._session_start = time.time()
        self._last_blink_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, face_result: FaceMeshResult) -> FrameMetrics:
        now = time.time()

        if not face_result.found:
            # Decay fatigue slowly when no face is visible, but don't reset.
            self._fatigue_score = utils.clamp(self._fatigue_score - 0.2, 0, 100)
            return FrameMetrics(
                timestamp=now,
                face_found=False,
                status=DriverStatus.NO_FACE,
                fatigue_score=self._fatigue_score,
                blink_count=self._blink_count,
                drowsy_count=self._drowsy_events,
            )

        lm = face_result.landmarks
        w, h = face_result.frame_w, face_result.frame_h

        left_eye = utils.landmarks_to_points(lm, utils.LEFT_EYE_EAR_IDX, w, h)
        right_eye = utils.landmarks_to_points(lm, utils.RIGHT_EYE_EAR_IDX, w, h)
        mouth_pts = utils.landmarks_to_points(lm, utils.MOUTH_MAR_IDX, w, h)

        left_ear = utils.eye_aspect_ratio(left_eye)
        right_ear = utils.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = utils.mouth_aspect_ratio(mouth_pts)

        self._ear_history.append(ear)
        if len(self._ear_history) > 60:
            self._ear_history.pop(0)

        # --- Blink / eye closure tracking -----------------------------
        eyes_closed = ear < self.settings.ear_threshold
        if eyes_closed:
            self._closed_frames += 1
        else:
            if self._was_eye_closed and 2 <= self._closed_frames < self.settings.ear_consec_frames:
                # A short closure that re-opened counts as a blink.
                self._blink_count += 1
                self._last_blink_time = now
            self._closed_frames = 0
        self._was_eye_closed = eyes_closed

        is_drowsy_eyes = self._closed_frames >= self.settings.ear_consec_frames
        if is_drowsy_eyes and self._closed_frames == self.settings.ear_consec_frames:
            self._drowsy_events += 1

        # --- Yawn tracking ----------------------------------------------
        is_yawning = mar > self.settings.mar_threshold
        if is_yawning and not self._yawn_in_progress:
            self._yawn_in_progress = True
        elif not is_yawning:
            self._yawn_in_progress = False

        # --- Head pose ----------------------------------------------------
        head_direction = "Center"
        pose = utils.estimate_head_pose(lm, w, h)
        if pose is not None:
            pitch, yaw, _roll = pose
            head_direction = utils.classify_head_direction(
                pitch, yaw,
                self.settings.head_yaw_threshold,
                self.settings.head_pitch_threshold,
            )

        # --- Fatigue score (0-100) ----------------------------------------
        # Combine: proportion of recent frames with low EAR + drowsy events
        # + yawn frequency, smoothed over time for a stable gauge reading.
        avg_recent_ear = utils.moving_average(self._ear_history, window=30)
        ear_component = utils.clamp(
            (self.settings.ear_threshold + 0.08 - avg_recent_ear) / 0.08 * 60,
            0, 60
        )
        closure_component = utils.clamp(
            (self._closed_frames / max(self.settings.ear_consec_frames, 1)) * 30,
            0, 30
        )
        yawn_component = 10.0 if is_yawning else 0.0
        target_fatigue = ear_component + closure_component + yawn_component
        # Smooth toward target to avoid jitter
        self._fatigue_score += (target_fatigue - self._fatigue_score) * 0.15
        self._fatigue_score = utils.clamp(self._fatigue_score, 0, 100)

        # --- Overall status -------------------------------------------------
        if is_drowsy_eyes:
            status = DriverStatus.DROWSY
        elif is_yawning:
            status = DriverStatus.YAWNING
        elif head_direction != "Center":
            status = DriverStatus.DISTRACTED
        elif self._fatigue_score > 55:
            status = DriverStatus.SLEEPY
        elif eyes_closed:
            status = DriverStatus.BLINKING
        else:
            status = DriverStatus.AWAKE

        # --- Bounding box from landmark extremes --------------------------
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        # --- Detection confidence (heuristic, based on landmark spread) ---
        confidence = 0.0
        if bbox:
            box_w = max(bbox[2] - bbox[0], 1)
            confidence = utils.clamp(box_w / w * 220, 40, 99)

        return FrameMetrics(
            timestamp=now,
            face_found=True,
            ear=ear,
            mar=mar,
            blink_count=self._blink_count,
            drowsy_count=self._drowsy_events,
            fatigue_score=self._fatigue_score,
            status=status,
            head_direction=head_direction,
            confidence=confidence,
            eyes_closed_frames=self._closed_frames,
            bbox=bbox,
            left_eye_pts=left_eye,
            right_eye_pts=right_eye,
            mouth_pts=mouth_pts,
        )

    def session_seconds(self) -> float:
        """Elapsed seconds since this detector was instantiated."""
        return time.time() - self._session_start

    def reset_session(self) -> None:
        """Reset all counters and timers for a fresh session."""
        self._closed_frames = 0
        self._blink_count = 0
        self._drowsy_events = 0
        self._was_eye_closed = False
        self._yawn_in_progress = False
        self._ear_history.clear()
        self._fatigue_score = 0.0
        self._session_start = time.time()
        self._last_blink_time = 0.0
