"""
mediapipe_detector.py
Thin wrapper around MediaPipe Face Mesh that handles model lifecycle and
returns raw landmark results for a given BGR frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import cv2
import mediapipe as mp


@dataclass
class FaceMeshResult:
    """Container for a single-frame face mesh detection result."""
    found: bool
    landmarks: Optional[Any] = None     # normalized landmark list (468 pts)
    frame_w: int = 0
    frame_h: int = 0


class MediaPipeFaceMeshDetector:
    """
    Wraps mediapipe.solutions.face_mesh.FaceMesh for convenient per-frame
    inference. Designed to be created once and reused across frames for
    performance (the underlying graph keeps temporal state internally).
    """

    def __init__(self,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr) -> FaceMeshResult:
        """
        Run face mesh inference on a single BGR frame.
        Returns a FaceMeshResult with normalized landmarks if a face
        was found, else found=False.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return FaceMeshResult(found=False, frame_w=w, frame_h=h)

        face_landmarks = results.multi_face_landmarks[0].landmark
        return FaceMeshResult(found=True, landmarks=face_landmarks,
                               frame_w=w, frame_h=h)

    def close(self) -> None:
        """Release the underlying MediaPipe graph resources."""
        self._face_mesh.close()

    def __enter__(self) -> "MediaPipeFaceMeshDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
