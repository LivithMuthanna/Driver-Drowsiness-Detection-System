"""
utils.py
Utility/math functions used across the Driver Drowsiness Detection System.
Contains Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head pose
estimation helpers, and small geometry utilities.

No global mutable state is kept here -- every function is pure given its
inputs, which keeps the detection logic easy to test and reason about.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
import cv2

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark index groups
# ---------------------------------------------------------------------------
# These indices refer to the 468-point MediaPipe Face Mesh topology.

LEFT_EYE_EAR_IDX: List[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX: List[int] = [362, 385, 387, 263, 373, 380]

LEFT_EYE_OUTLINE: List[int] = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                                173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE: List[int] = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                                 466, 388, 387, 386, 385, 384, 398]

MOUTH_MAR_IDX: List[int] = [61, 291, 39, 181, 0, 17, 269, 405]
MOUTH_OUTLINE: List[int] = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                             291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                             95, 78]

# 6 points used for solvePnP based head-pose estimation
HEAD_POSE_IDX: List[int] = [1, 152, 33, 263, 61, 291]  # nose tip, chin,
# left eye corner, right eye corner, left mouth corner, right mouth corner

# Generic 3D model points (in an arbitrary unit, millimetres) corresponding
# to HEAD_POSE_IDX, used for solvePnP.
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)


def euclidean(p1: Point, p2: Point) -> float:
    """Return the Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmarks_to_points(landmarks, indices: List[int],
                         frame_w: int, frame_h: int) -> List[Point]:
    """
    Convert a subset of normalized MediaPipe landmarks into pixel-space
    (x, y) tuples for the given frame dimensions.
    """
    pts: List[Point] = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))
    return pts


def eye_aspect_ratio(eye_pts: List[Point]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) given 6 eye landmark points ordered as:
    [p1 (left corner), p2 (top-left), p3 (top-right),
     p4 (right corner), p5 (bottom-right), p6 (bottom-left)]

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye_pts) != 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_pts: List[Point]) -> float:
    """
    Compute the Mouth Aspect Ratio (MAR) given 8 mouth landmark points
    ordered as: [left, right, top1, bottom1, top_center, bottom_center,
    top2, bottom2] matching MOUTH_MAR_IDX.
    """
    if len(mouth_pts) != 8:
        return 0.0
    left, right, top1, bottom1, top_c, bottom_c, top2, bottom2 = mouth_pts
    vertical = (euclidean(top1, bottom1) +
                euclidean(top_c, bottom_c) +
                euclidean(top2, bottom2))
    horizontal = euclidean(left, right)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def estimate_head_pose(landmarks, frame_w: int, frame_h: int
                        ) -> Optional[Tuple[float, float, float]]:
    """
    Estimate head pose (pitch, yaw, roll in degrees) using solvePnP with
    6 facial landmark correspondences. Returns None on failure.
    """
    image_points = np.array(
        landmarks_to_points(landmarks, HEAD_POSE_IDX, frame_w, frame_h),
        dtype=np.float64
    )

    focal_length = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

    success, rotation_vec, _translation_vec = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [float(a[0]) for a in euler_angles]

    if pitch < -90:
        pitch = 180 + pitch
    elif pitch > 90:
        pitch = pitch - 180

    return pitch, yaw, roll


def classify_head_direction(pitch: float, yaw: float,
                             yaw_thresh: float, pitch_thresh: float) -> str:
    """Classify head direction into Center / Left / Right / Up / Down."""
    if yaw > yaw_thresh:
        return "Right"
    if yaw < -yaw_thresh:
        return "Left"
    if pitch > pitch_thresh:
        return "Down"
    if pitch < -pitch_thresh:
        return "Up"
    return "Center"


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high."""
    return max(low, min(high, value))


def moving_average(values: List[float], window: int = 5) -> float:
    """Return the simple moving average of the last `window` values."""
    if not values:
        return 0.0
    recent = values[-window:]
    return sum(recent) / len(recent)


def cv2_to_qimage_array(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB for Qt display."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
