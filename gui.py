"""
gui.py
PyQt5 desktop dashboard for the Driver Drowsiness Detection System.

Contains:
- CameraThread: grabs frames in a background QThread, runs detection,
  emits processed frame + metrics to the GUI thread.
- SettingsDialog: lets the user tune EAR/MAR thresholds, alarm volume,
  camera index, and detection sensitivity.
- MainWindow: the premium dark dashboard described in the spec — header,
  live camera feed, right sidebar stats, bottom event log, drowsy overlay.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QPushButton, QListWidget, QListWidgetItem,
    QDialog, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox, QComboBox,
    QFormLayout, QDialogButtonBox, QGraphicsDropShadowEffect, QSizePolicy,
)

import utils
from mediapipe_detector import MediaPipeFaceMeshDetector, FaceMeshResult
from detector import DrowsinessDetector, DetectionSettings, DriverStatus, FrameMetrics
from alarm import AlarmManager
from logger import CSVLogger
from settings import AppSettings


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLOR_BG = "#15181f"
COLOR_CARD = "#1f232d"
COLOR_CARD_BORDER = "#2b303c"
COLOR_BLUE = "#3b82f6"
COLOR_GREEN = "#22c55e"
COLOR_ORANGE = "#f59e0b"
COLOR_RED = "#ef4444"
COLOR_TEXT = "#e5e7eb"
COLOR_SUBTEXT = "#9ca3af"

STATUS_COLORS = {
    DriverStatus.NO_FACE: COLOR_SUBTEXT,
    DriverStatus.AWAKE: COLOR_GREEN,
    DriverStatus.BLINKING: COLOR_BLUE,
    DriverStatus.SLEEPY: COLOR_ORANGE,
    DriverStatus.YAWNING: COLOR_ORANGE,
    DriverStatus.DISTRACTED: COLOR_ORANGE,
    DriverStatus.DROWSY: COLOR_RED,
}


def make_shadow(blur: int = 20, color: str = "#000000", alpha: int = 160) -> QGraphicsDropShadowEffect:
    """Build a soft drop shadow effect for card widgets."""
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    c = QColor(color)
    c.setAlpha(alpha)
    effect.setColor(c)
    effect.setOffset(0, 4)
    return effect


# ---------------------------------------------------------------------------
# Camera + detection worker thread
# ---------------------------------------------------------------------------
class CameraThread(QThread):
    """
    Background thread that owns the webcam, runs MediaPipe + the
    DrowsinessDetector, draws landmark overlays, and emits results to the
    GUI thread via Qt signals (never touches widgets directly).
    """
    frame_ready = pyqtSignal(np.ndarray, object)   # (annotated BGR frame, FrameMetrics)
    fps_ready = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._running = False
        self._paused = False

        det_settings = DetectionSettings(
            ear_threshold=settings.ear_threshold,
            mar_threshold=settings.mar_threshold,
            ear_consec_frames=settings.ear_consec_frames,
            head_yaw_threshold=settings.head_yaw_threshold,
            head_pitch_threshold=settings.head_pitch_threshold,
            sensitivity=settings.detection_sensitivity,
        )
        det_settings.apply_sensitivity()
        self.detector = DrowsinessDetector(det_settings)
        self.face_mesh = MediaPipeFaceMeshDetector()

    def update_detection_settings(self, settings: AppSettings) -> None:
        """Apply new thresholds live, without restarting the camera."""
        self.settings = settings
        self.detector.settings.ear_threshold = settings.ear_threshold
        self.detector.settings.mar_threshold = settings.mar_threshold
        self.detector.settings.ear_consec_frames = settings.ear_consec_frames
        self.detector.settings.head_yaw_threshold = settings.head_yaw_threshold
        self.detector.settings.head_pitch_threshold = settings.head_pitch_threshold
        self.detector.settings.sensitivity = settings.detection_sensitivity
        self.detector.settings.apply_sensitivity()

    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_DSHOW
                                if sys.platform.startswith("win") else 0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(
                f"Could not open camera index {self.settings.camera_index}."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        prev_time = time.time()
        fps = 0.0

        while self._running:
            if self._paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            face_result = self.face_mesh.process(frame)
            metrics = self.detector.process(face_result)

            annotated = self._draw_overlays(frame, face_result, metrics)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = fps * 0.9 + inst_fps * 0.1

            self.frame_ready.emit(annotated, metrics)
            self.fps_ready.emit(fps)

        cap.release()
        self.face_mesh.close()

    def _draw_overlays(self, frame: np.ndarray, face_result: FaceMeshResult,
                        metrics: FrameMetrics) -> np.ndarray:
        """Draw bounding box, eye/mouth landmarks, and status text."""
        out = frame.copy()
        color_hex = STATUS_COLORS.get(metrics.status, COLOR_GREEN)
        bgr = self._hex_to_bgr(color_hex)

        if metrics.bbox:
            x1, y1, x2, y2 = metrics.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        for pts in (metrics.left_eye_pts, metrics.right_eye_pts):
            for p in pts:
                cv2.circle(out, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

        for p in metrics.mouth_pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 2, (255, 180, 0), -1)

        if metrics.status == DriverStatus.DROWSY:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]),
                          (0, 0, 255), -1)
            out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
            text = "WAKE UP!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2.2
            thickness = 5
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx = (out.shape[1] - tw) // 2
            ty = (out.shape[0] + th) // 2
            cv2.putText(out, text, (tx, ty), font, scale, (0, 0, 255),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(out, text, (tx, ty), font, scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

        return out

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------
class SettingsDialog(QDialog):
    """Modal dialog for adjusting detection / alarm / camera settings."""

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)
        self.settings = settings

        layout = QFormLayout(self)

        self.ear_spin = QDoubleSpinBox()
        self.ear_spin.setRange(0.05, 0.45)
        self.ear_spin.setSingleStep(0.01)
        self.ear_spin.setValue(settings.ear_threshold)
        layout.addRow("EAR Threshold:", self.ear_spin)

        self.mar_spin = QDoubleSpinBox()
        self.mar_spin.setRange(0.2, 1.2)
        self.mar_spin.setSingleStep(0.01)
        self.mar_spin.setValue(settings.mar_threshold)
        layout.addRow("MAR Threshold:", self.mar_spin)

        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["low", "medium", "high"])
        self.sensitivity_combo.setCurrentText(settings.detection_sensitivity)
        layout.addRow("Detection Sensitivity:", self.sensitivity_combo)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(settings.alarm_volume * 100))
        layout.addRow("Alarm Volume:", self.volume_slider)

        self.alarm_checkbox = QCheckBox("Enable Alarm")
        self.alarm_checkbox.setChecked(settings.alarm_enabled)
        layout.addRow(self.alarm_checkbox)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(settings.camera_index)
        layout.addRow("Camera Index:", self.camera_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.setStyleSheet(f"""
            QDialog {{ background-color: {COLOR_CARD}; color: {COLOR_TEXT}; }}
            QLabel {{ color: {COLOR_TEXT}; }}
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background-color: {COLOR_BG}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_CARD_BORDER}; border-radius: 4px; padding: 3px;
            }}
            QCheckBox {{ color: {COLOR_TEXT}; }}
            QPushButton {{
                background-color: {COLOR_BLUE}; color: white; border-radius: 4px;
                padding: 6px 14px;
            }}
        """)

    def get_updated_settings(self) -> AppSettings:
        """Return a new AppSettings object reflecting the dialog's values."""
        self.settings.ear_threshold = self.ear_spin.value()
        self.settings.mar_threshold = self.mar_spin.value()
        self.settings.detection_sensitivity = self.sensitivity_combo.currentText()
        self.settings.alarm_volume = self.volume_slider.value() / 100.0
        self.settings.alarm_enabled = self.alarm_checkbox.isChecked()
        self.settings.camera_index = self.camera_spin.value()
        return self.settings


# ---------------------------------------------------------------------------
# Reusable stat card widget
# ---------------------------------------------------------------------------
class StatCard(QFrame):
    """A rounded card showing a label and a dynamic value, used in the sidebar."""

    def __init__(self, title: str, initial_value: str = "--") -> None:
        super().__init__()
        self.setObjectName("statCard")
        self.setGraphicsEffect(make_shadow())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("statTitle")
        self.value_label = QLabel(initial_value)
        self.value_label.setObjectName("statValue")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str, color: Optional[str] = None) -> None:
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """Top-level dashboard window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Driver Drowsiness Detection System")
        self.resize(1400, 860)

        self.app_settings = AppSettings.load()
        self.alarm = AlarmManager(
            sound_path="alarm.wav",
            volume=self.app_settings.alarm_volume,
            enabled=self.app_settings.alarm_enabled,
        )
        self.csv_logger = CSVLogger(csv_path=self.app_settings.log_csv_path)

        self._flash_state = False
        self._last_status: Optional[DriverStatus] = None
        self._session_start = time.time()

        self._build_ui()
        self._apply_stylesheet()
        self._start_camera()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock_and_session)
        self.clock_timer.start(1000)

        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._toggle_flash)
        self.flash_timer.start(500)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_camera_panel(), stretch=3)
        body.addWidget(self._build_sidebar(), stretch=1)
        root.addLayout(body, stretch=5)

        root.addWidget(self._build_event_log(), stretch=2)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("headerCard")
        header.setGraphicsEffect(make_shadow())
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 12, 20, 12)

        title = QLabel("AI DRIVER DROWSINESS DETECTION SYSTEM")
        title.setObjectName("headerTitle")
        layout.addWidget(title)
        layout.addStretch()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("headerStat")
        layout.addWidget(self.fps_label)

        self.time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.time_label.setObjectName("headerStat")
        layout.addWidget(self.time_label)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setObjectName("settingsButton")
        settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(settings_btn)

        return header

    def _build_camera_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("cameraCard")
        frame.setGraphicsEffect(make_shadow())
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)

        self.camera_label = QLabel("Initializing camera...")
        self.camera_label.setObjectName("cameraFeed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.camera_label)
        return frame

    def _build_sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("sidebar")
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status_card = StatCard("DRIVER STATUS", "No Face Detected")
        self.ear_card = StatCard("EAR VALUE", "0.000")
        self.mar_card = StatCard("MAR VALUE", "0.000")
        self.blink_card = StatCard("BLINK COUNT", "0")
        self.fatigue_card = StatCard("FATIGUE %", "0%")
        self.session_card = StatCard("SESSION TIME", "00:00:00")
        self.confidence_card = StatCard("DETECTION CONFIDENCE", "0%")
        self.alarm_card = StatCard("ALARM STATUS", "Silent")
        self.face_card = StatCard("FACE DETECTED", "No")
        self.head_card = StatCard("HEAD DIRECTION", "Center")

        for card in (self.status_card, self.ear_card, self.mar_card,
                     self.blink_card, self.fatigue_card, self.session_card,
                     self.confidence_card, self.alarm_card, self.face_card,
                     self.head_card):
            layout.addWidget(card)

        layout.addStretch()
        return frame

    def _build_event_log(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("logCard")
        frame.setGraphicsEffect(make_shadow())
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)

        title = QLabel("REAL-TIME EVENT LOG")
        title.setObjectName("logTitle")
        layout.addWidget(title)

        self.event_list = QListWidget()
        self.event_list.setObjectName("eventList")
        layout.addWidget(self.event_list)
        return frame

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def _start_camera(self) -> None:
        self.camera_thread = CameraThread(self.app_settings)
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.fps_ready.connect(self._on_fps_ready)
        self.camera_thread.error_occurred.connect(self._on_camera_error)
        self.camera_thread.start()

    def _restart_camera(self) -> None:
        if hasattr(self, "camera_thread"):
            self.camera_thread.stop()
        self._start_camera()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_camera_error(self, message: str) -> None:
        self.camera_label.setText(f"⚠ {message}")
        self._log_event(f"ERROR: {message}")

    def _on_fps_ready(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_frame_ready(self, frame: np.ndarray, metrics: FrameMetrics) -> None:
        self._render_frame(frame)
        self._update_sidebar(metrics)
        self._handle_alarm_and_effects(metrics)
        self._handle_event_transitions(metrics)
        self.csv_logger.log(metrics)

    def _render_frame(self, frame: np.ndarray) -> None:
        rgb = utils.cv2_to_qimage_array(frame)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _update_sidebar(self, metrics: FrameMetrics) -> None:
        color = STATUS_COLORS.get(metrics.status, COLOR_GREEN)
        self.status_card.set_value(metrics.status.value, color)
        self.ear_card.set_value(f"{metrics.ear:.3f}")
        self.mar_card.set_value(f"{metrics.mar:.3f}")
        self.blink_card.set_value(str(metrics.blink_count))
        self.fatigue_card.set_value(
            f"{metrics.fatigue_score:.0f}%",
            COLOR_RED if metrics.fatigue_score > 70 else
            (COLOR_ORANGE if metrics.fatigue_score > 40 else COLOR_GREEN)
        )
        self.confidence_card.set_value(f"{metrics.confidence:.0f}%")
        self.face_card.set_value(
            "Yes" if metrics.face_found else "No",
            COLOR_GREEN if metrics.face_found else COLOR_RED
        )
        self.head_card.set_value(
            metrics.head_direction,
            COLOR_GREEN if metrics.head_direction == "Center" else COLOR_ORANGE
        )

    def _handle_alarm_and_effects(self, metrics: FrameMetrics) -> None:
        if metrics.status == DriverStatus.DROWSY:
            self.alarm.start()
            self.alarm_card.set_value("ALARM ACTIVE", COLOR_RED)
        else:
            self.alarm.stop()
            self.alarm_card.set_value("Silent", COLOR_SUBTEXT)

        if metrics.status == DriverStatus.DROWSY and self._flash_state:
            self.status_card.setStyleSheet(
                f"#statCard {{ background-color: {COLOR_RED}; border-radius: 10px; }}"
            )
        else:
            self.status_card.setStyleSheet("")

    def _toggle_flash(self) -> None:
        self._flash_state = not self._flash_state

    def _handle_event_transitions(self, metrics: FrameMetrics) -> None:
        if metrics.status == self._last_status:
            return
        self._last_status = metrics.status

        messages = {
            DriverStatus.BLINKING: "Blink Detected",
            DriverStatus.YAWNING: "Yawning Detected",
            DriverStatus.DROWSY: "Drowsiness Alert!",
            DriverStatus.DISTRACTED: f"Driver Looking {metrics.head_direction}",
            DriverStatus.SLEEPY: "Driver Appears Sleepy",
            DriverStatus.NO_FACE: "Face Lost",
            DriverStatus.AWAKE: "Driver Awake",
        }
        message = messages.get(metrics.status)
        if message:
            self._log_event(message)
            self.csv_logger.log_event(message)

    def _log_event(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {text}")
        if "Alert" in text or "Drowsy" in text:
            item.setForeground(QColor(COLOR_RED))
        elif "Yawning" in text or "Sleepy" in text or "Looking" in text:
            item.setForeground(QColor(COLOR_ORANGE))
        else:
            item.setForeground(QColor(COLOR_TEXT))
        self.event_list.insertItem(0, item)
        while self.event_list.count() > 200:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _tick_clock_and_session(self) -> None:
        self.time_label.setText(datetime.now().strftime("%H:%M:%S"))
        elapsed = int(time.time() - self._session_start)
        hh, rem = divmod(elapsed, 3600)
        mm, ss = divmod(rem, 60)
        self.session_card.set_value(f"{hh:02d}:{mm:02d}:{ss:02d}")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.app_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.app_settings = dialog.get_updated_settings()
            self.app_settings.save()
            self.alarm.set_volume(self.app_settings.alarm_volume)
            self.alarm.set_enabled(self.app_settings.alarm_enabled)
            self.camera_thread.update_detection_settings(self.app_settings)
            self._log_event("Settings updated")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if hasattr(self, "camera_thread"):
            self.camera_thread.stop()
        self.alarm.shutdown()
        event.accept()

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------
    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {COLOR_BG}; }}
            QWidget {{ font-family: 'Segoe UI', Arial, sans-serif; }}

            #headerCard, #cameraCard, #logCard, #statCard {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_CARD_BORDER};
                border-radius: 12px;
            }}

            #headerTitle {{
                color: {COLOR_TEXT}; font-size: 18px; font-weight: 700;
                letter-spacing: 1px;
            }}
            #headerStat {{
                color: {COLOR_SUBTEXT}; font-size: 13px; font-weight: 600;
                padding: 0 14px;
            }}
            #settingsButton {{
                background-color: {COLOR_BLUE}; color: white;
                border-radius: 8px; padding: 6px 16px; font-weight: 600;
            }}
            #settingsButton:hover {{ background-color: #2563eb; }}

            #cameraFeed {{
                background-color: #0b0d12; border-radius: 8px;
                color: {COLOR_SUBTEXT}; font-size: 14px;
            }}

            #statTitle {{
                color: {COLOR_SUBTEXT}; font-size: 11px; font-weight: 700;
                letter-spacing: 0.5px;
            }}
            #statValue {{
                color: {COLOR_TEXT}; font-size: 20px; font-weight: 700;
            }}

            #logTitle {{
                color: {COLOR_SUBTEXT}; font-size: 12px; font-weight: 700;
                letter-spacing: 1px; padding-bottom: 4px;
            }}
            #eventList {{
                background-color: #11141a; border: none; border-radius: 8px;
                color: {COLOR_TEXT}; font-size: 13px;
            }}
            #eventList::item {{ padding: 4px 6px; }}
        """)


def run_app() -> None:
    """Entry point used by main.py to launch the Qt application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
