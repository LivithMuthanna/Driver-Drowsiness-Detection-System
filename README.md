# AI Driver Drowsiness Detection System

A real-time, AI-powered driver drowsiness monitoring desktop application built with
**MediaPipe Face Mesh**, **OpenCV**, **PyQt5**, and **pygame** — no `dlib` and no
`shape_predictor_68_face_landmarks.dat` required.

## Features

- Live webcam feed with face bounding box, eye and mouth landmark overlay
- Eye Aspect Ratio (EAR) based blink and prolonged-eye-closure detection
- Mouth Aspect Ratio (MAR) based yawning detection
- Head pose estimation (solvePnP) → Left / Right / Up / Down / Center
- Driver status classification: Awake, Blinking, Sleepy, Yawning, Distracted, Drowsy, No Face Detected
- Looping audible alarm that starts on drowsiness and stops automatically on eye reopening
- Live FPS counter, session timer, blink counter, drowsiness counter, fatigue score (0–100%), detection confidence
- Premium dark-themed dashboard: header, large camera panel, stat sidebar, real-time scrolling event log
- Visual drowsy alert: red screen overlay, flashing status card, large "WAKE UP!" banner
- In-app Settings dialog: EAR/MAR thresholds, alarm volume/enable, camera index, sensitivity
- Automatic CSV session logging (timestamp, EAR, MAR, blink count, fatigue, status) + discrete event log CSV

## Project Structure

```
driver_drowsiness/
├── main.py                 # Entry point
├── gui.py                  # PyQt5 dashboard, camera thread, settings dialog
├── detector.py              # Drowsiness/blink/yawn/fatigue state machine
├── mediapipe_detector.py     # MediaPipe Face Mesh wrapper
├── alarm.py                  # pygame-based looping alarm manager
├── utils.py                  # EAR/MAR/head-pose math helpers
├── logger.py                  # CSV session + event logger
├── settings.py                 # Settings load/save (config.json)
├── config.json                  # Default thresholds & preferences
├── alarm.wav                     # Alarm sound (placeholder tone included)
├── icons/                         # (optional) UI icons
├── assets/                         # (optional) extra assets
└── requirements.txt
```

## Installation

1. Install **Python 3.12** (Windows recommended, also works on macOS/Linux with a webcam).
2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        (Windows)
   source venv/bin/activate     (macOS/Linux)
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run

```
python main.py
```

The dashboard window will open and automatically start the default webcam
(camera index `0`). Use the **⚙ Settings** button in the header to change the
camera index, detection thresholds, sensitivity, and alarm options — changes
apply immediately and are saved to `config.json`.

Session data is written continuously to `logs/session_log.csv`, and discrete
events (blinks, yawns, alerts) are written to `logs/session_log_events.csv`.

## Replacing the Alarm Sound

A simple placeholder tone is included as `alarm.wav`. Replace this file with
any `.wav` alarm sound of your choice — the application will pick it up on
the next launch.

## Troubleshooting

| Problem | Solution |
|---|---|
| `Could not open camera index 0` | Try a different index in Settings, or check that no other app is using the webcam. |
| Black/frozen video feed | Confirm your webcam drivers are installed and the camera isn't privacy-blocked in Windows settings. |
| No alarm sound | Confirm `alarm.wav` exists in the project root and `pygame.mixer` initialized without errors (check console output). On Linux, ensure an audio backend (ALSA/PulseAudio) is available. |
| `ImportError: No module named PyQt5` | Run `pip install -r requirements.txt` inside your active virtual environment. |
| High CPU usage | Lower camera resolution in `gui.py` (`CameraThread.run`) or set sensitivity to "low" in Settings. |
| EAR/MAR values look off for your face | Tune `EAR Threshold` and `MAR Threshold` in Settings while watching the live values in the sidebar. |

## Future Improvements

- Add a calibration wizard that auto-tunes EAR/MAR thresholds per user
- Multi-face support with driver-only ROI selection
- Cloud/Edge logging dashboard with historical trend charts
- Seatbelt and phone-usage distraction detection via an auxiliary model
- Night-mode IR camera support for low-light driving conditions
- Package as a standalone Windows `.exe` via PyInstaller
