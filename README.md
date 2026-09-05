# Real-Time Hand Gesture Recognition

A real-time computer vision application that detects hand landmarks from a webcam feed and recognizes common hand gestures using landmark-based rules.

The project uses **OpenCV** for webcam capture and visualization, and **MediaPipe Hand Landmarker** to detect 21 landmarks on each hand.

## Features

- Real-time webcam hand tracking
- Detection of up to two hands
- 21-point hand landmark visualization
- Left/right hand identification
- Rule-based gesture recognition
- Gesture labels displayed alongside each detected hand

### Supported Gestures

| Gesture | Prediction |
| --- | --- |
| ✋ Open Palm | `OPEN PALM` |
| ✊ Fist | `FIST` |
| ✌️ Peace | `PEACE` |
| ☝️ Pointing | `POINTING` |
| 👍 Thumbs Up | `THUMBS UP` |

## How It Works

The application follows this pipeline:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Landmark-Based Gesture Rules
   ↓
Gesture Prediction
```

MediaPipe provides 21 normalized landmark coordinates for each detected hand.

For example, the index fingertip is landmark `8`, while its PIP joint is landmark `6`. For an upright hand, the application can approximately determine whether the index finger is raised by comparing their vertical positions:

```python
index_up = landmarks[8].y < landmarks[6].y
```

The states of multiple fingers are then combined to recognize gestures such as Peace, Open Palm, and Fist.

## Tech Stack

- Python
- OpenCV
- MediaPipe

## Project Structure

```text
real-time-hand-gesture-recognition/
├── README.md
├── requirements.txt
├── download_model.py
├── .gitignore
├── assets/
├── models/
│   └── hand_landmarker.task
└── src/
    ├── main.py
    └── gesture_recognizer.py
```

The MediaPipe model file is downloaded locally and is not stored in the repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/real-time-hand-gesture-recognition.git
cd real-time-hand-gesture-recognition
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

```bash
python download_model.py
```

### 5. Run the application

```bash
python src/main.py
```

Press `q` to exit the webcam window.

That's it! Once you are all set up, have fun trying out the different gestures!

## Limitations

Gesture recognition currently uses simple geometric rules based on landmark positions rather than a trained gesture classification model.

Recognition may therefore be affected by:

- Hand orientation
- Camera angle
- Partial occlusion
- Lighting conditions
- Ambiguous finger positions

## Future Improvements

- Improve gesture recognition using landmark angles and distances
- Add temporal smoothing for more stable predictions
- Support additional gestures
- Add gesture-controlled actions
- Explore a trained gesture classifier using landmark features

## Acknowledgements

Hand landmark detection is powered by Google's MediaPipe Hand Landmarker.
