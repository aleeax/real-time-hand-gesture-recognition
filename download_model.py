from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading MediaPipe Hand Landmarker model...")

    urlretrieve(MODEL_URL, MODEL_PATH)

    print(f"Model downloaded successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()