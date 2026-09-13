from pathlib import Path
import sys

import cv2
import joblib
import mediapipe as mp
import mediapipe.tasks as tasks
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HAND_MODEL_PATH = (
    PROJECT_ROOT / "models" / "hand_landmarker.task"
)

CLASSIFIER_PATH = (
    PROJECT_ROOT / "models" / "gesture_classifier.joblib"
)


def normalize_landmarks(hand_landmarks):
    """Convert landmarks to coordinates relative to the wrist."""

    wrist = hand_landmarks[0]
    features = []

    for landmark in hand_landmarks:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    return features


def create_feature_names():
    """Create the same feature names used during training."""

    feature_names = []

    for i in range(21):
        feature_names.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}",
        ])

    return feature_names


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/predict_image.py data/test_images/test_1.jpg")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    # Load image
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: Could not read image.")
        return

    # Set up MediaPipe
    base_options = tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    )

    options = tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
    )

    landmarker = tasks.vision.HandLandmarker.create_from_options(
        options
    )

    # OpenCV BGR -> RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image,
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        print("No hand detected.")
        landmarker.close()
        return

    print("Hand detected: Yes")

    # Extract exactly the same features used during training
    features = normalize_landmarks(
        result.hand_landmarks[0]
    )

    feature_names = create_feature_names()

    X_new = pd.DataFrame(
        [features],
        columns=feature_names
    )

    # Load our trained gesture classifier
    classifier = joblib.load(CLASSIFIER_PATH)

    prediction = classifier.predict(X_new)[0]

    print(f"\nPrediction: {prediction}")

    # Show probabilities
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_new)[0]

        print("\nPrediction probabilities:")

        results = sorted(
            zip(classifier.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True,
        )

        for label, probability in results:
            print(
                f"{label:12s}: "
                f"{probability:.3f}"
            )

    landmarker.close()


if __name__ == "__main__":
    main()