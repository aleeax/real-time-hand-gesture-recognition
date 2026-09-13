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

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (0, 17),
]


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

def draw_landmarks(image, hand_landmarks):
    """Draw MediaPipe hand landmarks and connections on an image."""

    height, width, _ = image.shape

    points = []

    # Convert normalized coordinates into image pixel coordinates
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Draw skeleton connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            image,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    # Draw the 21 landmarks
    for index, point in enumerate(points):
        cv2.circle(
            image,
            point,
            5,
            (0, 255, 0),
            -1,
        )

        # Add landmark number
        cv2.putText(
            image,
            str(index),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )


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
    hand_landmarks = result.hand_landmarks[0]

    features = normalize_landmarks(
        hand_landmarks
    )
    # this gave us a variable we can reuse for drawing

    feature_names = create_feature_names()

    X_new = pd.DataFrame(
        [features],
        columns=feature_names
    )

    # Load our trained gesture classifier
    classifier = joblib.load(CLASSIFIER_PATH)

    prediction = classifier.predict(X_new)[0]

    # Draw MediaPipe landmarks on the original image
    draw_landmarks(
        image,
        hand_landmarks
    )

    # Display classifier prediction
    cv2.putText(
        image,
        f"Prediction: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

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

    cv2.imshow(
        "Unseen Gesture Prediction",
        image
    )

    print("\nPress any key on the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    landmarker.close()


if __name__ == "__main__":
    main()