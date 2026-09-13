from pathlib import Path
import csv
import time

import cv2
import mediapipe as mp
import mediapipe.tasks as tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATA_PATH = PROJECT_ROOT / "data" / "gesture_landmarks.csv"

GESTURES = {
    ord("o"): "OPEN_PALM",
    ord("p"): "PEACE",
    ord("f"): "FIST",
    ord("i"): "POINTING",
}

RECORD_SECONDS = 3


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


def create_header():
    """Create CSV column names for 21 x/y/z landmarks."""

    header = []

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}",
        ])

    header.append("label")

    return header


def save_sample(writer, hand_landmarks, label):
    features = normalize_landmarks(hand_landmarks)
    writer.writerow(features + [label])


def main():
    DATA_PATH.parent.mkdir(exist_ok=True)

    # Create CSV and header if it doesn't exist yet
    file_exists = DATA_PATH.exists()

    csv_file = open(
        DATA_PATH,
        "a",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(create_header())

    # MediaPipe setup
    base_options = tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )

    landmarker = tasks.vision.HandLandmarker.create_from_options(
        options
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        csv_file.close()
        landmarker.close()
        return

    print("Gesture Data Collector")
    print("----------------------")
    print("O = Open Palm")
    print("P = Peace")
    print("F = Fist")
    print("I = Pointing")
    print("Q = Quit")

    recording_label = None
    recording_end = 0
    sample_count = 0

    while True:
        success, frame = camera.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        # Record detected hand while recording mode is active
        if recording_label and time.time() < recording_end:

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                save_sample(
                    writer,
                    hand_landmarks,
                    recording_label
                )

                sample_count += 1

            status = (
                f"Recording {recording_label}: "
                f"{sample_count} samples"
            )

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        elif recording_label:
            print(
                f"Finished {recording_label}: "
                f"{sample_count} samples"
            )

            csv_file.flush()

            recording_label = None
            sample_count = 0

        else:
            cv2.putText(
                frame,
                "O: Palm | P: Peace | F: Fist | I: Point | Q: Quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Gesture Data Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in GESTURES and recording_label is None:
            recording_label = GESTURES[key]
            recording_end = time.time() + RECORD_SECONDS
            sample_count = 0

            print(
                f"Recording {recording_label} "
                f"for {RECORD_SECONDS} seconds..."
            )

    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()
    csv_file.close()


if __name__ == "__main__":
    main()