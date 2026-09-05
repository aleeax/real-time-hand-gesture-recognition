import cv2
import mediapipe as mp
import mediapipe.tasks as tasks

MODEL_PATH = "models/hand_landmarker.task"

def draw_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape

    # Draw each of the 21 landmarks
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def main():
    # Configure MediaPipe Hand Landmarker
    base_options = tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )

    landmarker = tasks.vision.HandLandmarker.create_from_options(options)

    # 0 = default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera started!")
    print("Show your hand to the camera.")
    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        # Draw landmarks if hands are detected
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

        # Display number of detected hands
        cv2.putText(
            frame,
            f"Hands detected: {len(result.hand_landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
    # make sure a window appear showing your live webcam feed
    # this is testing python -> OpenCV -> webcam -> live frames
