import cv2
import mediapipe as mp
import mediapipe.tasks as tasks

MODEL_PATH = "models/hand_landmarker.task"

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (0, 17),
]

def draw_landmarks(frame, hand_landmarks):
    """Draw the hand skeleton and landmarks."""
    height, width, _ = frame.shape

    points = []

    # Convert normalized coordinates to pixel coordinates
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Draw connections first
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    # Draw landmarks on top
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

def recognize_gesture(hand_landmarks):
    """Recognize simple hand gestures using landmark positions."""
    # so when a fingertip is above its lower joint: tip.y < joint.y,
    # we can roughly consider that finger raised

    # Fingertip and PIP joint landmark indices
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    ring_up = hand_landmarks[16].y < hand_landmarks[14].y
    pinky_up = hand_landmarks[20].y < hand_landmarks[18].y

    # Peace sign: index + middle up, ring + pinky down
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # this is not using the thumb yet, it will be handled later

    return "Unknown"

def main():
    # Configure MediaPipe Hand Landmarker
    base_options = tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
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

        # Convert to numpy image into a mediapipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand landmarks
        result = landmarker.detect(mp_image)

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
    
        # Draw landmarks if hands are detected
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            gesture = recognize_gesture(hand_landmarks)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 80),
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
