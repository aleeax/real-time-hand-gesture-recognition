import cv2

def main():
    # 0 = default webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera started!")
    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # make sure a window appear showing your live webcam feed
    # this is testing python -> OpenCV -> webcam -> live frames
    