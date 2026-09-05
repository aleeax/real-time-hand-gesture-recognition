def recognize_gesture(hand_landmarks):
    """Recognize simple hand gestures using landmark positions."""
    # so when a fingertip is above its lower joint: tip.y < joint.y,
    # we can roughly consider that finger raised

    # Four fingers:
    # A finger is considered raised when its tip is above its PIP joint.
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    ring_up = hand_landmarks[16].y < hand_landmarks[14].y
    pinky_up = hand_landmarks[20].y < hand_landmarks[18].y

    # Thumb points upward
    thumb_vertical = (
        hand_landmarks[4].y < hand_landmarks[3].y
        and hand_landmarks[3].y < hand_landmarks[2].y
    )

    # Thumb tip should also be clearly above the index-finger base
    thumb_extended = (
        hand_landmarks[4].y < hand_landmarks[5].y
    )

    thumb_up = thumb_vertical and thumb_extended

    # ✋ Open palm
    if index_up and middle_up and ring_up and pinky_up:
        return "OPEN PALM"

    # ✌️ Peace
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # ☝️ Pointing
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINTING"

    # 👍 Thumbs up
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "THUMBS UP"

    # ✊ Fist
    if (
        not thumb_up
        and not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return "FIST"

    return "Unknown"
