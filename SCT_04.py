"""Real-time hand-landmark demo with simple rule-based gesture labels.

MediaPipe detects hand landmarks; this script applies transparent geometric
rules to label a few common gestures. It does not train a gesture-classification
machine-learning model.
"""
import cv2
import mediapipe as mp

WINDOW_NAME = "Hand Gesture Demo"

def classify_gesture(landmarks):
    """Classify a few upright-hand poses using normalized landmark positions."""
    # MediaPipe landmark indices: tips 8/12/16/20, PIP joints 6/10/14/18.
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = [landmarks[tip].y < landmarks[pip].y for tip, pip in finger_pairs]

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_up = thumb_tip.y < thumb_ip.y

    if all(extended):
        return "Open Palm"
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "Victory"
    if not any(extended) and thumb_up:
        return "Thumbs Up"
    if not any(extended):
        return "Fist"
    return "Unknown"

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: could not open the default webcam.")
        return

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            print("Hand Gesture Demo")
            print("Show an upright hand to the camera. Press q to quit.")

            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: could not read a frame from the webcam.")
                    break

                # Mirror the preview for a more natural webcam experience.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                label = "No hand detected"

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )
                    label = classify_gesture(hand_landmarks.landmark)

                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
