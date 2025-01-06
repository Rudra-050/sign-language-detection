#This script uses MediaPipe to track hands and their landmarks in real-time from a webcam feed
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initializes the HandTracker using MediaPipe Hands.

        Args:
            max_num_hands (int): Maximum number of hands to detect.
            detection_confidence (float): Minimum confidence value for hand detection.
            tracking_confidence (float): Minimum confidence value for hand tracking.
        """
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Processes a frame to detect hands and their landmarks.

        Args:
            frame (numpy.ndarray): Input image frame (BGR format).

        Returns:
            annotated_frame (numpy.ndarray): Frame with hand landmarks drawn.
            hand_landmarks (list): List of hand landmarks detected.
        """
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = self.hands.process(rgb_frame)

        annotated_frame = frame.copy()
        hand_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_draw.draw_landmarks(
                    annotated_frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS
                )
                hand_landmarks.append(hand_landmark)

        return annotated_frame, hand_landmarks

    def release(self):
        """Releases resources used by the HandTracker."""
        self.hands.close()

if __name__ == "__main__":
    # Example usage
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    print("Hand tracking is running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame
        annotated_frame, hand_landmarks = tracker.process_frame(frame)

        # Display the frame
        cv2.imshow("Hand Tracker", annotated_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
