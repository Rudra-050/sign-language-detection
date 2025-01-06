import mediapipe as mp
import cv2
import numpy as np

class HandLandmarksUtil:
    """
    Utility class for detecting and processing hand landmarks using MediaPipe.
    """
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the MediaPipe Hand module.

        Args:
            static_image_mode (bool): Whether to treat input images as static.
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence for hand detection.
            min_tracking_confidence (float): Minimum confidence for hand tracking.
        """
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Processes a single frame to detect hand landmarks.

        Args:
            frame (numpy.ndarray): The input frame from the camera.

        Returns:
            list: A list of detected hand landmarks.
            numpy.ndarray: The annotated frame with landmarks drawn.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list.append(hand_landmarks)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return landmarks_list, frame

    def extract_landmark_points(self, hand_landmarks):
        """
        Extracts landmark points from a single hand's landmarks.

        Args:
            hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                The landmarks of a detected hand.

        Returns:
            list: A list of (x, y, z) tuples for each landmark.
        """
        points = []
        for landmark in hand_landmarks.landmark:
            points.append((landmark.x, landmark.y, landmark.z))
        return points

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hand_util = HandLandmarksUtil()

    print("Hand landmarks detection is running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        landmarks, annotated_frame = hand_util.process_frame(frame)
        cv2.imshow("Hand Landmarks", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
