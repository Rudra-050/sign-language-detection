#The following script will handle any sort of error that occurs during hand tracking
class Errorhandler:
    """
    Utility class to handle and log errors during hand tracking or other processes.
    """
    def __init__(self, log_file="error_log.txt"):
        """
        Initializes the ErrorSorter with an optional log file.

        Args:
            log_file (str): Path to the file where errors will be logged.
        """
        self.log_file = log_file

    def log_error(self, error_message):
        """
        Logs an error message to the console and the log file.

        Args:
            error_message (str): The error message to log.
        """
        print(f"Error: {error_message}")
        with open(self.log_file, "a") as log:
            log.write(f"Error: {error_message}\n")

    def check_frame_error(self, frame):
        """
        Validates the frame captured from the camera.

        Args:
            frame (numpy.ndarray): The frame to check.

        Returns:
            bool: True if the frame is valid, False otherwise.
        """
        if frame is None:
            self.log_error("Failed to capture frame. Frame is None.")
            return False
        return True

    def check_camera_access(self, cap):
        """
        Validates if the camera can be accessed.

        Args:
            cap (cv2.VideoCapture): The VideoCapture object.

        Returns:
            bool: True if the camera is accessible, False otherwise.
        """
        if not cap.isOpened():
            self.log_error("Could not access the camera.")
            return False
        return True

if __name__ == "__main__":
    # Example usage
    import cv2

    error_handler = Errorhandler()
    cap = cv2.VideoCapture(0)

    if not error_handler.check_camera_access(cap):
        exit()

    print("Camera feed is running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or not error_handler.check_frame_error(frame):
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
