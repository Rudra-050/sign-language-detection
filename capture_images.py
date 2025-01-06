import cv2
import os

def capture_images(output_dir="dataset", image_prefix="img", camera_index=0):
    """
    Captures images from the camera and saves them to the specified directory.

    Args:
        output_dir (str): Directory where the captured images will be saved.
        image_prefix (str): Prefix for the saved image filenames.
        camera_index (int): Index of the camera to use (default: 0).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'c' to capture an image, 'q' to quit.")

    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the camera feed
        cv2.imshow("Capture Images", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Save the current frame as an image file
            image_path = os.path.join(output_dir, f"{image_prefix}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            image_count += 1

        elif key == ord('q'):
            # Quit the image capture loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Customize the output directory and prefix if needed
    capture_images(output_dir="dataset", image_prefix="sign")
