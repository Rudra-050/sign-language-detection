import cv2
import os

def collect_images_and_videos(base_dir="datasets", datasets=None, labels=None, camera_index=0):
    """
    Collects images and videos for each dataset and label, saving them in subdirectories.

    Args:
        base_dir (str): Base directory where datasets will be stored.
        datasets (list): List of dataset names (e.g., ['custom_dataset']).
        labels (list): List of labels for classification (e.g., ['hello', 'thanks']).
        camera_index (int): Index of the camera to use.
    """
    if datasets is None:
        datasets = []
    if labels is None:
        labels = []

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"Created dataset directory: {dataset_dir}")

        for label in labels:
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
                print(f"Created label directory: {label_dir}")

            print(f"Collecting data for dataset: {dataset}, label: {label}.")
            print("Press 'c' to capture image, 'v' to start/stop video recording, 'q' to quit.")

            image_count = 0
            video_count = 0
            recording = False
            video_writer = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Display the camera feed
                cv2.imshow(f"Collecting - {dataset}/{label}", frame)

                # Wait for key press
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    # Save the current frame as an image
                    image_path = os.path.join(label_dir, f"{label}_image_{image_count}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Image saved: {image_path}")
                    image_count += 1

                elif key == ord('v'):
                    if not recording:
                        # Start video recording
                        video_path = os.path.join(label_dir, f"{label}_video_{video_count}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
                        fps = 20.0  # Frames per second
                        frame_size = (frame.shape[1], frame.shape[0])  # Frame size
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
                        print(f"Started recording video: {video_path}")
                        recording = True
                    else:
                        # Stop video recording
                        video_writer.release()
                        print(f"Video saved: {video_path}")
                        video_count += 1
                        recording = False

                elif key == ord('q'):
                    # Quit collecting for the current label
                    if recording and video_writer is not None:
                        video_writer.release()
                        print(f"Stopped and saved video: {video_path}")
                        recording = False
                    break

                # Write frames to the video file if recording
                if recording and video_writer is not None:
                    video_writer.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")

if __name__ == "__main__":
    # Specify only the custom dataset
    datasets = ["custom_dataset"]  # Collect data only for the custom dataset
    labels = ["hello", "thanks", "yes", "no"]  # Add your labels here
    collect_images_and_videos(base_dir="datasets", datasets=datasets, labels=labels)
