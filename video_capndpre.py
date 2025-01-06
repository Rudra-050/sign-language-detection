import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir="frames", frame_interval=30):
    """
    Extracts frames from a video at regular intervals and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        frame_interval (int): Number of frames to skip between each saved frame.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame saved: {frame_path}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. Total frames saved: {saved_count}")

def resize_frames(input_dir, output_dir="resized_frames", size=(224, 224)):
    """
    Resizes all images in a directory to the specified size.

    Args:
        input_dir (str): Directory containing input frames.
        output_dir (str): Directory where resized frames will be saved.
        size (tuple): Desired size (width, height) for the resized images.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.isfile(input_path):
            img = cv2.imread(input_path)
            if img is None:
                print(f"Warning: Could not read {input_path}")
                continue

            resized_img = cv2.resize(img, size)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized image saved: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # Extract frames from a video
    extract_frames(video_path="input_video.mp4", output_dir="frames", frame_interval=30)

    # Resize the extracted frames
    resize_frames(input_dir="frames", output_dir="resized_frames", size=(224, 224))
