import os
import cv2
import albumentations as A

def augment_images_and_videos(input_dir, output_dir, augment_count=5):
    """
    Augments images and videos from the input directory and saves them in the output directory.

    Args:
        input_dir (str): Directory containing the original images and videos.
        output_dir (str): Directory where augmented images and videos will be stored.
        augment_count (int): Number of augmented versions to generate per original file.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Define augmentation pipeline using Albumentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5)
    ])

    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue

        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext in ['.jpg', '.png', '.jpeg']:  # Process images
                process_image(file_path, output_label_dir, file_name, transform, augment_count)

            elif file_ext in ['.mp4', '.avi', '.mov']:  # Process videos
                process_video(file_path, output_label_dir, file_name, transform, augment_count)

    print("Image and video augmentation complete.")

def process_image(file_path, output_label_dir, file_name, transform, augment_count):
    """Processes and augments a single image."""
    image = cv2.imread(file_path)

    if image is None:
        print(f"Warning: Failed to read image {file_path}")
        return

    # Save the original image
    original_output_path = os.path.join(output_label_dir, file_name)
    cv2.imwrite(original_output_path, image)

    # Generate augmented images
    for i in range(augment_count):
        augmented = transform(image=image)
        augmented_image = augmented["image"]
        augmented_file_name = f"{os.path.splitext(file_name)[0]}_aug{i + 1}.jpg"
        augmented_output_path = os.path.join(output_label_dir, augmented_file_name)
        cv2.imwrite(augmented_output_path, augmented_image)

    print(f"Augmented images for {file_name} saved in {output_label_dir}")

def process_video(file_path, output_label_dir, file_name, transform, augment_count):
    """Processes and augments a single video."""
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(f"Error: Failed to read video {file_path}")
        return

    # Extract original video frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Save the original video frames as a video file
    original_video_path = os.path.join(output_label_dir, file_name)
    save_video(frames, original_video_path, fps, (frame_width, frame_height))

    # Generate augmented versions
    for i in range(augment_count):
        augmented_frames = []
        for frame in frames:
            augmented = transform(image=frame)
            augmented_frames.append(augmented["image"])

        augmented_video_path = os.path.join(output_label_dir, f"{os.path.splitext(file_name)[0]}_aug{i + 1}.avi")
        save_video(augmented_frames, augmented_video_path, fps, (frame_width, frame_height))

    print(f"Augmented videos for {file_name} saved in {output_label_dir}")

def save_video(frames, output_path, fps, frame_size):
    """Saves a list of frames as a video file."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()

if __name__ == "__main__":
    # Example usage
    augment_images_and_videos(input_dir="dataset", output_dir="augmented_dataset", augment_count=3)
