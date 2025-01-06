#this script is used to distinguish between the video files and video files and organizes them accordingly in the output dictionary
import os
import shutil
import random

def split_dataset(dataset_dir, output_dir="split_dataset", train_ratio=0.8, val_ratio=0.1):
    """
    Splits the dataset into training, validation, and test sets for both images and videos.

    Args:
        dataset_dir (str): Directory containing the dataset with subdirectories for each label.
        output_dir (str): Directory where the split dataset will be stored.
        train_ratio (float): Proportion of data to include in the training set.
        val_ratio (float): Proportion of data to include in the validation set.
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        print("Error: Invalid split ratios. Ensure train_ratio + val_ratio < 1.0.")
        return

    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue

        # Create label subdirectories for each split and media type
        for split in ["train", "val", "test"]:
            for media_type in ["images", "videos"]:
                split_label_dir = os.path.join(output_dir, split, media_type, label)
                os.makedirs(split_label_dir, exist_ok=True)

        # Separate image and video files
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        video_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        # Shuffle and split files
        for files, media_type in zip([image_files, video_files], ["images", "videos"]):
            random.shuffle(files)
            train_count = int(len(files) * train_ratio)
            val_count = int(len(files) * val_ratio)

            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]

            # Move files to respective directories
            for file_list, split in zip([train_files, val_files, test_files], ["train", "val", "test"]):
                for file_name in file_list:
                    src_path = os.path.join(label_path, file_name)
                    dest_path = os.path.join(output_dir, split, media_type, label, file_name)
                    shutil.copy(src_path, dest_path)

        print(f"Split for label '{label}':")
        print(f"  Images: {len(image_files)} files ({len(train_files)} train, {len(val_files)} val, {len(test_files)} test)")
        print(f"  Videos: {len(video_files)} files ({len(train_files)} train, {len(val_files)} val, {len(test_files)} test)")

    print("Dataset splitting complete.")

if __name__ == "__main__":
    # Example usage
    split_dataset(dataset_dir="dataset", output_dir="split_dataset", train_ratio=0.8, val_ratio=0.1)
