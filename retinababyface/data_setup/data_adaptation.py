import os
import shutil
import random
from pathlib import Path
from PIL import Image


def add_child_prob_to_labels(dataset_root: str, child_prob: int = 1) -> None:
    """
    Adds a child_prob label (1=baby, 0=adult) to all .txt label files in the dataset.
    The child_prob value is inserted after the class index in each line of the label file.
    Lines that already contain the child_prob value are skipped.

    Args:
        dataset_root (str): Path to the root of the dataset directory containing train, val, and test subsets.
        child_prob (int): Value to insert for the child label. Default is 1 (baby).

    Returns:
        None

    This function processes label files in the 'labels' subdirectory of each subset (train, val, test).
    It updates lines with 10 elements by adding the child_prob value, resulting in 11 elements per line.
    Lines with malformed data or already containing the child_prob value are skipped.
    """
    subsets = ["train", "val", "test"]

    for subset in subsets:
        # Construct the path to the labels directory for the current subset
        label_dir = os.path.join(dataset_root, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"❌ Label directory not found: {label_dir}")
            continue

        for file_name in os.listdir(label_dir):
            # Skip non-.txt files
            if not file_name.endswith(".txt"):
                continue

            file_path = os.path.join(label_dir, file_name)

            # Read the contents of the label file
            with open(file_path, "r") as f:
                lines = f.readlines()

            updated_lines = []
            modified = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 11:
                    # Line already contains child_prob, skip it
                    updated_lines.append(line)
                elif len(parts) == 10:
                    # Add child_prob after the class index
                    class_idx = parts[0]
                    rest = parts[1:]
                    new_line = f"{class_idx} {child_prob} " + " ".join(rest) + "\n"
                    updated_lines.append(new_line)
                    modified = True
                else:
                    # Skip malformed lines and log a warning
                    print(f"⚠️ Malformed line skipped in {file_path}: {line.strip()}")
                    updated_lines.append(line)

            # Write updated lines back to the file if modifications were made
            if modified:
                with open(file_path, "w") as f:
                    f.writelines(updated_lines)
                print(f"✅ Updated: {file_path}")
            else:
                print(f"✔️ No changes needed: {file_path}")

    print("🎯 Processing completed.")


def convert_widerface_annotations(
    images_root: str, label_txt_path: str, output_root: str
) -> None:
    """
    Converts WIDERFace validation set annotations to a custom RetinaBabyFace format.
    Saves images and labels in organized folders and normalizes bounding box coordinates.

    Args:
        images_root (str): Path to the folder containing all images organized by category.
        label_txt_path (str): Path to the .txt file containing all annotations.
        output_root (str): Destination folder where images and new labels will be saved.
    """
    # Create directories for images and labels
    images_dir = os.path.join(output_root, "images")
    labels_dir = os.path.join(output_root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    total_faces = 0  # Counter for total faces processed
    current_image_path = None  # Current image relative path
    current_image_full_path = None  # Full path to the current image
    current_annotations = []  # List of bounding box annotations for the current image

    # Read all lines from the annotation file
    with open(label_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line represents an image path
        if line.endswith(".jpg"):
            # Save the previous image and its annotations
            if current_image_path is not None and current_annotations:
                bbox2obb(
                    current_image_path,
                    current_annotations,
                    images_root,
                    images_dir,
                    labels_dir,
                )
                total_faces += len(current_annotations)

            # Update current image path and reset annotations
            current_image_path = line
            current_annotations = []
            i += 1
            if i >= len(lines):
                break

            # Read the number of faces in the image
            num_faces = int(lines[i].strip())
            i += 1
            for _ in range(num_faces):
                if i >= len(lines):
                    break
                # Parse bounding box coordinates
                annotation = lines[i].strip().split()
                if len(annotation) >= 4:
                    current_annotations.append(
                        list(map(float, annotation[:4]))
                    )  # x, y, w, h
                i += 1
        else:
            i += 1

    # Save the last image and its annotations
    if current_image_path and current_annotations:
        bbox2obb(
            current_image_path, current_annotations, images_root, images_dir, labels_dir
        )
        total_faces += len(current_annotations)

    # Print summary of the process
    print(f"🎯 Total faces processed: {total_faces}")
    print(f"✅ Images and labels saved in: {output_root}")


def bbox2obb(
    relative_img_path: str,
    bboxes: list,
    root_img_dir: str,
    out_img_dir: str,
    out_lbl_dir: str,
) -> None:
    """
    Saves the image and its corresponding label file with normalized bounding boxes in the new format.

    Args:
        relative_img_path (str): Relative path to the image file.
        bboxes (list): List of bounding boxes (x, y, w, h).
        root_img_dir (str): Root directory containing the original images.
        out_img_dir (str): Directory to save the copied images.
        out_lbl_dir (str): Directory to save the label files.
    """
    img_path = os.path.join(root_img_dir, relative_img_path)
    image_name = os.path.basename(img_path)
    label_name = os.path.splitext(image_name)[0] + ".txt"

    try:
        # Open the image to retrieve its dimensions
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"❌ Error opening image {img_path}: {e}")
        return

    label_lines = []
    for x, y, w, h in bboxes:
        # Convert bounding box to vertices (axis-aligned box)
        x1, y1 = x, y
        x2, y2 = x + w, y
        x3, y3 = x + w, y + h
        x4, y4 = x, y + h

        # Normalize coordinates
        coords = [x1, y1, x2, y2, x3, y3, x4, y4]
        coords = [
            round(c / width, 6) if i % 2 == 0 else round(c / height, 6)
            for i, c in enumerate(coords)
        ]

        # Format the label line
        line = f"-1 0 " + " ".join(map(str, coords)) + " 0\n"
        label_lines.append(line)

    # Copy the image to the output directory
    shutil.copy(img_path, os.path.join(out_img_dir, image_name))

    # Save the label file
    with open(os.path.join(out_lbl_dir, label_name), "w") as f:
        f.writelines(label_lines)
    print(f"✅ Saved: {image_name} with {len(bboxes)} faces.")


def convert_face_detection_dataset(input_root: str, output_root: str) -> None:
    """
    Converts a FACE DETECTION DATASET to the RetinaBabyFace format.

    Args:
        input_root (str): Path to the original dataset, containing subfolders `images/train`, `labels/train`, etc.
        output_root (str): Path to the new dataset with structure `train/images`, `train/labels`, etc.

    This function processes the dataset by converting bounding box annotations from center-width-height format
    to axis-aligned bounding box format with normalized coordinates. It organizes the output dataset into
    separate folders for images and labels for each split (train, val). Invalid lines in label files are skipped.
    """
    total_faces_global = 0  # Counter for total faces across all splits

    for split in ["train", "val"]:
        # Define input and output directories for images and labels
        input_img_dir = os.path.join(input_root, "images", split)
        input_lbl_dir = os.path.join(input_root, "labels", split)

        output_img_dir = os.path.join(output_root, split, "images")
        output_lbl_dir = os.path.join(output_root, split, "labels")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)

        face_count = 0  # Counter for faces in the current split

        for file in os.listdir(input_lbl_dir):
            # Skip non-.txt files
            if not file.endswith(".txt"):
                continue

            label_path = os.path.join(input_lbl_dir, file)
            image_name = os.path.splitext(file)[0] + ".jpg"
            image_path = os.path.join(input_img_dir, image_name)

            try:
                # Open the image to retrieve its dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"❌ Failed to open image: {image_path}. Error: {e}")
                continue

            # Read the label file
            with open(label_path, "r") as f:
                lines = f.readlines()

            label_lines = []  # List to store converted label lines
            for line in lines:
                parts = line.strip().split()
                # Skip invalid lines that do not have the expected number of elements
                if len(parts) != 5:
                    continue

                # Parse bounding box coordinates (center-x, center-y, width, height)
                _, cx, cy, w, h = map(float, parts)

                # Convert from center-width-height to axis-aligned bounding box vertices
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy - h / 2
                x3 = cx + w / 2
                y3 = cy + h / 2
                x4 = cx - w / 2
                y4 = cy + h / 2

                # Normalize coordinates and format the label line
                coords = [x1, y1, x2, y2, x3, y3, x4, y4]
                coords = [
                    round(c, 6) for c in coords
                ]  # Coordinates are already normalized
                new_line = f"-1 0 " + " ".join(map(str, coords)) + " 0\n"
                label_lines.append(new_line)

            # Save the converted labels and copy the image to the output directory
            if label_lines:
                shutil.copy(image_path, os.path.join(output_img_dir, image_name))
                with open(os.path.join(output_lbl_dir, file), "w") as f:
                    f.writelines(label_lines)
                face_count += len(label_lines)

        print(f"✅ Split '{split}' processed with {face_count} faces.")
        total_faces_global += face_count

    print(f"\n🎯 Total faces in the entire dataset: {total_faces_global}")


def clean_and_split_dataset(root_dir: str, output_dir: str, seed: int = 42) -> None:
    """
    Cleans label files without matching images, counts annotations, and splits dataset into train/val/test.

    This function performs the following steps:
    1. Removes orphan label files (label files without corresponding images).
    2. Counts the total number of annotations in the dataset after cleanup.
    3. Splits the dataset into train, validation, and test subsets based on a predefined ratio (80% train, 10% val, 10% test).

    Args:
        root_dir (str): Path to the dataset containing 'images/*.jpg' and 'labels/*.txt'.
        output_dir (str): Destination root for the new split dataset.
        seed (int): Random seed for reproducibility of the dataset split.

    Returns:
        None: The function saves the cleaned and split dataset in the specified output directory.
    """
    # Define paths to the images and labels directories
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")

    # Get sets of image and label files
    image_files = set(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    label_files = set(f for f in os.listdir(label_dir) if f.endswith(".txt"))

    # Find matching image-label pairs
    matched = image_files.intersection(f.replace(".txt", ".jpg") for f in label_files)
    valid_image_files = sorted(matched)

    # Step 1: Remove orphan label files (labels without corresponding images)
    orphan_labels = label_files - set(f.replace(".jpg", ".txt") for f in matched)
    for label in orphan_labels:
        os.remove(os.path.join(label_dir, label))
    print(f"🧹 Removed {len(orphan_labels)} orphan label files.")

    # Step 2: Count total annotations in the remaining dataset
    total_annotations = 0
    for file in valid_image_files:
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            total_annotations += len(f.readlines())

    print(f"📌 Total remaining images: {len(valid_image_files)}")
    print(f"🎯 Total annotations after cleanup: {total_annotations}")

    # Step 3: Split dataset into train, validation, and test subsets
    random.seed(seed)  # Set random seed for reproducibility
    random.shuffle(valid_image_files)  # Shuffle the list of valid image files

    # Calculate the number of images for each subset
    n_total = len(valid_image_files)
    n_train = int(n_total * 0.8)  # 80% for training
    n_val = int(n_total * 0.1)  # 10% for validation
    n_test = n_total - n_train - n_val  # Remaining 10% for testing

    # Create splits dictionary
    splits = {
        "train": valid_image_files[:n_train],
        "val": valid_image_files[n_train : n_train + n_val],
        "test": valid_image_files[n_train + n_val :],
    }

    # Save each split into separate directories
    for split, files in splits.items():
        split_img_dir = os.path.join(output_dir, split, "images")
        split_lbl_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        # Copy images and labels to their respective directories
        for file in files:
            shutil.copy(
                os.path.join(image_dir, file), os.path.join(split_img_dir, file)
            )
            shutil.copy(
                os.path.join(label_dir, file.replace(".jpg", ".txt")),
                os.path.join(split_lbl_dir, file.replace(".jpg", ".txt")),
            )

        print(f"✅ {split}: {len(files)} images")

    print(f"\n📁 Split dataset saved in: {output_dir}")


def count_annotations_by_split(dataset_root: str) -> None:
    """
    Counts the number of annotations per partition (train/val/test) in a dataset.

    This function iterates through the 'train', 'val', and 'test' subdirectories of the dataset,
    counts the number of label files and the total number of annotations in each split, and
    prints a summary of the results.

    Args:
        dataset_root (str): Root directory of the dataset containing 'train', 'val', and 'test' subdirectories.

    Returns:
        None: The function prints the results to the console.
    """
    # Initialize a counter for total annotations across all splits
    total_annotations = 0

    # Define the dataset splits to process
    splits = ["train", "val", "test"]

    # Iterate through each split (train, val, test)
    for split in splits:
        # Construct the path to the labels directory for the current split
        label_dir = os.path.join(dataset_root, split, "labels")

        # Check if the labels directory exists
        if not os.path.exists(label_dir):
            print(f"⚠️ Label directory missing for split '{split}'")
            continue

        # Initialize counters for the current split
        num_files = 0  # Number of label files
        num_annots = 0  # Number of annotations

        # Iterate through all files in the labels directory
        for file in os.listdir(label_dir):
            # Skip non-.txt files
            if not file.endswith(".txt"):
                continue

            # Read the label file and count the number of annotations
            file_path = os.path.join(label_dir, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                num_annots += len(lines)
                num_files += 1

        # Update the total annotations counter
        total_annotations += num_annots

        # Print the summary for the current split
        print(f"📂 {split}: {num_files} label files | {num_annots} annotations")

    # Print the total annotations across all splits
    print(f"\n🎯 Total annotations across all splits: {total_annotations}")
