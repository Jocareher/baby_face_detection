import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy import ndimage

import re
import math
import os
import json
import random
import glob
import hashlib
from typing import List, Tuple, Dict
from pathlib import Path
import shutil

from collections import defaultdict


def process_and_save_json(file_path: str, output_directory: str):
    """
    Reads a JSON file containing annotations, filters out entries with a specific label,
    and saves individual JSON files for each valid entry.

    This function reads a master JSON file containing annotations for multiple images,
    filters out any image data marked with the label 'doubt', and then saves the remaining
    image data as individual JSON files in a specified directory.

    Args:
        - file_path: The file path to the master JSON file containing annotations for multiple images.
        - output_directory: The directory path where individual JSON files will be saved.

    Note:
        If the output directory does not exist, it will be created.
    """
    # Read the JSON file from the provided file path
    with open(file_path, "r") as file:
        annotations = json.load(file)

    # Check if the output directory exists, create if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for image_data in annotations:
        # Skip images with the "doubt" label
        if any(
            label.get("rectanglelabels", []) == ["doubt"]
            for label in image_data.get("label", [])
        ):
            continue

        # Generate and save individual JSON file for each valid image
        image_name = image_data["image"].split("/")[-1].split(".")[0]
        individual_json_path = os.path.join(output_directory, f"{image_name}.json")

        with open(individual_json_path, "w") as outfile:
            json.dump(image_data, outfile, indent=4)


def find_duplicated_images(source_dir: Path, destination_dir: Path) -> List:
    """
    Finds and moves duplicate images from one folder to another.

    Args:
        - source_dir: Folder where the original images are located.
        - destination_dir: Folder where the duplicate images will be moved.
    Returns:
        A list of filenames of the duplicate images found.

    """
    # Create the destination folder if it does not exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Dictionary to store image hashes
    hashes = {}
    # List to keep track of duplicate files
    duplicate_files = []

    # Iterate over all files in the source folder
    for filename in os.listdir(source_dir):
        # Check if the file is an image
        if filename.endswith(("png", "jpg", "jpeg", "JPG")):
            # Construct the full file path
            file_path = os.path.join(source_dir, filename)

            # Open the image and calculate its hash
            with Image.open(file_path) as img:
                # Convert the image to a byte sequence and calculate its SHA-256 hash
                hash_obj = hashlib.sha256(img.tobytes())
                # Retrieve the hash value in hexadecimal string format
                img_hash = hash_obj.hexdigest()

            # Check if the hash already exists (duplicate image)
            if img_hash in hashes:
                # Add to duplicate files list
                duplicate_files.append(filename)
                # Move the duplicate file to the destination folder
                os.rename(file_path, os.path.join(destination_dir, filename))
            else:
                # Store the hash of the new, unique image
                hashes[img_hash] = filename

    return duplicate_files


def remove_images_with_label_5(images_folder: str, labels_folder: str) -> None:
    """
    Remove images and labels containing the label '5' from the given folders.

    Args:
        - images_folder: Path to the folder containing images in JPG format.
        - labels_folder: Path to the folder containing label files in YOLO format.
    """
    # Iterate through all images and labels
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(images_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_file)

            # Check if the label file exists
            if os.path.exists(label_path):
                # Read the contents of the label file
                with open(label_path, "r") as label_file:
                    labels = label_file.readlines()

                # Check if any label contains the number 5
                has_target_label = any("5" in label.strip().split() for label in labels)

                if has_target_label:
                    # If it contains label 5, delete the image and label file
                    os.remove(image_path)
                    os.remove(label_path)
                    print(f"Image and labels removed due to label 5: {image_file}")
                else:
                    # If it doesn't contain label 5, keep the image and label file
                    print(f"Valid image and labels: {image_file}")
            else:
                # If the label file doesn't exist, delete the image
                os.remove(image_path)
                print(f"Image removed due to missing labels: {image_file}")


def extract_losses(file_path: str) -> tuple:
    """
    Extracts the loss values from a training log file.

    Args:
        file_path (str): Path to the log file containing training output.

    Returns:
        tuple: Two elements, a list of iteration numbers and a dictionary of loss types and their corresponding values.
    """
    # Define regex patterns to find iterations and loss values
    iter_pattern = re.compile(r"iter: (\d+)")
    loss_patterns = {
        "total_loss": re.compile(r"total_loss: ([\d.]+)"),
        "loss_cls": re.compile(r"loss_cls: ([\d.]+)"),
        "loss_box_reg": re.compile(r"loss_box_reg: ([\d.]+)"),
        "loss_rpn_cls": re.compile(r"loss_rpn_cls: ([\d.]+)"),
        "loss_rpn_loc": re.compile(r"loss_rpn_loc: ([\d.]+)"),
    }

    # Dictionary to store the extracted values
    losses = {key: [] for key in loss_patterns.keys()}
    iterations = []

    # Open and read the text file
    with open(file_path, "r") as file:
        for line in file:
            # Search and extract iteration
            iter_match = iter_pattern.search(line)
            if iter_match:
                iterations.append(int(iter_match.group(1)))
                # Search and extract each type of loss
                for loss_key, loss_regex in loss_patterns.items():
                    # Match the loss pattern in the line
                    loss_match = loss_regex.search(line)
                    if loss_match:
                        # Append the loss value to the corresponding key in the dictionary
                        losses[loss_key].append(float(loss_match.group(1)))

    return iterations, losses


def calculate_rotated_bbox_for_yolo_v8(
    data: dict,
    original_size: bool = False,
    image_resize: tuple = (640, 640),
    output_format: str = "absolute",
) -> list[tuple]:
    """
    Calculate the coordinates of the corners of a rotated bounding box based on Label Studio's annotation method,
    where rotation is around the top-left corner. Supports output in absolute coordinates, normalized relative to
    the original image size, or normalized relative to a resized image dimension.

    Args:
    - data (dict): Dictionary containing normalized bounding box information and rotation.
    - original_size (bool): Whether to use the original image dimensions.
    - image_resize (tuple): Resize dimensions (width, height), used if original_size is False.
    - output_format (str): Specifies the output format ("absolute", "normalized_original", "normalized_resized").

    Returns:
    - list of tuple: Coordinates of bounding box corners in the requested format.
    """
    # Choose scaling factors based on the specified dimensions and output format.
    width_factor = data["original_width"] if original_size else image_resize[0]
    height_factor = data["original_height"] if original_size else image_resize[1]

    # Convert normalized dimensions to absolute or normalized based on the chosen factors.
    width = data["width"] / 100 * width_factor
    height = data["height"] / 100 * height_factor
    top_left_x = data["x"] / 100 * width_factor
    top_left_y = data["y"] / 100 * height_factor

    # Convert rotation angle to radians.
    angle_rad = math.radians(data["rotation"])

    # Define corners based on top-left reference for rotation.
    corners = [(0, 0), (width, 0), (width, height), (0, height)]

    # Rotate each corner around the top-left corner
    rotated_corners = []
    for x, y in corners:
        # Apply the rotation matrix to each corner point.
        # For a point (x, y) and a rotation angle theta, the new position (rotated_x, rotated_y) is calculated as follows:
        # rotated_x = x * cos(theta) - y * sin(theta)
        # rotated_y = x * sin(theta) + y * cos(theta)
        # This formula is derived from the standard 2D rotation matrix.
        # After rotation, the new points are not relative to the top-left corner of the image anymore.
        # So we need to translate the rotated points back by adding the absolute coordinates of the top-left corner.
        rotated_x = (x * math.cos(angle_rad) - y * math.sin(angle_rad)) + top_left_x
        rotated_y = (x * math.sin(angle_rad) + y * math.cos(angle_rad)) + top_left_y

        # Adjust coordinates based on the output format.
        if output_format in ["normalized_original", "normalized_resized"]:
            rotated_x /= width_factor
            rotated_y /= height_factor

        rotated_corners.append((rotated_x, rotated_y))

    return rotated_corners


def convert_annotations_to_yolo_obb(
    json_folder_path: str,
    output_folder_path: str,
    class_list: list,
    original_size: bool = False,
    image_resize: tuple = (640, 640),
    output_format: str = "absolute",
):
    """
    Convert rotated bounding box annotations from JSON files to YOLO OBB format and save to TXT files.

    This function processes each JSON file in the specified directory, converts the annotation data
    to the YOLO OBB format, and writes the converted data to a corresponding TXT file in the output folder.

    Args:
    - json_folder_path (str): The path to the folder containing JSON annotation files.
    - output_folder_path (str): The path to the folder where TXT files will be saved.
    - class_list (list): A list of class names ordered according to their class index.
    - original_size (bool): Flag indicating whether to use the original image dimensions for calculations.
    - image_resize (tuple): The dimensions (width, height) to which the image has been resized.
                            This is used if original_size is False.
    - output_format (str): Specifies the output format ("absolute", "normalized_original", "normalized_resized").

    Outputs:
    - TXT files containing the annotations in YOLO OBB format, saved to the destination folder.
    """
    # Create the destination folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through all the files in the json directory
    for file_name in os.listdir(json_folder_path):
        if file_name.endswith(".json"):
            # Read the JSON file
            with open(os.path.join(json_folder_path, file_name), "r") as json_file:
                data = json.load(json_file)

            # Prepare the content for the TXT file
            txt_content = []
            for annotation in data["label"]:
                # Calculate the rotated bounding box coordinates
                corners = calculate_rotated_bbox_for_yolo_v8(
                    annotation, original_size, image_resize, output_format
                )
                # Get the class index
                class_index = class_list.index(annotation["rectanglelabels"][0])
                # Convert coordinates to the YOLO OBB format, absolute or normalized
                yolo_obb = [class_index] + [val for corner in corners for val in corner]
                txt_content.append(" ".join(map(str, yolo_obb)))

            # Write the content to the corresponding TXT file
            txt_file_name = os.path.splitext(data["image"].split("/")[-1])[0] + ".txt"
            with open(os.path.join(output_folder_path, txt_file_name), "w") as txt_file:
                txt_file.write("\n".join(txt_content))


def create_yolov8_pairs(root_directory: str) -> list[tuple]:
    """
    Traverse the 'images' and 'labels' subdirectories within the specified root directory.
    Pair each image with its corresponding annotation file, if available.

    Args:
    - root_directory (str): The path to the root directory containing 'images' and 'labels' folders.

    Returns:
    - List of tuples: Each tuple contains the path to an image file and a list of annotation strings.
    """
    # Path to the subdirectory containing image files
    images_dir = os.path.join(root_directory, "images")
    # Path to the subdirectory containing annotation files
    labels_dir = os.path.join(root_directory, "labels")

    pairs = []
    # Loop through all files in the images directory
    for image_name in os.listdir(images_dir):
        # Check if the file is a JPEG image
        if image_name.endswith(".jpg"):
            # Extract the base name without the file extension
            base_name = os.path.splitext(image_name)[0]
            # Construct the corresponding label file name
            label_name = base_name + ".txt"
            # Full paths to the image and label files
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, label_name)

            # Check if the annotation file exists
            if os.path.exists(label_path):
                # Read all lines from the annotation file
                with open(label_path, "r") as file:
                    annotations = file.readlines()
                # Append the image path and its annotations as a tuple to the pairs list
                pairs.append((image_path, annotations))
    return pairs


def normalize_annotations(labels_dir: str, target_size=(640, 640)):
    """
    Normalize annotations of oriented bounding boxes assuming images will be resized
    to a target input size when input to the YOLOv8 model. This version only requires
    the path to the labels directory.

    Args:
    - labels_dir (str): The path to the directory containing label files.
    - target_size (tuple): The target size (width, height) for normalization.
    """
    # Create a directory for normalized labels if it doesn't exist
    normalized_labels_dir = os.path.join(labels_dir, "normalized_labels")
    if not os.path.exists(normalized_labels_dir):
        os.makedirs(normalized_labels_dir)

    # Loop through each label file in the directory
    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            # Construct the path to the original and the new label file
            label_path = os.path.join(labels_dir, label_file)
            normalized_label_path = os.path.join(normalized_labels_dir, label_file)

            # Open and read the original label file
            with open(label_path, "r") as file:
                annotations = file.readlines()

            # List to store normalized annotations
            normalized_annotations = []
            for annotation in annotations:
                parts = annotation.strip().split()
                class_index = int(parts[0])  # Extract the class index
                coords = list(
                    map(float, parts[1:])
                )  # Extract and convert coordinates to float

                # Normalize each coordinate
                normalized_coords = [
                    coord / target_size[i % 2] for i, coord in enumerate(coords)
                ]

                # Construct the normalized annotation line
                normalized_line = f"{class_index} " + " ".join(
                    f"{coord}" for coord in normalized_coords
                )
                normalized_annotations.append(normalized_line)

            # Write the normalized annotations to a new file in the normalized_labels directory
            with open(normalized_label_path, "w") as file:
                file.write("\n".join(normalized_annotations))


def resize_and_save_images(
    input_path: str, output_path: str, target_size: tuple = (640, 640)
):
    """
    Resize all image files in the input_path to the target_size and save them to output_path.

    Args:
        - input_path: Path to the folder containing images to resize.
        - output_path: Path to the folder where resized images will be saved.
        - target_size: Tuple (width, height) indicating the target size for resizing.
    """
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all image files in the input directory
    image_files = glob.glob(os.path.join(input_path, "*"))
    image_files = [
        file
        for file in image_files
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    for image_file in image_files:
        with Image.open(image_file) as img:
            # Resize image
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Save resized image to the output directory
            # Extract filename and extension to construct the output filename
            filename = os.path.basename(image_file)
            output_file = os.path.join(output_path, filename)
            resized_img.save(output_file)

            print(f"Resized and saved {filename} to {output_path}")


def rename_files_in_folder(dir_path: str, new_name_base: str):
    """
    Renames all files in the specified directory to a new name followed by a sequential number.

    Args:
        - dir_path (str): Path to the directory containing the files to be renamed.
        - new_name_base (str): Base for the new file names.

    Returns:
        - None
    """

    # Initialize the counter
    counter = 0

    # Get a list of all files in the directory
    files = os.listdir(dir_path)

    # Sort the files by name to keep a sequential order
    files.sort()

    # Rename each file in the directory
    for file in files:
        # Construct the new file name with the counter, maintaining the file's original extension
        new_name = f"{new_name_base}{counter:02d}{os.path.splitext(file)[1]}"

        # Build the full path to the original and new file
        original_path = os.path.join(dir_path, file)
        new_path = os.path.join(dir_path, new_name)

        # Rename the file
        os.rename(original_path, new_path)

        # Increment the counter for the next file
        counter += 1

    print("All files have been successfully renamed.")


def check_json_structure(json_folder_path: str) -> list:
    """
    Check each JSON file in the specified folder for the presence of the 'label' key.

    Args:
    - json_folder_path (str): The path to the folder containing JSON files.

    Returns:
    - list: A list of file names that are missing the 'label' key.
    """
    missing_label_files = []

    # Loop through all the files in the specified folder
    for file_name in os.listdir(json_folder_path):
        if file_name.endswith(".json"):
            # Construct file path
            file_path = os.path.join(json_folder_path, file_name)
            # Read the JSON file
            try:
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                # Check if 'label' key is present
                if "label" not in data:
                    missing_label_files.append(file_name)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return missing_label_files


def count_images_per_label_from_metadata(metadata: str) -> dict:
    """
    Counts the number of images per label from a JSON file containing annotations. It returns a dictionary where
    each key is a label accompanied by a tuple containing the count of images and the names of the images. The
    dictionary is sorted by label name, and the image names are also sorted alphabetically.

    Args:
    - metadata (str): Path to the JSON file containing the annotations.

    Returns:
    - dict: A dictionary with labels as keys. Each key maps to a dictionary that contains the count of images
        under 'count' key and a list of image names under 'images' key, both sorted by label name.
    """

    # Initialize a defaultdict to keep count and list of image names per label
    label_counts = defaultdict(lambda: {"count": 0, "images": []})

    # Open and read the JSON file
    with open(metadata, "r") as file:
        json_data = json.load(file)

    # Iterate over each item in the JSON data if it's a list
    if isinstance(json_data, list):
        for annotation in json_data:
            # Extract image name from the 'image' field if available
            image_name = annotation.get("image", "").split("/")[-1]

            # Check if 'label' key exists and is not empty
            if "label" in annotation and annotation["label"]:
                # Assuming label structure matches provided example
                for label_info in annotation["label"]:
                    labels = label_info.get("rectanglelabels", [])
                    for label in labels:
                        # Increment the count and append the image name to the list if not already included
                        label_counts[label]["count"] += 1
                        if image_name not in label_counts[label]["images"]:
                            label_counts[label]["images"].append(image_name)

    # Convert defaultdict to dict for output
    label_counts = dict(label_counts)

    # Sort the dictionary by label name
    sorted_label_counts = {
        k: v for k, v in sorted(label_counts.items(), key=lambda item: item[0])
    }

    # Sort image names for each label
    for label in sorted_label_counts:
        sorted_label_counts[label]["images"].sort()

    return sorted_label_counts


def create_directory_structure(base_dir: str, destination: str) -> None:
    """
    Creates a directory structure for training, validation, and test sets within a specified destination.
    Each set will have 'images' and 'labels' subdirectories.

    Args:
    - base_dir (str): The base directory from which to create the structure. (not used in function but may be needed for extended functionality)
    - destination (str): The destination directory where the train, val, and test directories are created.
    """
    # Iterate through the set names and create directories
    for set_name in ["train", "val", "test"]:
        # Create 'images' subdirectory for each set
        os.makedirs(os.path.join(destination, set_name, "images"), exist_ok=True)
        # Create 'labels' subdirectory for each set
        os.makedirs(os.path.join(destination, set_name, "labels"), exist_ok=True)


def read_and_split_dataset(
    root_path: str, train_size=0.8, val_size=0.1, test_size=0.1
) -> tuple:
    """
    Reads the dataset from the root directory and splits it into training, validation, and test sets according to the specified proportions.

    Args:
    - root_path (str): The root directory containing 'images' and 'labels' folders.
    - train_size (float): Proportion of the dataset to include in the train split.
    - val_size (float): Proportion of the dataset to include in the validation split.
    - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    - tuple: Three tuples containing the paths to the images and labels for the train, validation, and test sets.
    """
    # Paths to 'images' and 'labels' directories
    path_images = os.path.join(root_path, "images")
    path_labels = os.path.join(root_path, "labels")

    data = []
    # Iterate through label files and match them with images
    for label_file in os.listdir(path_labels):
        # Extract base name without extension
        base_name = os.path.splitext(label_file)[0]
        # Construct image path with .jpg extension
        image_path = os.path.join(path_images, base_name + ".jpg")
        # Try with .JPG if .jpg does not exist
        if not os.path.isfile(image_path):
            image_path = os.path.join(path_images, base_name + ".JPG")
        # If image file exists, append it with its label to data list
        if os.path.isfile(image_path):
            data.append((image_path, os.path.join(path_labels, label_file)))

    # Ensure the sum of sizes equals 1
    if train_size + val_size + test_size != 1:
        raise ValueError("The sum of train_size, val_size, and test_size must be 1")

    # Split data into training and the rest; then split the rest into validation and test
    train, temp = train_test_split(
        data, test_size=(val_size + test_size), random_state=42
    )
    val, test = train_test_split(
        temp, test_size=test_size / (val_size + test_size), random_state=42
    )

    return train, val, test


def move_files(data: list, destination: str) -> None:
    """
    Copies image and label files to their respective destination directories.

    Args:
    - data (list): A list of tuples containing paths to images and their corresponding label files.
    - destination (str): The destination directory where the files are copied.
    """
    # Iterate through the data and copy each image and label to the destination
    for img_path, label_path in data:
        # Determine destination paths for image and label
        img_dest_path = os.path.join(destination, "images", os.path.basename(img_path))
        label_dest_path = os.path.join(
            destination, "labels", os.path.basename(label_path)
        )

        # Copy image and label to the destination
        shutil.copy(img_path, img_dest_path)
        shutil.copy(label_path, label_dest_path)


def distribute_dataset(root_path: str, destination: str) -> None:
    """
    Distributes the dataset into training, validation, and test sets and copies the files to the respective directories.

    Args:
    - root_path (str): The root directory containing the original dataset.
    - destination (str): The base directory where the distributed dataset will be copied.
    """
    # Create directory structure in the destination
    create_directory_structure(root_path, destination)

    # Read and split the dataset
    train, val, test = read_and_split_dataset(root_path)

    # Move files to their respective directories
    move_files(train, os.path.join(destination, "train"))
    move_files(val, os.path.join(destination, "val"))
    move_files(test, os.path.join(destination, "test"))


def count_labels_per_class_and_set(root_path: str) -> None:
    """
    Counts the number of labels per class across different dataset sets (train, val, test) and prints the counts.
    The function assumes that each set contains a 'labels' directory with text files, each line of which represents
    an annotation for an image in the format: class_index x1 y1 x2 y2 x3 y3 x4 y4.

    Args:
    - root_path (str): The root directory containing 'train', 'val', and 'test' subdirectories.

    Each subdirectory should have a 'labels' folder with .txt files for annotations. The function maps class indices
    to class names and prints the count of labels for each class within each dataset set.
    """
    # Mapping of class indices to their names for printing
    class_names = {
        0: "3/4_left_sideview",
        1: "3/4_right_sideview",
        2: "Frontal",
        3: "Left_sideview",
        4: "Right_sideview",
    }

    # Define subdirectories to explore
    subdirectories = ["train", "val", "test"]

    # Iterate through each subdirectory
    for subdirectory in subdirectories:
        path_labels = os.path.join(root_path, subdirectory, "labels")
        # Dictionary to count occurrences of each class in the current subdirectory
        class_counts = defaultdict(int)

        # List all .txt files in the labels subdirectory
        label_files = [f for f in os.listdir(path_labels) if f.endswith(".txt")]

        # Read each file and count the labels
        for file in label_files:
            with open(os.path.join(path_labels, file), "r") as f:
                for line in f:
                    # Ensure the line is not empty
                    data = line.strip().split()
                    if data:
                        class_index = int(data[0])
                        class_counts[class_index] += 1

        # Print the count of labels per class for the current subdirectory in the specified order
        print(f"Label count for the {subdirectory} set:")
        for class_index in sorted(class_names.keys()):
            class_name = class_names[class_index]
            count = class_counts[class_index]
            print(f"  {class_name}: {count}")
        print("")  # Blank line to separate the sets


def flip_coordinates(labels: List[str]) -> List[str]:
    """
    Updates the class and coordinates of a YOLO label after a horizontal flip of the image.

    Args:
        - labels: List of strings, where the first element is the class index, and the following eight are coordinates.

    Returns:
        - List of strings updated with the flipped class and modified coordinates.
    """
    # Update class
    class_id = int(labels[0])
    # Swap class indices for side and 3/4 views
    if class_id in [0, 1]:
        labels[0] = str(1 - class_id)
    elif class_id in [3, 4]:
        labels[0] = str(7 - class_id)

    # Update coordinates
    coords = np.array(labels[1:], dtype=float).reshape(4, 2)
    coords[:, 0] = 1 - coords[:, 0]  # Reflect the x coordinates
    coords = coords[
        [1, 0, 3, 2], :
    ]  # Reorder points to maintain bounding box consistency

    return [labels[0]] + coords.flatten().tolist()


def generate_horizontal_flipped_images(root_dir: str, output_dir: str) -> None:
    """
    Processes a directory of images and labels, applying a horizontal flip to images with a single annotation
    of certain classes and updating their labels accordingly.

    Parameters:
    - root_dir: Root directory containing 'images' and 'labels' folders.
    - output_dir: Output directory where the modified images and labels will be saved.
    """
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    # Create output directories if they do not exist
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):  # Ensure only .jpg images are processed
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

            if not os.path.exists(
                label_path
            ):  # Skip images without a corresponding label file
                continue  # Move to the next image in the directory

            # Read and process label
            with open(label_path, "r") as file:
                labels = file.readlines()

            if len(labels) == 1:  # Process only if there is a single annotation
                label = labels[0].strip().split()
                if int(label[0]) in [
                    0,
                    1,
                    3,
                    4,
                ]:  # Check if the annotation is of an interested class
                    image = Image.open(image_path)
                    flipped_image = image.transpose(
                        Image.FLIP_LEFT_RIGHT
                    )  # Apply horizontal flip

                    new_label = flip_coordinates(label)  # Update label

                    # Construct filename with 'flip_' prefix
                    new_filename = f"flip_{filename}"
                    flipped_image.save(
                        os.path.join(output_dir, "images", new_filename)
                    )  # Save modified image

                    new_label_str = " ".join(map(str, new_label))
                    with open(
                        os.path.join(
                            output_dir, "labels", new_filename.replace(".jpg", ".txt")
                        ),
                        "w",
                    ) as f:
                        f.write(new_label_str)  # Save modified label
    print(f"All images and labels have been flipped and saved in {output_dir}")


def convert_obb_to_aabb(root_dir: str, dest_dir: str, save_label: bool = False) -> None:
    """
    Converts Oriented Bounding Box (OBB) annotations to Axis-Aligned Bounding Box (AABB) annotations and adjusts the coordinates to the actual dimensions of the images.

    Args:
        - root_dir (str): The root directory containing 'images' and 'labels' folders.
        - dest_dir (str): The destination directory to save the converted annotations.
        - save_label (bool, optional): Whether to save the class label with the annotations. Defaults to False.
    """
    labels_dir = os.path.join(root_dir, "labels")
    images_dir = os.path.join(root_dir, "images")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate through all label files in the labels directory
    for label_file in os.listdir(labels_dir):
        # Check if the file is a text file (.txt)
        if label_file.endswith(".txt"):
            # Construct the corresponding image path
            image_path = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))
            # Path to the original label file
            label_path = os.path.join(labels_dir, label_file)
            # Destination path for the converted label file
            dest_path = os.path.join(dest_dir, label_file)

            # Read the dimensions of the image
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            with open(label_path, "r") as file:
                # Read all lines (annotations) in the label file
                lines = file.readlines()

            with open(dest_path, "w") as file:
                for line in lines:
                    # Split each line and convert coordinates to floats, skipping the class index if necessary
                    parts = line.strip().split()
                    obb_coords = list(
                        map(float, parts[1:])
                    )  # Convert to float and omit the class index

                    # Convert OBB coordinates to AABB
                    x_coords = obb_coords[0::2]  # Extract x coordinates
                    y_coords = obb_coords[1::2]  # Extract y coordinates
                    # Find the minimum and maximum x and y coordinates
                    x_min, y_min, x_max, y_max = (
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords),
                    )

                    # Denormalize the coordinates
                    x_min, x_max = int(x_min * width), int(x_max * width)
                    y_min, y_max = int(y_min * height), int(y_max * height)

                    # Write the converted annotation to the destination file
                    if save_label:
                        # Include the class label in the annotation
                        file.write(f"{parts[0]} {x_min} {y_min} {x_max} {y_max}\n")
                    else:
                        # Exclude the class label from the annotation
                        file.write(f"{x_min} {y_min} {x_max} {y_max}\n")


def copy_corresponding_jsons(json_dir: str, images_dir: str, output_dir: str) -> None:
    """
    Copies .json files corresponding to the images in dir_images to the output_dir.

    Args:
        - json_dir (str): The directory where the .json files are located.
        - images_dir (str): The directory where the images are located.
        - output_dir (str): The directory where the corresponding .json files will be saved.
    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the base names of the image files (without the extension)
    image_names = [
        os.path.splitext(file)[0]
        for file in os.listdir(images_dir)
        if file.endswith(".jpg")
    ]

    # Loop through the .json files in dir_json
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            # Get the base name of the .json file (without the extension)
            base_name_json: str = os.path.splitext(json_file)[0]

            # If the base name of the .json file is in the list of image names
            if base_name_json in image_names:
                # Build the full paths
                full_json_path: str = os.path.join(json_dir, json_file)
                destination_json_path: str = os.path.join(output_dir, json_file)

                # Copy the .json file to the output directory
                shutil.copy(full_json_path, destination_json_path)
                print(f"Copied: {json_file} to {output_dir}")

    print("Process completed.")


def create_directories(path: str) -> None:
    """
    Ensure the existence of a directory.

    This function checks if a directory exists at the specified path,
    and if not, it creates the directory, including any necessary parent directories.

    Args:
        - path (str): The path to the directory that should exist.

    Returns:
        - None
    """
    os.makedirs(path, exist_ok=True)


def split_annotations_and_create_new_files_per_face(
    root_dir: str, output_dir: str
) -> None:
    """
    Splits annotations in JSON and TXT files when there are multiple faces in a single image.
    It creates individual JSON, TXT, and copies of image files for each face.

    The function expects a directory structure with 'images', 'json', and 'labels' subdirectories
    in the provided root directory. It processes each JSON file and checks for multiple annotations (faces).
    For each face, it creates a new JSON and TXT file, and a copy of the corresponding image file, appending
    an identifier to the filename to distinguish between different faces.

    Args:
        - root_dir (str): The root directory that contains the 'images', 'json', and 'labels' directories.
        - output_dir (str): The output directory where the newly created files will be stored, maintaining the
                            original directory structure with 'images', 'json', and 'labels' subdirectories.

    Returns:
        - None
    """
    # Creating output images directory path
    output_images_dir = os.path.join(output_dir, "images")
    # Creating output JSON directory path
    output_json_dir = os.path.join(output_dir, "json")
    # Creating output labels directory path
    output_labels_dir = os.path.join(output_dir, "labels")

    # Creating output images directory
    create_directories(output_images_dir)
    # Creating output JSON directory
    create_directories(output_json_dir)
    # Creating output labels directory
    create_directories(output_labels_dir)

    # Setting path to images directory in root directory
    images_dir = os.path.join(root_dir, "images")
    # Setting path to JSON directory in root directory
    json_dir = os.path.join(root_dir, "json")
    # Setting path to labels directory in root directory
    labels_dir = os.path.join(root_dir, "labels")

    # Looping through files in JSON directory
    for file_name in os.listdir(json_dir):
        # Checking if file is a JSON file
        if file_name.endswith(".json"):
            # Extracting base name of file
            base_name = os.path.splitext(file_name)[0]
            # Setting path to current JSON file
            json_path = os.path.join(json_dir, file_name)

            # Opening current JSON file
            with open(json_path, "r") as f:
                # Loading JSON data
                data = json.load(f)

                # New JSON data with a single annotation
                # Creating new JSON data with single annotation
                if len(data["label"]) > 1:
                    # Looping through each annotation
                    for i, label in enumerate(data["label"]):
                        # Creating new JSON data with single annotation
                        new_data = {**data, "label": [label]}
                        # Setting path to new JSON file
                        new_json_path = os.path.join(
                            output_json_dir, f"{base_name}_copy_{i}.json"
                        )

                        # Opening new JSON file
                        with open(new_json_path, "w") as nf:
                            # Writing new JSON data to file
                            json.dump(new_data, nf, indent=4)

                        # Setting path to corresponding TXT file
                        txt_path = os.path.join(labels_dir, f"{base_name}.txt")
                        # Checking if TXT file exists
                        if os.path.exists(txt_path):
                            # Opening corresponding TXT file
                            with open(txt_path, "r") as tf:
                                # Reading lines from TXT file
                                lines = tf.readlines()
                                # Checking if current index is within the range of lines
                                if i < len(lines):
                                    # Setting path to new TXT file
                                    new_txt_path = os.path.join(
                                        output_labels_dir, f"{base_name}_copy_{i}.txt"
                                    )

                                    # Opening new TXT file
                                    with open(new_txt_path, "w") as ntf:
                                        # Writing corresponding line to new TXT file
                                        ntf.write(lines[i])

                        # Setting path to corresponding image file
                        img_path = os.path.join(images_dir, f"{base_name}.jpg")
                        # Checking if image file exists
                        if os.path.exists(img_path):
                            # Setting path to new image file
                            new_img_path = os.path.join(
                                output_images_dir, f"{base_name}_copy_{i}.jpg"
                            )
                            # Copying image file with identifier
                            shutil.copy(img_path, new_img_path)


def rotate_point(
    cx: float, cy: float, angle: float, px: float, py: float
) -> Tuple[float, float]:
    """
    Rotate a point around a given center.

    Args:
        - cx: x-coordinate of the center point to rotate around.
        - cy: y-coordinate of the center point to rotate around.
        - angle: Angle in degrees to rotate the point in a clockwise direction.
        - px: x-coordinate of the point to rotate.
        - py: y-coordinate of the point to rotate.

    Returns:
        - The new x and y coordinates of the rotated point.

    The function converts the angle from degrees to radians and applies a standard rotation matrix
    to the point coordinates. The rotation is performed in a clockwise direction by negating the angle.
    """
    # Convert angle to radians
    angle_rad = np.radians(-angle)  # Negative for clockwise rotation
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)

    # Translate point back to origin:
    px -= cx
    py -= cy

    # Rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # Translate point back
    px = xnew + cx
    py = ynew + cy
    return px, py


def get_rotated_bbox_coords(
    coords: list, rotation_angle: float, original_width: int, original_height: int
) -> Tuple[int, int, int, int]:
    """
    Calculate the coordinates of a rotated bounding box after the image has been rotated by a specific angle.

    Args:
        - coords (list): List of original bounding box coordinates (x1, y1, x2, y2, x3, y3, x4, y4) as percentages of the image dimensions.
        - rotation_angle (float): The angle in degrees by which the image is rotated clockwise.
        - original_width (int): The width of the original image in pixels.
        - original_height (int): The height of the original image in pixels.

    Returns:
        - Tuple[int, int, int, int]: The top-left x and y coordinates, width, and height of the new axis-aligned bounding box in pixels.

    This function first converts the percentage coordinates into pixel values.
    It then computes the center of the original image and uses it as the pivot point to calculate the rotated coordinates of the bounding box.
    Finally, it determines the minimum axis-aligned bounding box that encapsulates the rotated coordinates.
    """
    # Convert percentage coordinates to pixels
    x1, y1 = coords[0] * original_width, coords[1] * original_height
    x2, y2 = coords[2] * original_width, coords[3] * original_height
    x3, y3 = coords[4] * original_width, coords[5] * original_height
    x4, y4 = coords[6] * original_width, coords[7] * original_height

    # Calculate the center of the original image
    original_center_x, original_center_y = original_width / 2, original_height / 2

    # Rotate the bounding box corners around the original image's center
    rot_x1, rot_y1 = rotate_point(
        original_center_x, original_center_y, rotation_angle, x1, y1
    )
    rot_x2, rot_y2 = rotate_point(
        original_center_x, original_center_y, rotation_angle, x2, y2
    )
    rot_x3, rot_y3 = rotate_point(
        original_center_x, original_center_y, rotation_angle, x3, y3
    )
    rot_x4, rot_y4 = rotate_point(
        original_center_x, original_center_y, rotation_angle, x4, y4
    )

    # Calculate the bounding box's new location in the rotated image
    rotated_center_x, rotated_center_y = rotate_point(
        original_center_x,
        original_center_y,
        rotation_angle,
        original_center_x,
        original_center_y,
    )
    bbox_center_x, bbox_center_y = (rot_x1 + rot_x3) / 2, (rot_y1 + rot_y3) / 2
    new_cx, new_cy = rotated_center_x + (
        bbox_center_x - original_center_x
    ), rotated_center_y + (bbox_center_y - original_center_y)

    # Calculate the top-left corner of the new bounding box
    new_x1, new_y1 = new_cx - (rot_x2 - rot_x1) / 2, new_cy - (rot_y3 - rot_y1) / 2
    # Calculate the width and height of the new bounding box
    new_width = np.sqrt((rot_x2 - rot_x1) ** 2 + (rot_y2 - rot_y1) ** 2)
    new_height = np.sqrt((rot_x3 - rot_x2) ** 2 + (rot_y3 - rot_y2) ** 2)

    # Return the new bounding box coordinates
    return int(new_x1), int(new_y1), int(new_width), int(new_height)


def get_new_image_dimensions(w: int, h: int, angle: float):
    """
    Calculates the new dimensions of the image after rotation.

    Args:
        - w (int): The original width of the image in pixels.
        - h (int): The original height of the image in pixels.
        - angle (float): The angle in degrees by which the image is rotated clockwise.

    Returns:
        - Tuple[int, int]: The new width and height of the image after rotation.

    This function calculates the new dimensions of an image after rotation by a given angle.
    It uses the formula:
    new_width = h * sin(angle) + w * cos(angle)
    new_height = h * cos(angle) + w * sin(angle)
    """
    angle_rad = np.radians(angle)
    cos_angle = abs(np.cos(angle_rad))
    sin_angle = abs(np.sin(angle_rad))

    # Calculate the new width and height of the image
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))

    return new_w, new_h


def adjust_bbox_after_image_rotation(
    new_tl_x: int,
    new_tl_y: int,
    new_w: int,
    new_h: int,
    orig_cx: int,
    orig_cy: int,
    new_cx: int,
    new_cy: int,
):
    """
    Adjust the coordinates of the bounding box after the image has been rotated.

    Args:
        - new_tl_x (int): Top-left x-coordinate of the bounding box in the rotated image.
        - new_tl_y (int): Top-left y-coordinate of the bounding box in the rotated image.
        - new_w (int): Width of the bounding box.
        - new_h (int): Height of the bounding box.
        - orig_cx (int): Original center x-coordinate of the image before rotation.
        - orig_cy (int): Original center y-coordinate of the image before rotation.
        - new_cx (int): New center x-coordinate of the rotated image.
        - new_cy (int): New center y-coordinate of the rotated image.

    Returns:
        tuple: Adjusted coordinates of the bounding box (top-left x, top-left y, width, height).

    This function adjusts the coordinates of a bounding box after the image has been rotated.
    It calculates the shift of the center of the bounding box after rotation and adjusts the top-left coordinates accordingly.
    """
    # Calculate the shift of the center after rotation
    shift_x = new_cx - orig_cx
    shift_y = new_cy - orig_cy

    # Adjust the bounding box coordinates by the shift
    adjusted_x = int(new_tl_x + shift_x)
    adjusted_y = int(new_tl_y + shift_y)

    return adjusted_x, adjusted_y, new_w, new_h


def visualize_rotated_images_and_aabboxes(
    root_dir: str,
    output_dir: str,
    max_images_per_grid: int,
    display_grid: bool = True,
    normalized_coords: bool = True,
) -> None:
    """
    Processes images by rotating them, calculating new bounding box coordinates, saving them in specified directories,
    and optionally displaying a selection of them in a grid.

    Args:
        root_dir: The root directory containing 'json', 'images', and 'labels' subdirectories.
        output_dir: The output directory where 'images' and 'labels' folders will be created.
        max_images_per_grid: The maximum number of images to display or process per grid.
        display_grid: Flag to decide if a grid of images is to be displayed.
        normalized_coords: Flag to decide if bounding box coordinates should be saved as normalized values or pixels.
    """
    # Ensure output directories exist
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")
    create_directories(images_output_dir)
    create_directories(labels_output_dir)

    # JSON, Images and Labels directory paths
    json_dir = os.path.join(root_dir, "json")
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    # Get all image files and select random images for display
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.shuffle(image_files)
    images_to_display = image_files[:max_images_per_grid] if display_grid else []

    # Create grid for visualization if needed
    if display_grid:
        # Initialize the grid only if display is needed
        rows = cols = int(np.ceil(np.sqrt(len(images_to_display))))
        fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 5))
        axs = axs.ravel() if rows * cols > 1 else [axs]

    for idx, image_file in enumerate(image_files):
        # Paths for current image and annotation files
        image_path = os.path.join(images_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        json_path = os.path.join(json_dir, base_name + ".json")
        txt_path = os.path.join(labels_dir, base_name + ".txt")

        # Process JSON and TXT files
        with open(json_path, "r") as json_file:
            annotation_data = json.load(json_file)
            bbox_data = annotation_data["label"][0]

            # Process image
            image = cv2.imread(image_path)
            rotated_image = ndimage.rotate(image, bbox_data["rotation"], reshape=True)

            # Process bounding box
            with open(txt_path, "r") as f:
                coords = [float(i) for i in f.read().split()[1:]]
            new_tl_x, new_tl_y, new_w, new_h = get_rotated_bbox_coords(
                coords,
                bbox_data["rotation"],
                bbox_data["original_width"],
                bbox_data["original_height"],
            )

            new_img_width, new_img_height = get_new_image_dimensions(
                bbox_data["original_width"],
                bbox_data["original_height"],
                bbox_data["rotation"],
            )

            adjusted_bbox_coords = adjust_bbox_after_image_rotation(
                new_tl_x,
                new_tl_y,
                new_w,
                new_h,
                bbox_data["original_width"] / 2,
                bbox_data["original_height"] / 2,
                new_img_width / 2,
                new_img_height / 2,
            )

            # Save rotated image
            cv2.imwrite(os.path.join(images_output_dir, image_file), rotated_image)

            # Save bounding box coordinates
            bbox_file_path = os.path.join(labels_output_dir, f"{base_name}.txt")
            with open(bbox_file_path, "w") as f:
                if normalized_coords:
                    adjusted_bbox_coords_percentage = [
                        adjusted_bbox_coords[0] / new_img_width,
                        adjusted_bbox_coords[1] / new_img_height,
                        adjusted_bbox_coords[2] / new_img_width,
                        adjusted_bbox_coords[3] / new_img_height,
                    ]
                    f.write(" ".join(map(str, adjusted_bbox_coords_percentage)) + "\n")
                else:
                    f.write(
                        f"{adjusted_bbox_coords[0]} {adjusted_bbox_coords[1]} {adjusted_bbox_coords[2]} {adjusted_bbox_coords[3]}\n"
                    )

            # If display_grid is True, display only the selected images
            if display_grid and image_file in images_to_display:
                display_idx = images_to_display.index(image_file)
                # Draw bounding box on image for display
                display_img = cv2.rectangle(
                    rotated_image,
                    (adjusted_bbox_coords[0], adjusted_bbox_coords[1]),
                    (
                        adjusted_bbox_coords[0] + adjusted_bbox_coords[2],
                        adjusted_bbox_coords[1] + adjusted_bbox_coords[3],
                    ),
                    (0, 255, 0),
                    2,
                )
                # Convert image for display
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                # Display image in the grid
                ax = axs[display_idx]
                ax.imshow(display_img)
                ax.axis("on")
                # Set the title with the image name
                ax.set_title(base_name, fontsize=10)

    # Only show the grid if we're displaying
    if display_grid:
        # Hide any unused subplots if there are fewer images than grid cells
        for ax in axs[len(images_to_display) :]:
            ax.axis("off")
        # Tight layout often produces better-looking grids
        plt.tight_layout()
        plt.show()
