from detectron2.structures import BoxMode
from detectron2.data import transforms as T, DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils


import cv2
import torch
import numpy as np
from PIL import Image

import os
import json
import hashlib
from typing import List, Tuple
from pathlib import Path


def create_data_pairs(
    input_path: str,
    detectron_img_path: str,
    detectron_annot_path: str,
    dir_type: str = "train",
) -> list:
    """
    Creates pairs of image and annotation file paths.

    Args:
        - input_path (str): The base directory path of the dataset.
        - detectron_img_path (str): The directory path where Detectron2 will look for images.
        - detectron_annot_path (str): The directory path where Detectron2 will look for annotations.
        - dir_type (str, optional): The subdirectory within the dataset to use (default is 'train').

    Returns:
        - list: A list of pairs, where each pair contains the path to an image and its corresponding annotation file.
    """

    # Find all JPEG image paths in the specified directory
    img_paths = sorted(Path(input_path + dir_type + "/images/").glob("*.jpg"))

    # Initialize a list to hold the pairs
    pairs = []
    for img_path in img_paths:
        # Extract the base filename without extension
        file_name_tmp = str(img_path).split("/")[-1].split(".")
        # Remove the file extension
        file_name_tmp.pop(-1)
        # Rejoin the remaining filename parts
        file_name = ".".join((file_name_tmp))

        # Construct the full path to the corresponding annotation file
        label_path = Path(input_path + dir_type + "/labels/" + file_name + ".txt")

        # Check if the annotation file exists, and if so, add the image and annotation paths to the list
        if label_path.is_file():
            line_img = detectron_img_path + dir_type + "/images/" + file_name + ".jpg"
            line_annot = (
                detectron_annot_path + dir_type + "/labels/" + file_name + ".txt"
            )
            pairs.append([line_img, line_annot])

    return pairs


def create_coco_format(data_pairs: list) -> list:
    """
    Converts data pairs into COCO format for object detection.

    Args:
        - data_pairs (list): A list of pairs, where each pair contains the path to an image and its corresponding annotation file.

    Returns:
        - list: A list of dictionaries, where each dictionary represents an image and its annotations in COCO format.
    """

    # Initialize a list to hold data in COCO format
    data_list = []

    for i, path in enumerate(data_pairs):
        # Get the filename and dimensions of the image
        filename = path[0]
        img_h, img_w = cv2.imread(filename).shape[:2]

        # Initialize a dictionary for this image
        img_item = {
            "file_name": filename,
            "image_id": i,
            "height": img_h,
            "width": img_w,
        }

        # Print the image number and filename
        print(str(i), filename)

        # Initialize a list to hold annotations for this image
        annotations = []
        with open(path[1]) as annot_file:
            lines = annot_file.readlines()

            for line in lines:
                # Split the line into components and strip newline if present
                box = line[:-1].split(" ") if line[-1] == "\n" else line.split(" ")

                # Parse annotation components
                class_id, x_c, y_c, width, height = map(float, box[:5])

                # Calculate the bounding box coordinates in absolute terms
                x1 = (x_c - (width / 2)) * img_w
                y1 = (y_c - (height / 2)) * img_h
                x2 = (x_c + (width / 2)) * img_w
                y2 = (y_c + (height / 2)) * img_h

                # Create the annotation dictionary
                annotation = {
                    "bbox": [x1, y1, x2, y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(class_id),
                    "iscrowd": 0,
                }
                annotations.append(annotation)

            # Add the annotations to the image item
            img_item["annotations"] = annotations

        # Add the image item to the dataset list
        data_list.append(img_item)

    return data_list


def resize_image(image, target_size):
    """
    Resize an image maintaining its aspect ratio.

    Args:
        image (PIL.Image): The original image.
        target_size (int): The desired maximum size (either width or height).

    Returns:
        PIL.Image: The resized image.
        float: The scaling factor applied to the image dimensions.
    """
    original_size = max(image.size)
    if original_size <= target_size:
        return image, 1.0  # No scaling applied
    ratio = target_size / float(original_size)
    new_size = tuple([int(x * ratio) for x in image.size])
    return image.resize(new_size, Image.ANTIALIAS), ratio


def rotate_bbox(annotation: dict, transforms: List[T.Transform]) -> dict:
    """
    Rotates bounding boxes in the annotation according to the specified transformations.

    This function checks the bounding box mode in the annotation and applies rotation
    transformations if supported. It currently handles only the BoxMode.XYWHA_ABS mode.

    Args:
        annotation: A dictionary containing annotation details, including the bounding box
                    and its mode (e.g., XYWHA_ABS).
        transforms: A list of transformations to apply to the bounding box. These transformations
                    should have an 'apply_rotated_box' method if they are to be applied.

    Returns:
        dict: The updated annotation dictionary with the rotated bounding box.

    Note:
        This function currently only supports annotations in the XYWHA_ABS box mode and will
        skip transformations for other modes.
    """
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
        # Directly convert to a PyTorch tensor instead of first creating a list of numpy arrays
        rotated_bbox = torch.as_tensor(
            annotation["bbox"], dtype=torch.float32
        ).unsqueeze(0)

        for transform in transforms:
            if hasattr(transform, "apply_rotated_box"):
                # Ensure that rotated_bbox is a PyTorch tensor before applying the transformation
                rotated_bbox = transform.apply_rotated_box(rotated_bbox)

        annotation["bbox"] = rotated_bbox.squeeze(0).numpy()
    else:
        # Other box modes are currently not handled
        pass

    return annotation


def adjust_bbox_for_resizing(annotation, scale_factor):
    """
    Adjusts bounding box coordinates according to the scaling factor.

    Args:
        annotation (dict): The annotation containing bbox details.
        scale_factor (float): The scaling factor applied to the image dimensions.

    Returns:
        dict: The updated annotation with adjusted bounding box.
    """
    bbox = annotation["bbox"]
    # Adjust the bounding box coordinates
    adjusted_bbox = [coord * scale_factor for coord in bbox]
    annotation["bbox"] = adjusted_bbox
    return annotation


def get_shape_augmentations() -> List[T.Transform]:
    """
    Creates a list of optional shape augmentation transformations.

    This function returns a list of transformations that modify the shape aspects of the image
    such as flipping, rotation, and cropping.

    Returns:
        List[Transform]: A list of transformations for shape augmentation, including random flip,
                        rotation within a range, and relative range cropping.
    """
    # Define and return a list of shape-related augmentations
    return [
        T.RandomFlip(),  # Randomly flips the image
        T.RandomRotation(
            angle=[-30, 30], expand=True
        ),  # Randomly rotates the image within -30 to 30 degrees
        T.RandomCrop(
            "relative_range", (0.8, 0.8)
        ),  # Randomly crops the image in the specified relative range
    ]


def get_color_augmentations() -> T.AugmentationList:
    """
    Creates an augmentation list for optional color transformations.

    This function returns an AugmentationList object containing transformations that
    alter the color properties of the image, including brightness, contrast, saturation,
    and lighting.

    Returns:
        T.AugmentationList: A list of color-related transformations encapsulated in an
                            AugmentationList object.
    """
    # Define and return an augmentation list with color-related transformations
    return T.AugmentationList(
        [
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomLighting(0.8),
        ]
    )


def dataset_mapper(dataset_dict: dict, target_size=512) -> dict:
    """
    Maps and applies transformations to the dataset dictionary.

    This function reads an image from the dataset, applies color and shape augmentations,
    and processes the annotations accordingly. It is used to prepare the dataset for
    training with augmented data.

    Args:
        - dataset_dict: A dictionary containing dataset information, including file names and
                    annotations.

    Returns:
        dict: The updated dataset dictionary with transformed images and annotations.

    Note:
        This function applies transformations only to non-crowd annotations. In the context
        of Detectron2, non-crowd annotations refer to individual object instances in an image
        marked for detection. Crowd annotations, which denote densely packed objects of the
        same class where individual instances are not distinctly marked, are not transformed.
        This distinction is crucial for training models on images with both clear individual
        instances and densely populated areas.
    """
    # Read the image in BGR format
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # Resize image
    pil_image = Image.fromarray(image)
    pil_image, scale_factor = resize_image(pil_image, target_size)
    image = np.array(pil_image)

    # Apply color augmentations
    color_aug_input = T.AugInput(image)
    get_color_augmentations()(color_aug_input)
    image = color_aug_input.image

    # Apply shape augmentations and get the resulting transforms
    image, image_transforms = T.apply_transform_gens(get_shape_augmentations(), image)

    # Convert the image to a PyTorch tensor
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    # Delete image for releasing memory
    del image

    # Apply transformations to annotations, filtering out crowd objects
    # Apply transformations to annotations, adjusting bbox for resizing
    annotations = [
        adjust_bbox_for_resizing(rotate_bbox(obj, image_transforms), scale_factor)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    # Convert the updated annotations to rotated instances
    instances = utils.annotations_to_instances_rotated(annotations, image.shape[:2])

    # Filter out empty instances
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


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


def convert_bbox_format(
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float,
    original_width: int,
    original_height: int,
) -> tuple:
    """
    Converts bounding box format from Label Studio to Detectron2 specifications.

    Args:
        - x (float): Percentage x-coordinate of the top-left corner of the bounding box.
        - y (float): Percentage y-coordinate of the top-left corner of the bounding box.
        - width (float): Percentage width of the bounding box.
        - height (float): Percentage height of the bounding box.
        - rotation (float): Clockwise rotation angle in degrees (Label Studio format).
        - original_width (int): Original width of the image in pixels.
        - original_height (int): Original height of the image in pixels.

    Returns:
        tuple: Converted bounding box (center_x, center_y, width, height, angle) in Detectron2 format.
    """
    # Convert percentages to absolute coordinates in pixels for x, y, width, and height
    x_abs = x / 100.0 * original_width
    y_abs = y / 100.0 * original_height
    width_abs = width / 100.0 * original_width
    height_abs = height / 100.0 * original_height

    # Convert clockwise rotation to counter-clockwise for Detectron2 and adjust angle range
    angle_detectron = -rotation % 360
    if angle_detectron > 180:
        angle_detectron -= 360

    # Calculate the center of the unrotated box
    center_x = x_abs + width_abs / 2
    center_y = y_abs + height_abs / 2

    # Convert rotation angle to radians and compute cosine and sine components
    rotation_radians = np.deg2rad(rotation)
    cos_angle = np.cos(rotation_radians)
    sin_angle = np.sin(rotation_radians)

    # Calculate new center coordinates after applying rotation matrix to bounding box dimensions
    rotated_center_x = x_abs + (width_abs * cos_angle - height_abs * sin_angle) / 2
    rotated_center_y = y_abs + (width_abs * sin_angle + height_abs * cos_angle) / 2

    return rotated_center_x, rotated_center_y, width_abs, height_abs, angle_detectron


def label_studio_to_detectron_dataset(directory: str, class_labels: List[str]) -> tuple:
    """
    Converts a dataset from Label Studio JSON format to Detectron2 dataset format.

    Args:
        - directory: Directory where the JSON files and images are stored.
        - class_labels: List of class labels to be used in the dataset.

    Returns:
        List of dictionaries, each representing an image with its annotations in Detectron2 format.
    """
    # List the files in the provided directory
    files = os.listdir(directory)
    # Initialize list to store image data in Detectron2 format
    images = []
    # Initialize list to store class labels
    classes = []

    # Iterate over each file in the directory
    for filename in files:
        # Skip files that are not JSON files
        if ".json" not in filename:
            continue

        # Construct full path to the JSON file
        path = os.path.join(directory, filename)
        # Open and read the JSON file
        with open(path, "rt") as f:
            data = json.load(f)

        # Initialize list to store annotations for the current image
        annotations = []
        # Iterate over each annotation in the JSON file
        for annotation in data["label"]:
            # Extract the label (class) of the annotation
            label = annotation["rectanglelabels"][0]
            # Add label to the classes list if it's not already present
            if label not in classes:
                classes.append(label)

            # Convert Label Studio bbox format to Detectron2 bbox format
            cx, cy, w, h, a = convert_bbox_format(
                x=annotation["x"],
                y=annotation["y"],
                width=annotation["width"],
                height=annotation["height"],
                rotation=annotation["rotation"],
                original_width=annotation["original_width"],
                original_height=annotation["original_height"],
            )

            # Add the converted bbox to the annotations list
            annotations.append(
                {
                    "bbox_mode": 4,  # Oriented bounding box format
                    "category_id": class_labels.index(
                        label
                    ),  # Map label to category ID
                    "bbox": (cx, cy, w, h, a),  # Bounding box data
                }
            )

        # Construct the full path to the corresponding image file
        image_file_name = os.path.join(directory, data["image"].split("/")[-1])

        # Add image data to the images list
        images.append(
            {
                "id": data["id"],  # Image ID
                "file_name": image_file_name,  # Path to image file
                "height": annotation["original_height"],  # Image height
                "width": annotation["original_width"],  # Image width
                "annotations": annotations,  # Annotations for the image
            }
        )

    return images


def prepare_dataset_for_detectron(directory: str, class_labels: list) -> list:
    """
    Prepares and converts a dataset from Label Studio format to Detectron2 format.

    Args:
        - directory (str): Directory path where the Label Studio JSON files and images are stored.
        - class_labels (list): List of class labels used in the annotations.

    Returns:
        list: A list of dictionaries, each representing an image with its annotations in Detectron2 format.
    """
    return label_studio_to_detectron_dataset(directory, class_labels)


def register_datasets(root_dir: str, class_labels: list):
    """
    Registers and retrieves the sizes of training, validation, and test datasets in Detectron2,
    along with the number of classes.
    Args:
        root_dir (str): The root directory where dataset directories ('train', 'val', 'test') are located.
        class_labels (list): List of class labels for the dataset.

    Returns:
        tuple: A tuple containing the sizes of the training, validation, and test datasets,
            and the number of classes.

    This function registers each dataset (train, val, test) with Detectron2, preparing it for
    training and evaluation. If a dataset is already registered, it skips re-registration and
    retrieves the dataset size.
    """

    dataset_sizes = {}

    # Iterate over each dataset type (train, val, test)
    for dataset_name in ["train", "val", "test"]:
        # Construct the directory path for the current dataset type
        dataset_dir = os.path.join(root_dir, dataset_name, "images")

        # Check if the dataset is already registered in Detectron2
        if dataset_name not in DatasetCatalog:
            # Register the dataset if it's not already registered
            # The lambda function is used to delay the execution of prepare_dataset_for_detectron
            # until the dataset is actually needed
            DatasetCatalog.register(
                dataset_name,
                lambda d=dataset_dir: prepare_dataset_for_detectron(d, class_labels),
            )
            # Set metadata (like class labels) for the dataset
            MetadataCatalog.get(dataset_name).set(thing_classes=class_labels)
        else:
            # Print a message if the dataset is already registered
            print(f"Dataset '{dataset_name}' already registered.")

        # Retrieve the dataset and calculate its size
        dataset_dicts = DatasetCatalog.get(dataset_name)
        dataset_sizes[dataset_name] = len(dataset_dicts)

    return dataset_sizes["train"], dataset_sizes["val"], dataset_sizes["test"]


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
