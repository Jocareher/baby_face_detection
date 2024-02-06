from typing import List, Optional, Dict
import math
import random

from detectron2.structures import BoxMode
from detectron2.data.transforms import TransformList
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from data_setup import convert_bbox_format


def draw_bounding_boxes(image, annotations):
    """
    Draws bounding boxes on an image, including support for rotated boxes.

    Parameters:
    - image: The image on which to draw.
    - annotations: A list of annotations where each annotation contains the bbox.
                    Bbox format can be either [x1, y1, x2, y2] or [cx, cy, w, h, angle].
    """
    for annotation in annotations:
        bbox = annotation["bbox"]
        if len(bbox) == 4:  # Non-rotated bbox
            # Draw the bounding box
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )
        elif len(bbox) == 5:  # Rotated bbox
            cx, cy, w, h, angle = bbox
            # Calculate the four corners of the rotated bbox
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Draw the rotated bounding box
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        else:
            print(f"Invalid bbox format: {bbox}")
            continue

    return image


def visualize_transformations(
    image_path: str,
    annotations: List[Dict],
    transform_gens: List,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Visualizes the effect of transformations on an image, supporting both rotated and non-rotated bounding boxes.

    Parameters:
    - image_path (str): Path to the image to be transformed.
    - annotations (List[Dict]): List of annotations for the image. Each annotation is a dict with a 'bbox' key.
    - transform_gens (List): List of transformation generators to apply.
    - metadata (Optional[Dict]): Metadata for the image, can include information like class labels.

    The function does not return anything; it visualizes the original and transformed images.
    """
    # Load the original image from the specified path
    original_image = cv2.imread(image_path)
    transformed_image = original_image.copy()

    # Convert annotations to XYXY_ABS format if needed and check for rotation
    for annot in annotations:
        if "bbox_mode" in annot and annot["bbox_mode"] != BoxMode.XYXY_ABS:
            if len(annot["bbox"]) == 5:  # Check for rotated bbox format
                annot["bbox"] = convert_bbox_format(
                    *annot["bbox"], original_image.shape[1], original_image.shape[0]
                )
            else:
                annot["bbox"] = BoxMode.convert(
                    annot["bbox"], annot["bbox_mode"], BoxMode.XYXY_ABS
                )

    # Draw bounding boxes on the original image
    original_image_with_boxes = draw_bounding_boxes(original_image.copy(), annotations)

    # Apply transformations to the image and the bounding boxes
    transform_list = TransformList(
        [t.get_transform(original_image) for t in transform_gens]
    )
    transformed_image = transform_list.apply_image(transformed_image)

    # Update bounding boxes for the transformed image and draw them
    transformed_annotations = [
        dict(
            annot,
            bbox=transform_list.apply_box(np.array(annot["bbox"]).reshape(1, -1))[0],
        )
        for annot in annotations
    ]
    transformed_image_with_boxes = draw_bounding_boxes(
        transformed_image, transformed_annotations
    )

    # Display the original and transformed images using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(transformed_image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Transformed Image")
    plt.show()


def visualize_predictions(
    cfg_path: str,
    weights_path: str,
    dataset_name: str,
    num_classes: int,
    score_thresh: float,
    num_images: int,
    device: str = "cpu",
) -> None:
    """
    Visualizes random predictions on images from a specified dataset using a Detectron2 model.

    Args:
        cfg_path (str): Path to the configuration file for the model.
        weights_path (str): Path to the trained model weights.
        dataset_name (str): Name of the dataset to visualize (e.g., "val", "test").
        num_classes (int): Number of classes for the model.
        score_thresh (float): Score threshold for making predictions.
        num_images (int): Number of images to visualize randomly.
        device (str): Computation device to use ('cpu' or 'cuda').

    Displays a grid of images with model predictions including bounding boxes, labels, and confidence scores.
    """
    # Initialize Detectron2 configuration
    cfg = get_cfg()

    # Load model configuration
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
    cfg.merge_from_file("../configs/rotated_bbox_config.yaml")

    # Load trained model weights
    cfg.MODEL.WEIGHTS = weights_path

    # Set the computation device
    cfg.MODEL.DEVICE = device

    # Set the number of classes for the ROI heads
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Set the scoring threshold for object detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    # Define the test dataset
    cfg.DATASETS.TEST = (dataset_name,)

    # Initialize the predictor with the model configuration
    predictor = DefaultPredictor(cfg)

    # Retrieve dataset dictionaries from the registered dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Randomly select images from the dataset
    selected_dicts = random.sample(dataset_dicts, num_images)

    # Calculate the number of rows and columns for the grid
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Set up a matplotlib figure with the calculated grid size
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Initialize all subplots as invisible for non-square grid handling
    for ax in axs.flat:
        ax.axis("off")

    # Visualize the specified number of images
    for idx, d in enumerate(selected_dicts):
        # Read the image
        img = cv2.imread(d["file_name"])

        # Make prediction using the model
        outputs = predictor(img)

        # Convert BGR image to RGB for visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Visualize the predictions on the image
        v = Visualizer(
            img_rgb,
            metadata=MetadataCatalog.get(dataset_name).set(
                thing_classes=[
                    "3/4_left_sideview",
                    "3/4_rigth_sideview",
                    "Frontal",
                    "Left_sideview",
                    "Right_sideview",
                ]
            ),
            scale=0.5,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Find the corresponding subplot
        ax = axs[idx // num_cols, idx % num_cols]

        # Display the image with predictions in the grid
        ax.imshow(out.get_image())
        ax.axis("on")  # Or 'off' if you do not want to show the axes

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_losses(iterations: list, losses: dict) -> None:
    """
    Plots the extracted loss values over iterations.

    Args:
        iterations (list): List of iteration numbers.
        losses (dict): Dictionary containing different types of losses and their values.
    """
    # Create a figure with a specified size
    plt.figure(figsize=(10, 5))
    # Iterate through each loss type and plot it
    for loss_key, loss_values in losses.items():
        # Plot the loss values for each iteration
        plt.plot(iterations, loss_values, label=loss_key)
    # Label the x-axis as 'Iterations'
    plt.xlabel("Iterations")
    # Label the y-axis as 'Loss'
    plt.ylabel("Loss")
    # Title of the plot
    plt.title("Training Loss Over Iterations")
    # Display the legend to identify each loss type
    plt.legend()
    # Show the plot
    plt.show()

def draw_yolov8_annotations_on_images(pairs: list[tuple],
                               class_list: list[str],
                               num_images_to_display: int,
                               show_labels: bool = True,
                               show_axis: str = "on") -> None:
    """
    Draw annotations on images as polygons and display them in a grid.
    Optionally include class labels aligned with the top edge of the bounding box.

    Args:
    - pairs (list of tuples): A list containing tuples of image paths and annotation data.
    - class_list (list of str): A list of class names ordered by their corresponding class index.
    - num_images_to_display (int): The number of images to display on the grid.
    - show_labels (bool, optional): If True, display class labels. Default is True.
    - show_axis (str, optional): Control the visibility of the axis. Default is "on".

    Outputs:
    - Displays a grid of images with the respective annotations.
    """
    # Select a random subset of image-annotation pairs
    selected_pairs = random.sample(pairs, min(num_images_to_display, len(pairs)))
    
    # Determine the number of rows and columns for the grid based on the number of images
    grid_cols = int(np.ceil(np.sqrt(num_images_to_display)))
    grid_rows = int(np.ceil(num_images_to_display / grid_cols))
    # Create a grid of subplots
    fig, axs = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=(15, 15))
    axs = axs.flatten()  # Flatten to 1D array for easy indexing

    # Loop through the axes and hide any that won't be used
    for ax in axs[num_images_to_display:]:
        ax.axis('off')

    for idx, ax in enumerate(axs[:num_images_to_display]):
        # Extract the image path and annotations for the current index from the selected pairs
        image_path, annotation_data = selected_pairs[idx]
        # Open the image file and display it on the current axis
        img = Image.open(image_path)
        ax.imshow(img)
        
        # Iterate over each annotation for the current image
        for annotation in annotation_data:
            # Extract the class index and convert it to the class name
            class_index = int(annotation.split(' ')[0])
            class_name = class_list[class_index]
            # Parse the annotation coordinates and reshape them into a 2x4 matrix
            points = list(map(float, annotation.strip().split(' ')[1:]))
            points = np.array(points).reshape((4, 2))
            
            # Create a polygon patch from the annotation points and add it to the axis
            poly = patches.Polygon(points, closed=True, fill=False, edgecolor='blue')
            ax.add_patch(poly)
            
            # If labels should be shown, calculate the text properties and display it
            if show_labels:
                top_edge_vec = points[1] - points[0]  # Vector representing the top edge of the box
                angle = np.arctan2(top_edge_vec[1], top_edge_vec[0])
                
                # Set the position for the label text at the midpoint of the top edge
                label_pos = (points[0] + points[1]) / 2
                text_x, text_y = label_pos
                margin = 3  # Margin for the text position above the top edge
                
                # Adjust text position based on the orientation of the top edge
                if top_edge_vec[0] < 0:  # If the edge is oriented to the left
                    angle -= np.pi  # Adjust the angle to keep text orientation consistent

                # The text is placed above the top edge, considering the margin
                ax.text(text_x, text_y - margin, class_name, rotation=np.degrees(angle),
                        color='red', fontsize=9, ha='center', va='bottom', rotation_mode='anchor')
                
        # Display axis on the images
        ax.axis(show_axis)
    
    plt.tight_layout()
    plt.show()