import random
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_aabb_bboxes_and_save_images(root_dir: str, output_dir: str):
    """
    Read images and corresponding bounding box annotations from the root directory,
    draw bounding boxes on the images, and save the annotated images to the output directory.

    Args:
        - root_dir (str): The root directory containing 'images' and 'labels' subdirectories.
        - output_dir (str): The directory where annotated images will be saved.

    Returns:
        None
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List files in the root directory
    for filename in os.listdir(os.path.join(root_dir, "images")):
        if filename.endswith(".jpg"):
            image_path = os.path.join(root_dir, "images", filename)
            label_path = os.path.join(
                root_dir, "labels", filename.replace(".jpg", ".txt")
            )

            # Read image
            image = cv2.imread(image_path)

            # Read bounding box annotations
            with open(label_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    # Extract bounding box coordinates and class index
                    class_index, x, y, w, h = map(int, line.strip().split())

                    # Draw bounding box on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

                    # Add text with class index
                    cv2.putText(
                        image,
                        f"{class_index}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            # Save image with bounding boxes
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)

            print(f"Processed: {filename}")


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


def draw_yolov8_annotations_on_images(
    pairs: list[tuple],
    class_list: list[str],
    num_images_to_display: int,
    show_labels: bool = True,
    show_axis: str = "on",
) -> None:
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
        ax.axis("off")

    for idx, ax in enumerate(axs[:num_images_to_display]):
        # Extract the image path and annotations for the current index from the selected pairs
        image_path, annotation_data = selected_pairs[idx]
        # Open the image file and display it on the current axis
        img = Image.open(image_path)
        ax.imshow(img)

        # Iterate over each annotation for the current image
        for annotation in annotation_data:
            # Extract the class index and convert it to the class name
            class_index = int(annotation.split(" ")[0])
            class_name = class_list[class_index]
            # Parse the annotation coordinates and reshape them into a 2x4 matrix
            points = list(map(float, annotation.strip().split(" ")[1:]))
            points = np.array(points).reshape((4, 2))

            # Create a polygon patch from the annotation points and add it to the axis
            poly = patches.Polygon(points, closed=True, fill=False, edgecolor="blue")
            ax.add_patch(poly)

            # If labels should be shown, calculate the text properties and display it
            if show_labels:
                top_edge_vec = (
                    points[1] - points[0]
                )  # Vector representing the top edge of the box
                angle = np.arctan2(top_edge_vec[1], top_edge_vec[0])

                # Set the position for the label text at the midpoint of the top edge
                label_pos = (points[0] + points[1]) / 2
                text_x, text_y = label_pos
                margin = 3  # Margin for the text position above the top edge

                # Adjust text position based on the orientation of the top edge
                if top_edge_vec[0] < 0:  # If the edge is oriented to the left
                    angle -= (
                        np.pi
                    )  # Adjust the angle to keep text orientation consistent

                # The text is placed above the top edge, considering the margin
                ax.text(
                    text_x,
                    text_y - margin,
                    class_name,
                    rotation=np.degrees(angle),
                    color="red",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    rotation_mode="anchor",
                )

        # Display axis on the images
        ax.axis(show_axis)

    plt.tight_layout()
    plt.show()


def plot_images_with_labels_and_obboxes(
    root_dir: str, output_dir: str, line_thickness: int = 5, font_size: int = 20
):
    """
    Processes all images in a directory, draws oriented bounding boxes and the class index with specified line thickness and font size,
    and saves the resulting images using OpenCV.

    Args:
    - root_dir (str): Root directory containing 'images' and 'labels' folders.
    - output_dir (str): Directory where the resulting images will be saved.
    - line_thickness (int): Line thickness of the bounding boxes.
    - font_size (int): Font size for the class text.
    """
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each file in the images directory
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)

        # Skip files that are not regular image files
        try:
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape
        except:
            print(f"Skipping file: {image_name}")
            continue

        # Check if there is a corresponding label file
        base_name = os.path.splitext(image_name)[0]
        bbox_path = os.path.join(labels_dir, base_name + ".txt")

        if os.path.exists(bbox_path):
            with open(bbox_path, "r") as file:
                for line in file:
                    # Parse bounding box data from the label file
                    bbox_data = line.strip().split()
                    # Extract the class index from the bounding box data
                    class_index = bbox_data[0]
                    # Convert normalized coordinates to absolute
                    coordinates = [float(coord) for coord in bbox_data[1:]]
                    absolute_coordinates = [
                        (
                            int(coordinates[i] * image_width),
                            int(coordinates[i + 1] * image_height),
                        )
                        for i in range(0, len(coordinates), 2)
                    ]

                    # Draw bounding box on the image
                    points = [(x, y) for x, y in absolute_coordinates]
                    cv2.polylines(
                        image,
                        [np.array(points)],
                        isClosed=True,
                        color=(255, 0, 0),
                        thickness=line_thickness,
                    )

                    # Draw the class index on the image
                    label_x, label_y = absolute_coordinates[0]
                    # Add class index text near the bounding box
                    cv2.putText(
                        image,
                        class_index,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size,
                        (0, 0, 255),
                        thickness=2,
                    )

        # Save the resulting image
        output_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_image_path, image)

    print(f"All images have been processed and saved in {output_dir}")


def draw_bbox_polygons_from_file(image_path: str, bbox_file_path: str) -> None:
    """
    Draws bounding boxes on an image from a given bounding box file.

    Args:
        image_path (str): Path to the image file.
        bbox_file_path (str): Path to the file containing bounding box coordinates.
    """
    # Read the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Check if the image was loaded successfully
    if image is None:
        print("Error loading the image")
        return

    # Read the bounding boxes from the file
    with open(bbox_file_path, "r") as file:
        lines = file.readlines()

    # Draw each bounding box on the image
    for line in lines:
        parts = line.strip().split()
        # Assume that the last 8 numbers are the coordinates
        points = list(map(float, parts[-8:]))
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Draw the polygon on the image
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Display the image with the drawn bounding boxes
    plt.imshow(image)
    plt.axis("off")  # Hide the axis
    plt.show()
