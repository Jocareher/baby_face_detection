from typing import List, Optional, Dict

from detectron2.structures import BoxMode
from detectron2.data.transforms import TransformList

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
