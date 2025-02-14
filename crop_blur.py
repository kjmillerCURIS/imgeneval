import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def process_image(image, bboxes, sigma):
    """
    Processes the image by cropping tightly around the bounding boxes, centering as much as possible,
    and blurring the area outside the convex hull of the bounding boxes.

    Args:
        image (numpy array): The input image.
        bboxes (list of tuples): List of bounding boxes in (x1, y1, x2, y2) format, normalized [0,1].
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        cropped_blurred_image (numpy array): The processed image.
        new_bboxes (list of tuples): The bounding boxes relative to the cropped image.
    """
    h, w, _ = image.shape  # Get image dimensions

    # Convert normalized bbox coordinates to absolute pixel coordinates
    abs_bboxes = [(int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)) for x1, y1, x2, y2 in bboxes]

    # Compute the tightest bounding box containing all bboxes
    x_min = min(x1 for x1, _, _, _ in abs_bboxes)
    y_min = min(y1 for _, y1, _, _ in abs_bboxes)
    x_max = max(x2 for _, _, x2, _ in abs_bboxes)
    y_max = max(y2 for _, _, _, y2 in abs_bboxes)

    # Compute the average center of all bounding boxes
    center_x = int(np.mean([(x1 + x2) / 2 for x1, _, x2, _ in abs_bboxes]))
    center_y = int(np.mean([ (y1 + y2) / 2 for _, y1, _, y2 in abs_bboxes]))

    # Determine the required square size
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    crop_size = max(bbox_width, bbox_height)  # Take the tightest square that fits all bboxes

    # Ensure the crop stays within image bounds while centering as much as possible
    x_start = max(0, min(center_x - crop_size // 2, w - crop_size))
    y_start = max(0, min(center_y - crop_size // 2, h - crop_size))
    x_end = x_start + crop_size
    y_end = y_start + crop_size

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Compute the convex hull of all bounding boxes
    bbox_points = np.array(
        [(x1 - x_start, y1 - y_start) for x1, y1, _, _ in abs_bboxes] + 
        [(x2 - x_start, y1 - y_start) for _, y1, x2, _ in abs_bboxes] + 
        [(x2 - x_start, y2 - y_start) for _, _, x2, y2 in abs_bboxes] + 
        [(x1 - x_start, y2 - y_start) for x1, _, _, y2 in abs_bboxes]
    )

    hull = cv2.convexHull(bbox_points)  # Compute convex hull

    # Create a mask for the convex hull
    mask = np.zeros_like(cropped_image[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # Blur the entire cropped image
    blurred_image = cv2.GaussianBlur(cropped_image, (0, 0), sigma)

    # Apply mask: Keep inside the convex hull unchanged, use blurred image outside
    cropped_blurred_image = np.where(mask[:, :, None] == 255, cropped_image, blurred_image)

    # Compute new bounding boxes relative to the cropped image
    new_bboxes = [((x1 - x_start) / crop_size, (y1 - y_start) / crop_size, 
                   (x2 - x_start) / crop_size, (y2 - y_start) / crop_size) for x1, y1, x2, y2 in abs_bboxes]

    return cropped_blurred_image, new_bboxes

# Example usage
if __name__ == "__main__":
    image = cv2.imread("input.jpg")  # Load your image
    bboxes = [(0.2, 0.3, 0.5, 0.6), (0.4, 0.4, 0.7, 0.8)]  # Example normalized bboxes
    sigma = 10  # Blurring strength
    
    processed_image, updated_bboxes = process_image(image, bboxes, sigma)
    
    # Save or display the result
    cv2.imwrite("processed_image.jpg", processed_image)
    print("Updated bounding boxes:", updated_bboxes)
