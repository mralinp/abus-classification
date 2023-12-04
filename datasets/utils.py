import cv2
import numpy as np

def ResizeData(sample, target_size):
    # Get the original image dimensions
    x, m = sample
    
    original_height, original_width, _ = x.shape
    
    # Set the target size
    primary_target_size = (490, 490)

    # Calculate the padding needed
    padding_height = max(0, primary_target_size[0] - original_height)
    padding_width = max(0, primary_target_size[1] - original_width)

    # Calculate the padding amounts for top, bottom, left, and right
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding
    left_padding = padding_width // 2
    right_padding = padding_width - left_padding

    # Create a border around the image with zero-padding
    image_with_padding = cv2.copyMakeBorder(
        x,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Set the padding color to black
    )
    
    mask_with_padding = cv2.copyMakeBorder(
        m,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Set the padding color to black
    )

    # Resize the image to the target size
    resized_image = cv2.resize(image_with_padding, target_size)
    resized_mask = cv2.resize(mask_with_padding, target_size)
    
    return resized_image, resized_mask
