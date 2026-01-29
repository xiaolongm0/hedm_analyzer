"""
Image processing utilities for Dash GUI image display
"""

import numpy as np


def prepare_image_for_display(image, saturation_threshold=65535, saturation_highlight=True, lower_threshold=0):
    """
    Prepare grayscale image for Plotly display with saturation highlighting.

    Args:
        image: numpy array of grayscale image (H, W)
        saturation_threshold: threshold value for saturated pixels
        saturation_highlight: whether to highlight saturated pixels in red
        lower_threshold: exclude pixels below this value

    Returns:
        RGB numpy array (H, W, 3) ready for px.imshow()
    """
    if image is None:
        return None

    # Apply lower threshold - set below-threshold pixels to 0
    filtered_image = image.copy().astype(np.float64)
    if lower_threshold > 0:
        filtered_image[filtered_image < lower_threshold] = 0

    # Apply percentile-based contrast scaling (1-99 percentile)
    valid_pixels = filtered_image[filtered_image > 0]
    if len(valid_pixels) > 0:
        p1, p99 = np.percentile(valid_pixels, [1, 99])
    else:
        p1, p99 = 0, 1

    # Avoid division by zero
    if p99 <= p1:
        p99 = p1 + 1

    # Scale to 0-255
    scaled = np.clip((filtered_image - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

    # Convert to RGB
    rgb_image = np.stack([scaled, scaled, scaled], axis=2)

    # Apply saturation highlighting if enabled
    if saturation_highlight:
        saturated_mask = image >= saturation_threshold
        rgb_image[saturated_mask] = [255, 0, 0]  # Red for saturated pixels

    return rgb_image
