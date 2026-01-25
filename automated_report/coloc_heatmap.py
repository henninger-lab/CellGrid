import numpy as np

def z_score_transform(image):
    """
    Transforms a float image as np array into z-score space.

    Parameters:
    image : np.ndarray
        Input float image.
    
    Returns:
    np.ndarray
        Z-score transformed image.
    """
    mean = np.mean(image)
    sigma = np.std(image)
    if sigma == 0:
        raise ValueError("Standard deviation is zero, cannot perform z-score transformation.")
    z_image = (image - mean) / sigma
    return z_image

def coloc_heatmap(image, channels):
    """
    Returns a colocalization heatmap from a multi-channel image.

    Parameters:
    image : np.ndarray
        Input 3D multi-channel image (CZYX).
    channels : tuple of int (len 2)
        Tuple of channel indices to consider for colocalization.
    
    Returns:
    coloc_map : np.ndarray
        Colocalization heatmap as a 3D np array.
    """
    channel_1 = image[channels[0], :, :, :].astype(np.float32)
    channel_2 = image[channels[1], :, :, :].astype(np.float32)
    z_channel_1 = z_score_transform(channel_1)
    z_channel_2 = z_score_transform(channel_2)
    coloc_map = np.multiply(z_channel_1, z_channel_2) / (abs(z_channel_1) + abs(z_channel_2) + 1)
    return coloc_map

