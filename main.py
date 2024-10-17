import cv2
import numpy as np


def load_image(image_path):
    """
    Load an image from the specified file path.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Loaded image.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image from {image_path}")
    return image


def save_image(image, output_path):
    """
    Save an image to the specified file path.

    Args:
        image (np.ndarray): Image to be saved.
        output_path (str): Path to save the image.
    """
    cv2.imwrite(output_path, image)


def display_image(image, window_name='Image'):
    """
    Display an image in a window.

    Args:
        image (np.ndarray): Image to display.
        window_name (str): Title of the display window.
    """
    im_resized = cv2.resize(image, (960, 540))
    cv2.imshow(window_name, im_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def increase_exposure(image, factor):
    """
    Increase the exposure of the image by multiplying pixel values by the factor.

    Args:
        image (np.ndarray): Input image.
        factor (float): Exposure factor. Values >1 increase exposure.

    Returns:
        np.ndarray: Brightened image.
    """
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def noise_reduction(image):
    """
    Apply Non-Local Means Denoising to reduce noise while preserving details.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image with reduced noise.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def contrast_enhancement(image):
    """
    Apply CLAHE to enhance local contrast in the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Contrast-enhanced image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def generate_exposure_images(image, num_images=30, exposure_step=0.1):
    """
    Generate a series of images with increasing exposure.

    Args:
        image (np.ndarray): Input image.
        num_images (int): Number of images to generate.
        exposure_step (float): Step size for increasing exposure.

    Returns:
        list of np.ndarray: List of images with varied exposures.
    """
    images = []
    for i in range(1, num_images + 1):
        exposed_image = increase_exposure(image, 1 + i * exposure_step)
        images.append(exposed_image)
    return images


def hdr_night_mode(image_path, num_images=30, exposure_step=0.1, output_path=None):
    """
    Apply an HDR-like night mode effect using exposure fusion.

    Args:
        image_path (str): Path to the input image.
        num_images (int): Number of images to stack with different exposures.
        exposure_step (float): Step size for increasing exposure.
        output_path (str, optional): Path to save the result. Defaults to None.

    Returns:
        np.ndarray: Final HDR image processed with exposure fusion.
    """
    try:
        # Load the input image
        image = load_image(image_path)

        # Generate images with varying exposures
        exposure_images = generate_exposure_images(image, num_images, exposure_step)

        # Merge the exposure images using Mertens fusion
        merge_mertens = cv2.createMergeMertens()
        hdr_image = merge_mertens.process(exposure_images)

        # Convert HDR image to 8-bit
        hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)

        # Apply noise reduction
        denoised_image = noise_reduction(hdr_image_8bit)

        # Apply contrast enhancement
        final_image = contrast_enhancement(denoised_image)

        # Save the result if output path is provided
        if output_path:
            save_image(final_image, output_path)

        return final_image

    except FileNotFoundError as e:
        print(e)
        return None


image_path = "Input Images/dark_image_4.jpg"
output_path = "Output Images/hdr_night_mode_result4.jpg"
result_image = hdr_night_mode(image_path, output_path=output_path)

# if result_image is not None:
#     display_image(result_image, window_name='HDR Night Mode Image')
