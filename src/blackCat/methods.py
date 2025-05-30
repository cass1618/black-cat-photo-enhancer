import numpy as np
from skimage import exposure, io, color
from ultralytics import YOLO
import cv2


# provided by @vivaciousvivvy
def yolo(image):
    """
    Segments image using YOLOv8 to separate the cat from the background
    
    Args:
        image(numpy.ndarray): RBG image
    
    Returns:
        image that only includes the cat with background removed as RGB numpy.ndarray
    """
    if isinstance(image, str):
        from .process import load_display
        path = image
        load_display(yolo, path)
        
    else:
        # Sourced from chatgpt using the prompt: Proivde an example of segmentation with a black cat using YOLO where the image size is 3024 x 4032
        # Load the high-resolution image
        original_image = image
        height, width = original_image.shape[:2]

        # Load YOLOv8 model
        model = YOLO('yolov8x-seg.pt')  # Use 'yolov8n-seg.pt' for speed/testing

        # Run inference and retain the resized input shape
        results = model(original_image, verbose=False)[0]

        # Get class ID for 'cat'
        class_names = model.names
        cat_class_id = [k for k, v in class_names.items() if v == 'cat']
        if not cat_class_id:
            print("No 'cat' class found.")
            exit()
        cat_class_id = cat_class_id[0]

        # Retrieve model input dimensions (from results.orig_shape and results.img.shape)
        model_input_shape = results.orig_shape  # shape the model used (after resize/pad)
        processed_shape = results.orig_img.shape[:2]  # the resized image fed into model

        # Scaling factors to map back to original
        scale_x = width / processed_shape[1]
        scale_y = height / processed_shape[0]

        # Create blank mask
        mask_image = np.zeros((height, width), dtype=np.uint8)

        # Draw each 'cat' mask polygon after scaling coordinates
        for i, cls in enumerate(results.boxes.cls):
            if int(cls) == cat_class_id:
                # Scale the polygon coordinates
                xy = results.masks.xy[i]
                xy_scaled = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in xy], dtype=np.int32)
                cv2.fillPoly(mask_image, [xy_scaled], color=255)

        # Apply mask
        cat_pixels = cv2.bitwise_and(original_image, original_image, mask=mask_image)

        return cat_pixels


# modified version of yolo that returns the mask rather than segmented photo
def get_mask(image):
    """
    Segments image using YOLOv8 to separate the cat from the background
    
    Args:
        image(numpy.ndarray): RBG image
    
    Returns:
        grayscale mask of the cat portion of the image
    """
    height, width = image.shape[:2]

    model = YOLO('yolov8x-seg.pt')  # Use 'yolov8n-seg.pt' for speed/testing
    results = model(image, verbose=False)[0]

    # class_names = model.names
    cat_class_id = [k for k, v in model.names.items() if v == 'cat']
    if not cat_class_id:
        print("No 'cat' class found.")
        exit()
    cat_class_id = cat_class_id[0]

    processed_shape = results.orig_img.shape[:2]  # the resized image fed into model

    # Scaling factors to map back to original
    scale_x = width / processed_shape[1]
    scale_y = height / processed_shape[0]

    # Create blank mask
    mask_image = np.zeros((height, width), dtype=np.uint8)

    # Draw each 'cat' mask polygon after scaling coordinates
    for i, cls in enumerate(results.boxes.cls):
        if int(cls) == cat_class_id:
            # Scale the polygon coordinates
            xy = results.masks.xy[i]
            xy_scaled = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in xy], dtype=np.int32)
            cv2.fillPoly(mask_image, [xy_scaled], color=255)

    return mask_image


# written by @ruspedpdx
def bpdhe(image):
    """
    Applies brightness preserving dynamic histogram equalization to an image
    
    Args:
        image(numpy.ndarray): RBG image
    
    Returns:
        brightness preserved image as grayscale numpy.ndarray
    """
    if isinstance(image, str):
        from .process import load_display
        path = image
        load_display(yolo, path)
        
    else:

        # First, convert to grayscale
        # Compute the histogram equalization.
        # Preserve the brightness levels by calculating the mean
        # Adjust the intensity for the entire image

        gray = color.rgb2gray(image)
        hist_eq = exposure.equalize_hist(gray)
        mean_original = np.mean(gray)
        mean_eq = np.mean(hist_eq)
        brightness_preserved = hist_eq * (mean_original / mean_eq)

        return np.clip(brightness_preserved, 0, 1)


def lab_clahe(image):
    """
    Converts image to CIELAB color space before applying contrast limited adaptive histogram equalization to cat segment of the image

    Args:
        image(numpy.ndarray): RBG image
    
    Returns:
        image with clahe applied to cat portion as RGB numpy.ndarra
    """
    if isinstance(image, str):
        from .process import load_display
        path = image
        load_display(yolo, path)
        
    else:
        # obtain portion of photo that includes only the cat
        mask = get_mask(image)

        # convert image from RGB to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # separate lightness, red-green, and blue-yellow channels
        lightness, rg, by = cv2.split(lab_image)

        # apply clahe algorithm to the lightness channel
        clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_lightness = clahe_obj.apply(lightness)

        # copy the lightness channel and replace segmented portion with clahe version
        clahe_lightness_mask = lightness.copy()
        clahe_lightness_mask[mask == 255] = clahe_lightness[mask == 255]

        # merge the lightness channel with the color channels
        clahe_lab = cv2.merge((clahe_lightness_mask, rg, by))

        # return the image to RGB color space
        clahe_image = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)

        return clahe_image