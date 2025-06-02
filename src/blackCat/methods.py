import numpy as np
from skimage import exposure, io, color
from ultralytics import YOLO
import cv2


# provided by @vivaciousvivvy
def yolo_scaled(image):
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
def get_mask_scaled(image):
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


def yolo_unscaled(image):
    # Sourced from chatgpt using the prompt: Proivde an example of segmentation with a black cat using YOLO
    # Load YOLOv8 model with segmentation capability
    model = YOLO('yolov8x-seg.pt')  # You can use yolov8n-seg.pt for a smaller model

    # Run inference
    results = model(image)[0]

    # Get class names and masks
    class_names = model.names
    cat_class_id = [k for k, v in class_names.items() if v == 'cat']
    if not cat_class_id:
        print("No 'cat' class in the model.")
        exit()

    cat_class_id = cat_class_id[0]
    cat_masks = [mask for i, mask in enumerate(results.masks.data) if int(results.boxes.cls[i]) == cat_class_id]

    if not cat_masks:
        print("No cat detected.")
        exit()

    # Convert the PyTorch tensor mask to numpy
    cat_mask = cat_masks[0].cpu().numpy()

    # Resize mask if needed to match image size
    mask_resized = cv2.resize(cat_mask.astype(np.uint8), (image.shape[1], image.shape[0]))

    # Apply mask to the image
    cat_pixels = cv2.bitwise_and(image, image, mask=mask_resized)

    return cat_pixels


def get_mask_unscaled(image):
    # Load YOLOv8 model with segmentation capability
    model = YOLO('yolov8x-seg.pt')  # You can use yolov8n-seg.pt for a smaller model

    # Run inference
    results = model(image)[0]

    # Get class names and masks
    class_names = model.names
    cat_class_id = [k for k, v in class_names.items() if v == 'cat']
    if not cat_class_id:
        print("No 'cat' class in the model.")
        exit()

    cat_class_id = cat_class_id[0]
    cat_masks = [mask for i, mask in enumerate(results.masks.data) if int(results.boxes.cls[i]) == cat_class_id]

    if not cat_masks:
        print("No cat detected.")
        exit()

    # Convert the PyTorch tensor mask to numpy
    cat_mask = cat_masks[0].cpu().numpy()

    # Resize mask if needed to match image size
    mask_resized = cv2.resize(cat_mask.astype(np.uint8), (image.shape[1], image.shape[0]))
    return mask_resized


def negative_transform_unscaled(image):
    neg_img = 255 - image
    mask = get_mask_unscaled(image)
    neg_final = image.copy()
    for c in range(3):  # assuming image has 3 channels
        neg_final[:, :, c] = np.where(mask == 1, neg_img[:, :, c], image[:, :, c])
    return neg_final


def gamma_transform_unscaled(image):
    gamma_img = exposure.adjust_gamma(image, gamma=0.5, gain=1)
    mask = get_mask_unscaled(image)
    gamma_final = image.copy()
    for c in range(3):
        gamma_final[:, :, c] = np.where(mask == 1, gamma_img[:, :, c], image[:, :, c])
    return gamma_final


def log_transform_unscaled(image):
    log_img = exposure.adjust_log(image, gain=2, inv=False)
    mask = get_mask_unscaled(image)
    log_final = image.copy()
    for c in range(3):  # assuming image has 3 channels
        log_final[:, :, c] = np.where(mask == 1, log_img[:, :, c], image[:, :, c])
    return log_final


def log_transform_scaled(image):
    log_img = exposure.adjust_log(image, gain=2, inv=False)
    mask = get_mask_scaled(image)
    log_final = image.copy()
    for c in range(3):  # assuming image has 3 channels
        log_final[:, :, c] = np.where(mask == 255, log_img[:, :, c], image[:, :, c])
    return log_final

def negative_transform_scaled(image):
    neg_img = 255 - image
    mask = get_mask_scaled(image)
    neg_final = image.copy()
    for c in range(3):  # assuming image has 3 channels
        neg_final[:, :, c] = np.where(mask == 255, neg_img[:, :, c], image[:, :, c])
    return neg_final


def gamma_transform_scaled(image):
    gamma_img = exposure.adjust_gamma(image, gamma=0.5, gain=1)
    mask = get_mask_scaled(image)
    gamma_final = image.copy()
    for c in range(3):
        gamma_final[:, :, c] = np.where(mask == 255, gamma_img[:, :, c], image[:, :, c])
    return gamma_final


# written by @ruspedpdx
def bpdhe(image):
    """
    Applies brightness preserving dynamic histogram equalization to an image
    
    Args:
        image(numpy.ndarray): RGB image
    
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

        # copy the lightness channel and replace segmented portion
        clahe_lightness_mask = lightness.copy()
        clahe_lightness_mask[mask == 255] = clahe_lightness[mask == 255]

        # merge the lightness channel with the color channels
        clahe_lab = cv2.merge((clahe_lightness_mask, rg, by))

        # return the image to RGB color space
        clahe_image = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)

        return clahe_image