import cv2
import numpy as np
import base64


def generate_safety_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def encode_to_base64(img_rgb):
    """
    Encodes image to a clean base64 string.
    Removed the 'data:image/png' prefix for better Flutter compatibility.
    """
    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Use .jpg with quality 80 to reduce payload size (faster network transfer)
    _, buffer = cv2.imencode(
        '.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    # Return PURE base64 string only
    return base64.b64encode(buffer).decode('utf-8')


def auto_crop_logo(image_np):
    """
    Crops the image to the bounding box of the logo content.
    """
    if image_np is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Thresholding to find non-white/non-black areas
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour bounding box
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Add a little padding (10px)
        y_start = max(0, y-10)
        y_end = min(image_np.shape[0], y+h+10)
        x_start = max(0, x-10)
        x_end = min(image_np.shape[1], x+w+10)

        return image_np[y_start:y_end, x_start:x_end]

    return image_np
