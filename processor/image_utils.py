# processor/image_utils.py
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
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"


def auto_crop_logo(image_np):
    """
    Crops the image to the bounding box of the logo content.
    """
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
        # Crop with a little padding (10px)
        return image_np[max(0, y-10):y+h+10, max(0, x-10):x+w+10]

    return image_np
