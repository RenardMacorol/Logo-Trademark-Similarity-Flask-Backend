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


def auto_crop_logo(img_rgb):
    mask = generate_safety_mask(cv2.resize(img_rgb, (224, 224)))
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        sy, sx = img_rgb.shape[0]/224, img_rgb.shape[1]/224
        x, y, w, h = int(x*sx), int(y*sy), int(w*sx), int(h*sy)
        y1, y2 = max(0, y-20), min(img_rgb.shape[0], y+h+20)
        x1, x2 = max(0, x-20), min(img_rgb.shape[1], x+w+20)
        return cv2.resize(img_rgb[y1:y2, x1:x2], (224, 224))
    return cv2.resize(img_rgb, (224, 224))


def encode_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
