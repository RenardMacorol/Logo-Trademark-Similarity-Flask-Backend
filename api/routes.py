from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from processor.image_utils import auto_crop_logo

# We define the blueprint to separate it from the main app
api_bp = Blueprint('api', __name__)

# This is a helper we'll pass from the main app
search_engine = None


def init_api(engine_instance):
    global search_engine
    search_engine = engine_instance


@api_bp.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # 1. Image Processing
    img_raw = cv2.imdecode(np.frombuffer(
        file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # 2. Auto-Crop
    cropped_rgb = auto_crop_logo(img_rgb)

    # 3. Search via the global engine instance
    results, _ = search_engine.search(cropped_rgb)

    # 4. Return pure JSON for Mobile
    return jsonify({
        "status": "success",
        "predictions": results[:5]
    })
