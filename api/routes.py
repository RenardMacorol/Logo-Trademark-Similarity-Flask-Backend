from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import constant
from processor.image_utils import auto_crop_logo
from core.services import WonksNetService

api_bp = Blueprint('api', __name__)


@api_bp.route('/predict', methods=['POST'])
def predict():
    # 1. Get scope from Flutter (default to PH)
    # Flutter sends this via: request.fields['scope'] = 'BOTH';
    scope = request.form.get('scope', constant.DB_PH_SCOPE)

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # 2. Image Processing
    img_raw = cv2.imdecode(np.frombuffer(
        file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    cropped_rgb = auto_crop_logo(img_rgb)

    # 3. Use the Master Predict Service
    # This automatically handles engine selection and duplicate aggregation
    results, _ = WonksNetService.predict(cropped_rgb, scope=scope)

    # 4. Return pure JSON for Flutter
    return jsonify({
        "status": "success",
        "scope_used": scope,
        "predictions": results  # results is already sliced to top 5 in the service
    })
