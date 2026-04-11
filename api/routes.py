import cv2
import numpy as np
from flask import Blueprint, request, jsonify

import constant
from processor.image_utils import auto_crop_logo, encode_to_base64
from processor.post_processor import finalize_results  # Import the new logic
from core.services import WonksNetService

api_bp = Blueprint('api', __name__)


@api_bp.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    # 1. Setup Configuration
    scope = request.form.get('scope', constant.DB_PH_SCOPE)
    category_input = request.form.get('category', 'All')
    try:
        user_k = int(request.form.get('top_k', 10))
    except:
        user_k = 10

    try:
        # 2. Pre-processing
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img_raw = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img_raw is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 3. AI Inference
        # In-optimize natin ang categories list dito
        search_cats = [category_input] if category_input != 'All' else []
        results, ai_mask = WonksNetService.predict(
            cropped_rgb, k=user_k, scope=scope, categories=search_cats
        )

        # 4. Generate Flippable Visuals (300x300)
        display_size = (300, 300)
        orig_base = encode_to_base64(cv2.resize(cropped_rgb, display_size))

        # Heatmap Processing
        mask_data = (ai_mask.squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(cv2.resize(
            heatmap, display_size), cv2.COLOR_BGR2RGB)
        mask_base = encode_to_base64(heatmap_rgb)

        # 5. Fine-tuned Post-Processing (The "Defense" Logic)
        processed_matches = finalize_results(results, category_input)

        return jsonify({
            "status": "success",
            "predictions": processed_matches,
            "original_img_base64": orig_base,
            "mask_img_base64": mask_base,
            "total_found": len(processed_matches)
        }), 200

    except Exception as e:
        print(f"🔴 API Error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal processing failed"}), 500
