import cv2
import numpy as np
import traceback
from flask import Blueprint, request, jsonify

import constant
from processor.image_utils import auto_crop_logo, encode_to_base64
from processor.post_processor import finalize_results
from core.services import WonksNetService
from core.services import LatentMapper

# Define the blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Main Endpoint: Handles image upload, AI inference, 
    and Neural Segmentation Attention mapping.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    # 1. Setup Configuration from Request
    scope = request.form.get('scope', constant.DB_PH_SCOPE)
    category_input = request.form.get('category', 'All')

    try:
        user_k = int(request.form.get('top_k', 10))
    except (ValueError, TypeError):
        user_k = 10

    try:
        # 2. Pre-processing
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img_raw = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img_raw is None:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # 3. AI Inference (WonksNet Engine)
        search_cats = [category_input] if category_input != 'All' else []

        response_data = WonksNetService.predict(
            img_rgb,
            k=user_k,
            scope=scope,
            categories=search_cats
        )

        # 🟢 Capture Core Components + The Neural Mask
        raw_results = response_data.get('top_matches', [])
        latent_map = response_data.get('latent_map')
        # This is the missing link
        raw_mask = response_data.get('segmentation_mask')

        # 4. Generate Visuals for Flutter UI
        display_size = (300, 300)

        # Process Input Image
        orig_resized = cv2.resize(img_rgb, display_size)
        orig_base64 = encode_to_base64(orig_resized)

        # 🟢 Process Segmentation Attention (Heatmap)
        mask_base64 = ""
        if raw_mask is not None:
            # Resize mask to match display size and ensure it's in 0-255 range
            mask_resized = cv2.resize(raw_mask, display_size)
            if mask_resized.max() <= 1.0:  # Normalize if float
                mask_resized = (mask_resized * 255).astype(np.uint8)

            # Optional: Apply COLORMAP_JET if you want a colorful heatmap
            # heatmap = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
            # mask_base64 = encode_to_base64(heatmap)

            mask_base64 = encode_to_base64(mask_resized)

        # 5. Post-Processing (Neural Confidence Calibration)
        processed_matches = finalize_results(raw_results, category_input)

        # 6. Final JSON Response
        return jsonify({
            "status": "success",
            "predictions": processed_matches,
            "original_img_base64": orig_base64,
            "mask_img_base64": mask_base64,  # 🟢 Flutter now receives the attention map
            "latent_map": latent_map,
            "total_found": len(processed_matches),
            "meta": {
                "scope_used": scope,
                "category_filter": category_input,
                "tta_enabled": True
            }
        }), 200

    except Exception as e:
        print(f"🔴 API CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Check if server is alive"""
    return jsonify({
        "status": "online",
        "message": "WonksNet Backend is running",
        "version": "3.0.0"
    }), 200


@api_bp.route('/discovery', methods=['GET'])
def discovery():
    """
    Overview Mode: Pinapakita ang buong logo market landscape via PCA/t-SNE projection.
    """
    scope = request.args.get('scope', 'BOTH').upper()
    industry = request.args.get('industry', 'All')

    try:
        # 1. Kunin ang embeddings AT metadata
        all_embeddings, metadata = WonksNetService.get_market_embeddings(
            scope, industry)

        if all_embeddings is None or len(all_embeddings) < 2:
            return jsonify({
                "status": "success",
                "scope": scope,
                "industry": industry,
                "message": "Not enough data for projection",
                "points": [],
                "total_brands": 0
            })

        # 2. PCA Projection para sa Galaxy View
        pca_results = LatentMapper.map_to_2d_overview(all_embeddings, metadata)

        return jsonify({
            "status": "success",
            "scope": scope,
            "industry": industry,
            "total_brands": len(pca_results),
            "points": pca_results  # Array ng {x, y, brand_name, etc.}
        })

    except Exception as e:
        print(f"🔴 [DISCOVERY ERROR]: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal Server Error during latent mapping"
        }), 500
