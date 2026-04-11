import cv2
import numpy as np
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
    Main Endpoint: Handles image upload, TTA inference, 
    Heatmap generation, and Latent Space Mapping.
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
        # 2. Pre-processing (Byte to OpenCV)
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img_raw = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img_raw is None:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        # Convert to RGB and Crop
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 3. AI Inference (TTA + Search + PCA Mapping)
        # Tandaan: Ang WonksNetService.predict() ay nagbabalik na ngayon ng Dictionary
        search_cats = [category_input] if category_input != 'All' else []

        response_data = WonksNetService.predict(
            cropped_rgb,
            k=user_k,
            scope=scope,
            categories=search_cats
        )

        # Extract components from service response
        raw_results = response_data.get('top_matches', [])
        ai_mask = response_data.get('heatmap')
        latent_map = response_data.get('latent_map')

        # 4. Generate Visuals for Flutter (Base64)
        display_size = (300, 300)

        # Original Cropped Image
        orig_resized = cv2.resize(cropped_rgb, display_size)
        orig_base64 = encode_to_base64(orig_resized)

        # Heatmap Image
        mask_data = (ai_mask.squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(cv2.resize(
            heatmap, display_size), cv2.COLOR_BGR2RGB)
        mask_base64 = encode_to_base64(heatmap_rgb)

        # 5. Post-Processing / Defense Logic
        # Dito mo pinapakinis yung confidence scores at labels
        processed_matches = finalize_results(raw_results, category_input)

        # 6. Final JSON Response
        return jsonify({
            "status": "success",
            "predictions": processed_matches,
            "original_img_base64": orig_base64,
            "mask_img_base64": mask_base64,
            "latent_map": latent_map,  # Coordinates for Scatter Plot [x, y]
            "total_found": len(processed_matches),
            "meta": {
                "scope_used": scope,
                "category_filter": category_input,
                "tta_enabled": True
            }
        }), 200

    except Exception as e:
        # Detailed error log for Arch terminal debugging
        print(f"🔴 API CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "Internal processing failed",
            "details": str(e)
        }), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Check if server is alive"""
    return jsonify({"status": "online", "message": "WonksNet Backend is running"}), 200


@api_bp.route('/discovery', methods=['GET'])
def discovery():
    """
    Overview Mode: Pinapakita ang buong logo market landscape 
    nang walang kailangang i-upload na image.
    """
    # 🟢 FIX: Gawing case-insensitive ang scope default
    scope = request.args.get('scope', 'BOTH').upper()
    industry = request.args.get('industry', 'All')

    try:
        # 1. Kunin ang embeddings AT metadata mula sa Service
        all_embeddings, metadata = WonksNetService.get_market_embeddings(
            scope, industry)

        # 🟢 CHECK: Importante ang safety check na ito para sa PCA
        if all_embeddings is None or len(all_embeddings) < 2:
            print(f"⚠️ [DISCOVERY] Insufficient data: Found {
                  len(all_embeddings) if all_embeddings is not None else 0}")
            return jsonify({
                "status": "success",
                "scope": scope,
                "industry": industry,
                "message": "Not enough data for projection",
                "points": [],
                "total_brands": 0
            })

        # 2. PCA Projection
        # Siguraduhing ang output ng map_to_2d_overview ay list of dicts {x, y, metadata, image_path}
        pca_results = LatentMapper.map_to_2d_overview(all_embeddings, metadata)

        # 3. Final Response
        return jsonify({
            "status": "success",
            "scope": scope,
            "industry": industry,
            "total_brands": len(pca_results),
            "points": pca_results  # Ito yung array ng points para sa Flutter Galaxy
        })

    except Exception as e:
        print(f"🔴 [DISCOVERY ROUTE ERROR]: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal Server Error during PCA mapping"
        }), 500
