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
    Forensic generation, and Latent Space Mapping.
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

        # Convert to RGB and Crop (Dito nagfo-focus ang AI sa logo area)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 3. AI Inference (WonksNet Engine)
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

        # 🟢 Kunin ang Forensic Evidence (ORB/Feature Mapping)
        # Karaniwang Rank 1 lang ang may ganito mula sa service
        forensic_b64 = response_data.get('forensic_evidence')
        orb_score = response_data.get('orb_score', 0.0)

        # 4. Generate Visuals for Flutter UI (Base64)
        display_size = (300, 300)

        # Input Preview (Original Cropped)
        orig_resized = cv2.resize(cropped_rgb, display_size)
        orig_base64 = encode_to_base64(orig_resized)

        # AI Vision Heatmap (Segmentation)
        mask_data = (ai_mask.squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(cv2.resize(
            heatmap, display_size), cv2.COLOR_BGR2RGB)
        mask_base64 = encode_to_base64(heatmap_rgb)

        # 5. Post-Processing / Defense Logic
        # Nililinis ang confidence scores at inaayos ang labels
        processed_matches = finalize_results(raw_results, category_input)

        # 🟢 CRITICAL SYNC: Isalin ang forensic data sa Rank 1 match
        # Ito ang kailangan ng Flutter 'ResultCard' para lumitaw ang "View Forensic Evidence"
        for i in range(len(processed_matches)):
            if i < len(raw_results):
                # Kunin ang forensic_viz at orb_similarity na galing mismo sa raw result item
                processed_matches[i]['forensic_viz'] = raw_results[i].get(
                    'forensic_viz')
                processed_matches[i]['orb_similarity'] = raw_results[i].get(
                    'orb_similarity', 0.0)
        # 6. Final JSON Response
        return jsonify({
            "status": "success",
            "predictions": processed_matches,
            "original_img_base64": orig_base64,
            "mask_img_base64": mask_base64,
            "latent_map": latent_map,  # Coordinates para sa Scatter Plot dialog
            "total_found": len(processed_matches),
            "meta": {
                "scope_used": scope,
                "category_filter": category_input,
                "tta_enabled": True
            }
        }), 200

    except Exception as e:
        print(f"🔴 API CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "Internal processing failed",
            "details": str(e)
        }), 500


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
