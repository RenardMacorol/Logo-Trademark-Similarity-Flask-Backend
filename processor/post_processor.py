import os
import cv2
from processor.image_utils import encode_to_base64


def finalize_results(results, input_category):
    json_safe_results = []

    # 1. Sort results by neural confidence
    # We look for 'neural_confidence' from ResultProcessor, falling back to 'confidence'
    results = sorted(results, key=lambda x: x.get('metrics', {}).get(
        'neural_confidence', x.get('confidence', 0)), reverse=True)

    # 2. Neural Margin calculation
    margin = 0
    if len(results) >= 2:
        score1 = results[0].get('metrics', {}).get(
            'neural_confidence', results[0].get('confidence', 0))
        score2 = results[1].get('metrics', {}).get(
            'neural_confidence', results[1].get('confidence', 0))
        margin = float(score1) - float(score2)

    for i, match in enumerate(results):
        # --- [STAYED THE SAME: DIRECTORY INSTRUCTIONS] ---
        raw_path = match.get('File_Path') or match.get('path') or ""
        if 'datasetcopy/' in raw_path:
            final_path = f"static/database/{
                raw_path.split('datasetcopy/')[-1].lstrip('/')}"
        elif 'Dataset/' in raw_path:
            final_path = f"static/Dataset/{
                raw_path.split('Dataset/')[-1].lstrip('/')}"
        else:
            final_path = raw_path
        final_path = final_path.replace(" /", "/").replace("//", "/")
        # --- [END DIRECTORY INSTRUCTIONS] ---

        # 3. TOUCHED: Neural Confidence & Stability Calibration
        # Get raw score from engine
        raw_neural = float(match.get('metrics', {}).get(
            'neural_confidence', match.get('confidence', 0.0)))
        confidence = round(raw_neural, 2)

        # Stability logic
        stability = str(match.get('consensus', 'Stable'))
        brand_domain = match.get('Category') or match.get(
            'category') or "General"

        # Neutral Confidence "All" Check
        # If input_category is 'All', we skip the penalty to keep high Neural Confidence
        is_all = not input_category or str(input_category).lower() == 'all'

        if is_all:
            # Neural Level for 'All': Apply Defense Labels based on raw strength
            if i == 0 and margin > 15.0:
                stability = "Strong Contender"
            elif i == 0 and margin < 3.0:
                stability = "Ambiguous Match"
            else:
                stability = "Stable Match"
        else:
            # Neural Level for Targeted Category: Apply Mismatch Penalty
            if brand_domain.lower() != input_category.lower():
                confidence = round(confidence * 0.85, 2)
                stability = "Domain Mismatch"
            else:
                stability = "Targeted Match"

        # --- [STAYED THE SAME: THUMBNAIL PROCESSING] ---
        abs_path = os.path.join("/workspace/backend", final_path)
        thumb_base64 = ""
        if os.path.exists(abs_path):
            img_disk = cv2.imread(abs_path)
            if img_disk is not None:
                thumb = cv2.resize(img_disk, (120, 120))
                thumb_base64 = encode_to_base64(
                    cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        # --- [END THUMBNAIL PROCESSING] ---

        raw_brand = match.get('brand') or match.get('Brand') or "Unknown"

        json_safe_results.append({
            "confidence": confidence,
            # Pulling stability %
            "stability": round(float(match.get('stability', 0.0)), 2),
            "metadata": {
                "brand_name": str(raw_brand).strip(),
                "industry_domain": brand_domain,
                "margin_lead": round(margin, 2) if i == 0 else 0,
                "raw_neural_power": raw_neural
            },
            "image_path": final_path,
            "thumbnail_base64": thumb_base64,
            "stability_label": stability,
            "similarity_label": f"Neural Match: {confidence}%"
        })

    return json_safe_results
