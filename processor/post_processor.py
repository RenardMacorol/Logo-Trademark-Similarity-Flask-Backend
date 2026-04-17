import os
import cv2
from processor.image_utils import encode_to_base64

import os
import cv2
from processor.image_utils import encode_to_base64


def finalize_results(results, input_category):
    json_safe_results = []

    # Sort results by confidence first
    results = sorted(results, key=lambda x: x.get(
        'confidence', 0), reverse=True)

    # Margin calculation for Defense
    margin = 0
    if len(results) >= 2:
        margin = float(results[0].get('confidence', 0)) - \
            float(results[1].get('confidence', 0))

    for i, match in enumerate(results):
        raw_path = match.get('File_Path') or match.get('path') or ""

        # Path mapping (Cleaned up version)
        if 'datasetcopy/' in raw_path:
            final_path = f"static/database/{
                raw_path.split('datasetcopy/')[-1].lstrip('/')}"
        elif 'Dataset/' in raw_path:
            final_path = f"static/Dataset/{
                raw_path.split('Dataset/')[-1].lstrip('/')}"
        else:
            final_path = raw_path

        final_path = final_path.replace(" /", "/").replace("//", "/")

        # Defense Metrics Logic
        confidence = round(float(match.get('confidence', 0.0)), 2)
        stability = str(match.get('consensus', 'Stable'))

        # Margin Check
        if i == 0 and margin < 3.0:
            stability = "Ambiguous Match"
        elif i == 0 and margin > 15.0:
            stability = "Strong Contender"

        # Domain Penalty (Added safety catch for lowercase 'category' just in case)
        brand_domain = match.get('Category') or match.get(
            'category') or "General"
        if input_category and input_category != 'All':
            if brand_domain.lower() != input_category.lower():
                confidence = round(confidence * 0.85, 2)
                stability = "Domain Mismatch"

        # Thumbnail processing
        abs_path = os.path.join("/workspace/backend", final_path)
        thumb_base64 = ""
        if os.path.exists(abs_path):
            img_disk = cv2.imread(abs_path)
            if img_disk is not None:
                thumb = cv2.resize(img_disk, (120, 120))
                thumb_base64 = encode_to_base64(
                    cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

        # 🟢 THE FIX: Safely extract the brand name checking multiple case variations
        raw_brand = match.get('brand') or match.get('Brand') or "Unknown"

        json_safe_results.append({
            "confidence": confidence,
            "metadata": {
                "brand_name": str(raw_brand).strip(),
                "industry_domain": brand_domain,
                "margin_lead": round(margin, 2) if i == 0 else 0
            },
            "image_path": final_path,
            "thumbnail_base64": thumb_base64,
            "stability_label": stability,
            "similarity_label": f"Match Score: {confidence}%"
        })

    return json_safe_results
