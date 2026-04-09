from flask import Blueprint, render_template, request
import cv2
import numpy as np
import constant
from core.services import WonksNetService
from processor.image_utils import auto_crop_logo, encode_to_base64

web_bp = Blueprint('web', __name__)


@web_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Get scope from form (Defaults to PH)
        scope = request.form.get('scope', constant.DB_PH_SCOPE)

        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        # 2. Image Pipeline (Raw -> RGB -> Crop)
        img_raw = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 3. Use the Master Predict Service
        # This handles engine selection, search, and duplicate aggregation automatically
        results, ai_mask = WonksNetService.predict(cropped_rgb, scope=scope)

        # 4. Generate Heatmap Visualization
        # We process the raw mask returned by the service
        mask_data = (ai_mask.squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 5. Return to Template
        return render_template('index.html',
                               results=results,  # Already aggregated and sorted
                               mask_img=encode_to_base64(heatmap_rgb),
                               original_img=encode_to_base64(cropped_rgb),
                               current_scope=scope)

    # Initial GET request
    return render_template('index.html')
