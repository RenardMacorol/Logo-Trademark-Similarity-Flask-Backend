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
        # 1. Get scope AND the new top_k value from the form
        scope = request.form.get('scope', constant.DB_PH_SCOPE)

        # Capture top_k from the slider (defaulting to 5 if something goes wrong)
        try:
            user_k = int(request.form.get('top_k', 5))
        except (ValueError, TypeError):
            user_k = 5

        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        # 2. Image Pipeline (Raw -> RGB -> Crop)
        img_raw = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 3. Use the Master Predict Service with the new user_k
        # Now passing user_k so the service knows how many unique brands to return
        results, ai_mask = WonksNetService.predict(
            cropped_rgb,
            scope=scope,
            k=user_k
        )

        # 4. Generate Heatmap Visualization
        mask_data = (ai_mask.squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 5. Return to Template with the new ethical metrics
        return render_template('index.html',
                               # Contains confidence, stability, consensus, etc.
                               results=results,
                               mask_img=encode_to_base64(heatmap_rgb),
                               original_img=encode_to_base64(cropped_rgb),
                               current_scope=scope,
                               current_k=user_k)

    # Initial GET request
    return render_template('index.html')
