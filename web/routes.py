from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
from core.services import WonksNetService
from processor.image_utils import auto_crop_logo, encode_to_base64

web_bp = Blueprint('web', __name__)

# Global placeholder (In production, use session or Redis)
LAST_SESSION_IMAGE = None


@web_bp.route('/', methods=['GET', 'POST'])
def index():
    global LAST_SESSION_IMAGE
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file", 400

        # Pipeline: Raw -> RGB -> Crop
        img_raw = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # Save for real-time filtering
        LAST_SESSION_IMAGE = cropped_rgb.copy()

        # Initial Predict
        results, ai_mask = WonksNetService.predict(cropped_rgb, k=5)

        # Heatmap Gen
        mask_data = (ai_mask.squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # CHECK: Siguraduhin na 'components/results.html' ang filename mo sa folder
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            html_results = render_template(
                'components/results.html', results=results)
            return jsonify({
                "html": html_results,
                "original_img": encode_to_base64(cropped_rgb),
                "mask_img": encode_to_base64(heatmap_rgb),
                "total_found": len(results)
            })

        return render_template('index.html',
                               results=results,
                               original_img=encode_to_base64(cropped_rgb),
                               mask_img=encode_to_base64(heatmap_rgb))

    return render_template('index.html')


@web_bp.route('/filter-results', methods=['POST'])
def filter_results():
    global LAST_SESSION_IMAGE

    # Safety check kung may image sa session memory
    if LAST_SESSION_IMAGE is None:
        return jsonify({
            "html": "<div class='error-msg'>⚠️ Session expired. Please re-upload the logo.</div>",
            "total_found": 0
        }), 200

    try:
        # 1. Kunin ang values mula sa AJAX Form
        # Ginamit natin ang .getlist() para sa category at consensus dahil multiple checkboxes/selects sila
        sort_by = request.form.get('sort_by', 'confidence')
        scope = request.form.get('scope', 'BOTH')
        categories = request.form.getlist('category')
        consensus = request.form.getlist('consensus')

        # Kunin ang top_k value, default to 5 kung walang input
        try:
            user_k = int(request.form.get('top_k', 5))
        except:
            user_k = 5

        # 2. Tawagin ang Engine (Dito na lahat ang filtering logic)
        # Ang predict() method mo dapat ay tinatawag ang engine.search() sa loob
        results, _ = WonksNetService.predict(
            LAST_SESSION_IMAGE,
            k=user_k,
            sort_by=sort_by,
            categories=categories,
            consensus_levels=consensus,
            scope=scope
        )

        # 3. Render ang HTML fragment para sa AJAX
        # Siguraduhin na ang results list ay hindi empty bago i-render para iwas error sa UI
        html_results = render_template(
            'components/results.html', results=results)

        return jsonify({
            "html": html_results,
            "total_found": len(results)
        })

    except Exception as e:
        # I-print ang error sa terminal para sa debugging
        print(f"🔴 Route Error (Filter): {str(e)}")
        return jsonify({
            "html": f"<p style='color:red;'>System Error: {str(e)}</p>",
            "total_found": 0
        }), 500
