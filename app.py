from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import constant

# Modular Imports
from core.model import WonksNetModel
from core.search_engine import LogoSearchEngine
from processor.image_utils import auto_crop_logo, encode_to_base64
from api.routes import api_bp, init_api

app = Flask(__name__)

# --- 1. GPU CONFIG (GTX 1650 Optimized) ---
# We do this FIRST before initializing the model
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # We use Virtual Device / Memory Limit only.
        # DO NOT use set_memory_growth here; it will cause a crash.
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(constant.GPU_MEMORY_LIMIT)]
        )
        print("✅ GPU: 3GB Logical Limit Configured.")
    except RuntimeError as e:
        print(f"⚠️ GPU Initialization warning: {e}")

# --- 2. INITIALIZE SERVICES ---
print("🚀 Initializing WonksNet Brain...")

# Build the model instance
wonks_model = WonksNetModel(constant.MODEL_FILE)

# Build the search engine
engine = LogoSearchEngine(
    wonks_model.encoder,
    constant.VECTORS_PATH,
    constant.METADATA_PATH
)

# Register the API Blueprint for Mobile
init_api(engine)
app.register_blueprint(api_bp, url_prefix='/api')

# --- 3. WEB UI ROUTE ---


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file"

        # 1. Processing
        img_raw = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 2. Search
        results, ai_mask = engine.search(cropped_rgb)

        # 3. Visualization
        mask_viz = cv2.applyColorMap(
            (ai_mask.squeeze() * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        return render_template('index.html',
                               results=results,
                               mask_img=encode_to_base64(
                                   cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB)),
                               original_img=encode_to_base64(cropped_rgb))

    return render_template('index.html')


if __name__ == '__main__':
    # host 0.0.0.0 is necessary for Docker/Dashboard-1 access
    app.run(host=constant.HOST, port=constant.PORT, debug=False)
