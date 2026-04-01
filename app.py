from flask import Flask, render_template, request, jsonify
from tensorflow.keras.layers import InputLayer
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import json
from sklearn.metrics.pairwise import cosine_similarity


# 1. GPU INITIALIZATION
# This prevents TensorFlow from "hogging" all 4GB of your GTX 1650 VRAM immediately
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {gpus}")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
else:
    print("Running on CPU - Check NVIDIA Container Toolkit installation")
app = Flask(__name__)
model1_path = 'test_models/logo_segmentation_v1_94pct.keras'
model2_path = 'test_models/best_logo_triplet_model.keras'


print(model1_path)
print(model2_path)
# 1. Load models

SEG_MODEL = tf.keras.models.load_model(model1_path, compile=False)
FULL_TRIPLET = tf.keras.models.load_model(model2_path, compile=False)

# 2. Extract the specific 'functional' layer which is the actual encoder
# Based on your summary, the layer name is 'functional'
TRIPLET_MODEL = FULL_TRIPLET.get_layer('functional')

print("Successfully extracted 128-dim encoder from Triplet Model")
print("Loading Database...")
VECTORS_DB = np.load('lognet_vectors.npy')
LABELS_DB = np.load('lognet_labels.npy')
with open('brand_metadata.json', 'r') as f:
    METADATA_DB = json.load(f)


def process_image(img, target_size):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('index.html', error="No file uploaded")

        # Read image
        img_bytes = file.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # --- STEP 1: Segmentation ---
        seg_input = process_image(original_img, (224, 224))
        mask = SEG_MODEL.predict(seg_input)[0]
        mask_2d = np.squeeze(mask)
        mask_uint8 = (mask_2d * 255).astype(np.uint8)
        # --- NEW: Visualizing the Mask ---
        # Convert the 0-1 float mask to a 0-255 grayscale image
        mask_visual = Image.fromarray(mask_uint8, mode='L')
        mask_base64 = encode_image(mask_visual)

        mask = mask.squeeze()

        # --- STEP 2: Cropping Logic ---
        mask_binary = (mask > 0.5).astype(np.uint8)
        coords = np.argwhere(mask_binary > 0)

        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Scale coordinates to original image size
            scale_x = original_img.width / 224
            scale_y = original_img.height / 224

            x_min = int(x_min * scale_x)
            x_max = int(x_max * scale_x)
            y_min = int(y_min * scale_y)
            y_max = int(y_max * scale_y)

            cropped_img = original_img.crop((x_min, y_min, x_max, y_max))
            crop_base64 = encode_image(cropped_img)

            # --- STEP 3: Triplet Embedding ---
            triplet_input = process_image(cropped_img, (224, 224))
            embedding_tensor = TRIPLET_MODEL.predict(triplet_input)
            embedding = embedding_tensor[0].tolist()

            # --- STEP 4: SEARCH ENGINE (NEW!) ---
            # Compare the embedding to all 29,000 vectors in the database instantly
            similarities = cosine_similarity(
                embedding.reshape(1, -1), VECTORS_DB)[0]
            top_indices = similarities.argsort()[-5:][::-1]  # Top 5 matches

            results = []
            for idx in top_indices:
                brand = LABELS_DB[idx]
                confidence = round(similarities[idx] * 100, 2)
                info = METADATA_DB.get(brand, {"domain": "Unknown"})
                results.append({
                    "brand": brand,
                    "confidence": confidence,
                    "domain": info['domain']
                })

            return render_template('index.html', results=results, mask_img=mask_base64, crop_img=crop_base64)
        else:
            return render_template('index.html', error="No logo detected in the image!")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
