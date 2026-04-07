from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import base64
import json
import os
import cv2

app = Flask(__name__)

# --- 1. GPU CONFIG (GTX 1650 Optimized) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Limit VRAM to 3GB to leave room for the OS/Display
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3000)])
    except RuntimeError as e:
        print(f"GPU error: {e}")

# --- 2. PATHS ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE = os.path.join('test_models/initial_model.keras')
VECTORS_PATH = os.path.join(BASE_DIR, 'databases', 'registry_embeddings.npy')
METADATA_PATH = os.path.join(BASE_DIR, 'databases', 'registry_metadata.json')

# --- 3. ARCHITECTURE (With Lambda Fix) ---


def build_multi_task_encoder(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_shape=input_shape)

    shared_features = layers.BatchNormalization(
        name="backbone_norm")(base_model.output)

    # Segmentation Head
    x = layers.Conv2DTranspose(512, (3, 3), strides=(
        2, 2), padding='same', activation='relu')(shared_features)
    x = layers.Conv2DTranspose(256, (3, 3), strides=(
        2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(
        2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(
        2, 2), padding='same', activation='relu')(x)
    seg_mask = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same',
                                      activation='sigmoid', name='segmentation_output')(x)

    # Attention
    small_mask = layers.Resizing(7, 7, name="attention_resize")(seg_mask)
    weighted_features = layers.Multiply(name="spatial_attention")([
        shared_features, small_mask])

    # Embedding Head
    pooled = layers.GlobalAveragePooling2D()(weighted_features)
    pooled = layers.BatchNormalization()(pooled)
    embedding = layers.Dense(512, activation='softplus',
                             name="embedding_dense")(pooled)

    # CRITICAL: Added output_shape=(512,) to prevent Lambda loading crash
    embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),
                              name='embedding_output',
                              output_shape=(512,))(embedding)

    return Model(inputs=base_model.input, outputs=[embedding, seg_mask])


# --- 4. SECURE LOAD ---
print("🚀 Initializing WonksNet Brain...")

# Build architecture first to define the Lambda output_shape
logo_encoder = build_multi_task_encoder()

try:
    # Use load_weights instead of load_model to bypass functional API errors
    logo_encoder.load_weights(MODEL_FILE)
    print("✅ Model weights loaded successfully.")
except Exception as e:
    print(f"⚠️ Direct weight load failed, trying surgical extraction: {e}")
    try:
        source = tf.keras.models.load_model(
            MODEL_FILE, compile=False, safe_mode=False, custom_objects={'tf': tf})
        for layer in logo_encoder.layers:
            try:
                logo_encoder.get_layer(layer.name).set_weights(
                    source.get_layer(layer.name).get_weights())
            except:
                continue
        print("✅ Surgical transfer successful.")
    except Exception as final_e:
        print(f"❌ ALL LOADING FAILED: {final_e}")

VECTORS_DB = np.load(VECTORS_PATH)
with open(METADATA_PATH, 'r') as f:
    METADATA_DB = json.load(f)

# --- 5. LOGIC HELPERS ---


def generate_safety_mask(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def auto_crop_logo(img_rgb):
    mask = generate_safety_mask(cv2.resize(img_rgb, (224, 224)))
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Scale coords back to original image size
        sy, sx = img_rgb.shape[0]/224, img_rgb.shape[1]/224
        x, y, w, h = int(x*sx), int(y*sy), int(w*sx), int(h*sy)

        y1, y2 = max(0, y-20), min(img_rgb.shape[0], y+h+20)
        x1, x2 = max(0, x-20), min(img_rgb.shape[1], x+w+20)
        return cv2.resize(img_rgb[y1:y2, x1:x2], (224, 224))
    return cv2.resize(img_rgb, (224, 224))


def encode_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

# --- 6. ROUTES ---


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file"

        # 1. Image Conversion
        img_raw = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # 2. Auto-Crop (Fixes 30% confidence issue)
        cropped_rgb = auto_crop_logo(img_rgb)

        # 3. AI Prediction
        input_tensor = np.expand_dims(
            cropped_rgb / 255.0, axis=0).astype(np.float32)
        vectors, masks = logo_encoder.predict(input_tensor, verbose=0)

        # 4. Chi-Square Top-K Matching
        query_vec = vectors[0]
        distances = 0.5 * \
            np.sum(((VECTORS_DB - query_vec)**2) /
                   (VECTORS_DB + query_vec + 1e-10), axis=1)

        results = []
        for i in range(len(VECTORS_DB)):
            results.append({
                "brand": METADATA_DB[i]['Brand'],
                "category": METADATA_DB[i]['Category'],
                "confidence": round((1 / (1 + distances[i])) * 100, 2),
                "dist": float(distances[i])
            })
        results.sort(key=lambda x: x['dist'])

        # 5. Mask Visualization (Jet Heatmap style)
        mask_data = (masks[0].squeeze() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return render_template('index.html',
                               results=results[:5],
                               mask_img=encode_to_base64(heatmap_rgb),
                               original_img=encode_to_base64(cropped_rgb))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
