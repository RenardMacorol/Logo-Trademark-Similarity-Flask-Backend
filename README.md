# 🧠 WonksNet: Multi-Task Trademark Intelligence

**WonksNet** is a high-performance, full-stack AI engine that detects, segmentates, and identifies company logos in milliseconds. By utilizing a **Multi-Task Learning (MTL)** architecture with a ResNet50 backbone, it isolates trademarks from background noise and matches them against a registry of 30,000+ global brands.

---

## ✨ Key Features

* **Multi-Task Vision Pipeline**: A single unified encoder that simultaneously handles **Logo Segmentation** (finding the logo) and **Feature Embedding** (understanding the brand).
* **Latent Spatial Attention**: Uses the generated segmentation mask to "weigh" the features, forcing the AI to ignore background clutter and focus only on the trademark.
* **Sub-Millisecond Vector Search**: Optimized **Chi-Square Distance** matching against a pre-computed database for industry-leading retrieval speed.
* **Robust Pre-Processing**: Built-in **Auto-Crop & Pseudo-Mask** logic that intelligently zooms into detected logos, drastically increasing identification confidence.
* **Mobile-First REST API**: Dedicated JSON endpoints designed for seamless integration with **Flutter** or **React Native** applications.

---

## 🛠️ Tech Stack

* **Backend**: Python, Flask, Werkzeug
* **Deep Learning**: TensorFlow 2.x (Keras 3), ResNet50, NumPy
* **Computer Vision**: OpenCV (Otsu Thresholding, Morphological Processing)
* **Data**: Scikit-Learn (Vector math), Wikipedia API
* **Deployment**: Optimized for **NVIDIA GTX 1650** (VRAM-efficient inference)

---

## 📂 Project Structure

```text
logo_app/
├── core/               # AI Engine (Model architecture & Surgical Weight Loading)
├── api/                # Mobile REST API Blueprints (JSON-first)
├── processor/          # OpenCV Image Processing (Auto-Crop & Safety Mask)
├── databases/          # Pre-computed Registry Vectors & Metadata
├── static/             # Web UI assets (CSS/JS)
├── templates/          # Bootstrap 5 Interface
└── test_models/        # Trained .keras files (Excluded from Git)
