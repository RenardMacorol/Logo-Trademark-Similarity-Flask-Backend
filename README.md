# 🧠 AI Logo Trademark Search Engine

A full-stack, dual-model AI application that detects, isolates, and identifies company logos from user-uploaded images in milliseconds. It searches against a pre-computed database of 30,000 global brands and instantly enriches the results with Wikipedia metadata.

## ✨ Key Features
* **Two-Stage AI Vision Pipeline:**
  1. **Logo Segmentation:** A TensorFlow model isolates the logo in the image and generates a cropping mask.
  2. **Feature Extraction:** A Triplet Encoder model converts the cropped logo into a dense 128-dimensional mathematical vector.
* **Lightning-Fast Vector Search:** Uses Scikit-Learn's `cosine_similarity` to instantly match the extracted vector against 30,000 pre-computed brand vectors.
* **Contextual Enrichment:** Automatically scrapes Wikipedia to provide industry context and brand summaries.
* **Cross-Platform Ready:** Includes a clean Bootstrap web interface and a dedicated JSON REST API (`/api/search`) designed for a mobile app (Flutter).

## 🛠️ Tech Stack
* **Backend:** Python, Flask, Werkzeug
* **Machine Learning:** TensorFlow / Keras, NumPy, Scikit-Learn, Pillow (PIL)
* **Data Integration:** Wikipedia API, JSON
* **Frontend:** HTML5, CSS3, Bootstrap 5

## ⚠️ Important Note Regarding Setup
To keep this Git repository lightweight, **the heavy machine learning models and database files are not included in this repository** (they are listed in `.gitignore`). 

To run this project locally, you must provide your own trained `.keras` models and numpy `.npy` databases in the root directory:
* `test_models/logo_segmentation_v1_94pct.keras`
* `test_models/best_logo_triplet_model.keras`
* `lognet_vectors.npy`
* `lognet_labels.npy`
* `wikipedia_metadata.json`

## 🚀 Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/ai-logo-search-flask.git](https://github.com/YourUsername/ai-logo-search-flask.git)
   cd ai-logo-search-flask
