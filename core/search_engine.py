import numpy as np
import json


class LogoSearchEngine:
    def __init__(self, model, vectors_path, metadata_path):
        self.model = model
        print("💾 Search Engine: Loading Vector Database...")
        self.vectors_db = np.load(vectors_path)

        print("📄 Search Engine: Loading Metadata Registry...")
        with open(metadata_path, 'r') as f:
            self.metadata_db = json.load(f)

    def search(self, cropped_rgb, k=5):
        # 1. AI Prediction (Embedding + Mask)
        input_tensor = np.expand_dims(
            cropped_rgb / 255.0, axis=0).astype(np.float32)
        vectors, masks = self.model.predict(input_tensor, verbose=0)

        # 2. Chi-Square Distance Calculation
        query_vec = vectors[0]
        distances = 0.5 * np.sum(((self.vectors_db - query_vec)**2) /
                                 (self.vectors_db + query_vec + 1e-10), axis=1)

        # 3. Zip and Sort Results
        results = []
        for i in range(len(self.vectors_db)):
            results.append({
                "brand": self.metadata_db[i]['Brand'],
                "category": self.metadata_db[i].get('Category', 'N/A'),
                "confidence": round((1 / (1 + distances[i])) * 100, 2),
                "dist": float(distances[i])
            })

        results.sort(key=lambda x: x['dist'])

        # Return Top-K results and the raw mask for visualization
        return results[:k], masks[0]
