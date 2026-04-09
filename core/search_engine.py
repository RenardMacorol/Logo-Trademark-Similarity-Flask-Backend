import numpy as np
import json


class LogoSearchEngine:
    def __init__(self, model, vectors_path, metadata_path, distance_strategy):
        self.model = model
        self.strategy = distance_strategy
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
        query_vec = vectors[0]

        # 2. Distance Calculation
        distances = self.strategy(query_vec, self.vectors_db)

        # 3. Group ALL raw matches by Brand
        # We store all scores for each brand to calculate the weighted average later
        brand_data = {}

        for i in range(len(self.vectors_db)):
            brand_name = self.metadata_db[i]['Brand']
            confidence = (1 / (1 + distances[i])) * 100

            if brand_name not in brand_data:
                brand_data[brand_name] = {
                    "all_scores": [],
                    "best_dist": distances[i],
                    "domain": self.metadata_db[i].get('Category', 'N/A')
                }

            brand_data[brand_name]["all_scores"].append(confidence)

            # Track the absolute best distance for metadata accuracy
            if distances[i] < brand_data[brand_name]["best_dist"]:
                brand_data[brand_name]["best_dist"] = distances[i]

        # 4. Apply Weighted Equation & Collapse Duplicates
        final_results = []

        for brand, data in brand_data.items():
            scores = data["all_scores"]

            if len(scores) > 1:
                # Weighted Logic: 80% Top Score + 20% Avg of next best matches
                scores.sort(reverse=True)
                s_max = scores[0]

                # We limit s_others to the top 4 secondary matches to avoid
                # noise from low-confidence matches deep in the DB
                s_others = scores[1:5]
                s_others_avg = sum(s_others) / len(s_others)

                final_conf = (0.8 * s_max) + (0.2 * s_others_avg)
            else:
                # Only one match found, no weighting possible
                final_conf = scores[0]

            final_results.append({
                "brand": brand,
                "domain": data["domain"],
                "confidence": round(final_conf, 2),
                "dist": float(data["best_dist"])
            })

        # 5. Sort by Weighted Confidence (Highest First)
        final_results.sort(key=lambda x: x['confidence'], reverse=True)

        return final_results[:k], masks[0]
