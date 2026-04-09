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

    def search(self, cropped_rgb, user_k=5):
        # 1. AI Prediction (Embedding + Heatmap Mask)
        input_tensor = np.expand_dims(
            cropped_rgb / 255.0, axis=0).astype(np.float32)
        vectors, masks = self.model.predict(input_tensor, verbose=0)
        query_vec = vectors[0]

        # 2. Distance Calculation
        distances = self.strategy(query_vec, self.vectors_db)

        # 3. Deep Scan (Top 100 for Ethical Context)
        # We look at 100 neighbors even if the user only wants to see 5.
        global_k = min(100, len(self.vectors_db))
        top_indices = np.argpartition(distances, global_k)[:global_k]

        brand_data = {}

        # 4. Global Context Aggregation
        for i in top_indices:
            brand_name = self.metadata_db[i]['Brand']
            # Raw confidence score (Inverse of distance)
            confidence = (1 / (1 + distances[i])) * 100

            if brand_name not in brand_data:
                brand_data[brand_name] = {
                    "all_scores": [],
                    "best_dist": distances[i],
                    "domain": self.metadata_db[i].get('Category', 'N/A')
                }

            brand_data[brand_name]["all_scores"].append(confidence)

            # Track the most accurate match instance
            if distances[i] < brand_data[brand_name]["best_dist"]:
                brand_data[brand_name]["best_dist"] = distances[i]

        # 5. Ethical Weighted Calculation & Stability Metrics
        final_results = []
        for brand, data in brand_data.items():
            # Sort scores descending: [Best match, 2nd best, ..., worst match]
            scores = sorted(data["all_scores"], reverse=True)

            # --- A. Weighted Confidence (The Quality Metric) ---
            s_max = scores[0]
            # Verify against top 9 secondary matches (Top 10 pool)
            s_others = scores[1:10]
            s_others_avg = sum(s_others) / len(s_others) if s_others else s_max
            weighted_conf = (0.8 * s_max) + (0.2 * s_others_avg)

            # --- B. Stability Score (The Quantity/Ethical Metric) ---
            # How many times did this brand appear in the Top 100 scan?
            match_count = len(scores)

            # Formula: Normalize 20 matches as "100% Stable".
            # If a brand appears 20+ times in the top 100, the match is statistically rock-solid.
            stability_score = min(100, (match_count / 20) * 100)

            # --- C. Consensus Labeling (Explainability) ---
            if match_count >= 15:
                consensus = "Strong"
            elif match_count >= 5:
                consensus = "Moderate"
            else:
                consensus = "Weak"

            final_results.append({
                "brand": brand,
                "domain": data["domain"],
                "confidence": round(weighted_conf, 2),
                "stability": round(stability_score, 2),
                "consensus": consensus,
                "match_count": match_count,
                "dist": float(data["best_dist"]),
                "is_stable": match_count >= 5  # Boolean helper for UI color logic
            })

        # 6. Final Sort by Confidence and Slice by user_k
        final_results.sort(key=lambda x: x['confidence'], reverse=True)

        return final_results[:user_k], masks[0]
