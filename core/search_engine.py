import numpy as np
import json
import cv2


class LogoSearchEngine:
    def __init__(self, model, vectors_path, metadata_path, distance_strategy):
        self.model = model
        self.strategy = distance_strategy

        print(f"💾 Search Engine: Loading Vector Database from {
              vectors_path}...")
        self.vectors_db = np.load(vectors_path)

        print(f"📄 Search Engine: Loading Metadata Registry from {
              metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.metadata_db = json.load(f)

    def search(self, cropped_rgb, user_k=5, sort_by="confidence", categories=None, consensus_levels=None, scope=None):
        """
        Performs a deep neural search with weighted confidence and multi-level filtering.
        """

        # --- 1. PRE-PROCESSING ---
        input_resized = cv2.resize(cropped_rgb, (224, 224))
        input_tensor = np.expand_dims(
            input_resized / 255.0, axis=0).astype(np.float32)

        # --- 2. AI PREDICTION ---
        # Ginagamit ang model para makakuha ng feature vector (embedding)
        vectors, masks = self.model.predict(input_tensor, verbose=0)
        query_vec = vectors[0]

        # --- 3. DISTANCE CALCULATION ---
        # Kinukumpara ang query vector sa buong database
        distances = self.strategy(query_vec, self.vectors_db)

        # --- 4. DEEP SCAN ---
        # Kukuha tayo ng top 100 entries para sa ethical weighting at aggregation
        global_k = min(100, len(self.vectors_db))
        top_indices = np.argpartition(distances, global_k)[:global_k]

        brand_data = {}

        # --- 5. GLOBAL CONTEXT AGGREGATION & FILTERING ---
        for i in top_indices:
            metadata = self.metadata_db[i]
            brand_name = metadata.get('Brand', 'Unknown')
            category = metadata.get('Category', 'N/A')
            brand_scope = metadata.get('Scope', 'GLOBAL')

            # --- A. DATABASE SCOPE FILTER ---
            if scope and scope != "BOTH":
                if brand_scope.upper() != scope.upper():
                    continue

            # --- B. SMART CATEGORY FILTER (Partial Matching) ---
            if categories and "all" not in [c.lower() for c in categories]:
                category_match = False
                for selected_cat in categories:
                    if selected_cat.lower() in category.lower():
                        category_match = True
                        break
                if not category_match:
                    continue

            # Inverse distance to get confidence percentage
            confidence = (1 / (1 + distances[i])) * 100

            if brand_name not in brand_data:
                brand_data[brand_name] = {
                    "all_scores": [],
                    "best_dist": distances[i],
                    "domain": category,
                    "scope": brand_scope
                }

            brand_data[brand_name]["all_scores"].append(confidence)

            if distances[i] < brand_data[brand_name]["best_dist"]:
                brand_data[brand_name]["best_dist"] = distances[i]

        # --- 6. ETHICAL WEIGHTED CALCULATION ---
        final_results = []
        for brand, data in brand_data.items():
            scores = sorted(data["all_scores"], reverse=True)

            # Quality Score: 80% weight sa best match, 20% sa supporting matches
            s_max = scores[0]
            s_others = scores[1:10]
            s_others_avg = sum(s_others) / len(s_others) if s_others else s_max
            weighted_conf = (0.8 * s_max) + (0.2 * s_others_avg)

            # Stability Score: Quantity of evidence in top 100
            match_count = len(scores)
            stability_score = min(100, (match_count / 20) * 100)

            # Consensus Labeling based on match frequency
            if match_count >= 15:
                consensus = "Strong"
            elif match_count >= 5:
                consensus = "Moderate"
            else:
                consensus = "Weak"

            # --- C. CONSENSUS FILTER ---
            if consensus_levels and consensus not in consensus_levels:
                continue

            final_results.append({
                "brand": brand,
                "domain": data["domain"],
                "scope": data["scope"],
                "confidence": round(weighted_conf, 2),
                "stability": round(stability_score, 2),
                "consensus": consensus,
                "match_count": match_count,
                "dist": float(data["best_dist"]),
                "is_stable": match_count >= 5
            })

        # --- 7. DYNAMIC SORTING ---
        sort_map = {
            "confidence": "confidence",
            "stability": "stability",
            "matches": "match_count"
        }
        target_key = sort_map.get(sort_by, "confidence")

        # Siguraduhin na descending order (highest score first)
        final_results.sort(key=lambda x: x.get(target_key, 0), reverse=True)

        # I-slice base sa user_k at ibalik kasama ang Grad-CAM mask
        return final_results[:user_k], masks[0]
