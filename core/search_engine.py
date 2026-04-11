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
        Performs a deep neural search with weighted confidence and path preservation.
        """

        # --- 1. PRE-PROCESSING ---
        input_resized = cv2.resize(cropped_rgb, (224, 224))
        input_tensor = np.expand_dims(
            input_resized / 255.0, axis=0).astype(np.float32)

        # --- 2. AI PREDICTION ---
        # Makakuha ng feature vector (embedding) at attention mask
        vectors, masks = self.model.predict(input_tensor, verbose=0)
        query_vec = vectors[0]

        # --- 3. DISTANCE CALCULATION ---
        # Vector comparison laban sa buong database
        distances = self.strategy(query_vec, self.vectors_db)

        # --- 4. DEEP SCAN ---
        # Top 100 entries ang kukunin para sa aggregation logic
        global_k = min(100, len(self.vectors_db))
        top_indices = np.argpartition(distances, global_k)[:global_k]

        brand_data = {}

        # --- 5. GLOBAL CONTEXT AGGREGATION & FILTERING ---
        for i in top_indices:
            metadata = self.metadata_db[i]

            # SAKTONG KEYS PARA SA METADATA MO
            brand_name = metadata.get(
                'Brand') or metadata.get('brand') or 'Unknown'
            category = metadata.get(
                'Category') or metadata.get('domain') or 'N/A'
            brand_scope = metadata.get('Scope') or 'GLOBAL'
            file_path = metadata.get('File_Path') or metadata.get('path') or ""

            # --- A. DATABASE SCOPE FILTER ---
            if scope and scope != "BOTH":
                if str(brand_scope).upper() != str(scope).upper():
                    continue

            # --- B. SMART CATEGORY FILTER ---
            if categories and "all" not in [c.lower() for c in categories]:
                category_match = False
                for selected_cat in categories:
                    if selected_cat.lower() in category.lower():
                        category_match = True
                        break
                if not category_match:
                    continue

            # Confidence conversion
            confidence = (1 / (1 + distances[i])) * 100

            # Dito natin i-store ang brand info at ang Path
            if brand_name not in brand_data:
                brand_data[brand_name] = {
                    "all_scores": [],
                    "best_dist": distances[i],
                    "domain": category,
                    "scope": brand_scope,
                    "path": file_path  # Inisyal na path
                }

            brand_data[brand_name]["all_scores"].append(confidence)

            # Kung mas 'accurate' ang current distance, i-update ang reference path
            if distances[i] < brand_data[brand_name]["best_dist"]:
                brand_data[brand_name]["best_dist"] = distances[i]
                brand_data[brand_name]["path"] = file_path

        # --- 6. ETHICAL WEIGHTED CALCULATION ---
        final_results = []
        for brand, data in brand_data.items():
            scores = sorted(data["all_scores"], reverse=True)

            # Weighted Scoring (80% Top match, 20% Support)
            s_max = scores[0]
            s_others = scores[1:10]
            s_others_avg = sum(s_others) / len(s_others) if s_others else s_max
            weighted_conf = (0.8 * s_max) + (0.2 * s_others_avg)

            # Stability/Match Count
            match_count = len(scores)
            stability_score = min(100, (match_count / 20) * 100)

            # Consensus Labeling
            if match_count >= 15:
                consensus = "Strong"
            elif match_count >= 5:
                consensus = "Moderate"
            else:
                consensus = "Weak"

            # --- C. CONSENSUS FILTER ---
            if consensus_levels and consensus not in consensus_levels:
                continue

            # BUBALIK NA TAYO SA FLASK: Isasama na natin lahat ng kailangan
            final_results.append({
                "brand": brand,           # Lowercase version
                "Brand": brand,           # Capital version (for compatibility)
                "Category": data["domain"],
                "File_Path": data["path"],  # HETO ANG NAWAWALA KANINA!
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
        final_results.sort(key=lambda x: x.get(target_key, 0), reverse=True)

        # Ibalik ang Top K results at ang Segmentation Mask
        return final_results[:user_k], masks[0]
