import numpy as np
import json
import cv2
import os


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

    def get_random_vectors(self, n=30):
        """Kukuha ng random vectors para sa background ng Latent Map"""
        indices = np.random.choice(len(self.vectors_db), min(
            n, len(self.vectors_db)), replace=False)
        return self.vectors_db[indices].tolist()

    def _prepare_tensor(self, img_rgb):
        """Internal helper para siguruhing (1, 224, 224, 3) ang shape na papasok sa model"""
        resized = cv2.resize(img_rgb, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def search(self, cropped_rgb, query_vector=None, user_k=5, sort_by="confidence", categories=None, consensus_levels=None, scope=None):
        # --- 1. PRE-PROCESSING & AI PREDICTION ---
        input_tensor = self._prepare_tensor(cropped_rgb)

        if query_vector is None:
            vectors, masks = self.model.predict(input_tensor, verbose=0)
            query_vec = vectors[0]
            best_mask = masks[0]
        else:
            query_vec = query_vector
            _, masks = self.model.predict(input_tensor, verbose=0)
            best_mask = masks[0]

        # --- 2. DISTANCE CALCULATION ---
        distances = self.strategy(query_vec, self.vectors_db)

        # --- 3. DEEP SCAN ---
        global_k = min(100, len(self.vectors_db))
        top_indices = np.argpartition(distances, global_k)[:global_k]

        brand_data = {}

        # --- 4. AGGREGATION & FILTERING ---
        for i in top_indices:
            metadata = self.metadata_db[i]
            brand_name = metadata.get(
                'Brand') or metadata.get('brand') or 'Unknown'
            category = metadata.get(
                'Category') or metadata.get('domain') or 'N/A'
            brand_scope = metadata.get('Scope') or 'GLOBAL'
            file_path = metadata.get('File_Path') or metadata.get('path') or ""

            if scope and scope != "BOTH":
                if str(brand_scope).upper() != str(scope).upper():
                    continue

            if categories and "all" not in [c.lower() for c in categories]:
                if not any(cat.lower() in category.lower() for cat in categories):
                    continue

            confidence = (1 / (1 + distances[i])) * 100

            if brand_name not in brand_data:
                brand_data[brand_name] = {
                    "all_scores": [],
                    "best_dist": distances[i],
                    "domain": category,
                    "scope": brand_scope,
                    "path": file_path,
                    "vector": self.vectors_db[i].tolist()
                }

            brand_data[brand_name]["all_scores"].append(confidence)

            if distances[i] < brand_data[brand_name]["best_dist"]:
                brand_data[brand_name]["best_dist"] = distances[i]
                brand_data[brand_name]["path"] = file_path
                brand_data[brand_name]["vector"] = self.vectors_db[i].tolist()

        # --- 5. WEIGHTED CALCULATION ---
        final_results = []
        for brand, data in brand_data.items():
            scores = sorted(data["all_scores"], reverse=True)
            s_max = scores[0]
            s_others = scores[1:10]
            s_others_avg = sum(s_others) / len(s_others) if s_others else s_max

            weighted_conf = (0.8 * s_max) + (0.2 * s_others_avg)

            match_count = len(scores)
            stability_score = min(100, (match_count / 20) * 100)

            if match_count >= 15:
                consensus = "Strong"
            elif match_count >= 5:
                consensus = "Moderate"
            else:
                consensus = "Weak"

            if consensus_levels and consensus not in consensus_levels:
                continue

            final_results.append({
                "brand": brand,
                "Brand": brand,
                "Category": data["domain"],
                "File_Path": data["path"],
                "vector": data["vector"],
                "confidence": round(weighted_conf, 2),
                "stability": round(stability_score, 2),
                "consensus": consensus,
                "match_count": match_count,
                "is_stable": match_count >= 5,
                "orb_similarity": 0,
                "forensic_viz": None
            })

        # --- 6. DYNAMIC SORTING ---
        sort_map = {"confidence": "confidence",
                    "stability": "stability", "matches": "match_count"}
        target_key = sort_map.get(sort_by, "confidence")
        final_results.sort(key=lambda x: x.get(target_key, 0), reverse=True)

        # Slice results to top K
        top_k_list = final_results[:user_k]

        # --- 🟢 STAGE 2: MULTI-FORENSIC TRIGGER ---
        for match in top_k_list:
            try:
                from processor.forensic_engine import ForensicEngine

                raw_path = match["File_Path"]

                # 🛠️ DOCKER PATH MAPPING
                # Ito yung fix para mabasa ng cv2.imread ang files sa loob ng container
                fixed_path = raw_path.replace(
                    "/workspace/Logo2K_Dataset_Permanent/datasetcopy/",
                    "/workspace/backend/static/database/"
                )

                if os.path.exists(fixed_path):
                    viz, orb_score = ForensicEngine.generate_evidence(
                        cropped_rgb,
                        fixed_path
                    )
                    match["orb_similarity"] = orb_score
                    match["forensic_viz"] = viz
                else:
                    print(f"❌ File Not Found at: {fixed_path}")

            except Exception as e:
                print(f"⚠️ Forensic Engine Error for {match['brand']}: {e}")

        # Ibabalik ang lahat para sa Service layer
        first_forensic = top_k_list[0]["forensic_viz"] if top_k_list else None

        return top_k_list, best_mask, first_forensic
