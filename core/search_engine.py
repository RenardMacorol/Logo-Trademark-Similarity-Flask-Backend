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

    def search(self, cropped_rgb, query_vector=None, query_attention=None, user_k=5, sort_by="confidence", categories=None, consensus_levels=None, scope=None):
        # --- 1. OPTIMIZED AI PREDICTION (Capturing Attention) ---
        # We now capture the mask/attention output to feed the "Neural Level" UI
        mask_output = None

        if query_vector is not None:
            query_vec = query_vector
            # Use pre-computed attention if available, otherwise remains None
            mask_output = query_attention
        else:
            input_tensor = self._prepare_tensor(cropped_rgb)
            # 🟢 inference call
            vectors, masks = self.model.predict(input_tensor, verbose=0)
            query_vec = vectors[0]

            # 🟢 THE CLEANUP:
            # Most Keras/TF models return masks as (H, W, 1).
            # We need to squeeze it to (H, W) for the encoder to work perfectly.
            if masks is not None and len(masks) > 0:
                raw_mask = masks[0]
                # Remove any extra dimensions (e.g., (224, 224, 1) -> (224, 224))
                if len(raw_mask.shape) == 3:
                    mask_output = np.squeeze(raw_mask, axis=-1)
                else:
                    mask_output = raw_mask
                # --- 2. DISTANCE CALCULATION ---
        distances = self.strategy(query_vec, self.vectors_db)

        # --- 3. DEEP SCAN ---
        # We scan a wider range (100) to find all variants of a brand for stability
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

            # Scope and Category Filters
            if scope and scope != "BOTH" and str(brand_scope).upper() != str(scope).upper():
                continue
            if categories and "all" not in [c.lower() for c in categories]:
                if not any(cat.lower() in category.lower() for cat in categories):
                    continue

            # Convert distance to a percentage confidence
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

            # Ensure we keep the path and vector for the absolute closest visual match
            if distances[i] < brand_data[brand_name]["best_dist"]:
                brand_data[brand_name]["best_dist"] = distances[i]
                brand_data[brand_name]["path"] = file_path
                brand_data[brand_name]["vector"] = self.vectors_db[i].tolist()

        # --- 5. WEIGHTED NEURAL CALCULATION ---
        final_results = []
        for brand, data in brand_data.items():
            scores = sorted(data["all_scores"], reverse=True)
            s_max = scores[0]

            # Aggregate top 10 variants to calculate stability/consensus
            s_others_avg = sum(
                scores[1:10]) / len(scores[1:10]) if len(scores) > 1 else s_max

            # Neural weighted confidence: 80% primary hit, 20% ensemble stability
            weighted_conf = (0.8 * s_max) + (0.2 * s_others_avg)

            match_count = len(scores)
            stability_score = min(100, (match_count / 20) * 100)

            # Define consensus levels for the UI
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
                "Category": data["domain"],
                "File_Path": data["path"],
                "vector": data["vector"],
                "confidence": round(weighted_conf, 2),
                "stability": round(stability_score, 2),
                "consensus": consensus,
                "match_count": match_count,
                "is_stable": match_count >= 5
            })

        # --- 6. FINAL SORT ---
        # Sort by confidence and slice to requested user_k
        final_results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
        top_candidates = final_results[:user_k]

        # 🟢 THE FIX: We now return the mask_output instead of None
        # Result Format: (Matches List, Segmentation Attention, Forensic Viz [Removed])
        return top_candidates, mask_output, None
