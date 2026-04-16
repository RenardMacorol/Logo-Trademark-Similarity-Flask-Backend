import numpy as np


class ResultProcessor:
    @staticmethod
    def aggregate_and_score(top_indices, distances, metadata_db, vectors_db, scope=None, categories=None):
        brand_data = {}

        for i in top_indices:
            meta = metadata_db[i]
            # Metadata normalization
            brand = meta.get('Brand') or meta.get('brand') or 'Unknown'
            cat = meta.get('Category') or meta.get('domain') or 'N/A'
            b_scope = meta.get('Scope', 'GLOBAL')

            # Path correction for database access
            path = meta.get('File_Path') or meta.get('path', "")
            if "/workspace/Logo2K_Dataset_Permanent/datasetcopy/" in path:
                path = path.replace(
                    "/workspace/Logo2K_Dataset_Permanent/datasetcopy/", "/workspace/backend/static/database/")

            # 1. Filtering Logic
            if scope and scope != "BOTH" and b_scope.upper() != scope.upper():
                continue
            if categories and "all" not in [c.lower() for c in categories]:
                if not any(c.lower() in cat.lower() for c in categories):
                    continue

            # 2. Neural Score Calculation (The Blue Bar)
            # Inverse distance normalized to percentage
            raw_conf = (1 / (1 + distances[i])) * 100

            # 3. DNA Match Estimation (The Orange Bar)
            # This simulates the "Pixel DNA Overlap" based on vector proximity
            # until the ForensicEngine provides the actual spatial overlap.
            dna_estimate = max(5.0, raw_conf * 0.1)  # Baseline DNA overlap

            # 4. Grouping by Brand
            if brand not in brand_data:
                brand_data[brand] = {
                    "scores": [],
                    "dna_scores": [],
                    "best_dist": distances[i],
                    "meta": meta,
                    "path": path,
                    "vec": vectors_db[i]
                }

            brand_data[brand]["scores"].append(raw_conf)
            brand_data[brand]["dna_scores"].append(dna_estimate)

            # Keep the visually closest image as the primary representative
            if distances[i] < brand_data[brand]["best_dist"]:
                brand_data[brand].update(
                    {"best_dist": distances[i], "path": path})

        return ResultProcessor._finalize(brand_data)

    @staticmethod
    def _finalize(brand_data):
        final = []
        for brand, data in brand_data.items():
            scores = sorted(data["scores"], reverse=True)
            dna_scores = sorted(data["dna_scores"], reverse=True)

            # --- 🟢 AUDIT REPORT METRICS ---

            # A. Neural Confidence (Blue Bar)
            # Weighted average of top hits for that brand
            neural_conf = (
                0.8 * scores[0]) + (0.2 * (np.mean(scores[1:5]) if len(scores) > 1 else scores[0]))

            # B. Pixel DNA Overlap (Orange Bar)
            pixel_dna = (0.9 * dna_scores[0]) + (
                0.1 * (np.mean(dna_scores[1:5]) if len(dna_scores) > 1 else dna_scores[0]))

            # C. Consensus/Stability (The metadata metrics)
            count = len(scores)
            stability = min(100, (count / 20) * 100)

            # Initial placeholder for Overall Similarity.
            # Note: The ForensicEngine will later inject the 'Geometric Match'
            # and recalculate the 'Overall Similarity' to be accurate.
            overall = (neural_conf * 0.4) + (pixel_dna * 0.3)

            final.append({
                "brand": brand,
                "category": data["meta"].get("Category", "N/A"),
                "File_Path": data["path"],
                "vector": data["vec"].tolist(),
                "match_count": count,
                "stability": round(stability, 2),
                "metrics": {
                    "neural_confidence": round(neural_conf, 2),
                    "pixel_dna_overlap": round(pixel_dna, 2),
                    "geometric_structural_match": 0.0,  # Will be filled by ForensicEngine
                    "overall_similarity": round(overall, 2)
                }
            })

        return final
