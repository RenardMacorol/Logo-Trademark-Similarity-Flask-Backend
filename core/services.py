import numpy as np
import cv2
import constant
import random
from core.model import WonksNetModel
from core.search_engine import LogoSearchEngine
from core.distance_metric import chi_square_distance
from core.latent_service import LatentMapper


class WonksNetService:
    _model_instance = None
    _engines = {}

    @classmethod
    def get_model(cls):
        """Singleton pattern for model loading."""
        if cls._model_instance is None:
            print("🧠 [SINGLETON] Loading Model...")
            cls._model_instance = WonksNetModel(constant.MODEL_FILE)
        return cls._model_instance

    @classmethod
    def _get_scope(cls, scope_key):
        """Loads and caches LogoSearchEngines."""
        if scope_key not in cls._engines:
            paths = {
                constant.DB_PH_SCOPE: (constant.PH_VECTORS_PATH, constant.PH_METADATA_PATH),
                constant.DB_GLOBAL_SCOPE: (
                    constant.GLOBAL_VECTORS_PATH, constant.GLOBAL_METADATA_PATH)
            }
            if scope_key not in paths:
                raise ValueError(f"Invalid scope key: {scope_key}")

            v_path, m_path = paths[scope_key]
            model = cls.get_model()

            print(f"💾 [INIT] Loading {scope_key} dataset into RAM...")
            cls._engines[scope_key] = LogoSearchEngine(
                model.encoder, v_path, m_path,
                distance_strategy=chi_square_distance
            )
        return cls._engines[scope_key]

    @classmethod
    def warm_up(cls):
        """🔥 SYSTEM PRE-HEAT."""
        print("🔥 [WARMUP] Starting System Pre-heat...")
        try:
            cls.get_model()
            cls._get_scope(constant.DB_PH_SCOPE)
            cls._get_scope(constant.DB_GLOBAL_SCOPE)
            print("✅ [WARMUP] System fully optimized.")
        except Exception as e:
            print(f"❌ [WARMUP] Error: {e}")

    # --- PREDICTION LOGIC (UNTOUCHED AS REQUESTED) ---
    @classmethod
    def _predict_with_tta(cls, model, img_rgb):
        resized = cv2.resize(img_rgb, (224, 224))
        input_norm = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_norm, axis=0)

        feat1 = model.encoder.predict(input_tensor, verbose=0)
        flipped_img = cv2.flip(input_norm, 1)
        flipped_tensor = np.expand_dims(flipped_img, axis=0)
        feat2 = model.encoder.predict(flipped_tensor, verbose=0)

        return np.mean([feat1[0], feat2[0]], axis=0)

    @classmethod
    def predict(cls, full_image_rgb, scope="PH", k=5, sort_by="confidence", categories=None):
        model = cls.get_model()

        # 1. Search Vector using your exact TTA logic
        query_vector = cls._predict_with_tta(model, full_image_rgb)

        # 2. Forensic Heatmap logic
        resized = cv2.resize(full_image_rgb, (224, 224))
        input_tensor = np.expand_dims(
            resized.astype(np.float32) / 255.0, axis=0)
        preds = model.predict(input_tensor, verbose=0)

        raw_seg = np.squeeze(preds[1][0])
        heatmap_norm = cv2.normalize(
            raw_seg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        query_attention = cv2.GaussianBlur(heatmap_norm, (15, 15), 0)

        # 3. Target Scopes
        targets = [cls._get_scope(constant.DB_PH_SCOPE), cls._get_scope(constant.DB_GLOBAL_SCOPE)] if scope == "BOTH" else \
                  [cls._get_scope(constant.DB_GLOBAL_SCOPE if scope ==
                                  "GLOBAL" else constant.DB_PH_SCOPE)]

        # 4. Search & Ranking
        raw_combined = []
        best_forensic = None  # To store the visual evidence from the engine

        for engine in targets:
            # FIX: Catch all 3 values (res, mask, forensic)
            res, _, forensic = engine.search(
                full_image_rgb,
                query_vector=query_vector,
                query_attention=query_attention,
                user_k=k,
                categories=categories
            )
            raw_combined.extend(res)

            # Keep the forensic viz from the highest engine result if not already set
            if forensic and not best_forensic:
                best_forensic = forensic

        # 5. Final Re-ranking
        raw_combined.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
        top_k = raw_combined[:k]

        # 6. Latent Mapping (Discovery Mode)
        neighbor_vectors = [item.get('vector')
                            for item in top_k if 'vector' in item]
        industry_filter = categories[0] if categories else 'All'
        bg_vectors, _ = cls.get_market_embeddings(
            scope=scope, industry=industry_filter)

        if len(bg_vectors) > 50:
            idx = np.random.choice(len(bg_vectors), 50, replace=False)
            bg_vectors = bg_vectors[idx]

        latent_data = LatentMapper.map_to_2d(
            query_vector, neighbor_vectors, bg_vectors)

        return {
            "top_matches": top_k,
            "latent_map": latent_data,
            "heatmap": query_attention,
            "forensic_evidence": top_k[0].get("forensic_viz") if top_k else best_forensic
        }

    @classmethod
    def get_market_embeddings(cls, scope='PH', industry='All'):
        """Discovery Mode: Fetching and cleaning for Latent Space."""
        all_feats, all_metadata = [], []
        scope_input = str(scope).upper()

        target_keys = [constant.DB_PH_SCOPE, constant.DB_GLOBAL_SCOPE] if scope_input == "BOTH" else \
                      [constant.DB_GLOBAL_SCOPE] if scope_input == "GLOBAL" else [
                          constant.DB_PH_SCOPE]

        for skey in target_keys:
            try:
                engine = cls._get_scope(skey)
                vectors = np.array(engine.vectors_db if hasattr(
                    engine, 'vectors_db') else engine.vectors)
                metadata_list = engine.metadata_db

                for i in range(min(len(vectors), len(metadata_list))):
                    meta = metadata_list[i]
                    ind = (meta.get('Category') or meta.get(
                        'category') or "General")
                    brand = (meta.get('Brand') or meta.get(
                        'brand') or "Unknown")
                    path = (meta.get('file_path')
                            or meta.get('File_Path') or "")

                    # Metadata Cleaning
                    brand = str(brand).replace('_', ' ').title().strip()
                    ind = str(ind).strip().title()

                    if industry.lower() == 'all' or ind.lower() == industry.lower():
                        all_feats.append(vectors[i])
                        all_metadata.append({
                            "image_path": str(path),
                            "metadata": {"brand_name": brand, "industry_domain": ind, "origin": skey}
                        })
            except Exception as e:
                print(f"🔴 ERROR in {skey} scope: {e}")

        return np.array(all_feats), all_metadata
