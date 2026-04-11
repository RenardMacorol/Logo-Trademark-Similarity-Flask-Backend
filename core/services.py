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
        if cls._model_instance is None:
            print("🧠 [SINGLETON] Loading Model...")
            cls._model_instance = WonksNetModel(constant.MODEL_FILE)
        return cls._model_instance

    @classmethod
    def _get_scope(cls, scope_key):
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
        print("🔥 [WARMUP] Initializing System Engines...")
        cls._get_scope(constant.DB_PH_SCOPE)
        cls._get_scope(constant.DB_GLOBAL_SCOPE)
        print("✅ [WARMUP] System fully loaded.")

    # --- DISCOVERY MODE LOGIC ---
    @classmethod
    def get_market_embeddings(cls, scope='PH', industry='All'):
        all_feats = []
        all_metadata = []

        scope_input = scope.upper()
        if scope_input == "BOTH":
            target_keys = [constant.DB_PH_SCOPE, constant.DB_GLOBAL_SCOPE]
        elif scope_input == "GLOBAL":
            target_keys = [constant.DB_GLOBAL_SCOPE]
        else:
            target_keys = [constant.DB_PH_SCOPE]

        for skey in target_keys:
            try:
                engine = cls._get_scope(skey)
                vectors = np.array(engine.vectors_db if hasattr(
                    engine, 'vectors_db') else engine.vectors)
                metadata_list = engine.metadata_db

                print(f"📦 [SYNC] {skey}: Vectors={
                      len(vectors)}, Meta={len(metadata_list)}")

                for i in range(len(vectors)):
                    if i >= len(metadata_list):
                        break

                    meta = metadata_list[i]

                    # 1. Extraction with Key Detection
                    industry_domain = (meta.get('Category') or meta.get('category') or
                                       meta.get('industry_domain') or "General")
                    brand_name = (meta.get('Brand') or meta.get('brand') or
                                  meta.get('name') or "Unknown")
                    image_path = (meta.get('File_Path') or meta.get('file_path') or
                                  meta.get('path') or "")

                    # 2. Smart Recovery from Path
                    if (brand_name == "Unknown" or industry_domain == "General") and image_path:
                        parts = image_path.split('/')
                        if len(parts) >= 3:
                            if industry_domain == "General":
                                industry_domain = parts[-3]
                            if brand_name == "Unknown":
                                brand_name = parts[-2]

                    # 3. Cleaning & Filtering
                    brand_name = str(brand_name).replace(
                        '_', ' ').title().strip()
                    industry_domain = str(industry_domain).strip().title()

                    query = str(industry).strip().lower()
                    if query == 'all' or industry_domain.lower() == query:
                        all_feats.append(vectors[i])
                        all_metadata.append({
                            "image_path": str(image_path),
                            "metadata": {
                                "brand_name": brand_name,
                                "industry_domain": industry_domain,
                                "origin": skey
                            }
                        })
            except Exception as e:
                print(f"🔴 ERROR in {skey} scope: {str(e)}")

        return np.array(all_feats), all_metadata

    # --- PREDICTION LOGIC ---
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
    def predict(cls, cropped_rgb, scope=constant.DB_PH_SCOPE, k=5, sort_by="confidence", categories=None):
        model = cls.get_model()
        query_vector = cls._predict_with_tta(model, cropped_rgb)

        targets = []
        if scope == "BOTH":
            targets = [cls._get_scope(constant.DB_PH_SCOPE), cls._get_scope(
                constant.DB_GLOBAL_SCOPE)]
        else:
            targets = [cls._get_scope(scope)]

        raw_combined = []
        best_mask = None

        for engine in targets:
            res, mask = engine.search(
                cropped_rgb,
                query_vector=query_vector,
                user_k=100,
                categories=categories
            )
            raw_combined.extend(res)
            if best_mask is None:
                best_mask = mask

        raw_combined.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
        top_k = raw_combined[:k]

        neighbor_vectors = [item.get('vector')
                            for item in top_k if 'vector' in item]

        # Background context logic
        industry_filter = categories[0] if categories and len(
            categories) > 0 else 'All'
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
            "heatmap": best_mask
        }
