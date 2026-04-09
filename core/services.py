import constant
from core.model import WonksNetModel
from core.search_engine import LogoSearchEngine
from core.distance_metric import chi_square_distance


class WonksNetService:
    _model_instance = None
    _engines = {}

    @classmethod
    def get_model(cls):
        """Singleton Accessor for the AI Model"""
        if cls._model_instance is None:
            print("🧠 [SINGLETON] Loading Model...")
            cls._model_instance = WonksNetModel(constant.MODEL_FILE)
        return cls._model_instance

    @classmethod
    def _get_scope(cls, scope_key):
        """Internal helper to manage specific dataset instances"""
        if scope_key not in cls._engines:
            paths = {
                constant.DB_PH_SCOPE: (constant.PH_VECTORS_PATH, constant.PH_METADATA_PATH),
                constant.DB_GLOBAL_SCOPE: (
                    constant.GLOBAL_VECTORS_PATH, constant.GLOBAL_METADATA_PATH)
            }

            v_path, m_path = paths[scope_key]
            model = cls.get_model()

            print(f"💾 [INIT] Loading {scope_key} dataset into RAM...")
            cls._engines[scope_key] = LogoSearchEngine(
                model.encoder, v_path, m_path,
                distance_strategy=chi_square_distance
            )
        return cls._engines[scope_key]

    @classmethod
    def predict(cls, cropped_rgb, scope=constant.DB_PH_SCOPE, k=5):
        """
        The Master Predict Function:
        1. Identifies which engines to use.
        2. Gathers results.
        3. Collapses duplicates if searching across BOTH databases.
        """
        # Determine engines to use
        if scope == constant.DB_BOTH_SCOPE:
            targets = [cls._get_scope(constant.DB_PH_SCOPE), cls._get_scope(
                constant.DB_GLOBAL_SCOPE)]
        else:
            # Check if scope exists, fallback to PH
            targets = [cls._get_scope(scope if scope in [
                                      constant.DB_PH_SCOPE, constant.DB_GLOBAL_SCOPE] else constant.DB_PH_SCOPE)]

        raw_results = []
        best_mask = None
        max_conf = -1

        # 1. Gather results from all target engines
        for engine in targets:
            results, mask = engine.search(cropped_rgb, k=k)
            raw_results.extend(results)

            # Keep the mask from the engine that found the most confident match
            if results and results[0]['confidence'] > max_conf:
                max_conf = results[0]['confidence']
                best_mask = mask

        # 2. Global Aggregation (Collapse duplicates across databases)
        # If Jollibee is in PH and Global, we only want the best one.
        final_map = {}
        for item in raw_results:
            brand = item['brand']
            if brand not in final_map or item['confidence'] > final_map[brand]['confidence']:
                final_map[brand] = item

        # 3. Final Sort & Slice
        final_list = sorted(final_map.values(),
                            key=lambda x: x['confidence'], reverse=True)

        return final_list[:k], best_mask

    @classmethod
    def warm_up(cls):
        """Pre-loads both databases at server start"""
        cls._get_scope(constant.DB_PH_SCOPE)
        cls._get_scope(constant.DB_GLOBAL_SCOPE)
        print("✅ [WARMUP] System fully loaded.")
