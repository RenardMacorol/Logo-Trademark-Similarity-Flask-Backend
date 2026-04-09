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

            if scope_key not in paths:
                raise ValueError(f"Invalid scope key: {scope_key}")

            v_path, m_path = paths[scope_key]
            model = cls.get_model()

            print(f"💾 [INIT] Loading {scope_key} dataset into RAM...")
            # We use model.encoder because the search engine needs the vector generator
            cls._engines[scope_key] = LogoSearchEngine(
                model.encoder, v_path, m_path,
                distance_strategy=chi_square_distance
            )
        return cls._engines[scope_key]

    @classmethod
    def predict(cls, cropped_rgb, scope=constant.DB_PH_SCOPE, k=5):
        # 1. Select engines based on scope using the internal _get_scope helper
        targets = []
        if scope == constant.DB_PH_SCOPE:
            targets = [cls._get_scope(constant.DB_PH_SCOPE)]
        elif scope == constant.DB_GLOBAL_SCOPE:
            targets = [cls._get_scope(constant.DB_GLOBAL_SCOPE)]
        else:
            # BOTH scope
            targets = [
                cls._get_scope(constant.DB_PH_SCOPE),
                cls._get_scope(constant.DB_GLOBAL_SCOPE)
            ]

        raw_combined = []
        best_mask = None

        # 2. Collect results from engines
        for engine in targets:
            res, mask = engine.search(cropped_rgb, user_k=k)
            raw_combined.extend(res)
            if best_mask is None:
                best_mask = mask

        # 3. Cross-Database Aggregation (Merging PH + Global info)
        final_map = {}
        for item in raw_combined:
            name = item['brand']
            if name not in final_map:
                final_map[name] = {
                    "scores": [item['confidence']],
                    "stabilities": [item['stability']],
                    "match_counts": [item['match_count']],
                    "item": item
                }
            else:
                final_map[name]["scores"].append(item['confidence'])
                final_map[name]["stabilities"].append(item['stability'])
                final_map[name]["match_counts"].append(item['match_count'])

        # 4. Final Re-calculation for Merged Results
        final_output = []
        for name, data in final_map.items():
            # Weighted Confidence Merge
            scores = data["scores"]
            if len(scores) > 1:
                f_max = max(scores)
                f_others_avg = sum(s for s in scores if s !=
                                   f_max) / (len(scores)-1)
                final_conf = (0.8 * f_max) + (0.2 * f_others_avg)
            else:
                final_conf = scores[0]

            # Ethical Metrics Merge (Summing counts and averaging stability)
            total_matches = sum(data["match_counts"])
            avg_stability = sum(data["stabilities"]) / len(data["stabilities"])

            # Create the final unified object
            item = data["item"]
            item["confidence"] = round(final_conf, 2)
            item["stability"] = round(avg_stability, 2)
            item["match_count"] = total_matches

            # Re-verify stability across both DBs (Ethical Logic)
            item["is_stable"] = total_matches >= 5

            # Determine final consensus string
            if total_matches >= 15:
                item["consensus"] = "Strong"
            elif total_matches >= 5:
                item["consensus"] = "Moderate"
            else:
                item["consensus"] = "Weak"

            final_output.append(item)

        # 5. Global Sort
        final_output.sort(key=lambda x: x['confidence'], reverse=True)
        return final_output[:k], best_mask

    @classmethod
    def warm_up(cls):
        """Pre-loads both databases at server start"""
        cls._get_scope(constant.DB_PH_SCOPE)
        cls._get_scope(constant.DB_GLOBAL_SCOPE)
        print("✅ [WARMUP] System fully loaded.")
