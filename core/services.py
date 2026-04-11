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
            cls._engines[scope_key] = LogoSearchEngine(
                model.encoder, v_path, m_path,
                distance_strategy=chi_square_distance
            )
        return cls._engines[scope_key]

    @classmethod
    def predict(cls, cropped_rgb, scope=constant.DB_PH_SCOPE, k=5, sort_by="confidence", categories=None, consensus_levels=None):
        # 1. Select engines based on scope
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
            # Kukuha tayo ng more candidates para sa aggregation (top 100)
            res, mask = engine.search(
                cropped_rgb,
                user_k=100,
                sort_by=sort_by,
                categories=categories,
                consensus_levels=None,
                scope=None
            )
            raw_combined.extend(res)
            if best_mask is None:
                best_mask = mask

        # 3. Cross-Database Aggregation (Preserving all metadata keys)
        final_map = {}
        for item in raw_combined:
            # Gumamit ng .get() para flexible sa Brand/brand keys
            name = item.get('Brand') or item.get('brand') or "Unknown"

            if name not in final_map:
                final_map[name] = {
                    "scores": [item.get('confidence', 0)],
                    "stabilities": [item.get('stability', 0)],
                    "match_counts": [item.get('match_count', 0)],
                    "item": item  # Dito naka-store ang buong dictionary kasama ang File_Path
                }
            else:
                final_map[name]["scores"].append(item.get('confidence', 0))
                final_map[name]["stabilities"].append(item.get('stability', 0))
                final_map[name]["match_counts"].append(
                    item.get('match_count', 0))

                # Kung mas mataas ang confidence ng bagong item, ito ang gawing primary metadata
                if item.get('confidence', 0) > max(final_map[name]["scores"][:-1]):
                    final_map[name]["item"] = item

        # 4. Final Re-calculation & Filtering
        final_output = []
        for name, data in final_map.items():
            scores = data["scores"]
            if len(scores) > 1:
                f_max = max(scores)
                f_others_avg = sum(s for s in scores if s !=
                                   f_max) / (len(scores)-1)
                final_conf = (0.8 * f_max) + (0.2 * f_others_avg)
            else:
                final_conf = scores[0]

            total_matches = sum(data["match_counts"])
            avg_stability = sum(data["stabilities"]) / len(data["stabilities"])

            # Sync ang calculated values sa original item
            item = data["item"]
            item["confidence"] = round(final_conf, 2)
            item["stability"] = round(avg_stability, 2)
            item["match_count"] = total_matches

            # Determine final consensus string
            if total_matches >= 15:
                res_consensus = "Strong"
            elif total_matches >= 5:
                res_consensus = "Moderate"
            else:
                res_consensus = "Weak"

            item["consensus"] = res_consensus
            item["is_stable"] = total_matches >= 5

            if consensus_levels and res_consensus not in consensus_levels:
                continue

            final_output.append(item)

        # 5. Global Sort based on user preference
        sort_map = {
            "confidence": "confidence",
            "stability": "stability",
            "matches": "match_count"
        }
        target_key = sort_map.get(sort_by, "confidence")
        final_output.sort(key=lambda x: x.get(target_key, 0), reverse=True)

        return final_output[:k], best_mask

    @classmethod
    def warm_up(cls):
        """Pre-loads both databases at server start"""
        cls._get_scope(constant.DB_PH_SCOPE)
        cls._get_scope(constant.DB_GLOBAL_SCOPE)
        print("✅ [WARMUP] System fully loaded.")
