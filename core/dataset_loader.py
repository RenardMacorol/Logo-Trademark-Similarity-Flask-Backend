from core.model import WonksNetModel
from core.search_engine import LogoSearchEngine
from core.distance_metric import chi_square_distance
# assuming you have a model loader class
from core.model_loader import ModelLoader
import constant


class DatasetLoader:
    _engines = {}

    @classmethod
    def get_scope(cls, scope_key):
        if scope_key not in cls._engines:
            paths = {
                constant.DB_PH_SCOPE: (constant.PH_VECTORS_PATH, constant.PH_METADATA_PATH),
                constant.DB_GLOBAL_SCOPE: (
                    constant.GLOBAL_VECTORS_PATH, constant.GLOBAL_METADATA_PATH)
            }

            if scope_key not in paths:
                raise ValueError(f"Invalid scope key:  {scope_key}")

            v_path, m_path = paths[scope_key]
            model = ModelLoader.get_model()   # Load the model using ModelLoader
            print(f"💾 [INIT] Loading {scope_key} dataset into RAM...")
            cls._engines[scope_key] = LogoSearchEngine(
                model.encoder, v_path, m_path,
                distance_strategy=chi_square_distance
            )
        return cls._engines[scope_key]
