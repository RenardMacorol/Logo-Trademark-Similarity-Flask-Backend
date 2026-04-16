from core.model import WonksNetModel
import constant


class ModelLoader:
    _model_instance = None

    @classmethod
    def get_model(cls):
        if cls._model_instance is None:
            print("🧠 [SINGLETON] Loading Model...")
            cls._model_instance = WonksNetModel(constant.MODEL_FILE)
        return cls._model_instance
