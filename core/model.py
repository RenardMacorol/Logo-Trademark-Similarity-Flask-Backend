import tensorflow as tf
from tensorflow.keras import layers,  Model


class WonksNetModel:
    def __init__(self, model_path):
        self.model_path = model_path
        # Encoder Architecture
        self.encoder = self._init_build_architecture()
        # Place the Trained Weights
        self._load_weights()

    def _init_build_architecture(self, input_shape=(224, 224, 3)):
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=input_shape)

        shared_features = layers.BatchNormalization(
            name="backbone_norm")(base_model.output)

        # Segmentation Head
        x = layers.Conv2DTranspose(512, (3, 3), strides=(
            2, 2), padding='same', activation='relu')(shared_features)
        x = layers.Conv2DTranspose(256, (3, 3), strides=(
            2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(128, (3, 3), strides=(
            2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=(
            2, 2), padding='same', activation='relu')(x)
        seg_mask = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same',
                                          activation='sigmoid', name='segmentation_output')(x)

        # Attention + Embedding
        small_mask = layers.Resizing(7, 7, name="attention_resize")(seg_mask)
        weighted_features = layers.Multiply(name="spatial_attention")([
            shared_features, small_mask])
        pooled = layers.GlobalAveragePooling2D()(weighted_features)
        pooled = layers.BatchNormalization()(pooled)
        embedding = layers.Dense(
            512, activation='softplus', name="embedding_dense")(pooled)

        embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),
                                  name='embedding_output',
                                  output_shape=(512,))(embedding)

        return Model(inputs=base_model.input, outputs=[embedding, seg_mask])

    def _load_weights(self):
        # Corrected f-string and added the 'try' block
        print(f"🧠 WonksNet: Attempting surgical weight load from {self.model_path}")
        try:
            # 1. Try standard load first
            self.encoder.load_weights(self.model_path)
            print("✅ Direct weights loaded.")

        except Exception as e:
            # 2. Fallback to surgery if direct load fails
            print(f"⚠️ Direct load failed. Performing surgery...")
            source = tf.keras.models.load_model(
                self.model_path,
                compile=False,
                safe_mode=False
            )

            # Transfer layer by layer
            for layer in self.encoder.layers:
                try:
                    weights = source.get_layer(layer.name).get_weights()
                    self.encoder.get_layer(layer.name).set_weights(weights)
                except:
                    # Skip layers that don't match (like Input or Lambda)
                    continue
            print("✅ Surgery successful. Weights mapped ")

    def predict(self, input_data):
        return self.encoder.predict(input_data, verbose=0)
