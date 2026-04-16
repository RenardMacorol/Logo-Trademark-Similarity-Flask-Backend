import tensorflow as tf
from tensorflow.keras import layers, Model


class WonksNetModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.encoder = self._init_build_architecture()
        self._load_weights()

    @property
    def model(self): return self.encoder

    @property
    def input(self): return self.encoder.input

    @property
    def output(self): return self.encoder.output

    def get_layer(self, name):
        """Proxy to allow ForensicEngine to access ResNet50 layers directly."""
        return self.encoder.get_layer(name)

    def _init_build_architecture(self, input_shape=(224, 224, 3)):
        base = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=input_shape)
        shared = layers.BatchNormalization(name="backbone_norm")(base.output)

        # Segmentation Head
        x = shared
        for filters in [512, 256, 128, 64]:
            x = layers.Conv2DTranspose(filters, (3, 3), strides=(
                2, 2), padding='same', activation='relu')(x)
        seg_mask = layers.Conv2DTranspose(1, (3, 3), strides=(
            2, 2), padding='same', activation='sigmoid', name='segmentation_output')(x)

        # Attention Embedding
        small_mask = layers.Resizing(7, 7, name="attention_resize")(seg_mask)
        weighted = layers.Multiply(
            name="spatial_attention")([shared, small_mask])
        pooled = layers.GlobalAveragePooling2D()(weighted)
        pooled = layers.BatchNormalization()(pooled)
        emb = layers.Dense(512, activation='softplus',
                           name="embedding_dense")(pooled)
        emb = layers.Lambda(lambda x: tf.math.l2_normalize(
            x, axis=1), name='embedding_output')(emb)

        return Model(inputs=base.input, outputs=[emb, seg_mask])

    def _load_weights(self):
        try:
            self.encoder.load_weights(self.model_path)
        except:
            source = tf.keras.models.load_model(self.model_path, compile=False)
            for layer in self.encoder.layers:
                try:
                    self.encoder.get_layer(layer.name).set_weights(
                        source.get_layer(layer.name).get_weights())
                except:
                    continue

    def predict(self, input_data, verbose=0):
        return self.encoder.predict(input_data, verbose=verbose)
