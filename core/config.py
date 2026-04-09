import tensorflow as tf
import constant


def initialize_gpu(memory_limit=constant.GPU_MEMORY_LIMIT):
    """
    Configures TensorFlow to respect Arch Linux / Hardware constraints.
    Prevents TF from allocating 100% of VRAM by default.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Setting a hard limit to leave room for your Arch desktop/DE
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=memory_limit)]
            )
            print(f"✅ GPU: {memory_limit}MB Logical Limit Configured")
        except RuntimeError as e:
            # This happens if GPU is initialized before the config is set
            print(f"⚠️ GPU Configuration Error: {e}")
    else:
        print("💡 Hardware: No GPU found, falling back to CPU mode.")
