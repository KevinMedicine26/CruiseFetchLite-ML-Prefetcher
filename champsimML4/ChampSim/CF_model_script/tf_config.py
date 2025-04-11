import tensorflow as tf

def configure_tensorflow():
    """Configure TensorFlow to use GPU optimally"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"GPU devices available: {len(physical_devices)}")
            print(f"Using GPU: {physical_devices}")
            
            # Enable XLA (Accelerated Linear Algebra) compilation
            tf.config.optimizer.set_jit(True)
            
            # Enable mixed precision for faster computation on newer GPUs
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("GPU optimizations enabled: XLA and mixed precision")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found, using CPU only")
