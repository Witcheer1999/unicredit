import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf

print(f"Keras Version: {keras.__version__}")
print(f"TensorFlow Version: {tf.__version__}")

try:
    from tensorflow.keras import layers
    print("Successfully imported tensorflow.keras.layers")
except ImportError as e:
    print(f"Error importing tensorflow.keras.layers: {e}")

try:
    model = keras.Sequential([keras.layers.Dense(1)])
    print("Successfully created a Keras model")
except Exception as e:
    print(f"Error creating Keras model: {e}")
