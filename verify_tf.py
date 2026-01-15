import tensorflow as tf
import sys

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU detected: {gpu_devices}")
else:
    print("No GPU detected.")
