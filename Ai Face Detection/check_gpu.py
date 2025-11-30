import tensorflow as tf
import sys

def check_gpu_availability():
    """
    Checks for NVIDIA GPU availability using TensorFlow and prints the result.
    """
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version}")

    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n[SUCCESS] GPU is available and detected by TensorFlow!")
        print(f"Number of GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Name: {details.get('device_name', 'N/A')}")
                print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
            except:
                print("    Could not retrieve device details.")
    else:
        print("\n[FAILURE] No GPU detected by TensorFlow.")
        print("Please ensure the following:")
        print("1. You have an NVIDIA GPU.")
        print("2. You have installed the correct NVIDIA drivers.")
        print("3. You have installed the CUDA Toolkit and cuDNN library compatible with your TensorFlow version.")
        print("4. You have installed the GPU-enabled version of TensorFlow (e.g., 'pip install tensorflow[and-cuda]').")

if __name__ == "__main__":
    check_gpu_availability()
