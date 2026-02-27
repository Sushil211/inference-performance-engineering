import torch
import sys
import os

def capture_environment(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        f.write("=== Environment Metadata ===\n")
        
        # Python & PyTorch
        f.write(f"Python Version: {sys.version.split()[0]}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"cuDNN Version: {torch.backends.cudnn.version()}\n")
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            
            # Convert bytes to GB
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            f.write(f"Total VRAM: {total_vram:.2f} GB\n")
        else:
            f.write("GPU Name: None (Running on CPU/MPS)\n")

    print(f"✅ Environment metadata saved to {save_path}")

if __name__ == "__main__":
    # Test it locally first
    capture_environment("results/01_batching_cliff/environment.txt")