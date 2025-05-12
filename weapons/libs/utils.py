import torch

def get_gpu_device():
    """
    Get the GPU device.
    
    if NVIDIA GPU is available,  return "gpu"
    if MPS (Apple Silicon GPU) is available, return "mps"
    otherwise, return "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        return "cuda"
    else:
        return "cpu"