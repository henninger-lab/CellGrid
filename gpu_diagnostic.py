import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Cuda version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_name(0)}")