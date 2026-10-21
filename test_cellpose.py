import torch

print("=" * 50)
print("GPU STATUS")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available!")

print("\n" + "=" * 50)
print("CELLPOSE MODEL TEST")
print("=" * 50)

from cellpose import models

# Test 1: nuclei model (classic, reliable)
print("\nLoading 'nuclei' model...")
try:
    model_nuclei = models.CellposeModel(pretrained_model='nuclei', gpu=torch.cuda.is_available())
    print("  ✓ 'nuclei' model loaded successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 2: cpsam model (newer, SAM-based)
print("\nLoading 'cpsam' model...")
try:
    model_cpsam = models.CellposeModel(pretrained_model='cpsam', gpu=torch.cuda.is_available())
    print("  ✓ 'cpsam' model loaded successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)