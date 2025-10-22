# Edit these paths to your model locations
HOI_MODEL_PATH = "src/models/hoi_fractional_not_finetuned.pt"   # does person + hand + object
POSE_MODEL_PATH = "src/models/yolo11m-pose.pt"

# Device configuration - will auto-detect CUDA if available
def get_device():
    """Auto-detect the best available device (CUDA > CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            print(f"CUDA detected! Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            return device
        else:
            print("CUDA not available, using CPU")
            return "cpu"
    except ImportError:
        print("PyTorch not available, defaulting to CPU")
        return "cpu"
    except Exception as e:
        print(f"Error detecting device: {e}, defaulting to CPU")
        return "cpu"

DEVICE = get_device()

# Pose estimation parameters
POSE_CONF_THRESHOLD = 0.3  # Minimum confidence for pose keypoints

# Pocket region parameters
POCKET_PADDING_PX = 25  # Padding in pixels to expand pocket regions
