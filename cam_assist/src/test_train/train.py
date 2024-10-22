import torch
from ultralytics import YOLO
from pathlib import Path


script_dir = Path(__file__).parent


# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a pre-trained YOLOv8 model (small version)
model = YOLO("yolo11s.pt").to(device)

# Display model information (optional)
model.info()

# Define custom training settings
custom_train_settings = {
    'epochs': 75,                # Increase the number of epochs
    'imgsz': (480, 384),         # Increase the image size for better accuracy
    'batch': 16,                 # Increase batch size if memory allows
    'device': device,
    'optimizer': 'AdamW',        # Switch to AdamW optimizer
    'lr0': 0.0025,               # Initial learning rate
    'lrf': 0.01,                 # Final learning rate multiplier
    'momentum': 0.937,           # Momentum parameter
    'weight_decay': 0.0005,      # Weight decay for regularization
    'warmup_epochs': 3.0,        # Warm-up epochs
    'warmup_momentum': 0.8,      # Momentum during warm-up
    'warmup_bias_lr': 0.1,       # Bias learning rate during warm-up
    'cos_lr': True,              # Enable cosine learning rate scheduler
    'degrees': 0.0,              # Rotation degrees
    'translate': 0.1,            # Translation augmentation
    'scale': 0.5,                # Scaling augmentation
    'shear': 0.0,                # Shear augmentation
    'perspective': 0.0,          # Perspective augmentation
    'flipud': 0.0,               # Vertical flip
    'fliplr': 0.5,               # Horizontal flip
    'mosaic': 1.0,               # Mosaic augmentation
    'mixup': 0.1,                # Mixup augmentation
    'copy_paste': 0.0,           # Copy-paste augmentation
    'amp': True                  # Enable mixed precision training
}

# Train the model using the custom settings
results = model.train(data="config.yaml", **custom_train_settings)

# Save the trained model weights
model.save("yolov11s_trained_optimized.pt")