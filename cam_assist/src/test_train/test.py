#!/usr/bin/env python
from ultralytics import YOLO
import torch
from pathlib import Path


script_dir = Path(__file__).parent
print('Script Directory:', script_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the saved model
model_path = script_dir / "yolov11s_trained_optimized.pt"
model = YOLO(model_path)

# Define the path to the folder containing the images
image_folder = script_dir / "images" / "test"

# Iterate over all files in the folder using Path
for image_path in image_folder.iterdir():
    print(f"Processing image: {image_path}")
    
    # Run inference using the loaded model
    results = model(str(image_path), device=device, imgsz=(480, 384))
    
    for result in results:
        result.show()
        boxes = result.boxes  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get the corners (top-left and bottom-right)
            print(f"Top-left corner: ({x1}, {y1}), Bottom-right corner: ({x2}, {y2})")