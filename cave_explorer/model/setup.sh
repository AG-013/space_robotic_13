# This script sets up yolov5 for use to train a model
# This tool can help create yolo anotations: https://www.makesense.ai/

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install opencv-python numpy
pip install torch torchvision
pip install -r requirements.txt

