#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path


script_dir = Path(__file__).parent
print('Script Directory:', script_dir)


class YoloImageProcessor:
    
    def __init__(self, confidence_threshold=0.5):
        # Initialize YOLO model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(script_dir / 'test_train' / "yolov11s_trained_optimized.pt")
        rospy.loginfo(f"Using device: {self.device}")

        # Set confidence threshold
        self.confidence_threshold = confidence_threshold

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create a subscriber to the camera topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # Create a publisher for the processed image
        self.image_pub = rospy.Publisher("/yolo_processed_image", Image, queue_size=5)


    def image_callback(self, msg):
        try:
            # Convert the ROS image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Process the image using YOLO
        results = self.model(cv_image, device=self.device, imgsz=(480, 384))

        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0].item()  # Confidence score

                # Only process boxes with confidence above the threshold
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = box.cls[0].item()  # Class label

                    # Draw rectangle and label
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the modified CV2 image back to a ROS Image message
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Publish the processed image
        self.image_pub.publish(processed_msg)
        rospy.loginfo("Published processed image")


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('yolo_image_processor', anonymous=True)
    yolo_processor = YoloImageProcessor(confidence_threshold=0.5)  # Set your desired threshold
    yolo_processor.run()