import argparse
import sys
import cv2
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 object detection using an external phone camera stream."
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Direct URL to the camera stream (e.g., http://192.168.1.50:8080/video)",
    )
    parser.add_argument(
        "--ip",
        type=str,
        help="IP address of the phone (e.g., 192.168.1.50)",
    )
    parser.add_argument(
        "--port",
        type=str,
        help="Port of the streaming service (e.g., 8080 for IP Webcam, 4747 for DroidCam)",
    )
    return parser.parse_args()
