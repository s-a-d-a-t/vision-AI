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

def interactive_get_url():
    print("=" * 60)
    print("           Vision AI - Phone Camera Setup Guide")
    print("=" * 60)
    print("To use your phone's camera, you need a mobile app that streams")
    print("video over Wi-Fi. Common apps include:")
    print("  1. IP Webcam (Android) - typically streams on port 8080")
    print("  2. DroidCam (Android/iOS) - typically streams on port 4747")
    print("\nEnsure your phone and computer are connected to the SAME Wi-Fi network.")
    print("-" * 60)

    url = input("Enter stream URL (or press Enter to configure via IP/Port): ").strip()
    if url:
        return url

    ip = input("Enter phone IP address (e.g., 192.168.1.50): ").strip()
    if not ip:
        print("Error: IP address is required if direct URL is not provided.")
        sys.exit(1)

    port = input("Enter port (default 8080): ").strip()
    if not port:
        port = "8080"

    # Default to standard IP Webcam /video endpoint.
    # DroidCam usually uses /video or /mjpegfeed
    print("\nChoose camera stream type:")
    print("  1. IP Webcam (/video)")
    print("  2. DroidCam (/video or /mjpegfeed)")
    print("  3. Custom URL path")
    choice = input("Select option (1-3) [default: 1]: ").strip()

    if choice == "2":
        return f"http://{ip}:{port}/video"
    elif choice == "3":
        path = input("Enter path (e.g., /mjpegfeed): ").strip()
        if not path.startswith("/"):
            path = "/" + path
        return f"http://{ip}:{port}{path}"
    else:
        return f"http://{ip}:{port}/video"

