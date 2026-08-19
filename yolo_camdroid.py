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

def main():
    args = get_args()

    # Determine connection URL
    if args.url:
        stream_url = args.url
    elif args.ip:
        port = args.port if args.port else "8080"
        stream_url = f"http://{args.ip}:{port}/video"
    else:
        stream_url = interactive_get_url()

    print(f"\n[INFO] Connecting to phone camera stream at: {stream_url}")
    print("[INFO] Loading YOLOv11 Nano model...")
    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(stream_url)

    # Check if stream opened successfully
    if not camera.isOpened():
        print("\n[ERROR] Could not connect to camera stream.")
        print("Troubleshooting steps:")
        print("  1. Double check that the IP address and Port are correct.")
        print("  2. Make sure the streaming app is actively running on your phone.")
        print("  3. Ensure both phone and computer are on the exact same Wi-Fi network.")
        print("  4. Verify if you can access the stream in a browser on your computer.")
        sys.exit(1)

    print("[INFO] Connection established successfully! Press 'q' on the video window to quit.\n")

    while True:
        success, frame = camera.read()

        if not success:
            print("\n[ERROR] Lost connection to camera stream.")
            break

        results = model(frame)

        # Print detection results to terminal
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"{class_name} {round(confidence * 100, 2)}%")

        annotated_frame = results[0].plot()

        # Display the annotated feed
        cv2.imshow("Vision AI - Phone Camera Detection", annotated_frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")

if __name__ == "__main__":
    main()


