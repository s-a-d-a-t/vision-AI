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
    print("Select your mobile streaming app:")
    print("  1. DroidCam (default)")
    print("  2. IP Webcam")
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()

    ip = input("\nEnter the IP address shown on your phone (e.g., 10.0.2.13): ").strip()
    if not ip:
        print("[ERROR] IP address is required.")
        sys.exit(1)

    # Clean the input in case the user accidentally pasted a full URL or port
    if "://" in ip:
        ip = ip.split("://")[1]
    if ":" in ip:
        ip = ip.split(":")[0]
    if "/" in ip:
        ip = ip.split("/")[0]

    # Build the exact stream URL based on the selected app
    if choice == "2":
        # IP Webcam uses port 8080 and /video path
        return f"http://{ip}:8080/video"
    else:
        # DroidCam uses port 4747 and /video path
        return f"http://{ip}:4747/video"



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


