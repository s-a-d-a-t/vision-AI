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
    print("Select your connection method:")
    print("  1. DroidCam PC Client (Local Virtual Webcam - e.g. /dev/video4) [default]")
    print("  2. DroidCam Direct IP (Direct Wi-Fi connection to phone)")
    print("  3. IP Webcam Direct IP (Direct Wi-Fi connection to phone)")
    choice = input("Enter choice (1-3) [default: 1]: ").strip()

    if choice == "2" or choice == "3":
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

        if choice == "3":
            # IP Webcam uses port 8080 and /video path
            return f"http://{ip}:8080/video"
        else:
            # DroidCam uses port 4747 and /video path
            return f"http://{ip}:4747/video"
    else:
        # Local device index (DroidCam PC Client)
        print("\nNote: Your recently active DroidCam camera index is likely 4.")
        device_idx = input("Enter camera device index [default: 4]: ").strip()
        if not device_idx:
            device_idx = "4"
        return device_idx


def main():
    args = get_args()

    # Determine connection URL or device index
    if args.url:
        stream_source = args.url
    elif args.ip:
        port = args.port if args.port else "8080"
        stream_source = f"http://{args.ip}:{port}/video"
    else:
        stream_source = interactive_get_url()

    # If stream_source can be parsed as an integer, convert it
    try:
        source = int(stream_source)
        print(f"\n[INFO] Connecting to local camera device index: {source}")
    except ValueError:
        source = stream_source
        print(f"\n[INFO] Connecting to phone camera stream at: {source}")

    print("[INFO] Loading YOLOv11 Nano model...")
    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(source)

    # Check if stream opened successfully
    if not camera.isOpened():
        print(f"\n[ERROR] Could not connect to camera source: {source}")
        print("Troubleshooting steps:")
        if isinstance(source, int):
            print("  1. Make sure DroidCam client app on your PC is open and streaming.")
            print("  2. Try another device index (e.g. 0, 1, 2, or 3) if 4 is not correct.")
        else:
            print("  1. Double check that the IP address and Port are correct.")
            print("  2. Make sure the streaming app is actively running on your phone.")
            print("  3. Ensure both phone and computer are on the exact same Wi-Fi network.")
            print("  4. Verify if you can access the stream in a browser on your computer.")
        sys.exit(1)

    print("[INFO] Connection established successfully! Press 'q' on the video window to quit.\n")

    while True:
        success, frame = camera.read()

        if not success:
            print("\n[ERROR] Lost connection to camera source.")
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


