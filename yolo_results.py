import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                class_name,
                round(confidence * 100, 2),
                "%"
            )

    annotated_frame = results[0].plot()

    cv2.imshow("Vision AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()