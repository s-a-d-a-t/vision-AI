from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("images/test.jpg")

for result in results:
    result.show()