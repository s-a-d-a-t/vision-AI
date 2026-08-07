# Vision AI — Real-time Object Detection with YOLOv11

A lightweight and efficient computer vision project leveraging **Ultralytics YOLOv11** (Nano) and **OpenCV** to perform real-time object detection via webcam and static image analysis.

---

## 📌 What is this Project?

This project is a starting point for building Vision AI applications. It contains four main scripts:
1. **`camera.py`**: A diagnostic script that verifies your webcam/camera connection, reads frames, prints their dimensions, and displays the raw video stream.
2. **`yolo_camera.py`**: A real-time object detection pipeline that captures frames from your webcam, runs them through the YOLOv11 Nano model, annotates the frames with bounding boxes and labels (e.g., person, laptop, cup), and displays the live annotated video stream.
3. **`yolo_results.py`**: A script similar to `yolo_camera.py` but with console output enabled. It detects objects in real-time, displays the annotated webcam stream, and prints each detected object's class name and confidence score to the terminal.
4. **`yolo_image.py`**: A script that runs YOLOv11 inference on static images (e.g., `images/test.jpg`) and opens a window displaying the detected objects.

The project uses the lightweight **`yolo11n.pt`** (YOLOv11 Nano) model weights, which are optimized for fast inference on standard CPUs and edge devices.

---

## 💡 Why Do I Need It?

Computer Vision and Object Detection are core pillars of modern AI. You might need this project to:
* **Learn and Prototype**: Understand how to integrate deep learning models (YOLO) with live video feeds in Python.
* **Build Smarter Systems**: Lay the foundation for surveillance systems, custom object counters, motion detectors, or smart robotics.
* **Optimized Performance**: YOLOv11 is state-of-the-art in both accuracy and speed. Using the Nano version allows you to achieve real-time detection without requiring expensive GPU hardware.

---

## 📂 Project Structure

```bash
vision-AI/
├── images/
│   └── test.jpg           # Sample image for static testing
├── venv/                  # Python virtual environment (local dependencies)
├── camera.py              # Camera connection tester
├── yolo_camera.py         # Real-time webcam object detection script
├── yolo_results.py        # Webcam object detection with terminal logs
├── yolo_image.py          # Static image object detection script
├── yolo11n.pt             # YOLOv11 Nano pre-trained model weights (PyTorch)
├── requirements.txt       # Project dependencies and libraries
└── README.md              # Project documentation (this file)
```

---

## 🛠️ How to Test it on Your Local Computer

Follow these steps to set up and run the project locally.

### Prerequisites
* **Python 3.12** or higher (This project was set up with Python 3.12.3).
* A working webcam (built-in or USB-connected).

### Step 1: Navigate to the Project Directory
Open your terminal and ensure you are in the project folder:
```bash
cd /home/sadat/Documents/vision-AI
```

### Step 2: Activate the Virtual Environment
Activate the pre-configured Python virtual environment:
```bash
source venv/bin/activate
```
*Your terminal prompt should now show `(venv)` at the beginning.*

### Step 3: Install Dependencies
If you ever need to reinstall or set up this environment from scratch:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Scripts

#### 1. Test Your Camera Connection
To verify that Python can access your webcam:
```bash
python camera.py
```
* **Expectation**: A window titled "Vision AI" will open showing your webcam feed. In the terminal, you will see `Frame shape: (height, width, channels)` printed continuously.
* **Exit**: Press **`q`** on your keyboard while focusing the video window to close it.

#### 2. Run Real-time Object Detection
To start detecting objects in real-time:
```bash
python yolo_camera.py
```
* **Expectation**: A window titled "Vision AI - YOLO Detection" will display your webcam feed, highlighting recognized objects (like yourself, your chair, cell phone, books, etc.) with colorful bounding boxes, class names, and confidence scores.
* **Exit**: Press **`q`** on your keyboard while focusing the video window to close it.

#### 3. Run Object Detection with Terminal Output
To detect objects in real-time and view confidence scores printed in the console:
```bash
python yolo_results.py
```
* **Expectation**: A webcam window will display the annotated feed, and the terminal will continuously print identified classes and their confidence percentages (e.g. `person 94.2 %`).
* **Exit**: Press **`q`** on your keyboard while focusing the video window to close it.

#### 4. Test Static Image Inference
To run detection on a static image:
```bash
python yolo_image.py
```
* **Expectation**: The script will run the model on `images/test.jpg` and display the annotated result in an image viewer.
* **Exit**: Close the image window or press any key to exit.

---

## ⚠️ Troubleshooting & Customizations

### 1. Camera Access Issues
If you see the error message **`Could not access camera`**:
* Make sure your webcam is not being used by another application (e.g., Zoom, Teams, Discord).
* If you have multiple webcams connected, you may need to change the camera index in `camera.py` and `yolo_camera.py`:
  ```python
  # Change 0 to 1, 2, etc., if the default camera index is incorrect
  camera = cv2.VideoCapture(1)
  ```

### 2. Speeding up Inference (GPU Support)
By default, the scripts run on your CPU. If you have an NVIDIA GPU and want to run YOLO with CUDA acceleration:
1. Ensure CUDA is installed on your Linux system.
2. PyTorch will automatically detect CUDA if installed correctly. You can confirm this inside your activated virtual environment:
   ```bash
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
   ```
3. YOLO will automatically utilize the GPU if `torch.cuda.is_available()` returns `True`.

### 3. Using Different YOLO Models
If you want higher accuracy (at the expense of frame rate), you can download larger models (e.g., `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`). Just change the model instantiation in your script:
```python
model = YOLO("yolo11s.pt")  # Will automatically download the model on its first run
```
