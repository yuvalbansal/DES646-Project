# Sign Language to Text Converter

This project uses a deep learning model (MobileNetV3) to recognize American Sign Language (ASL) gestures from a live camera feed and convert them into text in real time.  
It also supports MediaPipe-based hand detection and tracking for improved accuracy.

---

## 🧩 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yuvalbansal/DES646-Project.git
cd DES646-Project
```

---

### 2. Create a Virtual Environment

```bash
python -m venv myenv
```

---

### 3. Activate the Virtual Environment

**On Windows:**

```bash
myenv\Scripts\activate
```

**On Ubuntu/Linux/macOS:**

```bash
source myenv/bin/activate
```

---

### 4. Install Dependencies

Make sure you are inside the activated virtual environment, then run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Program

### 5. Start Real-Time ASL Detection

```bash
python run_camera_inference.py
```

If you are using multiple cameras (e.g., an external USB camera),  
you can change the camera source by editing the line inside `run_camera_inference.py`:

```python
CAMERA_INDEX = 0  # change this to 0, 1, or 2 depending on your setup
```

---

## 🎮 Controls

| Key   | Action                            |
| ----- | --------------------------------- |
| **q** | Quit the program                  |
| **c** | Clear all committed text          |
| **b** | Backspace (delete last character) |

---

## 🧠 Notes

- Make sure your trained model and metadata are located in `runs/checkpoints/`:
  - `best.pt`
  - `metadata.json`
- The model currently supports ASL alphabet and digits, and can be extended by training on more gestures.
- If MediaPipe is installed, it will automatically be used for hand landmark detection.

---
