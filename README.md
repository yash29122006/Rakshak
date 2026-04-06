# SafeRoad AI — Traffic Accident Detection Prototype  
### *A minute saved now can save a life*

**SafeRoad AI** is a real-time, local accident detection system designed as part of **Project Rakshak**. It utilizes computer vision (YOLOv8) combined with multi-stage rule engines and AI verification (Google Gemini) to detect, confirm, and alert users about traffic accidents instantly via Telegram.

## Features

- **Live Webcam/Feed Processing:** Captures and processes continuous video frames.
- **YOLOv8 Object Detection:** Identifies cars, motorcycles, bicycles, buses, trucks, and persons using the highly efficient `yolov8n` neural network.
- **Rule-Based Accident Logic:**
  - **Rule 1 (Collision):** High Intersection-over-Union (IoU) overlap between two vehicles.
  - **Rule 2 (Pedestrian Impact):** IoU overlap between a pedestrian and a vehicle.
  - **Rule 3 (Sudden Abnormal Motion):** Pixel displacement tracking across frames to identify crash speeds.
- **Multi-Frame Confirmation:** Prevents false positives by ensuring triggers last for a sustained number of frames.
- **AI Dual-Verification:** Passes suspected accident frames to *Google Gemini 2.0 Flash Vision* to confirm actual visual damage or impact, dramatically reducing false alarms.
- **Telegram Alerting:**
  - Instantly sends an annotated **snapshot photo**.
  - Asynchronously compiles and sends a **~10-second video clip** (5s before, 5s after the event).

## 📹 Demonstrations

### 1. Python Implementation & Live Feed
Watch the system analyze the live feed and trigger the detection logic upon identifying an accident.

https://github.com/user-attachments/assets/97c56651-21bc-4f6f-9a37-e2d4fcab900d
<br>

### 2. Telegram Alert Received
See how the system dispatches the snapshot and video clip immediately to the specified Telegram Chat.

https://github.com/user-attachments/assets/9efd4a33-1112-495e-99cd-c9cf99e390cf
<br>

---

## 📁 File Structure

```text
accident-detector/
├── main.py               # Main orchestration, webcam loop, video buffer & background threads
├── detector.py           # YOLOv8 inference, IoU math, AI verification (Gemini), & drawing utilities
├── alert.py              # Telegram API integration for photos and multipart video uploads
├── config.py             # Centralised configurations (API keys, thresholds, framerates)
├── requirements.txt      # Python package dependencies
├── clips/                # Auto-generated directory for ~10s accident video outputs
└── venv/                 # Python Virtual Environment
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A Telegram Bot Token + Chat ID
- A Google Gemini API Key (Free Tier)

### 1. Clone the Repository
```bash
git clone https://github.com/yash29122006/Rakshak.git
cd Rakshak/accident-detector
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys
Edit `config.py` and populate your credentials:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
GEMINI_API_KEY = "your_gemini_vision_key"
```

### 4. Run the Application
```bash
python main.py
```
*(Press `Q` on the OpenCV window to cleanly exit the application)*

---

## ⚙️ How the Logic Pipeline Works

1. **Capture:** Reads the webcam (or video stream) into a rolling buffer.
2. **Detect:** YOLOv8 extracts bounding boxes for relevant classes.
3. **Analyze:** Custom Python heuristics check IoU overlap and rapid displacement.
4. **Confirm (Local):** The rule must persist for `ACCIDENT_CONFIRM_FRAMES`.
5. **Verify (Cloud):** The annotated frame is sent to Google Gemini Vision. If Gemini spots an accident, it proceeds.
6. **Alert (Background Thread):**
   - A Photo is sent to Telegram immediately.
   - The system records post-event frames for `VIDEO_POST_SECONDS`.
   - The pre/post frames are stitched into an `.avi` file.
   - The video is sent to Telegram.

## 🤝 Contributions
Built specifically for the Project Rakshak initiative. Pull requests, optimization suggestions, and new rule ideas are welcome!
