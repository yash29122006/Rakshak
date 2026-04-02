# ═══════════════════════════════════════════════════════════════
# config.py — Central configuration for SafeRoad AI Prototype
# ═══════════════════════════════════════════════════════════════
# All tuneable constants live here. No imports allowed.
# Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before running.
# ═══════════════════════════════════════════════════════════════

# ── Telegram Bot credentials ──────────────────────────────────
TELEGRAM_BOT_TOKEN = "8711636556:AAEace1glHXdOUxiav4IptoKj9E6frkx_Hk"
TELEGRAM_CHAT_ID = "1685455588"

# ── Google Gemini AI verification (free tier) ─────────────────
# Get a free key from: https://aistudio.google.com/apikey
# If empty, AI verification is skipped and rules alone are used.
GEMINI_API_KEY = "AIzaSyCEmF5MzLdQqTvbL9z4T4ZK91khbRRGlzg"

# ── YOLOv8 detection confidence ───────────────────────────────
# 0.45 = catches bikes/motorcycles which YOLO is less confident on
CONFIDENCE_THRESHOLD = 0.45

# ── Rule 1: Vehicle-vehicle collision IoU threshold ───────────
# 0.2 = balanced — real collisions overlap this much, parked cars don't
IOU_OVERLAP_THRESHOLD = 0.2

# ── Rule 2: Person-vehicle overlap IoU threshold ──────────────
# 0.08 = catches person-vehicle contact without triggering on passersby
PERSON_VEHICLE_IOU_THRESHOLD = 0.08

# ── Rule 3: Sudden motion pixel displacement threshold ────────
# 100px = catches crash-speed jumps, ignores normal driving
SUDDEN_MOTION_PIXEL_THRESHOLD = 100

# ── Multi-frame confirmation ──────────────────────────────────
# 2 consecutive frames = faster response while still filtering glitches
ACCIDENT_CONFIRM_FRAMES = 2

# ── Minimum bounding box area (pixels²) ───────────────────────
# 1200 = small enough to catch motorcycles/bikes at a distance
MIN_BOX_AREA = 1200

# ── Alert cooldown to prevent spam (seconds) ──────────────────
ALERT_COOLDOWN_SECONDS = 15

# ── Process every Nth frame (1 = every frame, 2 = skip one) ──
FRAME_SKIP = 2

# ── Video clip recording ─────────────────────────────────────
# Seconds of footage BEFORE accident to include in the clip
VIDEO_PRE_SECONDS = 5
# Seconds of footage AFTER accident confirmation to keep recording
VIDEO_POST_SECONDS = 5
# FPS for the output video file (match your webcam roughly)
VIDEO_FPS = 15

# ── COCO class IDs for vehicles ───────────────────────────────
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}

# ── COCO class ID for person ──────────────────────────────────
PERSON_CLASS = 0
