# ═══════════════════════════════════════════════════════════════
# alert.py — Telegram alert sender with image support & cooldown
# ═══════════════════════════════════════════════════════════════
# Depends on: config.py
# Must NOT import from detector.py or main.py.
# ═══════════════════════════════════════════════════════════════

import time
from datetime import datetime

import requests

import config

# ── Global cooldown tracker ───────────────────────────────────
last_alert_time = 0.0


def is_cooldown_active():
    """Check whether the alert cooldown period is still active.

    Returns:
        bool: True if a new alert should be suppressed, False if sending is OK.
    """
    global last_alert_time
    elapsed = time.time() - last_alert_time
    return elapsed < config.ALERT_COOLDOWN_SECONDS


def _build_caption(reason):
    """Build the Markdown-formatted Telegram caption.

    Args:
        reason (str): Human-readable description of which rule triggered.

    Returns:
        str: Formatted message string ready for Telegram.
    """
    # Format current time as "01 Apr 2026, 09:45:23 PM"
    now = datetime.now()
    timestamp = now.strftime("%d %b %Y, %I:%M:%S %p")

    caption = (
        "🚨 *ACCIDENT DETECTED!*\n"
        "\n"
        f"📍 *Location:* Laptop Webcam\n"
        f"🕐 *Time:* {timestamp}\n"
        f"⚠️ *Reason:* {reason}\n"
        "\n"
        "_SafeRoad AI Prototype Alert_"
    )
    return caption


def send_telegram_alert(image_bytes, reason):
    """Send an accident alert with a snapshot image to Telegram.

    Respects the cooldown timer — silently skips if called too soon
    after a previous alert.

    Args:
        image_bytes (bytes): JPEG-encoded image data of the annotated frame.
        reason (str): Description of the accident rule that fired.

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """
    global last_alert_time

    # ── Cooldown gate ─────────────────────────────────────────
    if is_cooldown_active():
        return False

    # ── Validate credentials ──────────────────────────────────
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[ALERT] Telegram credentials not configured in config.py — skipping alert.")
        return False

    # ── Build the API URL and payload ─────────────────────────
    api_url = (
        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
    )

    caption = _build_caption(reason)

    # The photo is sent as a multipart file upload
    files = {
        "photo": ("accident_snapshot.jpg", image_bytes, "image/jpeg"),
    }
    data = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "Markdown",
    }

    # ── Send the request ──────────────────────────────────────
    try:
        response = requests.post(api_url, data=data, files=files, timeout=10)

        if response.status_code == 200:
            print(f"[ALERT] Telegram alert sent — {reason}")
            last_alert_time = time.time()  # reset cooldown
            return True
        else:
            print(f"[ALERT] Telegram API error {response.status_code}: {response.text}")
            return False

    except requests.RequestException as error:
        print(f"[ALERT] Failed to send Telegram alert: {error}")
        return False


def send_telegram_video(video_path, reason):
    """Send an accident video clip to Telegram.

    This is called AFTER the photo alert has already been sent,
    so it does NOT check cooldown — the photo handles that gate.

    Args:
        video_path (str): Absolute path to the .avi video file.
        reason (str): Description of the accident rule that fired.

    Returns:
        bool: True if the video was sent successfully, False otherwise.
    """
    # ── Validate credentials ──────────────────────────────────
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("[VIDEO] Telegram credentials not configured — skipping video.")
        return False

    api_url = (
        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendVideo"
    )

    caption = "🎥 *Incident Video Clip*\n\n_Footage around the time of detection_"

    try:
        # Open the video file and upload it as multipart
        with open(video_path, "rb") as video_file:
            files = {
                "video": ("accident_clip.avi", video_file, "video/x-msvideo"),
            }
            data = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "Markdown",
            }

            response = requests.post(api_url, data=data, files=files, timeout=60)

        if response.status_code == 200:
            print(f"[VIDEO] Telegram video sent — {reason}")
            return True
        else:
            print(f"[VIDEO] Telegram API error {response.status_code}: {response.text}")
            return False

    except requests.RequestException as error:
        print(f"[VIDEO] Failed to send Telegram video: {error}")
        return False

