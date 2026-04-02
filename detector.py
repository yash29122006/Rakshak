# ═══════════════════════════════════════════════════════════════
# detector.py — YOLOv8 inference + accident rules + AI verification
# ═══════════════════════════════════════════════════════════════
# Depends on: config.py
# Must NOT import from main.py or alert.py.
# ═══════════════════════════════════════════════════════════════

import base64

import cv2
import numpy as np
import requests
from ultralytics import YOLO

import config

# ── Load YOLOv8 nano model once at import time ────────────────
# The model file is downloaded automatically on first run.
model = YOLO("yolov8n.pt")


# ═══════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════

def compute_iou(box_a, box_b):
    """Compute Intersection-over-Union between two bounding boxes.

    Each box is a tuple/list of (x1, y1, x2, y2) in pixel coordinates.

    Args:
        box_a: First bounding box [x1, y1, x2, y2].
        box_b: Second bounding box [x1, y1, x2, y2].

    Returns:
        float: IoU value in [0.0, 1.0].
    """
    # Intersection rectangle
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    # Union area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def compute_center(box):
    """Compute the centre point of a bounding box.

    Args:
        box: Bounding box [x1, y1, x2, y2].

    Returns:
        tuple: (cx, cy) centre coordinates.
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return (cx, cy)


def frame_to_bytes(frame):
    """Encode an OpenCV BGR frame as JPEG bytes.

    Args:
        frame (numpy.ndarray): The BGR image array.

    Returns:
        bytes: JPEG-encoded image data, or None on failure.
    """
    success, buffer = cv2.imencode(".jpg", frame)
    if success:
        return buffer.tobytes()
    return None


def save_frames_to_video(frames, fps, output_path):
    """Write a list of BGR frames to a video file.

    Uses the XVID codec wrapped in an .avi container for
    maximum compatibility across platforms.

    Args:
        frames (list[numpy.ndarray]): Ordered list of BGR frames.
        fps (int): Frames per second for the output video.
        output_path (str): Full file path for the output video.

    Returns:
        bool: True if the video was written successfully, False otherwise.
    """
    if not frames:
        return False

    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]

    # XVID codec works on Windows, Mac, and Linux without extra installs
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {output_path}")
        return False

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"[VIDEO] Saved {len(frames)} frames to {output_path}")
    return True


# ═══════════════════════════════════════════════════════════════
# Detection & parsing
# ═══════════════════════════════════════════════════════════════

def run_detection(frame):
    """Run YOLOv8 inference on a single frame.

    Filters results to only vehicles and persons above the
    configured confidence threshold.

    Args:
        frame (numpy.ndarray): BGR image from the webcam.

    Returns:
        tuple: (vehicles, persons) where each is a list of dicts
               with keys 'box' [x1,y1,x2,y2], 'class_id', 'label',
               and 'confidence'.
    """
    # Run inference with the confidence threshold from config
    results = model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

    vehicles = []
    persons = []

    # YOLOv8 returns a list; take the first (and only) result
    for result in results:
        boxes = result.boxes
        for idx in range(len(boxes)):
            class_id = int(boxes.cls[idx].item())
            confidence = float(boxes.conf[idx].item())
            # Extract pixel coordinates as plain Python ints
            x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
            box = [int(x1), int(y1), int(x2), int(y2)]

            # Filter out tiny/distant detections that are unreliable
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area < config.MIN_BOX_AREA:
                continue

            if class_id in config.VEHICLE_CLASSES:
                vehicles.append({
                    "box": box,
                    "class_id": class_id,
                    "label": config.VEHICLE_CLASSES[class_id],
                    "confidence": confidence,
                })
            elif class_id == config.PERSON_CLASS:
                persons.append({
                    "box": box,
                    "class_id": class_id,
                    "label": "person",
                    "confidence": confidence,
                })

    return vehicles, persons


# ═══════════════════════════════════════════════════════════════
# Three accident detection rules
# ═══════════════════════════════════════════════════════════════

def check_vehicle_collision(vehicles):
    """Rule 1 — Detect heavy overlap between two vehicle bounding boxes.

    If any pair of vehicles has IoU > IOU_OVERLAP_THRESHOLD, classify
    the event as a collision.

    Args:
        vehicles (list[dict]): Detected vehicle objects.

    Returns:
        tuple: (triggered: bool, reason: str).
    """
    vehicle_count = len(vehicles)
    for i in range(vehicle_count):
        for j in range(i + 1, vehicle_count):
            iou = compute_iou(vehicles[i]["box"], vehicles[j]["box"])
            if iou > config.IOU_OVERLAP_THRESHOLD:
                label_a = vehicles[i]["label"]
                label_b = vehicles[j]["label"]
                reason = (
                    f"Vehicle collision detected — "
                    f"{label_a} and {label_b} overlap (IoU={iou:.2f})"
                )
                return True, reason

    return False, ""


def check_person_vehicle_overlap(persons, vehicles):
    """Rule 2 — Detect a person bounding box overlapping a vehicle.

    If any person–vehicle pair has IoU > PERSON_VEHICLE_IOU_THRESHOLD,
    classify the event as a person being hit.

    Args:
        persons (list[dict]): Detected person objects.
        vehicles (list[dict]): Detected vehicle objects.

    Returns:
        tuple: (triggered: bool, reason: str).
    """
    for person in persons:
        for vehicle in vehicles:
            iou = compute_iou(person["box"], vehicle["box"])
            if iou > config.PERSON_VEHICLE_IOU_THRESHOLD:
                vehicle_label = vehicle["label"]
                reason = (
                    f"Person-vehicle collision — "
                    f"person overlaps with {vehicle_label} (IoU={iou:.2f})"
                )
                return True, reason

    return False, ""


def check_sudden_motion(vehicles, previous_vehicles):
    """Rule 3 — Detect sudden abnormal motion of any vehicle.

    Compares the centre-point displacement of each current vehicle
    against its nearest match in the previous frame. If any vehicle
    moved more than SUDDEN_MOTION_PIXEL_THRESHOLD pixels, flag it.

    Args:
        vehicles (list[dict]): Current-frame vehicle detections.
        previous_vehicles (list[dict]): Previous-frame vehicle detections.

    Returns:
        tuple: (triggered: bool, reason: str).
    """
    # Cannot evaluate motion without a previous frame
    if not previous_vehicles or not vehicles:
        return False, ""

    for current in vehicles:
        current_center = compute_center(current["box"])

        # Find the nearest previous vehicle of the same class
        min_distance = float("inf")
        for previous in previous_vehicles:
            if previous["class_id"] != current["class_id"]:
                continue
            prev_center = compute_center(previous["box"])
            distance = np.sqrt(
                (current_center[0] - prev_center[0]) ** 2
                + (current_center[1] - prev_center[1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance

        # Check threshold only when a same-class match was found
        if min_distance != float("inf") and min_distance > config.SUDDEN_MOTION_PIXEL_THRESHOLD:
            reason = (
                f"Sudden abnormal motion — "
                f"{current['label']} displaced {min_distance:.0f}px"
            )
            return True, reason

    return False, ""


# ═══════════════════════════════════════════════════════════════
# Combined rule evaluator
# ═══════════════════════════════════════════════════════════════

def evaluate_accident_rules(vehicles, persons, previous_vehicles):
    """Run all three accident rules and return the first one that fires.

    Priority order: collision → person-hit → sudden motion.

    Args:
        vehicles (list[dict]): Current-frame vehicle detections.
        persons (list[dict]): Current-frame person detections.
        previous_vehicles (list[dict]): Previous-frame vehicle detections.

    Returns:
        tuple: (accident_detected: bool, reason: str).
    """
    # Rule 1 — vehicle-vehicle collision
    triggered, reason = check_vehicle_collision(vehicles)
    if triggered:
        return True, reason

    # Rule 2 — person hit by vehicle
    triggered, reason = check_person_vehicle_overlap(persons, vehicles)
    if triggered:
        return True, reason

    # Rule 3 — sudden abnormal motion
    triggered, reason = check_sudden_motion(vehicles, previous_vehicles)
    if triggered:
        return True, reason

    return False, ""


# ═══════════════════════════════════════════════════════════════
# Drawing / annotation
# ═══════════════════════════════════════════════════════════════

def draw_detections(frame, vehicles, persons, accident_detected, reason=""):
    """Annotate a frame with bounding boxes and status text.

    Vehicles are drawn in blue, persons in green.
    If an accident is detected a red banner is overlaid;
    otherwise a green "Normal" label is shown.

    Args:
        frame (numpy.ndarray): The BGR image to annotate (modified in place).
        vehicles (list[dict]): Detected vehicles.
        persons (list[dict]): Detected persons.
        accident_detected (bool): Whether an accident rule fired.
        reason (str): Description of the triggered rule.

    Returns:
        numpy.ndarray: The annotated frame (same object as input).
    """
    # ── Draw vehicle boxes (blue) ─────────────────────────────
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["box"]
        label_text = f"{vehicle['label']} {vehicle['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            frame, label_text, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2,
        )

    # ── Draw person boxes (green) ─────────────────────────────
    for person in persons:
        x1, y1, x2, y2 = person["box"]
        label_text = f"person {person['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label_text, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    # ── Status banner ─────────────────────────────────────────
    if accident_detected:
        # Red warning banner at the top
        cv2.putText(
            frame, "ACCIDENT DETECTED", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
        )
        # Show the specific reason below the banner
        if reason:
            cv2.putText(
                frame, reason, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
    else:
        # Green "Normal" indicator
        cv2.putText(
            frame, "Normal", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3,
        )

    return frame


# ═══════════════════════════════════════════════════════════════
# AI-based accident verification (Google Gemini Vision)
# ═══════════════════════════════════════════════════════════════

def verify_accident_with_ai(image_bytes):
    """Double-check a suspected accident using Google Gemini Vision AI.

    Sends the annotated frame to Gemini and asks whether an accident
    is actually visible.  This dramatically reduces false positives
    because a vision-language model can understand scene context
    (parked cars, pedestrians, normal traffic) far better than
    simple IoU rules.

    If GEMINI_API_KEY is empty or the API call fails, the function
    returns True (trust the rules as a fallback).

    Args:
        image_bytes (bytes): JPEG-encoded image of the suspected frame.

    Returns:
        tuple: (is_accident: bool, ai_reason: str).
    """
    # ── Skip if no API key configured ─────────────────────────
    if not config.GEMINI_API_KEY:
        return True, "AI verification skipped (no API key)"

    # ── Build the Gemini API request ──────────────────────────
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.0-flash:generateContent"
        f"?key={config.GEMINI_API_KEY}"
    )

    # Encode image as base64 for the API
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a traffic accident detection AI. "
        "Analyze this traffic camera image carefully.\n\n"
        "Is there an ACTUAL vehicle accident, collision, crash, "
        "or a person being hit by a vehicle in this image?\n\n"
        "Consider:\n"
        "- Vehicles merely close together or parked is NOT an accident\n"
        "- A person walking near a vehicle is NOT an accident\n"
        "- Normal traffic flow is NOT an accident\n"
        "- Only visible damage, impact, or collision is an accident\n\n"
        "Respond with EXACTLY this format:\n"
        "VERDICT: YES or NO\n"
        "REASON: [brief 10-word explanation]"
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64,
                }},
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 60,
            "temperature": 0.1,
        },
    }

    # ── Call the API ──────────────────────────────────────────
    try:
        response = requests.post(api_url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"[AI] Gemini says: {text}")

            # Parse the verdict
            is_accident = "YES" in text.upper().split("VERDICT")[-1].split("REASON")[0]

            # Extract the reason
            ai_reason = text
            if "REASON:" in text.upper():
                ai_reason = text.split("REASON:")[-1].strip()

            return is_accident, ai_reason
        else:
            print(f"[AI] Gemini API error {response.status_code} — trusting rules")
            return True, "AI verification failed (API error)"

    except requests.RequestException as error:
        print(f"[AI] Gemini call failed: {error} — trusting rules")
        return True, "AI verification failed (network error)"

