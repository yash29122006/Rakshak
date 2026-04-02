# ═══════════════════════════════════════════════════════════════
# main.py — Webcam loop + orchestration for SafeRoad AI Prototype
# ═══════════════════════════════════════════════════════════════
# Entry point. Opens the webcam, captures frames, runs detection,
# evaluates accident rules, verifies with Gemini AI, sends alerts
# with snapshot + video, and shows a live window.
# Press Q to quit cleanly.
#
# Threading is used ONLY for alert sending (network I/O) so the
# webcam feed never freezes during Telegram uploads.
# ═══════════════════════════════════════════════════════════════

import os
import time
import threading
from collections import deque

import cv2

import config
from detector import (
    run_detection,
    evaluate_accident_rules,
    draw_detections,
    frame_to_bytes,
    save_frames_to_video,
    verify_accident_with_ai,
)
from alert import send_telegram_alert, send_telegram_video


# ── Global lock to prevent overlapping alert threads ──────────
alert_lock = threading.Lock()


def _send_alert_background(snapshot_bytes, reason, clip_frames, video_dir):
    """Background worker: AI verify → send photo → save video → send video.

    Runs in a separate thread so the main webcam loop stays smooth.
    Uses alert_lock to prevent multiple alerts from overlapping.

    Args:
        snapshot_bytes (bytes): JPEG snapshot of the accident frame.
        reason (str): Rule-based reason string.
        clip_frames (list): All frames for the video clip.
        video_dir (str): Directory to save the video file.
    """
    # Only one alert pipeline at a time
    if not alert_lock.acquire(blocking=False):
        print("[ALERT] Another alert is still being sent — skipping.")
        return

    try:
        # ── Step 1: AI verification ───────────────────────────
        ai_confirmed, ai_reason = verify_accident_with_ai(snapshot_bytes)

        if not ai_confirmed:
            print(f"[AI] Gemini rejected alert — {ai_reason}")
            return

        print(f"[AI] Gemini confirmed accident — {ai_reason}")

        # ── Step 2: Send snapshot photo to Telegram ───────────
        send_telegram_alert(snapshot_bytes, reason)

        # ── Step 3: Save video clip to file ───────────────────
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"accident_{timestamp_str}.avi"
        video_path = os.path.join(video_dir, video_filename)

        video_saved = save_frames_to_video(
            clip_frames, config.VIDEO_FPS, video_path
        )

        # ── Step 4: Send video to Telegram ────────────────────
        if video_saved:
            send_telegram_video(video_path, reason)

    finally:
        alert_lock.release()


def main():
    """Run the live accident detection pipeline.

    Steps:
        1. Open the default webcam (index 0).
        2. Loop: read frame → buffer → detect → evaluate → display.
        3. Keep a rolling buffer of the last N seconds of frames.
        4. On confirmed accident:
           a. Capture snapshot + copy buffer frames.
           b. Continue recording post-event frames (non-blocking).
           c. Spawn background thread for: AI verify → send photo → send video.
        5. Show annotated live feed — camera NEVER freezes.
        6. Press 'Q' to exit cleanly.
    """
    # ── Open webcam ───────────────────────────────────────────
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        return

    print("=" * 55)
    print("  SafeRoad AI — Accident Detection Prototype")
    print("  Press Q to quit")
    print("=" * 55)

    # ── Create the display window explicitly ──────────────────
    window_name = "SafeRoad AI — Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.moveWindow(window_name, 100, 100)

    # ── Rolling frame buffer (last N seconds of raw frames) ───
    buffer_max_frames = config.VIDEO_PRE_SECONDS * config.VIDEO_FPS
    frame_buffer = deque(maxlen=buffer_max_frames)

    # ── State variables ───────────────────────────────────────
    frame_count = 0
    previous_vehicles = []
    accident_streak = 0

    # ── Post-event video recording state ──────────────────────
    # States: "idle" → "recording_post" → (spawn thread) → "idle"
    video_state = "idle"
    post_event_frames = []
    post_event_start_time = 0.0
    video_accident_reason = ""
    pre_event_snapshot = None     # snapshot bytes for the alert
    pre_event_buffer_copy = []    # copy of frame buffer at trigger time

    # ── Output directory for video clips ──────────────────────
    video_output_dir = os.path.join(os.path.dirname(__file__), "clips")
    os.makedirs(video_output_dir, exist_ok=True)

    # ── Main loop ─────────────────────────────────────────────
    while True:
        success, frame = capture.read()
        if not success:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame_count += 1

        # ── Always buffer the raw frame ───────────────────────
        frame_buffer.append(frame.copy())

        # ── Frame skipping for detection ──────────────────────
        if frame_count % config.FRAME_SKIP != 0:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Run YOLOv8 detection ──────────────────────────────
        vehicles, persons = run_detection(frame)

        # ── Evaluate accident rules ───────────────────────────
        rule_triggered, reason = evaluate_accident_rules(
            vehicles, persons, previous_vehicles
        )

        # ── Multi-frame confirmation ──────────────────────────
        if rule_triggered:
            accident_streak += 1
        else:
            accident_streak = 0

        accident_confirmed = accident_streak >= config.ACCIDENT_CONFIRM_FRAMES

        # ══════════════════════════════════════════════════════
        # Video recording state machine (non-blocking)
        # ══════════════════════════════════════════════════════

        if video_state == "idle":
            if accident_confirmed and not alert_lock.locked():
                # ── Capture snapshot + freeze the pre-event buffer ──
                annotated = draw_detections(
                    frame.copy(), vehicles, persons, True, reason
                )
                pre_event_snapshot = frame_to_bytes(annotated)
                pre_event_buffer_copy = list(frame_buffer)
                video_accident_reason = reason

                # Start collecting post-event frames
                video_state = "recording_post"
                post_event_frames = []
                post_event_start_time = time.time()
                print(f"[VIDEO] Recording post-event ({config.VIDEO_POST_SECONDS}s)...")

        elif video_state == "recording_post":
            # ── Collect post-event frames (non-blocking) ──────
            post_event_frames.append(frame.copy())

            elapsed = time.time() - post_event_start_time
            if elapsed >= config.VIDEO_POST_SECONDS:
                # ── Enough footage — spawn background thread ──
                all_clip_frames = pre_event_buffer_copy + post_event_frames

                alert_thread = threading.Thread(
                    target=_send_alert_background,
                    args=(
                        pre_event_snapshot,
                        video_accident_reason,
                        all_clip_frames,
                        video_output_dir,
                    ),
                    daemon=True,
                )
                alert_thread.start()

                # Reset state — main loop continues immediately
                video_state = "idle"
                post_event_frames = []
                pre_event_snapshot = None
                pre_event_buffer_copy = []

        # ── Draw bounding boxes on live frame ─────────────────
        show_accident = accident_confirmed or video_state == "recording_post"
        display_reason = reason if accident_confirmed else video_accident_reason

        display_frame = draw_detections(
            frame, vehicles, persons, show_accident, display_reason
        )

        # ── REC indicator during post-event recording ─────────
        if video_state == "recording_post":
            elapsed = time.time() - post_event_start_time
            rec_text = f"REC {elapsed:.1f}s / {config.VIDEO_POST_SECONDS}s"
            cv2.putText(
                display_frame, rec_text, (display_frame.shape[1] - 280, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        # ── Show "AI Verifying..." when thread is running ─────
        if alert_lock.locked():
            cv2.putText(
                display_frame, "AI Verifying...", (display_frame.shape[1] - 280, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
            )

        # ── Show live feed (NEVER blocks) ─────────────────────
        cv2.imshow(window_name, display_frame)

        # ── Update previous vehicles ──────────────────────────
        previous_vehicles = vehicles

        # ── Quit on Q ─────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────
    capture.release()
    cv2.destroyAllWindows()
    print("[INFO] SafeRoad AI shut down cleanly.")


if __name__ == "__main__":
    main()
