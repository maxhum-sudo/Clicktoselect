"""
MediaPipe hand-pointer control.

Pointer: index fingertip (landmark 8) drives cursor.
Pinch thumb+index acts as mouse hold (down/up).
Click also supported via mic spike (tongue click) or space key.
"""

import math
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.02

SMOOTHING = 0.65
# Map center 70% of camera view to full screen (less hand travel to reach edges)
HAND_ZONE_MIN = 0.15
HAND_ZONE_MAX = 0.85
CLICK_COOLDOWN = 0.8
PINCH_CLOSE = 0.045
PINCH_OPEN = 0.065
# Freeze cursor when fingers are this close (avoids jump during pinch)
PINCH_FREEZE = 0.08

MIC_RATE = 16000
MIC_CHUNK = 1024
MIC_CALIBRATION_CHUNKS = 25

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def announce_select():
    try:
        # Use a short system click-like sound instead of speech.
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
    except Exception:
        print("\a", end="", flush=True)


def get_screen_size() -> tuple[int, int]:
    return pyautogui.size()


def hand_to_screen(norm_x: float, norm_y: float, screen_width: int, screen_height: int) -> tuple[int, int]:
    zone_w = HAND_ZONE_MAX - HAND_ZONE_MIN
    x = (norm_x - HAND_ZONE_MIN) / zone_w
    y = (norm_y - HAND_ZONE_MIN) / zone_w
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return int(x * screen_width), int(y * screen_height)


def ensure_hand_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


def init_mic():
    mic = pyaudio.PyAudio()
    stream = mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=MIC_CHUNK,
    )
    samples = []
    for _ in range(MIC_CALIBRATION_CHUNKS):
        data = stream.read(MIC_CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        samples.append(float(np.sqrt(np.mean(np.square(audio)))))
    baseline = max(1.0, float(np.median(samples)))
    threshold = max(2200.0, baseline * 4.0)
    return mic, stream, baseline, threshold


def main():
    ensure_hand_model()

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    screen_width, screen_height = get_screen_size()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smooth_x, smooth_y = 0.5, 0.5
    pinch_down = False
    mouse_held_by_pinch = False
    last_click_time = 0.0
    mic_enabled = True
    mic = None
    mic_stream = None
    mic_threshold = None

    print("MediaPipe hand pointer started.")
    print("Pinch thumb+index to click. Space also clicks.")
    print("Mic tongue-click enabled by default; press 'm' to toggle.")
    print("Press 'q' to quit.")

    try:
        mic, mic_stream, mic_baseline, mic_threshold = init_mic()
        print(f"Mic calibrated. baseline={mic_baseline:.0f}, threshold={mic_threshold:.0f}")
    except Exception as exc:
        mic_enabled = False
        mic_stream = None
        print(f"Mic unavailable ({exc}).")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    index_tip = lm[8]
                    thumb_tip = lm[4]

                    pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    # Freeze cursor only when entering pinch (avoids jump on click); allow movement during drag
                    if pinch_dist > PINCH_FREEZE or pinch_down:
                        raw_x, raw_y = index_tip.x, index_tip.y
                        smooth_x = SMOOTHING * smooth_x + (1 - SMOOTHING) * raw_x
                        smooth_y = SMOOTHING * smooth_y + (1 - SMOOTHING) * raw_y
                    sx, sy = hand_to_screen(smooth_x, smooth_y, screen_width, screen_height)
                    pyautogui.moveTo(sx, sy, _pause=False)
                    now = time.monotonic()
                    if (not pinch_down) and pinch_dist < PINCH_CLOSE:
                        pyautogui.mouseDown()
                        pinch_down = True
                        mouse_held_by_pinch = True
                        print(f"  -> mouse down (pinch {pinch_dist:.3f})")
                    elif pinch_down and pinch_dist > PINCH_OPEN:
                        pinch_down = False
                        if mouse_held_by_pinch:
                            pyautogui.mouseUp()
                            announce_select()
                            mouse_held_by_pinch = False
                            print("  -> mouse up (release)")

                    cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 6, (255, 180, 0), -1)
                    if pinch_down:
                        status = "DRAG (move to drag)"
                    elif pinch_dist < PINCH_FREEZE:
                        status = "LOCKED (click)"
                    else:
                        status = "OPEN=MOVE"
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 240, 50), 2)
                    cv2.putText(
                        frame,
                        f"pinch:{pinch_dist:.3f} close<{PINCH_CLOSE:.3f}",
                        (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 220, 255),
                        2,
                    )
                else:
                    if mouse_held_by_pinch:
                        pyautogui.mouseUp()
                        mouse_held_by_pinch = False
                        print("  -> mouse up (hand lost)")
                    pinch_down = False
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 200, 255),
                        2,
                    )

                if mic_enabled and mic_stream is not None and mic_threshold is not None:
                    try:
                        data = mic_stream.read(MIC_CHUNK, exception_on_overflow=False)
                        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(np.square(audio))))
                        now = time.monotonic()
                        if rms > mic_threshold and (now - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.click()
                            announce_select()
                            last_click_time = now
                            print(f"  -> click (mic spike {rms:.0f})")
                        cv2.putText(
                            frame,
                            f"MIC on rms:{int(rms)} thr:{int(mic_threshold)}",
                            (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 255),
                            2,
                        )
                    except Exception:
                        pass
                else:
                    cv2.putText(
                        frame,
                        "MIC off (press m to toggle)",
                        (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 180, 255),
                        2,
                    )

                cv2.imshow("MediaPipe Hand Pointer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    mic_enabled = not mic_enabled
                    print(f"  -> mic {'enabled' if mic_enabled else 'disabled'}")
                if key == 32:
                    now = time.monotonic()
                    if (now - last_click_time) > CLICK_COOLDOWN:
                        pyautogui.click()
                        announce_select()
                        last_click_time = now
                        print("  -> click (space)")
    finally:
        if mouse_held_by_pinch:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        if mic_stream is not None:
            try:
                mic_stream.stop_stream()
                mic_stream.close()
            except Exception:
                pass
        if mic is not None:
            try:
                mic.terminate()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
