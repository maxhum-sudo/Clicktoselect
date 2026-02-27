"""
Hand/object cursor control via manual tracker.

Press 's' to select your hand (or black hat) once, then the tracker follows it.
Press 'b' for automatic black-object tracking mode.
Press space to click. Press 'r' to reselect target. Press 'q' to quit.
Mic loud-click mode: make a tongue click (high volume spike) to trigger click.
"""

import time
import subprocess

import cv2
import pyaudio
import numpy as np
import pyautogui

# Disable pyautogui fail-safe (moving mouse to corner) so gaze can move freely
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.02

SMOOTHING = 0.65
HAND_ZONE_MARGIN = 0.10
CLICK_COOLDOWN = 0.8
MIC_RATE = 16000
MIC_CHUNK = 1024
MIC_CALIBRATION_CHUNKS = 25


def get_screen_size() -> tuple[int, int]:
    return pyautogui.size()


def hand_to_screen(norm_x: float, norm_y: float, screen_width: int, screen_height: int) -> tuple[int, int]:
    """Map normalized hand position (0..1) to screen coordinates."""
    margin = HAND_ZONE_MARGIN
    x = margin + norm_x * (1 - 2 * margin)
    y = margin + norm_y * (1 - 2 * margin)
    x = max(0, min(1, x))
    y = max(0, min(1, y))
    sx = int(x * screen_width)
    sy = int(y * screen_height)
    return sx, sy


def create_tracker():
    """Create best available OpenCV tracker for selected target."""
    creators = [
        ("legacy", "TrackerCSRT_create"),
        ("legacy", "TrackerKCF_create"),
        ("legacy", "TrackerMOSSE_create"),
        ("root", "TrackerCSRT_create"),
        ("root", "TrackerKCF_create"),
        ("root", "TrackerMOSSE_create"),
    ]
    for namespace, fn_name in creators:
        try:
            fn = getattr(cv2.legacy if namespace == "legacy" else cv2, fn_name)
            return fn()
        except Exception:
            continue
    raise RuntimeError("No supported OpenCV tracker found in this build.")


def announce_select():
    """Play an audible confirmation for click actions."""
    try:
        subprocess.Popen(["say", "select"])
    except Exception:
        # Terminal bell fallback if speech isn't available.
        print("\a", end="", flush=True)


def detect_black_target(frame_bgr):
    """Detect largest dark object (useful for a black hat)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Dark pixels: low V. Limit S to avoid some bright saturated colors.
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 120, 60))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 2500:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


def main():
    screen_width, screen_height = get_screen_size()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smooth_x, smooth_y = 0.5, 0.5
    last_click_time = 0.0
    tracker = None
    tracking_on = False
    black_mode = False
    mic = None
    mic_stream = None
    mic_baseline = None
    mic_threshold = None

    print("Hand/object cursor control started.")
    print("Press 's' to select your hand or hat with a box.")
    print("Press 'b' for auto black-hat tracking mode (no selection needed).")
    print("Press space to click, 'r' to reselect, 'q' to quit.")
    print("Mic click enabled: make a loud tongue click to trigger select.")
    print("Press 'q' in the camera window to quit.\n")

    try:
        mic = pyaudio.PyAudio()
        mic_stream = mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=MIC_RATE,
            input=True,
            frames_per_buffer=MIC_CHUNK,
        )
        # Calibrate baseline on startup ambient sound.
        samples = []
        for _ in range(MIC_CALIBRATION_CHUNKS):
            data = mic_stream.read(MIC_CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            samples.append(float(np.sqrt(np.mean(np.square(audio.astype(np.float32))))))
        mic_baseline = max(1.0, float(np.median(samples)))
        mic_threshold = max(2200.0, mic_baseline * 4.0)
        print(f"Mic calibrated. baseline={mic_baseline:.0f}, threshold={mic_threshold:.0f}")
    except Exception as exc:
        print(f"Mic unavailable ({exc}). Keyboard click still works.")
        mic_stream = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if black_mode:
                bbox = detect_black_target(frame)
                if bbox is not None:
                    x, y, bw, bh = [int(v) for v in bbox]
                    cx = x + bw // 2
                    cy = y + bh // 2
                    raw_x = cx / float(w)
                    raw_y = cy / float(h)
                    smooth_x = SMOOTHING * smooth_x + (1 - SMOOTHING) * raw_x
                    smooth_y = SMOOTHING * smooth_y + (1 - SMOOTHING) * raw_y
                    sx, sy = hand_to_screen(smooth_x, smooth_y, screen_width, screen_height)
                    pyautogui.moveTo(sx, sy, _pause=False)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 80, 80), 2)
                    cv2.putText(
                        frame,
                        "BLACK MODE (space=click, s=manual)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 120, 120),
                        2,
                    )
                else:
                    cv2.putText(frame, "Black target not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            elif tracking_on and tracker is not None:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, bw, bh = [int(v) for v in bbox]
                    cx = x + bw // 2
                    cy = y + bh // 2
                    raw_x = cx / float(w)
                    raw_y = cy / float(h)
                    smooth_x = SMOOTHING * smooth_x + (1 - SMOOTHING) * raw_x
                    smooth_y = SMOOTHING * smooth_y + (1 - SMOOTHING) * raw_y

                    sx, sy = hand_to_screen(smooth_x, smooth_y, screen_width, screen_height)
                    pyautogui.moveTo(sx, sy, _pause=False)

                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        "TRACKING (space=click, r=reselect, b=black)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (50, 240, 50),
                        2,
                    )
                else:
                    tracking_on = False
                    cv2.putText(frame, "Lost target. Press 's' to select again.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                cv2.putText(
                    frame,
                    "Press 's' select target or 'b' black mode",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )

            # Mic-based click on volume spike (tongue click).
            if mic_stream is not None and mic_threshold is not None:
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
                        f"MIC rms:{int(rms)} thr:{int(mic_threshold)}",
                        (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 180, 255),
                        2,
                    )
                except Exception:
                    pass

            cv2.imshow("Hand Gesture Cursor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") or key == ord("r"):
                black_mode = False
                bbox = cv2.selectROI("Hand Gesture Cursor", frame, fromCenter=False, showCrosshair=True)
                if bbox[2] > 0 and bbox[3] > 0:
                    tracker = create_tracker()
                    tracker.init(frame, bbox)
                    tracking_on = True
                    print("  -> target selected")
            elif key == ord("b"):
                tracking_on = False
                tracker = None
                black_mode = True
                print("  -> black mode enabled")
            elif key == 32:
                now = time.monotonic()
                if now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    announce_select()
                    last_click_time = now
                    print("  -> click (space)")
            elif key == ord("q"):
                break

    finally:
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
