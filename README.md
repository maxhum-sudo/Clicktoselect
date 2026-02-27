# Eye-Gaze + Voice Cursor Control

Control the mouse cursor with your eyes (via webcam) and voice commands — look where you want to point, then say **"click"**, **"open"**, **"scroll up"**, etc.

## How it works

- **Gaze:** The webcam feeds into MediaPipe Face Mesh. A combined “gaze” point is derived from your nose tip and both eye centers, smoothed and mapped to screen coordinates. Moving your head moves the cursor.
- **Voice:** A background thread listens for short phrases (e.g. “click”, “scroll down”) using Google Speech Recognition and runs the corresponding action (click, double-click, scroll, right-click).

## Setup

1. **Create a virtual environment (recommended):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Microphone (for voice):**  
   `SpeechRecognition` needs `pyaudio` for live microphone input:

   - **macOS:** `brew install portaudio` then `pip install pyaudio`
   - **Linux:** install `portaudio19-dev` (or equivalent) then `pip install pyaudio`
   - **Windows:** `pip install pyaudio` often works as-is

4. **Webcam:** A camera (built-in or USB) must be available and allowed for the script.

## Run

```bash
python gaze_voice_cursor.py
```

- Position your face in front of the camera; the cursor will follow.
- Say commands like: **click**, **double click**, **open**, **scroll up**, **scroll down**, **right click**.
- Press **`q`** in the camera preview window to quit.

## Voice commands

| Say            | Action        |
|----------------|---------------|
| click          | Left click    |
| double click   | Double-click  |
| open           | Double-click  |
| scroll up      | Scroll up     |
| scroll down    | Scroll down   |
| right click    | Right-click   |

## Possible extensions

- **Hand gestures** — Use MediaPipe Hands (or OpenCV) to trigger click/scroll with gestures instead of or in addition to voice.
- **True gaze** — Use iris landmarks (e.g. MediaPipe with iris refinement) to drive cursor from eye direction rather than head position.
- **Blink to click** — Detect deliberate double-blink and map to double-click.
- **AR-style feedback** — Overlay a cursor or highlight in the camera view that matches the on-screen cursor.

## Requirements

- Python 3.8+
- Webcam
- Microphone (for voice)
- Internet (for Google Speech Recognition)
