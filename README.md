# Cookie Monster Challenge + Hand Cursor

A Cookie Monster drag-and-drop game playable in the browser, with optional **hand-tracking cursor control** for hands-free play.

## 🎮 Play the game (live)

**→ [https://maxhum-sudo.github.io/Clicktoselect/](https://maxhum-sudo.github.io/Clicktoselect/)**

The game runs entirely in the browser. No install needed.

---

## 🖐️ Hand-tracking cursor (Python)

Control your mouse with your hand via webcam. Point with your index finger; pinch thumb+index to click/drag.

### Easiest way to get it

1. **Clone this repo:**
   ```bash
   git clone https://github.com/maxhum-sudo/Clicktoselect.git
   cd Clicktoselect
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   On macOS you may need: `brew install portaudio` before `pip install pyaudio`

3. **Run:**
   ```bash
   python gaze_voice_cursor.py
   ```

4. Point your index finger at the camera; pinch to click. Press **Space** or use a tongue-click into the mic for extra clicks. Press **q** to quit.

### Requirements

- Python 3.8+
- Webcam
- macOS / Linux / Windows

---

## Deploying the hand-tracking software

| Option | Feasibility | Notes |
|--------|-------------|-------|
| **App Store (iOS)** | Not directly | The Python script is desktop-only. A native iOS app would need Swift + Vision/ARKit or a cross‑platform framework (e.g. Kivy). |
| **Chrome Web Store** | Yes | MediaPipe has a JavaScript version. You’d rewrite the hand tracking in JS and package it as a Chrome extension. Requires HTTPS for webcam. |
| **Web app** | Yes | Use MediaPipe’s `@mediapipe/tasks-vision` in the browser. Host on any HTTPS site. Works on desktop and mobile. |
| **Electron / PyInstaller** | Yes | Package the Python app as a standalone desktop executable for Windows/Mac/Linux. |
| **pip package** | Yes | Publish to PyPI so users can `pip install` and run a CLI entry point. |

**Simplest distribution today:** Clone the repo and run `python gaze_voice_cursor.py` as above.

---

## Game modes

- **Timed:** 15s, 30s, or 45s rounds
- **Cookies Unlimited:** Start with 60s, earn more time by collecting cookies (+15s at 10, +10s at 20, +5s at 30, etc.)
- **Bad cookies:** Some cookies have 3 dots instead of 4 — click one and all cookies start moving
- **Share:** Generate a square image with your score to share
