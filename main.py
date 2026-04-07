"""
═══════════════════════════════════════════════════════════════════════════
  Finger Gesture Detection + Dobot Nova 5 Controller + Canon EOS 6D
  Auto-starts detection │ Web dashboard │ Physical keyboard controls
═══════════════════════════════════════════════════════════════════════════

Keyboard Controls:
  [SPACE]  Toggle tracking ON / OFF
  [S]      Emergency stop robot
  [ESC]    Exit program safely

Gestures:
  Index finger (1)         → Move robot to Preset 1
  Index + Middle (2)       → Move robot to Preset 2
  Thumb + Index (7)        → (unused — shutter fires automatically after preset move)
  Fist (10)                → Start continuous scan (oscillate between 2 positions)
  All Fingers / Open Palm (5) → Stop scanning

Usage:
  python main.py                    # Full mode
  python main.py --no-robot         # Vision + camera only
  python main.py --no-camera        # Robot only, no camera
  python main.py --ip 192.168.5.1   # Custom robot IP
  python main.py --photo-dir D:\\Photos  # Custom save dir

Open http://localhost:5001 in browser for web dashboard.

Requirements:
  pip install flask flask-cors ultralytics mediapipe opencv-python numpy joblib keyboard

Camera Setup (Canon EOS 6D via digiCamControl):
  1. Install digiCamControl: https://digicamcontrol.com/download
  2. CLOSE Canon EOS Utility (it locks the USB connection!)
  3. Open digiCamControl → verify camera appears
  4. Enable webserver: File → Settings → Webserver → Enable → RESTART app
  5. Leave digiCamControl running in background while this script runs
  6. Set camera to Manual (M), disable auto-sleep
"""

import cv2
import base64
import threading
import time
import os
import sys
import argparse

from flask import Flask, jsonify, Response, request
from flask_cors import CORS

from component.detection import GestureDetector, GESTURE_ID_TO_NAME
from component.dobot_controller import (
    DobotController, ScanController,
    load_presets_from_json, build_gesture_map,
    DOBOTSTUDIO_PRESETS_JSON, _FALLBACK_INITIAL_POSE, _FALLBACK_PRESETS,
)
from component.camera_connection import (
    CanonCamera, countdown_and_capture, abort_countdown,
    COUNTDOWN_COOLDOWN_FRAMES, DIGICAM_URL, DIGICAM_AUTOLAUNCH,
)

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("[WARN] 'keyboard' not installed. Install with: pip install keyboard")
    print("[WARN] Physical hotkeys disabled. Use web dashboard instead.")
    KEYBOARD_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CAMERA_INDEX     = 1
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
JPEG_QUALITY     = 75
CAMERA_SOURCE    = "laptop"   # "laptop" or "realsense"
DETECTION_METHOD = "auto"

MODEL_PATH         = r"C:\My Websites\hand_tracking\runs\train\finger_gestures\weights\best.pt"
SKLEARN_MODEL_PATH = None

MP_MAX_HANDS      = 1
MP_DETECTION_CONF = 0.7
MP_TRACKING_CONF  = 0.5

DOBOT_IP       = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT      = 30003

CANON_ENABLED        = True
CANON_SAVE_DIR       = r"C:\CapturedPhotos"
CANON_CAPTURE_GESTURE = 7

DEBOUNCE_FRAMES    = 90   # ~3s at 30fps — must hold gesture steadily before it fires
COOLDOWN_FRAMES    = 15
NO_HAND_STOP_DELAY = 10

# Safety gate — user must hold gesture 5 (All Fingers) for this long to unlock
SAFETY_HOLD_SEC = 2.0    # seconds gesture 5 must be held continuously
SAFETY_TIMEOUT  = 30.0   # seconds after unlock to pick a preset before gate auto-locks

FLASK_PORT = 5001


# ═══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════

is_running      = False
tracking_active = True   # Auto-start ON
dry_run         = False

INITIAL_POSE    = _FALLBACK_INITIAL_POSE
PRESETS         = _FALLBACK_PRESETS
GESTURE_TO_PRESET = build_gesture_map(PRESETS)

detector: GestureDetector | None = None
robot:    DobotController | None = None
scanner:  ScanController         = ScanController()
canon:    CanonCamera | None     = None

# Debounce / cooldown state
_debounce_counter      = 0
_debounce_last_gesture = None
_cooldown_counter      = 0
_no_hand_counter       = 0

# Safety gate state
_safety_unlocked    = False  # True after gesture-5 held for SAFETY_HOLD_SEC
_safety_gate5_start = 0.0   # time.time() when current gesture-5 hold began
_safety_unlocked_at = 0.0   # time.time() when gate was last unlocked

# Countdown state
_countdown_state = {
    "active": False,
    "phase":  None,
    "number": None,
    "preset": None,
}
_countdown_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
#  TRACKING CONTROL
# ═══════════════════════════════════════════════════════════════════

def reset_debounce():
    global _debounce_counter, _debounce_last_gesture, _cooldown_counter, _no_hand_counter
    _debounce_counter = 0
    _debounce_last_gesture = None
    _cooldown_counter = 0
    _no_hand_counter = 0


def start_tracking():
    global tracking_active
    if not tracking_active:
        tracking_active = True
        reset_debounce()
        print("  [TRACKING] ► Started")


def stop_tracking():
    global tracking_active
    if tracking_active:
        tracking_active = False
        reset_debounce()
        if scanner.active:
            scanner.stop()
        abort_countdown(_countdown_state, _countdown_lock)
        if robot and robot.connected and robot.enabled:
            robot.stop()
        print("  [TRACKING] ■ Stopped")


def toggle_tracking():
    if tracking_active:
        stop_tracking()
    else:
        start_tracking()


def emergency_stop():
    stop_tracking()
    if scanner.active:
        scanner.stop()
    abort_countdown(_countdown_state, _countdown_lock)
    if robot and robot.connected:
        robot.stop()
    print("  [EMERGENCY] ■ Robot stopped + tracking paused")


# ═══════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER
# ═══════════════════════════════════════════════════════════════════

def keyboard_listener():
    """Listen for physical keyboard presses."""
    if not KEYBOARD_AVAILABLE:
        return

    print("  [KEYS] Keyboard hotkeys active:")
    print("         [SPACE] = Toggle tracking")
    print("         [S]     = Emergency stop")
    print("         [ESC]   = Exit program")
    print()

    keyboard.add_hotkey('space', toggle_tracking, suppress=False)
    keyboard.add_hotkey('s', emergency_stop, suppress=False)

    while is_running:
        if keyboard.is_pressed('esc'):
            print("\n  [ESC] Exit requested...")
            break
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════
#  ROBOT BRIDGE
# ═══════════════════════════════════════════════════════════════════

def process_robot(gesture_id):
    global _debounce_counter, _debounce_last_gesture, _cooldown_counter, _no_hand_counter
    global _safety_unlocked, _safety_gate5_start, _safety_unlocked_at

    robot_ok = robot is not None and robot.connected and robot.enabled

    if _cooldown_counter > 0:
        _cooldown_counter -= 1
        return robot.current_preset if robot_ok else None

    if gesture_id is None:
        _debounce_counter = 0
        _debounce_last_gesture = None
        _no_hand_counter += 1
        _safety_gate5_start = 0.0   # reset hold timer when hand leaves frame
        return None

    _no_hand_counter = 0

    # ── Safety gate ─────────────────────────────────────────────────
    # Track continuous gesture-5 hold (independent of debounce)
    if gesture_id == 5:
        if _safety_gate5_start == 0.0:
            _safety_gate5_start = time.time()
        held = time.time() - _safety_gate5_start
        if not _safety_unlocked and held >= SAFETY_HOLD_SEC:
            _safety_unlocked    = True
            _safety_unlocked_at = time.time()
            print("  [SAFETY] ✓ Gate UNLOCKED — show your target gesture now")
    else:
        _safety_gate5_start = 0.0   # switched away from gesture-5, reset timer

    # Auto-lock if unlocked but no action taken within timeout
    if _safety_unlocked and (time.time() - _safety_unlocked_at) > SAFETY_TIMEOUT:
        _safety_unlocked    = False
        _safety_gate5_start = 0.0
        print("  [SAFETY] Gate auto-locked (timeout)")

    # Gate still locked → block all robot actions
    if not _safety_unlocked:
        return None
    # ── End safety gate ─────────────────────────────────────────────

    if gesture_id == _debounce_last_gesture:
        _debounce_counter += 1
    else:
        _debounce_counter = 1
        _debounce_last_gesture = gesture_id

    if _debounce_counter >= DEBOUNCE_FRAMES:
        action = GESTURE_TO_PRESET.get(gesture_id)

        # ── Start continuous scan (Fist) ────────────────────────
        if action == "scan":
            if robot_ok and not scanner.active:
                scanner.start(robot)
                print(f"  [GESTURE] Fist → START CONTINUOUS SCAN")
            _debounce_counter = 0
            _cooldown_counter = COOLDOWN_FRAMES * 3
            _safety_unlocked    = False          # lock gate after action
            _safety_gate5_start = 0.0
            print("  [SAFETY] Gate locked after action")
            return "scan"

        # ── Stop continuous scan (Open Palm) ────────────────────
        elif action == "stop_scan":
            if scanner.active:
                scanner.stop()
                print(f"  [GESTURE] All Fingers → STOP SCAN")
            _debounce_counter = 0
            _cooldown_counter = COOLDOWN_FRAMES * 2
            _safety_unlocked    = False          # lock gate after action
            _safety_gate5_start = 0.0
            print("  [SAFETY] Gate locked after action")
            return None

        # ── Robot movement + auto countdown + capture ───────────
        elif action and isinstance(action, int):
            if scanner.active:
                scanner.stop()
                time.sleep(0.2)

            # Move robot (if connected)
            if robot_ok:
                robot.move_to_preset(action, PRESETS)
                print(f"  [GESTURE] Gesture {gesture_id} → Robot Preset {action}")
            else:
                print(f"  [GESTURE] Gesture {gesture_id} → Preset {action} (robot offline — camera only)")

            # ── Always start countdown + auto capture ──────────
            abort_countdown(_countdown_state, _countdown_lock)
            threading.Thread(
                target=countdown_and_capture,
                args=(action, canon, _countdown_state, _countdown_lock),
                daemon=True,
                name=f"countdown_preset_{action}"
            ).start()

            _debounce_counter = 0
            _cooldown_counter = COUNTDOWN_COOLDOWN_FRAMES
            _safety_unlocked    = False          # lock gate after action
            _safety_gate5_start = 0.0
            print("  [SAFETY] Gate locked after action")
            return action

        _debounce_counter = 0

    return robot.current_preset if robot_ok else None


# ═══════════════════════════════════════════════════════════════════
#  FLASK API + WEB DASHBOARD
# ═══════════════════════════════════════════════════════════════════

flask_app = Flask(__name__)
CORS(flask_app)


@flask_app.route("/", methods=["GET"])
def dashboard():
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Nova 5 — Gesture Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #07090b;
  --surface:  #0e1215;
  --border:   #1a2028;
  --border2:  #253040;
  --green:    #00e676;
  --green-dim:#00e67630;
  --amber:    #ffab00;
  --red:      #ff3d3d;
  --blue:     #40c4ff;
  --muted:    #4a5568;
  --text:     #d0dce8;
  --mono: 'JetBrains Mono', monospace;
  --display: 'Syne', sans-serif;
}
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  min-height: 100vh;
  overflow-x: hidden;
}

/* Subtle scanline texture */
body::before {
  content:'';
  position:fixed; inset:0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,230,118,0.012) 2px, rgba(0,230,118,0.012) 4px);
  pointer-events:none;
  z-index:0;
}

/* ── Header ─────────────────────────────────────── */
.hdr {
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position:sticky; top:0; z-index:100;
  backdrop-filter: blur(8px);
}
.hdr-left { display:flex; flex-direction:column; gap:2px; }
.hdr h1 {
  font-family: var(--display);
  font-size: clamp(15px, 2.5vw, 20px);
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.5px;
  line-height: 1;
}
.hdr h1 em { color: var(--green); font-style:normal; }
.hdr-sub { font-size:10px; color: var(--muted); letter-spacing:1px; text-transform:uppercase; }
.hdr-right { display:flex; align-items:center; gap:16px; }
.badge {
  display:flex; align-items:center; gap:7px;
  font-size:11px; font-weight:600; letter-spacing:1px;
  text-transform:uppercase;
}
.pulse {
  width:9px; height:9px; border-radius:50%;
  background: var(--green);
  box-shadow: 0 0 0 0 var(--green);
  animation: pulse 2s infinite;
}
.pulse.off { background: var(--red); box-shadow: 0 0 0 0 var(--red); animation: none; }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.6); }
  70%  { box-shadow: 0 0 0 7px rgba(0,230,118,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}
#sysTime { font-size:12px; color: var(--muted); font-family: var(--mono); }

/* ── Layout ─────────────────────────────────────── */
.layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  grid-template-rows: auto;
  gap: 12px;
  padding: 12px;
  max-width: 1400px;
  margin: 0 auto;
  position: relative; z-index:1;
}

/* ── Video ──────────────────────────────────────── */
.video-wrap {
  grid-column:1; grid-row: 1 / 3;
  display:flex; flex-direction:column; gap:10px;
}
.video-box {
  position:relative;
  background: #000;
  border: 1px solid var(--border2);
  border-radius: 6px;
  overflow: hidden;
  aspect-ratio: 4/3;
}
.video-box img {
  width:100%; height:100%;
  object-fit:cover;
  display:block;
}
.video-box .vid-overlay {
  position:absolute; inset:0;
  display:flex; align-items:center; justify-content:center;
  background: rgba(0,0,0,0.7);
  flex-direction:column; gap:8px;
  font-size:12px; color:var(--muted);
  transition: opacity 0.3s;
}
.video-box .vid-overlay.hidden { opacity:0; pointer-events:none; }
.vid-corner {
  position:absolute;
  width:16px; height:16px;
  border-color: var(--green);
  border-style: solid;
  opacity:0.5;
}
.vid-corner.tl { top:8px; left:8px;  border-width:2px 0 0 2px; }
.vid-corner.tr { top:8px; right:8px; border-width:2px 2px 0 0; }
.vid-corner.bl { bottom:8px; left:8px;  border-width:0 0 2px 2px; }
.vid-corner.br { bottom:8px; right:8px; border-width:0 2px 2px 0; }
.vid-badge {
  position:absolute; top:10px; left:50%; transform:translateX(-50%);
  background: rgba(0,230,118,0.12);
  border: 1px solid var(--green);
  color: var(--green);
  font-size:10px; font-weight:600; letter-spacing:1.5px;
  padding: 3px 10px; border-radius:2px;
  text-transform:uppercase;
}

/* Gesture bar under video */
.gesture-bar {
  display:grid; grid-template-columns:1fr 1fr 1fr;
  gap:8px;
}
.g-cell {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius:6px;
  padding: 12px 14px;
  display:flex; flex-direction:column; gap:4px;
}
.g-cell .gc-label { font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); }
.g-cell .gc-val { font-size:22px; font-weight:700; font-family:var(--display); color:#fff; line-height:1; }
.g-cell .gc-sub { font-size:11px; color:var(--muted); }
#gNum { color: var(--green); }
#gPreset { color: var(--amber); }

/* ── Side panels ────────────────────────────────── */
.side { display:flex; flex-direction:column; gap:10px; }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius:6px;
  padding: 14px;
}
.card-title {
  font-size:9px; font-weight:600; letter-spacing:2px;
  text-transform:uppercase; color:var(--muted);
  margin-bottom:12px;
  display:flex; align-items:center; gap:8px;
}
.card-title::after {
  content:'';flex:1;height:1px;background:var(--border);
}

/* Status rows */
.srow {
  display:flex; justify-content:space-between; align-items:center;
  padding: 5px 0;
  border-bottom: 1px solid var(--border);
}
.srow:last-child { border-bottom:none; }
.srow .slabel { color:var(--muted); font-size:11px; }
.srow .sval { font-size:12px; font-weight:600; display:flex; align-items:center; gap:5px; }
.dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.dot.g { background:var(--green); box-shadow:0 0 5px var(--green); }
.dot.r { background:var(--red);   box-shadow:0 0 5px var(--red); }
.dot.y { background:var(--amber); box-shadow:0 0 5px var(--amber); }

/* Buttons */
.btn {
  width:100%; padding:10px 14px;
  border:none; border-radius:4px;
  font-family:var(--mono); font-size:11px; font-weight:600;
  letter-spacing:0.5px; text-transform:uppercase;
  cursor:pointer; transition:all 0.15s;
  display:flex; align-items:center; justify-content:center; gap:6px;
}
.btn:active { transform:scale(0.97); }
.btn-red   { background:#3d0a0a; color:var(--red);   border:1px solid var(--red); }
.btn-red:hover   { background:var(--red); color:#fff; }
.btn-green { background:#003d1a; color:var(--green); border:1px solid var(--green); }
.btn-green:hover { background:var(--green); color:#000; }
.btn-blue  { background:#0a1e2e; color:var(--blue);  border:1px solid var(--blue); }
.btn-blue:hover  { background:var(--blue); color:#000; }
.btn-amber { background:#2e1a00; color:var(--amber); border:1px solid var(--amber); }
.btn-amber:hover { background:var(--amber); color:#000; }
.btn-ghost { background:transparent; color:var(--muted); border:1px solid var(--border2); }
.btn-ghost:hover { border-color:var(--text); color:var(--text); }

.btn-row { display:flex; gap:8px; }
.btn-row .btn { flex:1; }

/* Gesture map table */
.gmap { width:100%; border-collapse:collapse; }
.gmap tr { border-bottom:1px solid var(--border); }
.gmap tr:last-child { border-bottom:none; }
.gmap td { padding:5px 4px; font-size:11px; }
.gmap .gm-id { color:var(--blue); font-weight:700; width:24px; }
.gmap .gm-name { color:var(--text); }
.gmap .gm-action { text-align:right; font-weight:600; }
.gmap .gm-action.active  { color:var(--green); }
.gmap .gm-action.scan    { color:var(--amber); }
.gmap .gm-action.stop    { color:var(--red); }
.gmap .gm-action.dim     { color:var(--muted); }

/* Hotkeys */
.hkeys { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.hk {
  background:#111820; border:1px solid var(--border2);
  border-radius:3px; padding:3px 8px;
  font-size:10px; color:var(--text); font-family:var(--mono);
}
.hk span { color:var(--muted); font-size:9px; margin-left:4px; }

/* Scan status bar */
.scan-bar {
  display:none;
  background: #1a1200;
  border:1px solid var(--amber);
  border-radius:4px;
  padding:8px 12px;
  font-size:11px; color:var(--amber);
  align-items:center; gap:8px;
}
.scan-bar.active { display:flex; }
.scan-dot { width:6px; height:6px; border-radius:50%; background:var(--amber); animation:pulse-amber 1s infinite; }
@keyframes pulse-amber {
  0%,100% { opacity:1; } 50% { opacity:0.3; }
}

/* ── Responsive ─────────────────────────────────── */
@media (max-width:900px) {
  .layout { grid-template-columns:1fr; }
  .video-wrap { grid-column:1; grid-row:1; }
  .side { grid-column:1; }
  .gesture-bar { grid-template-columns:1fr 1fr 1fr; }
}
@media (max-width:480px) {
  .hdr { padding:10px 14px; }
  .layout { padding:8px; gap:8px; }
  .gesture-bar { grid-template-columns:1fr 1fr; }
  .g-cell:last-child { display:none; }
  .btn-row { flex-direction:column; }
}
</style>
</head>
<body>

<!-- Header -->
<header class="hdr">
  <div class="hdr-left">
    <h1>Nova 5 &mdash; <em>Gesture</em> Control</h1>
    <div class="hdr-sub">MediaPipe · TCP/IP · digiCamControl</div>
  </div>
  <div class="hdr-right">
    <span id="sysTime"></span>
    <div class="badge" id="trackBadge"><div class="pulse" id="pulseDot"></div><span id="trackLabel">LIVE</span></div>
  </div>
</header>

<!-- Main layout -->
<div class="layout">

  <!-- Left: Video + gesture bar -->
  <div class="video-wrap">
    <div class="video-box">
      <img id="streamImg" src="/stream" alt="feed" onload="streamOk()" onerror="streamErr()">
      <div class="vid-corner tl"></div>
      <div class="vid-corner tr"></div>
      <div class="vid-corner bl"></div>
      <div class="vid-corner br"></div>
      <div class="vid-badge" id="streamStatus">LIVE</div>
      <div class="vid-overlay hidden" id="vidOverlay">
        <svg width="32" height="32" fill="none" stroke="#4a5568" stroke-width="1.5" viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="15" rx="2"/><path d="M16 3l-4 4-4-4"/></svg>
        <span>Reconnecting feed…</span>
      </div>
    </div>

    <div class="scan-bar" id="scanBar">
      <div class="scan-dot"></div>
      SCANNING — <span id="scanPos">Position 1</span>
    </div>

    <div class="gesture-bar">
      <div class="g-cell">
        <div class="gc-label">Gesture ID</div>
        <div class="gc-val" id="gNum">—</div>
        <div class="gc-sub" id="gName">No hand</div>
      </div>
      <div class="g-cell">
        <div class="gc-label">Confidence</div>
        <div class="gc-val" id="gConf" style="color:var(--blue)">—</div>
        <div class="gc-sub" id="gMethod">—</div>
      </div>
      <div class="g-cell">
        <div class="gc-label">Robot Action</div>
        <div class="gc-val" id="gPreset">—</div>
        <div class="gc-sub" id="gPresetName">—</div>
      </div>
    </div>
  </div>

  <!-- Right: Control panels -->
  <div class="side">

    <!-- Tracking control -->
    <div class="card">
      <div class="card-title">Tracking Control</div>
      <div class="btn-row" style="margin-bottom:8px">
        <button class="btn btn-red" id="btnTrack" onclick="toggleTracking()">■ Stop</button>
        <button class="btn btn-ghost" onclick="emergencyStop()">⚡ E-Stop</button>
      </div>
      <div class="hkeys">
        <div class="hk">SPACE<span>toggle</span></div>
        <div class="hk">S<span>e-stop</span></div>
        <div class="hk">ESC<span>exit</span></div>
      </div>
    </div>

    <!-- Robot status -->
    <div class="card">
      <div class="card-title">Robot Status</div>
      <div class="srow"><span class="slabel">Connection</span><span class="sval" id="rConn"><span class="dot r"></span>—</span></div>
      <div class="srow"><span class="slabel">Enabled</span><span class="sval" id="rEnabled">—</span></div>
      <div class="srow"><span class="slabel">Current Preset</span><span class="sval" id="rPreset" style="color:var(--green)">—</span></div>
      <div class="srow"><span class="slabel">Robot IP</span><span class="sval" id="rIp" style="color:var(--muted)">—</span></div>
      <div class="btn-row" style="margin-top:10px">
        <button class="btn btn-green" onclick="enableRobot()" style="font-size:10px">Enable</button>
        <button class="btn btn-red"   onclick="disableRobot()" style="font-size:10px">Disable</button>
      </div>
    </div>

    <!-- Canon camera -->
    <div class="card">
      <div class="card-title">Canon EOS 6D</div>
      <div class="srow"><span class="slabel">Connection</span><span class="sval" id="cConn"><span class="dot r"></span>Offline</span></div>
      <div class="srow"><span class="slabel">Method</span><span class="sval" id="cMethod" style="color:var(--muted)">—</span></div>
      <div class="srow"><span class="slabel">Photos Taken</span><span class="sval" id="cCount" style="color:var(--green)">0</span></div>
      <div class="srow" id="cErrRow" style="display:none"><span class="slabel" style="color:var(--red)">Error</span><span class="sval" id="cErr" style="color:var(--red);font-size:10px;max-width:160px;overflow:hidden;text-overflow:ellipsis;display:block">—</span></div>
      <div class="btn-row" style="margin-top:10px">
        <button class="btn btn-amber" onclick="manualCapture()" style="font-size:10px">📷 Capture</button>
        <button class="btn btn-blue" id="btnReconn" onclick="reconnectCamera()" style="font-size:10px">↺ Reconnect</button>
      </div>
    </div>

    <!-- Gesture map -->
    <div class="card">
      <div class="card-title">Gesture Map</div>
      <table class="gmap" id="gmapTable"></table>
    </div>

  </div><!-- /side -->
</div><!-- /layout -->

<script>
// ── State ────────────────────────────────────────
let isTracking = true;
let streamRetries = 0;
let streamTimer = null;

// ── Clock ────────────────────────────────────────
function tick() {
  const n = new Date();
  document.getElementById('sysTime').textContent =
    n.toTimeString().slice(0,8);
}
tick(); setInterval(tick, 1000);

// ── Stream watchdog ───────────────────────────────
function streamOk() {
  streamRetries = 0;
  document.getElementById('vidOverlay').classList.add('hidden');
  document.getElementById('streamStatus').textContent = 'LIVE';
  document.getElementById('streamStatus').style.color = 'var(--green)';
  document.getElementById('streamStatus').style.borderColor = 'var(--green)';
  if (streamTimer) { clearTimeout(streamTimer); streamTimer = null; }
}
function streamErr() {
  document.getElementById('vidOverlay').classList.remove('hidden');
  document.getElementById('streamStatus').textContent = 'NO FEED';
  document.getElementById('streamStatus').style.color = 'var(--red)';
  document.getElementById('streamStatus').style.borderColor = 'var(--red)';
  // Retry with cache-bust
  streamRetries++;
  const delay = Math.min(1000 * streamRetries, 5000);
  streamTimer = setTimeout(() => {
    const img = document.getElementById('streamImg');
    img.src = '/stream?' + Date.now();
  }, delay);
}

// ── Tracking ──────────────────────────────────────
function toggleTracking() {
  const url = isTracking ? '/tracking/stop' : '/tracking/start';
  fetch(url, {method:'POST'}).then(r=>r.json()).then(() => setTracking(!isTracking));
}
function setTracking(on) {
  isTracking = on;
  const btn = document.getElementById('btnTrack');
  const dot = document.getElementById('pulseDot');
  const lbl = document.getElementById('trackLabel');
  if (on) {
    btn.className = 'btn btn-red';
    btn.textContent = '■ Stop';
    dot.className = 'pulse';
    lbl.textContent = 'LIVE';
  } else {
    btn.className = 'btn btn-green';
    btn.textContent = '► Start';
    dot.className = 'pulse off';
    lbl.textContent = 'PAUSED';
  }
}
function emergencyStop() {
  fetch('/robot/stop', {method:'POST'});
  fetch('/tracking/stop', {method:'POST'}).then(() => setTracking(false));
}
function enableRobot()  { fetch('/robot/enable',  {method:'POST'}); }
function disableRobot() { fetch('/robot/disable', {method:'POST'}); }

// ── Camera ────────────────────────────────────────
function manualCapture() {
  fetch('/camera/capture', {method:'POST'}).then(r=>r.json()).then(d => {
    if (d.status === 'captured') {
      document.getElementById('cCount').textContent = d.count;
      flashElement('cCount');
    }
  }).catch(() => {});
}
function reconnectCamera() {
  const btn = document.getElementById('btnReconn');
  btn.textContent = '… Trying'; btn.disabled = true;
  fetch('/camera/reconnect', {method:'POST'}).then(r=>r.json()).then(d => {
    btn.disabled = false;
    btn.textContent = d.connected ? '✓ Connected' : '↺ Reconnect';
    setTimeout(() => { btn.textContent = '↺ Reconnect'; }, 3000);
  }).catch(() => { btn.disabled=false; btn.textContent='↺ Reconnect'; });
}
function flashElement(id) {
  const el = document.getElementById(id);
  el.style.transition = 'color 0.1s';
  el.style.color = '#fff';
  setTimeout(() => { el.style.color = ''; }, 400);
}

// ── Poll detection ────────────────────────────────
function poll() {
  fetch('/detection').then(r=>r.json()).then(d => {
    const hand = d.hand_detected;

    // Gesture panel
    document.getElementById('gNum').textContent  = hand ? d.gesture_id : '—';
    document.getElementById('gNum').style.color  = hand ? 'var(--green)' : 'var(--muted)';
    document.getElementById('gName').textContent = hand ? d.gesture_name : 'No hand detected';
    document.getElementById('gConf').textContent = hand ? Math.round((d.confidence||0)*100)+'%' : '—';
    document.getElementById('gMethod').textContent = d.method || '—';

    // Preset / action
    const preset = d.robot_preset;
    let pLabel = '—', pSub = '—', pColor = 'var(--muted)';
    if (preset === 'scan') { pLabel='SCAN'; pSub='Oscillating'; pColor='var(--amber)'; }
    else if (preset === 'camera') { pLabel='📷'; pSub='Capturing'; pColor='var(--blue)'; }
    else if (preset) { pLabel='P'+preset; pSub='Moving'; pColor='var(--green)'; }
    document.getElementById('gPreset').textContent = pLabel;
    document.getElementById('gPreset').style.color = pColor;
    document.getElementById('gPresetName').textContent = pSub;

    // Scan bar
    const scanBar = document.getElementById('scanBar');
    if (d.scan_active) {
      scanBar.classList.add('active');
      document.getElementById('scanPos').textContent = d.scan_position === 'pos2' ? 'Position 2' : 'Position 1';
    } else {
      scanBar.classList.remove('active');
    }

    // Tracking state sync
    if (d.method === 'paused' && isTracking) setTracking(false);
    if (d.method !== 'paused' && !isTracking) setTracking(true);

    // Robot status
    if (d.robot) {
      const r = d.robot;
      document.getElementById('rConn').innerHTML = r.connected
        ? '<span class="dot g"></span>Connected'
        : '<span class="dot r"></span>Disconnected';
      document.getElementById('rEnabled').innerHTML = r.enabled
        ? '<span class="dot g"></span>Enabled'
        : '<span class="dot y"></span>Disabled';
      const pn = r.current_preset;
      document.getElementById('rPreset').textContent = pn === 'scan' ? 'SCANNING' : pn ? '→ P'+pn : 'Idle';
      document.getElementById('rIp').textContent = r.ip || '—';
    }
  }).catch(() => {});
}

// ── Poll camera ───────────────────────────────────
function pollCamera() {
  fetch('/camera/status').then(r=>r.json()).then(d => {
    document.getElementById('cConn').innerHTML = d.connected
      ? '<span class="dot g"></span>Connected'
      : '<span class="dot r"></span>Offline';
    document.getElementById('cMethod').textContent = d.capture_method || '—';
    document.getElementById('cCount').textContent = d.capture_count || 0;
    const errRow = document.getElementById('cErrRow');
    if (d.last_error) {
      document.getElementById('cErr').textContent = d.last_error;
      errRow.style.display = 'flex';
    } else {
      errRow.style.display = 'none';
    }
  }).catch(() => {});
}

// ── Gesture map ───────────────────────────────────
const GNAMES = {1:'Index',2:'Index+Mid',3:'Idx+Mid+Ring',4:'Idx+Mid+Rng+Pnk',
  5:'All Fingers',6:'Thumb',7:'Thumb+Idx',8:'Thm+Idx+Mid',9:'Thm+Idx+Mid+Rng',10:'Fist'};

function loadMap() {
  fetch('/config/gesture_map').then(r=>r.json()).then(m => {
    const overrides = {5:'STOP',10:'SCAN'};
    const tbl = document.getElementById('gmapTable');
    let html = '';
    for (let i=1;i<=10;i++) {
      const raw = m.gesture_to_preset ? m.gesture_to_preset[String(i)] : null;
      let action = '—', cls = 'dim';
      if (overrides[i]) { action=overrides[i]; cls=i===5?'stop':'scan'; }
      else if (raw && !isNaN(raw)) {
        const nm = m.presets && m.presets[String(raw)] ? m.presets[String(raw)] : 'P'+raw;
        action=nm; cls='active';
      }
      html += `<tr><td class="gm-id">${i}</td><td class="gm-name">${GNAMES[i]||'?'}</td><td class="gm-action ${cls}">${action}</td></tr>`;
    }
    tbl.innerHTML = html;
  });
}

// ── Keyboard ──────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); toggleTracking(); }
  if (e.key === 's' || e.key === 'S') emergencyStop();
});

// ── Start ─────────────────────────────────────────
setInterval(poll, 150);
setInterval(pollCamera, 2500);
loadMap();
</script>
</body>
</html>"""
    return html


@flask_app.route("/tracking/start", methods=["POST"])
def api_tracking_start():
    start_tracking()
    return jsonify({"tracking": True})

@flask_app.route("/tracking/stop", methods=["POST"])
def api_tracking_stop():
    stop_tracking()
    return jsonify({"tracking": False})

@flask_app.route("/tracking/status", methods=["GET"])
def api_tracking_status():
    return jsonify({"tracking": tracking_active})

@flask_app.route("/status", methods=["GET"])
def api_status():
    rs = robot.get_status(PRESETS) if robot and robot.connected else {"connected":False,"dry_run":dry_run}
    cs = canon.get_status() if canon else {"connected":False}
    return jsonify({"running":is_running,"tracking":tracking_active,"detection_method":DETECTION_METHOD,
                    "robot":rs,"canon":cs,"gestures":GESTURE_ID_TO_NAME,
                    "gesture_to_preset":{str(k):v for k,v in GESTURE_TO_PRESET.items()},
                    "presets":{str(k):v["name"] for k,v in PRESETS.items()}})

@flask_app.route("/detection", methods=["GET"])
def api_detection():
    d = detector.get_latest_detection() if detector else {}
    d["tracking"] = tracking_active
    d["scan_active"] = scanner.active
    d["scan_position"] = scanner.current_pos
    if robot: d["robot"] = robot.get_status(PRESETS)
    return jsonify(d)

@flask_app.route("/stream", methods=["GET"])
def api_stream():
    def gen():
        while is_running:
            if detector:
                frame = detector.get_latest_frame()
                if frame is not None:
                    annotated = detector.draw_overlay(
                        frame,
                        tracking_active=tracking_active,
                        robot=robot,
                        canon=canon,
                        scan_active=scanner.active,
                        scan_current_pos=scanner.current_pos,
                        countdown_state=_countdown_state,
                        dry_run=dry_run,
                        safety_state={
                            "unlocked":       _safety_unlocked,
                            "hold_sec":       (time.time() - _safety_gate5_start) if _safety_gate5_start > 0 else 0.0,
                            "hold_total":     SAFETY_HOLD_SEC,
                            "timeout_left":   max(0.0, SAFETY_TIMEOUT - (time.time() - _safety_unlocked_at)) if _safety_unlocked else 0.0,
                            "timeout_total":  SAFETY_TIMEOUT,
                            "presets":        PRESETS,
                            "gesture_map":    GESTURE_TO_PRESET,
                            "debounce_ratio": _debounce_counter / DEBOUNCE_FRAMES if DEBOUNCE_FRAMES > 0 else 0.0,
                            "debounce_gid":   _debounce_last_gesture,
                        },
                    )
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/robot/stop", methods=["POST"])
def api_stop():
    if scanner.active: scanner.stop()
    if robot and robot.connected: robot.stop(); return jsonify({"status":"stopped"})
    return jsonify({"error":"not connected"}),503

@flask_app.route("/robot/enable", methods=["POST"])
def api_enable():
    if robot and robot.connected: robot.initialize(); return jsonify({"status":"enabled"})
    return jsonify({"error":"not connected"}),503

@flask_app.route("/robot/disable", methods=["POST"])
def api_disable():
    if robot and robot.connected: robot.disable(); return jsonify({"status":"disabled"})
    return jsonify({"error":"not connected"}),503

@flask_app.route("/robot/preset", methods=["POST"])
def api_preset():
    if not robot or not robot.connected: return jsonify({"error":"not connected"}),503
    data = request.get_json(silent=True) or {}
    preset_num = data.get("preset")
    if preset_num:
        success = robot.move_to_preset(int(preset_num), PRESETS)
        return jsonify({"status":"ok" if success else "failed","preset":preset_num})
    return jsonify({"error":"invalid preset"}),400

@flask_app.route("/config/gesture_map", methods=["GET"])
def api_map():
    return jsonify({
        "gesture_to_preset": {str(k):v for k,v in GESTURE_TO_PRESET.items()},
        "presets": {str(k):v["name"] for k,v in PRESETS.items()}
    })

# ── Canon Camera API Endpoints ──────────────────────────────────

@flask_app.route("/camera/status", methods=["GET"])
def api_camera_status():
    if canon:
        return jsonify(canon.get_status())
    return jsonify({"connected": False, "error": "Camera not initialized"}), 503

@flask_app.route("/camera/reconnect", methods=["POST"])
def api_camera_reconnect():
    """Try to reconnect to digiCamControl — will auto-launch it if not running."""
    if not canon:
        return jsonify({"error": "Camera not initialized"}), 503
    canon._last_reconnect_attempt = 0
    canon.connected = False
    success = canon.try_reconnect()
    return jsonify({"connected": success, "error": canon.last_error})

@flask_app.route("/camera/capture", methods=["POST"])
def api_camera_capture():
    """Manually trigger a photo capture from the web dashboard."""
    if not canon or not canon.connected:
        return jsonify({"error": "Camera not connected"}), 503
    filepath = canon.capture()
    if filepath:
        return jsonify({"status": "captured", "path": filepath, "count": canon.capture_count})
    return jsonify({"error": "Capture failed"}), 500

@flask_app.route("/camera/last", methods=["GET"])
def api_camera_last_photo():
    """Return the last captured photo as JPEG (for dashboard preview)."""
    if not canon or not canon.last_capture_path:
        return jsonify({"error": "No photos captured yet"}), 404
    try:
        import urllib.request as _urlreq
        preview_url = f"{canon.base_url}/preview.jpg"
        try:
            with _urlreq.urlopen(preview_url, timeout=5) as resp:
                img_data = resp.read()
                return jsonify({
                    "frame": base64.b64encode(img_data).decode(),
                    "path": canon.last_capture_path or "unknown",
                    "count": canon.capture_count
                })
        except Exception:
            pass

        if os.path.exists(canon.last_capture_path):
            img = cv2.imread(canon.last_capture_path)
            if img is not None:
                h, w = img.shape[:2]
                scale = min(800/w, 600/h)
                preview = cv2.resize(img, (int(w*scale), int(h*scale)))
                _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jsonify({
                    "frame": base64.b64encode(buf).decode(),
                    "path": canon.last_capture_path,
                    "count": canon.capture_count
                })

        return jsonify({"error": "Preview not available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/shutdown", methods=["POST"])
def api_shutdown():
    global is_running; is_running = False; return jsonify({"status":"shutting down"})


def run_flask(port):
    flask_app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    global DOBOT_IP, dry_run, robot, canon, is_running
    global INITIAL_POSE, PRESETS, GESTURE_TO_PRESET
    global CAMERA_SOURCE, detector

    parser = argparse.ArgumentParser(description="Finger Gesture + Dobot Nova 5 + Canon EOS 6D (All-in-One)")
    parser.add_argument("--no-robot",   action="store_true", help="Vision only, no robot")
    parser.add_argument("--no-camera",  action="store_true", help="Disable Canon camera capture")
    parser.add_argument("--ip",         default=DOBOT_IP, help=f"Dobot IP (default: {DOBOT_IP})")
    parser.add_argument("--port",       type=int, default=FLASK_PORT, help=f"Web port (default: {FLASK_PORT})")
    parser.add_argument("--photo-dir",  default=CANON_SAVE_DIR, help="Photo save directory")
    parser.add_argument("--presets-json", default=DOBOTSTUDIO_PRESETS_JSON,
                        help=f"Path to DobotStudio Pro presets.json")
    parser.add_argument("--camera", choices=["laptop", "realsense"], default=CAMERA_SOURCE,
                        help="Camera source: 'laptop' (default webcam) or 'realsense' (Intel RealSense SR305)")
    args = parser.parse_args()

    CAMERA_SOURCE = args.camera
    DOBOT_IP      = args.ip
    dry_run       = args.no_robot

    print()
    print("═"*64)
    print("  Finger Gesture → Dobot Nova 5 + Canon EOS 6D")
    print("  Auto-start │ Web dashboard │ Keyboard hotkeys")
    print("═"*64)
    print()

    # ── Load presets from DobotStudio Pro presets.json ────────────
    print(f"  Loading presets from: {args.presets_json}")
    INITIAL_POSE, PRESETS = load_presets_from_json(args.presets_json)
    GESTURE_TO_PRESET = build_gesture_map(PRESETS)
    print(f"  Gesture map: { {k: (PRESETS[v]['name'] if isinstance(v,int) and v in PRESETS else v) for k,v in GESTURE_TO_PRESET.items() if v} }")
    print()

    # ── Gesture detector ──────────────────────────────────────────
    detector = GestureDetector(
        camera_index=CAMERA_INDEX,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        camera_source=CAMERA_SOURCE,
        detection_method=DETECTION_METHOD,
        model_path=MODEL_PATH,
        sklearn_model_path=SKLEARN_MODEL_PATH,
        mp_max_hands=MP_MAX_HANDS,
        mp_detection_conf=MP_DETECTION_CONF,
        mp_tracking_conf=MP_TRACKING_CONF,
    )

    # ── Robot setup ───────────────────────────────────────────────
    if dry_run:
        print("\n  [DRY RUN] --no-robot: commands printed, not sent")
        robot = DobotController(DOBOT_IP, DASHBOARD_PORT, MOVE_PORT)
        robot.connected = True; robot.enabled = True
        def fake_move_to_joints(joints, name="Custom"):
            joint_str = ",".join([f"{j:.2f}" for j in joints])
            print(f"  [DRY RUN] JointMovJ({joint_str})  # {name}")
            robot.last_move_time = time.time()
            return True
        robot.move_to_joints = fake_move_to_joints
    else:
        print(f"\n  Connecting to Dobot Nova at {DOBOT_IP}...")
        robot = DobotController(DOBOT_IP, DASHBOARD_PORT, MOVE_PORT)
        if robot.connect():
            robot.initialize()
        else:
            print("  Running in vision-only mode (robot offline)")

    # ── Canon EOS 6D Setup ────────────────────────────────────────
    if CANON_ENABLED and not args.no_camera:
        print(f"\n  Connecting to Canon EOS 6D via digiCamControl...")
        canon = CanonCamera(save_dir=args.photo_dir, digicam_url=DIGICAM_URL)
        if DIGICAM_AUTOLAUNCH and not canon._is_digicam_running():
            print("  [CANON] digiCamControl not running — attempting auto-launch...")
            canon._launch_digicam()
        if canon.detect():
            print(f"  [CANON] ✓ Camera ready! Photos → {canon.save_dir}")
            print(f"  [CANON] Auto-shutter fires after every preset move (countdown timer)")
        else:
            print("  [CANON] ⚠ Camera not detected at startup — will retry before each capture")
    else:
        print("\n  [CANON] Camera disabled (--no-camera or CANON_ENABLED=False)")

    # ── Start threads ─────────────────────────────────────────────
    is_running = True

    # Override detection analyze callback to also process robot
    _orig_analyze = detector._analyze_frame

    def _patched_analyze(frame, tracking_active=True):
        det = _orig_analyze(frame, tracking_active)
        active_preset = process_robot(det.get("gesture_id") if det.get("hand_detected") else None)
        if det:
            det["robot_preset"] = active_preset
        return det

    detector._analyze_frame = _patched_analyze

    detector.start(tracking_active_fn=lambda: tracking_active)

    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()

    key_thread = threading.Thread(target=keyboard_listener, daemon=True)
    key_thread.start()

    print()
    print(f"  ✓ Detection auto-started")
    print(f"  ✓ Dashboard → http://localhost:{args.port}")
    print(f"  ✓ Stream    → http://localhost:{args.port}/stream")
    if canon and canon.connected:
        print(f"  ✓ Canon EOS 6D → auto-capture after each preset countdown (method: {canon.capture_method})")
    print()
    print("  ┌────────────────────────────────────┐")
    print("  │  Keyboard Controls:                 │")
    print("  │    [SPACE]  Toggle tracking ON/OFF  │")
    print("  │    [S]      Emergency stop robot    │")
    print("  │    [ESC]    Exit program             │")
    print("  └────────────────────────────────────┘")
    print()

    # Main loop — keyboard_listener will break on ESC
    try:
        while is_running:
            if KEYBOARD_AVAILABLE and not key_thread.is_alive():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  [!] Ctrl+C detected")
    finally:
        is_running = False
        time.sleep(0.3)
        if scanner.active: scanner.stop()
        if detector: detector.stop()
        if robot and not dry_run: robot.disconnect()
        if KEYBOARD_AVAILABLE:
            try: keyboard.unhook_all()
            except: pass
        print("  ✓ Shutdown complete.")


if __name__ == "__main__":
    main()
