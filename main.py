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
import argparse

from flask import Flask, jsonify, Response, request, render_template
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

DEBOUNCE_FRAMES    = 90   # ~3s at 30fps — must hold gesture steadily before it fires
COOLDOWN_FRAMES    = 15

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


def _robot_connected():
    return robot is not None and robot.connected


def _robot_ready():
    return _robot_connected() and robot.enabled


def _lock_safety_gate(message=None):
    global _safety_unlocked, _safety_gate5_start
    _safety_unlocked = False
    _safety_gate5_start = 0.0
    if message:
        print(message)


def _stop_scan_if_active():
    if scanner.active:
        scanner.stop()


def _start_countdown_for_preset(preset_num):
    abort_countdown(_countdown_state, _countdown_lock)
    threading.Thread(
        target=countdown_and_capture,
        args=(preset_num, canon, _countdown_state, _countdown_lock),
        daemon=True,
        name=f"countdown_preset_{preset_num}",
    ).start()


def _build_safety_overlay_state():
    return {
        "unlocked": _safety_unlocked,
        "hold_sec": (time.time() - _safety_gate5_start) if _safety_gate5_start > 0 else 0.0,
        "hold_total": SAFETY_HOLD_SEC,
        "timeout_left": max(0.0, SAFETY_TIMEOUT - (time.time() - _safety_unlocked_at)) if _safety_unlocked else 0.0,
        "timeout_total": SAFETY_TIMEOUT,
        "presets": PRESETS,
        "gesture_map": GESTURE_TO_PRESET,
        "debounce_ratio": _debounce_counter / DEBOUNCE_FRAMES if DEBOUNCE_FRAMES > 0 else 0.0,
        "debounce_gid": _debounce_last_gesture,
    }


def _json_error(message, status=400):
    return jsonify({"error": message}), status


def _robot_status_payload():
    return robot.get_status(PRESETS) if _robot_connected() else {"connected": False, "dry_run": dry_run}


def _canon_status_payload():
    return canon.get_status() if canon else {"connected": False}


# ═══════════════════════════════════════════════════════════════════
#  TRACKING CONTROL
# ═══════════════════════════════════════════════════════════════════

def reset_debounce():
    global _debounce_counter, _debounce_last_gesture, _cooldown_counter
    _debounce_counter = 0
    _debounce_last_gesture = None
    _cooldown_counter = 0


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
        _stop_scan_if_active()
        abort_countdown(_countdown_state, _countdown_lock)
        if _robot_ready():
            robot.stop()
        print("  [TRACKING] ■ Stopped")


def toggle_tracking():
    if tracking_active:
        stop_tracking()
    else:
        start_tracking()


def emergency_stop():
    stop_tracking()
    _stop_scan_if_active()
    abort_countdown(_countdown_state, _countdown_lock)
    if _robot_connected():
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
    global _debounce_counter, _debounce_last_gesture, _cooldown_counter
    global _safety_unlocked, _safety_gate5_start, _safety_unlocked_at

    robot_ok = _robot_ready()

    if _cooldown_counter > 0:
        _cooldown_counter -= 1
        return robot.current_preset if robot_ok else None

    if gesture_id is None:
        _debounce_counter = 0
        _debounce_last_gesture = None
        _safety_gate5_start = 0.0   # reset hold timer when hand leaves frame
        return None

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
        _lock_safety_gate("  [SAFETY] Gate auto-locked (timeout)")

    # Gate still locked → block all robot actions
    if not _safety_unlocked:
        return None
    # ── End safety gate ─────────────────────────────────────────────

    if gesture_id == _debounce_last_gesture:
        _debounce_counter += 1
    else:
        _debounce_counter = 1
        _debounce_last_gesture = gesture_id

    if _debounce_counter < DEBOUNCE_FRAMES:
        return robot.current_preset if robot_ok else None

    action = GESTURE_TO_PRESET.get(gesture_id)
    _debounce_counter = 0

    if action == "scan":
        if robot_ok and not scanner.active:
            scanner.start(robot)
            print("  [GESTURE] Fist → START CONTINUOUS SCAN")
        _cooldown_counter = COOLDOWN_FRAMES * 3
        _lock_safety_gate("  [SAFETY] Gate locked after action")
        return "scan"

    if action == "stop_scan":
        if scanner.active:
            scanner.stop()
            print("  [GESTURE] All Fingers → STOP SCAN")
        _cooldown_counter = COOLDOWN_FRAMES * 2
        _lock_safety_gate("  [SAFETY] Gate locked after action")
        return None

    if isinstance(action, int):
        if scanner.active:
            scanner.stop()
            time.sleep(0.2)

        if robot_ok:
            robot.move_to_preset(action, PRESETS)
            print(f"  [GESTURE] Gesture {gesture_id} → Robot Preset {action}")
        else:
            print(f"  [GESTURE] Gesture {gesture_id} → Preset {action} (robot offline — camera only)")

        _start_countdown_for_preset(action)
        _cooldown_counter = COUNTDOWN_COOLDOWN_FRAMES
        _lock_safety_gate("  [SAFETY] Gate locked after action")
        return action

    return robot.current_preset if robot_ok else None


# ═══════════════════════════════════════════════════════════════════
#  FLASK API + WEB DASHBOARD
# ═══════════════════════════════════════════════════════════════════

flask_app = Flask(__name__)
CORS(flask_app)


@flask_app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")

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
    return jsonify({
        "running": is_running,
        "tracking": tracking_active,
        "detection_method": DETECTION_METHOD,
        "robot": _robot_status_payload(),
        "canon": _canon_status_payload(),
        "gestures": GESTURE_ID_TO_NAME,
        "gesture_to_preset": {str(k): v for k, v in GESTURE_TO_PRESET.items()},
        "presets": {str(k): v["name"] for k, v in PRESETS.items()},
    })

@flask_app.route("/detection", methods=["GET"])
def api_detection():
    d = detector.get_latest_detection() if detector else {}
    d["tracking"] = tracking_active
    d["scan_active"] = scanner.active
    d["scan_position"] = scanner.current_pos
    if robot:
        d["robot"] = robot.get_status(PRESETS)
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
                        safety_state=_build_safety_overlay_state(),
                    )
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/robot/stop", methods=["POST"])
def api_stop():
    _stop_scan_if_active()
    if not _robot_connected():
        return _json_error("not connected", 503)
    robot.stop()
    return jsonify({"status": "stopped"})

@flask_app.route("/robot/enable", methods=["POST"])
def api_enable():
    if not _robot_connected():
        return _json_error("not connected", 503)
    robot.initialize()
    return jsonify({"status": "enabled"})

@flask_app.route("/robot/disable", methods=["POST"])
def api_disable():
    if not _robot_connected():
        return _json_error("not connected", 503)
    robot.disable()
    return jsonify({"status": "disabled"})

@flask_app.route("/robot/preset", methods=["POST"])
def api_preset():
    if not _robot_connected():
        return _json_error("not connected", 503)

    data = request.get_json(silent=True) or {}
    preset_num = data.get("preset")
    if preset_num is None:
        return _json_error("invalid preset", 400)

    try:
        preset_num = int(preset_num)
    except (TypeError, ValueError):
        return _json_error("preset must be an integer", 400)

    success = robot.move_to_preset(preset_num, PRESETS)
    return jsonify({"status": "ok" if success else "failed", "preset": preset_num})

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
    return _json_error("Camera not initialized", 503)

@flask_app.route("/camera/reconnect", methods=["POST"])
def api_camera_reconnect():
    """Try to reconnect to digiCamControl — will auto-launch it if not running."""
    if not canon:
        return _json_error("Camera not initialized", 503)
    canon._last_reconnect_attempt = 0
    canon.connected = False
    success = canon.try_reconnect()
    return jsonify({"connected": success, "error": canon.last_error})

@flask_app.route("/camera/capture", methods=["POST"])
def api_camera_capture():
    """Manually trigger a photo capture from the web dashboard."""
    if not canon:
        return _json_error("Camera not initialized", 503)
    if not canon.connected:
        canon.try_reconnect()
    if not canon.connected:
        return _json_error("Camera not connected", 503)

    filepath = canon.capture()
    if filepath:
        return jsonify({"status": "captured", "path": filepath, "count": canon.capture_count})
    return _json_error("Capture failed", 500)

@flask_app.route("/camera/last", methods=["GET"])
def api_camera_last_photo():
    """Return the last captured photo as JPEG (for dashboard preview)."""
    if not canon or not canon.last_capture_path:
        return _json_error("No photos captured yet", 404)
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
                scale = min(800 / max(1, w), 600 / max(1, h))
                preview = cv2.resize(img, (int(w*scale), int(h*scale)))
                _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return jsonify({
                    "frame": base64.b64encode(buf).decode(),
                    "path": canon.last_capture_path,
                    "count": canon.capture_count
                })

        return _json_error("Preview not available", 404)
    except Exception as e:
        return _json_error(str(e), 500)

@flask_app.route("/shutdown", methods=["POST"])
def api_shutdown():
    global is_running; is_running = False; return jsonify({"status":"shutting down"})


def run_flask(port):
    flask_app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


def _parse_args():
    parser = argparse.ArgumentParser(description="Finger Gesture + Dobot Nova 5 + Canon EOS 6D (All-in-One)")
    parser.add_argument("--no-robot", action="store_true", help="Vision only, no robot")
    parser.add_argument("--no-camera", action="store_true", help="Disable Canon camera capture")
    parser.add_argument("--ip", default=DOBOT_IP, help=f"Dobot IP (default: {DOBOT_IP})")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help=f"Web port (default: {FLASK_PORT})")
    parser.add_argument("--photo-dir", default=CANON_SAVE_DIR, help="Photo save directory")
    parser.add_argument("--presets-json", default=DOBOTSTUDIO_PRESETS_JSON,
                        help="Path to DobotStudio Pro presets.json")
    parser.add_argument("--camera", choices=["laptop", "realsense"], default=CAMERA_SOURCE,
                        help="Camera source: 'laptop' (default webcam) or 'realsense' (Intel RealSense SR305)")
    return parser.parse_args()


def _print_banner():
    print()
    print("═" * 64)
    print("  Finger Gesture → Dobot Nova 5 + Canon EOS 6D")
    print("  Auto-start │ Web dashboard │ Keyboard hotkeys")
    print("═" * 64)
    print()


def _load_runtime_presets(presets_path):
    global INITIAL_POSE, PRESETS, GESTURE_TO_PRESET
    print(f"  Loading presets from: {presets_path}")
    INITIAL_POSE, PRESETS = load_presets_from_json(presets_path)
    GESTURE_TO_PRESET = build_gesture_map(PRESETS)
    active_map = {
        k: (PRESETS[v]["name"] if isinstance(v, int) and v in PRESETS else v)
        for k, v in GESTURE_TO_PRESET.items() if v
    }
    print(f"  Gesture map: {active_map}")
    print()


def _setup_detector():
    global detector
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


def _setup_robot_controller():
    global robot
    if dry_run:
        print("\n  [DRY RUN] --no-robot: commands printed, not sent")
        robot = DobotController(DOBOT_IP, DASHBOARD_PORT, MOVE_PORT)
        robot.connected = True
        robot.enabled = True

        def fake_move_to_joints(joints, name="Custom"):
            joint_str = ",".join([f"{j:.2f}" for j in joints])
            print(f"  [DRY RUN] JointMovJ({joint_str})  # {name}")
            robot.last_move_time = time.time()
            return True

        robot.move_to_joints = fake_move_to_joints
        return

    print(f"\n  Connecting to Dobot Nova at {DOBOT_IP}...")
    robot = DobotController(DOBOT_IP, DASHBOARD_PORT, MOVE_PORT)
    if robot.connect():
        robot.initialize()
    else:
        print("  Running in vision-only mode (robot offline)")


def _setup_canon_camera(photo_dir, camera_disabled):
    global canon
    if camera_disabled:
        print("\n  [CANON] Camera disabled (--no-camera or CANON_ENABLED=False)")
        canon = None
        return

    print("\n  Connecting to Canon EOS 6D via digiCamControl...")
    canon = CanonCamera(save_dir=photo_dir, digicam_url=DIGICAM_URL)
    if DIGICAM_AUTOLAUNCH and not canon._is_digicam_running():
        print("  [CANON] digiCamControl not running — attempting auto-launch...")
        canon._launch_digicam()

    if canon.detect():
        print(f"  [CANON] ✓ Camera ready! Photos → {canon.save_dir}")
        print("  [CANON] Auto-shutter fires after every preset move (countdown timer)")
    else:
        print("  [CANON] ⚠ Camera not detected at startup — will retry before each capture")


def _patch_and_start_detector():
    _orig_analyze = detector._analyze_frame

    def _patched_analyze(frame, tracking_active=True):
        det = _orig_analyze(frame, tracking_active)
        active_preset = process_robot(det.get("gesture_id") if det.get("hand_detected") else None)
        if det:
            det["robot_preset"] = active_preset
        return det

    detector._analyze_frame = _patched_analyze
    detector.start(tracking_active_fn=lambda: tracking_active)


def _start_background_threads(port):
    flask_thread = threading.Thread(target=run_flask, args=(port,), daemon=True)
    flask_thread.start()

    key_thread = None
    if KEYBOARD_AVAILABLE:
        key_thread = threading.Thread(target=keyboard_listener, daemon=True)
        key_thread.start()
    return key_thread


def _print_runtime_summary(port):
    print()
    print("  ✓ Detection auto-started")
    print(f"  ✓ Dashboard → http://localhost:{port}")
    print(f"  ✓ Stream    → http://localhost:{port}/stream")
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


def _shutdown_runtime():
    global is_running
    is_running = False
    time.sleep(0.3)
    _stop_scan_if_active()
    if detector:
        detector.stop()
    if robot and not dry_run:
        robot.disconnect()
    if KEYBOARD_AVAILABLE:
        try:
            keyboard.unhook_all()
        except Exception:
            pass
    print("  ✓ Shutdown complete.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    global DOBOT_IP, dry_run, is_running, CAMERA_SOURCE

    args = _parse_args()
    CAMERA_SOURCE = args.camera
    DOBOT_IP = args.ip
    dry_run = args.no_robot

    _print_banner()
    _load_runtime_presets(args.presets_json)
    _setup_detector()
    _setup_robot_controller()
    _setup_canon_camera(photo_dir=args.photo_dir, camera_disabled=(not CANON_ENABLED or args.no_camera))

    is_running = True
    _patch_and_start_detector()
    key_thread = _start_background_threads(args.port)
    _print_runtime_summary(args.port)

    try:
        while is_running:
            if key_thread and not key_thread.is_alive():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  [!] Ctrl+C detected")
    finally:
        _shutdown_runtime()


if __name__ == "__main__":
    main()
