import time
import threading
import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from detection import GestureDetector
from dobot_controller import DobotController, load_presets, find_reset_position_index
from camera_connection import CanonCamera

# ── Konfigurasi ──────────────────────────────────────────────────────
DOBOT_IP   = "192.168.1.6"
DOBOT_PORT = 29999
JSON_FILE  = "presets.json"
FLASK_PORT = 5001

GESTURE_STABLE_SEC = 2.0
ROBOT_MOVE_TIMEOUT = 30.0
COUNTDOWN_SECONDS = 5

RESET_POSITION_NAME = "InitialPose"

GESTURE_TO_PRESET = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}

CANON_CONFIG = {
    "CANON_ENABLED": True,
    "CANON_SAVE_DIR": r"C:\CapturedPhotos",
    "CANON_CAPTURE_GESTURE": 7,
    "CANON_CAPTURE_COOLDOWN": 90,
    "DIGICAM_URL": "http://localhost:5513",
    "DIGICAM_CAPTURE_TIMEOUT": 15,
    "DIGICAM_EXE": r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe",
    "DIGICAM_APP_EXE": r"C:\Program Files (x86)\digiCamControl\digiCamControl.exe",
    "DIGICAM_AUTOLAUNCH": True,
    "DIGICAM_LAUNCH_WAIT": 8,
}

# ── Global Objects & State Machine ───────────────────────────────────
app = Flask(__name__, template_folder='templates')
CORS(app)

detector: GestureDetector | None = None
robot: DobotController | None = None
canon: CanonCamera | None = None
presets: list = []
reset_preset: dict | None = None
reset_index: int | None = None
tracking_active = True

# State machine
class AppState:
    IDLE = "IDLE"
    MOVING = "MOVING"
    COUNTDOWN = "COUNTDOWN"

current_state = AppState.IDLE
state_info = {}
state_lock = threading.Lock()
shutdown_event = threading.Event()

# ── Web Routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global detector
    while True:
        if detector and detector.is_running():
            frame = detector.get_latest_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1/30)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    detection_status = detector.get_status() if detector else {}
    robot_status = robot.get_status() if robot else {}
    with state_lock:
        state_data = {
            "app_state": current_state,
            "state_info": state_info
        }
    return jsonify({
        "detection": detection_status,
        "robot": robot_status,
        "tracking_active": tracking_active,
        "presets": {i+1: p['name'] for i, p in enumerate(presets)},
        "state": state_data
    })

@app.route('/control', methods=['POST'])
def control():
    global tracking_active, robot
    action = request.json.get('action')
    if action == 'toggle_tracking':
        tracking_active = not tracking_active
        print(f"[WEB] Tracking {'activated' if tracking_active else 'deactivated'}")
        return jsonify(success=True, tracking_active=tracking_active)
    elif action == 'emergency_stop':
        if robot:
            robot.shutdown()
            print("[WEB] EMERGENCY STOP ACTIVATED")
        return jsonify(success=True, message="Emergency stop activated")
    return jsonify(success=False, message="Invalid action"), 400

@app.route('/capture', methods=['POST'])
def capture():
    if canon and canon.enabled and canon.is_ready_for_capture():
        success = canon.capture_photo()
        return jsonify(success=success)
    return jsonify(success=False, message="Camera not ready or on cooldown"), 429

# ── Main Logic Thread ────────────────────────────────────────────────
def main_logic():
    global detector, robot, canon, presets, reset_preset, reset_index, tracking_active
    global current_state, state_info

    # 1. Initialization
    presets = load_presets(JSON_FILE)
    reset_index = find_reset_position_index(presets, RESET_POSITION_NAME)
    if reset_index is None:
        print(f"[MAIN] ❌ Reset position '{RESET_POSITION_NAME}' not found!")
        return
    reset_preset = presets[reset_index]
    
    robot = DobotController(DOBOT_IP, DOBOT_PORT)
    if not robot.connect(): return
    robot.initialize()
    robot.set_reset_position(reset_index, reset_preset['name'])
    
    canon = CanonCamera(CANON_CONFIG)
    canon.detect()
    
    print(f"\n[MAIN] Initializing: Moving to reset position ({reset_preset['name']})...")
    robot.move_to_preset(reset_preset['joint_str'], reset_preset['name'], reset_index)
    robot.wait_until_done(ROBOT_MOVE_TIMEOUT)
    
    detector = GestureDetector(camera_index=0, min_confidence=0.55)
    detector.start()
    print(f"[MAIN] Main logic loop running. Open http://localhost:{FLASK_PORT}")

    last_sent_gesture = None

    while not shutdown_event.is_set():
        try:
            if not tracking_active or current_state != AppState.IDLE:
                time.sleep(0.1)
                continue

            gesture = detector.get_stable_gesture(GESTURE_STABLE_SEC)
            if gesture is None or gesture == last_sent_gesture:
                time.sleep(0.1)
                continue
            if gesture not in GESTURE_TO_PRESET:
                continue

            preset_idx = GESTURE_TO_PRESET[gesture] - 1
            if not (0 <= preset_idx < len(presets)):
                continue
            
            preset = presets[preset_idx]
            
            # ── Start State Transition: IDLE -> MOVING ──
            with state_lock:
                current_state = AppState.MOVING
                state_info = {"preset_name": preset['name'], "preset_idx": preset_idx}
            
            print(f"\n[MAIN] ✅ Gesture '{gesture}' stable → moving to {preset['name']}")
            detector.set_robot_busy(True)

            success = robot.move_to_preset_with_reset(
                preset=preset,
                preset_index=preset_idx,
                reset_preset=reset_preset,
                timeout=ROBOT_MOVE_TIMEOUT
            )
            detector.set_robot_busy(False)

            if success:
                # ── State Transition: MOVING -> COUNTDOWN ──
                with state_lock:
                    current_state = AppState.COUNTDOWN
                    state_info['start_time'] = time.time()
                
                print(f"[MAIN] ✓ Move complete. Starting {COUNTDOWN_SECONDS}s countdown.")
                detector.start_countdown(COUNTDOWN_SECONDS)
                
                # Non-blocking wait for countdown
                countdown_start_time = time.time()
                while time.time() - countdown_start_time < COUNTDOWN_SECONDS:
                    time.sleep(0.1)

                if canon.connected:
                    print("[MAIN] Countdown finished. Triggering camera.")
                    canon.capture()
                
                last_sent_gesture = gesture
                print(f"[MAIN] ✓ Photo taken. Returning to idle state.\n")
            else:
                print(f"[MAIN] ❌ Failed to move to preset {preset['name']}\n")

            # ── State Transition: -> IDLE ──
            with state_lock:
                current_state = AppState.IDLE
                state_info = {}

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[MAIN] An error occurred: {e}")
            with state_lock:
                current_state = AppState.IDLE
            time.sleep(1)

    # Cleanup
    if detector: detector.stop()
    if robot: robot.shutdown()
    print("[MAIN] ✓ Main logic stopped.")

if __name__ == "__main__":
    logic_thread = threading.Thread(target=main_logic, daemon=False)
    logic_thread.start()
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C detected. Initiating graceful shutdown...")
    finally:
        print("\n[MAIN] Flask server shutting down. Cleaning up resources...")
        shutdown_event.set()  # Signal the main logic thread to stop
        if robot and reset_preset and robot.client is not None:
            print("[MAIN] Returning robot to InitialPose before shutdown...")
            robot.move_to_preset(reset_preset['joint_str'], reset_preset['name'], reset_index)
            robot.wait_until_done(ROBOT_MOVE_TIMEOUT)
            robot.shutdown()
        if detector:
            detector.stop()
        
        logic_thread.join()
        print("[MAIN] ✓ Cleanup complete. Goodbye!")