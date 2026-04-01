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

from detection import GestureDetector, GESTURE_ID_TO_NAME
from dobot_controller import (
    DobotController, ScanController,
    load_presets_from_json, build_gesture_map,
    DOBOTSTUDIO_PRESETS_JSON, _FALLBACK_INITIAL_POSE, _FALLBACK_PRESETS,
)
from camera_connection import (
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

DEBOUNCE_FRAMES    = 8
COOLDOWN_FRAMES    = 15
NO_HAND_STOP_DELAY = 10

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

    robot_ok = robot is not None and robot.connected and robot.enabled

    if _cooldown_counter > 0:
        _cooldown_counter -= 1
        return robot.current_preset if robot_ok else None

    if gesture_id is None:
        _debounce_counter = 0
        _debounce_last_gesture = None
        _no_hand_counter += 1
        return None

    _no_hand_counter = 0

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
            return "scan"

        # ── Stop continuous scan (Open Palm) ────────────────────
        elif action == "stop_scan":
            if scanner.active:
                scanner.stop()
                print(f"  [GESTURE] All Fingers → STOP SCAN")
            _debounce_counter = 0
            _cooldown_counter = COOLDOWN_FRAMES * 2
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
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Finger Gesture → Dobot Nova 5</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif}
  .header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;justify-content:space-between}
  .header h1{font-size:18px;font-weight:600} .header h1 span{color:#58a6ff}
  .header .subtitle{color:#8b949e;font-size:13px}
  .main{display:flex;gap:20px;padding:20px 24px;max-width:1200px;margin:0 auto}
  .video-panel{flex:1;min-width:0}
  .video-panel img{width:100%;border-radius:12px;border:2px solid #30363d;display:block;background:#000}
  .side-panel{width:320px;flex-shrink:0;display:flex;flex-direction:column;gap:14px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px}
  .card h3{font-size:14px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px}
  .status-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;font-size:14px}
  .status-row .label{color:#8b949e} .status-row .value{font-weight:600}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
  .dot.green{background:#3fb950;box-shadow:0 0 6px #3fb950}
  .dot.red{background:#f85149;box-shadow:0 0 6px #f85149}
  .dot.yellow{background:#d29922;box-shadow:0 0 6px #d29922}
  .btn{width:100%;padding:12px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all 0.2s;display:flex;align-items:center;justify-content:center;gap:8px}
  .btn:active{transform:scale(0.97)}
  .btn-stop{background:#da3633;color:#fff} .btn-stop:hover{background:#f85149}
  .btn-start{background:#238636;color:#fff} .btn-start:hover{background:#3fb950}
  .btn-outline{background:transparent;color:#58a6ff;border:1px solid #58a6ff} .btn-outline:hover{background:#58a6ff22}
  .btn-danger{background:transparent;color:#f85149;border:1px solid #f85149} .btn-danger:hover{background:#f8514922}
  .btn-group{display:flex;gap:8px} .btn-group .btn{flex:1}
  .gesture-grid{display:grid;grid-template-columns:40px 1fr 50px;gap:4px 10px;font-size:13px;align-items:center}
  .gesture-grid .gid{color:#58a6ff;font-weight:700;text-align:center}
  .gesture-grid .gname{color:#c9d1d9} .gesture-grid .gjog{color:#3fb950;font-family:monospace;font-weight:600;text-align:center}
  .big-gesture{text-align:center;padding:8px 0}
  .big-gesture .number{font-size:48px;font-weight:800;line-height:1}
  .big-gesture .name{font-size:16px;color:#8b949e;margin-top:4px}
  .big-gesture .jog{font-size:20px;font-weight:700;margin-top:2px}
  .hotkey{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:6px;padding:2px 8px;font-family:monospace;font-size:12px;color:#c9d1d9;margin:0 2px}
  @media(max-width:768px){.main{flex-direction:column}.side-panel{width:100%}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Finger Gesture → <span>Dobot Nova 5</span></h1>
    <div class="subtitle">YOLOv8 + MediaPipe │ TCP/IP Robot Control</div>
  </div>
  <div id="trackingBadge" style="font-size:14px;font-weight:600"><span class="dot green"></span> TRACKING</div>
</div>
<div class="main">
  <div class="video-panel"><img id="stream" src="/stream" alt="Live Stream" /></div>
  <div class="side-panel">
    <div class="card">
      <h3>Current Gesture</h3>
      <div class="big-gesture">
        <div class="number" id="gestureNum">—</div>
        <div class="name" id="gestureName">Waiting...</div>
        <div class="jog" id="gestureJog" style="color:#3fb950">—</div>
      </div>
    </div>
    <div class="card">
      <h3>Tracking Control</h3>
      <button class="btn btn-stop" id="btnTracking" onclick="toggleTracking()">■ Stop Tracking</button>
      <div style="height:8px"></div>
      <button class="btn btn-outline" onclick="emergencyStop()">Emergency Stop Robot</button>
      <div style="margin-top:12px;font-size:12px;color:#8b949e;line-height:1.6">
        Keyboard: <span class="hotkey">SPACE</span> Toggle tracking
        <span class="hotkey">S</span> Emergency stop
        <span class="hotkey">ESC</span> Exit
      </div>
    </div>
    <div class="card">
      <h3>Robot Status</h3>
      <div class="status-row"><span class="label">Connection</span><span class="value" id="robotConn">—</span></div>
      <div class="status-row"><span class="label">Enabled</span><span class="value" id="robotEnabled">—</span></div>
      <div class="status-row"><span class="label">Current Preset</span><span class="value" id="robotJog">—</span></div>
      <div class="status-row"><span class="label">IP</span><span class="value" id="robotIp">—</span></div>
      <div style="height:8px"></div>
      <div class="btn-group">
        <button class="btn btn-start" onclick="enableRobot()" style="font-size:12px;padding:8px">Enable</button>
        <button class="btn btn-danger" onclick="disableRobot()" style="font-size:12px;padding:8px">Disable</button>
      </div>
    </div>
    <div class="card">
      <h3>Gesture → Preset Map</h3>
      <div class="gesture-grid" id="gestureMap"></div>
    </div>
    <div class="card" id="cameraCard">
      <h3>Canon EOS 6D</h3>
      <div class="status-row"><span class="label">Connection</span><span class="value" id="canonConn">—</span></div>
      <div class="status-row"><span class="label">Photos Taken</span><span class="value" id="canonCount">0</span></div>
      <div class="status-row"><span class="label">Last Photo</span><span class="value" id="canonLast" style="font-size:11px;max-width:180px;overflow:hidden;text-overflow:ellipsis">—</span></div>
      <div class="status-row" id="canonErrRow" style="display:none"><span class="label" style="color:#f85149">Error</span><span class="value" id="canonErr" style="color:#f85149;font-size:11px;max-width:180px;overflow:hidden;text-overflow:ellipsis">—</span></div>
      <div style="height:8px"></div>
      <button class="btn btn-outline" onclick="manualCapture()" style="border-color:#d29922;color:#d29922">📷 Manual Capture</button>
      <div style="height:6px"></div>
      <button class="btn btn-outline" id="btnReconnect" onclick="reconnectCamera()" style="border-color:#58a6ff;color:#58a6ff;font-size:12px;padding:8px">↺ Reconnect Camera</button>
      <div style="height:8px"></div>
      <img id="lastPhoto" style="width:100%;border-radius:8px;display:none;border:1px solid #30363d" />
    </div>
  </div>
</div>
<script>
let isTracking=true;
function toggleTracking(){
  fetch(isTracking?'/tracking/stop':'/tracking/start',{method:'POST'}).then(r=>r.json()).then(()=>updateUI(!isTracking));
}
function updateUI(a){isTracking=a;const b=document.getElementById('btnTracking'),g=document.getElementById('trackingBadge');
  if(a){b.className='btn btn-stop';b.innerHTML='■ Stop Tracking';g.innerHTML='<span class="dot green"></span> TRACKING'}
  else{b.className='btn btn-start';b.innerHTML='► Start Tracking';g.innerHTML='<span class="dot red"></span> PAUSED'}}
function emergencyStop(){fetch('/robot/stop',{method:'POST'});fetch('/tracking/stop',{method:'POST'}).then(()=>updateUI(false))}
function enableRobot(){fetch('/robot/enable',{method:'POST'})}
function disableRobot(){fetch('/robot/disable',{method:'POST'})}
function manualCapture(){
  fetch('/camera/capture',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.status==='captured'){document.getElementById('canonCount').textContent=d.count;refreshLastPhoto()}
    else{alert('Capture failed: '+(d.error||'unknown'))}
  }).catch(()=>alert('Camera not available'))
}
function refreshLastPhoto(){
  fetch('/camera/last').then(r=>r.json()).then(d=>{
    if(d.frame){const img=document.getElementById('lastPhoto');img.src='data:image/jpeg;base64,'+d.frame;img.style.display='block';
    document.getElementById('canonLast').textContent=d.path.split(/[\\\/]/).pop()}
  }).catch(()=>{})
}
function poll(){fetch('/detection').then(r=>r.json()).then(d=>{
  const n=document.getElementById('gestureNum'),m=document.getElementById('gestureName'),j=document.getElementById('gestureJog');
  if(d.hand_detected){n.textContent=d.gesture_id;n.style.color='#58a6ff';m.textContent=d.gesture_name;
    if(d.robot_preset==='camera'){j.textContent='SNAP!';j.style.color='#d29922'}
    else if(d.robot_preset==='scan'){j.textContent='SCANNING';j.style.color='#ff6b35'}
    else{j.textContent=d.robot_preset?'PRESET: '+d.robot_preset:'READY';j.style.color=d.robot_preset?'#3fb950':'#f85149'}}
  else{n.textContent='—';n.style.color='#484f58';m.textContent=d.gesture_name||'No hand';j.textContent='—';j.style.color='#484f58'}
  if(d.method==='paused'&&isTracking)updateUI(false);
  if(d.method!=='paused'&&!isTracking)updateUI(true);
  if(d.robot){const r=d.robot;
    document.getElementById('robotConn').innerHTML=r.connected?'<span class="dot green"></span>Connected':'<span class="dot red"></span>Disconnected';
    document.getElementById('robotEnabled').innerHTML=r.enabled?'<span class="dot green"></span>Yes':'<span class="dot yellow"></span>No';
    document.getElementById('robotJog').textContent=r.current_preset==='scan'?'SCANNING':r.current_preset?(r.presets&&r.presets[r.current_preset]?'→ '+r.presets[r.current_preset]:'→ Preset '+r.current_preset):'None';
    document.getElementById('robotIp').textContent=r.ip||'—'}}).catch(()=>{})}
function loadMap(){fetch('/config/gesture_map').then(r=>r.json()).then(m=>{
  const names={1:'Index',2:'Index+Mid',3:'Idx+Mid+Ring',4:'Idx+Mid+Rng+Pnk',5:'All Fingers',6:'Thumb',7:'Thumb+Idx',8:'Thm+Idx+Mid',9:'Thm+Idx+Mid+Rng',10:'Fist'};
  const presets={1:'-',2:'-',3:'-',4:'-',5:'STOP',6:'-',7:'AUTO',8:'-',9:'-',10:'SCAN'};
  if(m.presets){for(const[k,v] of Object.entries(m.presets)){const s=m.gesture_to_preset[k];if(s&&!isNaN(s))presets[parseInt(k)]=v;}}
  const specials={'AUTO':'color:#8b949e;font-style:italic','SCAN':'color:#ff6b35;font-weight:700','STOP':'color:#f85149;font-weight:700'};
  const g=document.getElementById('gestureMap');g.innerHTML='';
  for(let i=1;i<=10;i++){const act=presets[i]||'-';const style=specials[act]||'';
    g.innerHTML+='<div class="gid">'+i+'</div><div class="gname">'+(names[i]||'?')+'</div><div class="gjog" style="'+style+'">'+act+'</div>'}});}
function reconnectCamera(){
  const btn=document.getElementById('btnReconnect');
  btn.textContent='🚀 Launching...';btn.disabled=true;
  fetch('/camera/reconnect',{method:'POST'}).then(r=>r.json()).then(d=>{
    btn.disabled=false;
    if(d.connected){btn.textContent='✓ Connected!';btn.style.borderColor='#3fb950';btn.style.color='#3fb950';
      setTimeout(()=>{btn.textContent='↺ Reconnect Camera';btn.style.borderColor='#58a6ff';btn.style.color='#58a6ff'},3000)
    }else{btn.textContent='✗ Failed — retry';btn.style.borderColor='#f85149';btn.style.color='#f85149';
      setTimeout(()=>{btn.textContent='↺ Reconnect Camera';btn.style.borderColor='#58a6ff';btn.style.color='#58a6ff'},4000)}
  }).catch(()=>{btn.disabled=false;btn.textContent='↺ Reconnect Camera'})
}
function pollCamera(){fetch('/camera/status').then(r=>r.json()).then(d=>{
  document.getElementById('canonConn').innerHTML=d.connected?'<span class="dot green"></span>Connected':'<span class="dot red"></span>Offline';
  document.getElementById('canonCount').textContent=d.capture_count||0;
  if(d.last_capture){document.getElementById('canonLast').textContent=d.last_capture.split(/[\\\/]/).pop()}
  const errRow=document.getElementById('canonErrRow');
  if(d.last_error){document.getElementById('canonErr').textContent=d.last_error;errRow.style.display='flex'}
  else{errRow.style.display='none'}
}).catch(()=>{})}
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();toggleTracking()}if(e.key==='s'||e.key==='S'){emergencyStop()}});
setInterval(poll,150);setInterval(pollCamera,2000);loadMap();
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
