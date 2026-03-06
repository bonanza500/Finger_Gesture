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
  Thumb + Index (7)        → Snap photo with Canon EOS 6D
  Fist (10) / All (5)      → Stop

Usage:
  python dobot_gesture_control_camera.py                    # Full mode
  python dobot_gesture_control_camera.py --no-robot         # Vision + camera only
  python dobot_gesture_control_camera.py --no-camera        # Robot only, no camera
  python dobot_gesture_control_camera.py --ip 192.168.5.1   # Custom robot IP
  python dobot_gesture_control_camera.py --photo-dir D:\\Photos  # Custom save dir

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
import numpy as np
import base64
import json
import threading
import time
import os
import sys
import socket
import argparse

from flask import Flask, jsonify, Response, request
from flask_cors import CORS

import mediapipe as mp

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("[WARN] 'keyboard' not installed. Install with: pip install keyboard")
    print("[WARN] Physical hotkeys disabled. Use web dashboard instead.")
    KEYBOARD_AVAILABLE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

THUMB_TIP  = mp_hands.HandLandmark.THUMB_TIP
THUMB_IP   = mp_hands.HandLandmark.THUMB_IP
INDEX_TIP  = mp_hands.HandLandmark.INDEX_FINGER_TIP
INDEX_PIP  = mp_hands.HandLandmark.INDEX_FINGER_PIP
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
MIDDLE_PIP = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
RING_TIP   = mp_hands.HandLandmark.RING_FINGER_TIP
RING_PIP   = mp_hands.HandLandmark.RING_FINGER_PIP
PINKY_TIP  = mp_hands.HandLandmark.PINKY_TIP
PINKY_PIP  = mp_hands.HandLandmark.PINKY_PIP
WRIST      = mp_hands.HandLandmark.WRIST

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not installed")
    YOLO_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 75

DETECTION_METHOD = "auto"
MODEL_PATH = r"C:\My Websites\hand_tracking\runs\train\finger_gestures\weights\best.pt"
SKLEARN_MODEL_PATH = None

MP_MAX_HANDS = 1
MP_DETECTION_CONF = 0.7
MP_TRACKING_CONF = 0.5

DOBOT_IP = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003

# ── Canon EOS 6D Configuration (via digiCamControl) ─────────────
CANON_ENABLED = True                      # Set False to disable camera features
CANON_SAVE_DIR = r"C:\CapturedPhotos"     # Where digiCamControl saves photos
CANON_CAPTURE_GESTURE = 7                 # Gesture 7 = "Thumb + Index" → snap photo
CANON_CAPTURE_COOLDOWN = 30               # Frames to wait between captures (~1 sec)
DIGICAM_URL = "http://localhost:5513"     # digiCamControl webserver address
DIGICAM_CAPTURE_TIMEOUT = 15             # Max seconds to wait for a capture

DEBOUNCE_FRAMES = 8
COOLDOWN_FRAMES = 15
NO_HAND_STOP_DELAY = 10

FLASK_PORT = 5001

# Preset positions (joint angles in degrees)
PRESETS = {
    1: {
        "name": "Preset 1",
        "joints": [-198.39, -85.32, 32.59, 22.86, -0.01, 1.99]
    },
    2: {
        "name": "Preset 2", 
        "joints": [-252.05, -22.05, -99.95, 22.86, -0.01, 1.99]
    }
}

# Gesture to preset mapping
# Use "camera" as the action for the photo capture gesture
GESTURE_TO_PRESET = {
    1: 1,        # Index -> Preset 1
    2: 2,        # Index + Middle -> Preset 2
    3: None,     # Index + Middle + Ring -> (unused)
    4: None,     # Index + Middle + Ring + Pinky -> (unused)
    5: None,     # All Fingers -> STOP
    6: None,     # Thumb -> (unused)
    7: "camera", # Thumb + Index -> SNAP PHOTO
    8: None,     # Thumb + Index + Middle -> (unused)
    9: None,     # Thumb + Index + Middle + Ring -> (unused)
    10: None,    # Fist -> STOP
}

GESTURE_LABELS = {
    0: "Fist", 1: "Index", 2: "Index + Middle",
    3: "Index + Middle + Ring", 4: "Index + Middle + Ring + Pinky",
    5: "All Fingers", 6: "Thumb", 7: "Thumb + Index",
    8: "Thumb + Index + Middle", 9: "Thumb + Index + Middle + Ring",
}

CLASS_TO_GESTURE = {0:10, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}

GESTURE_ID_TO_NAME = {
    1: "Index", 2: "Index + Middle", 3: "Index + Middle + Ring",
    4: "Index + Middle + Ring + Pinky", 5: "All Fingers",
    6: "Thumb", 7: "Thumb + Index", 8: "Thumb + Index + Middle",
    9: "Thumb + Index + Middle + Ring", 10: "Fist",
}

GESTURE_COLORS = {
    1:(0,165,255), 2:(0,255,255), 3:(0,255,0), 4:(255,165,0), 5:(255,0,255),
    6:(0,100,255), 7:(100,255,255), 8:(100,255,100), 9:(255,200,100), 10:(80,80,255),
}


# ═══════════════════════════════════════════════════════════════════
#  DOBOT CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class DobotController:
    def __init__(self, ip, dashboard_port=29999, move_port=30003):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.dashboard = None
        self.mover = None
        self.connected = False
        self.enabled = False
        self.current_preset = None  # Current preset number
        self.last_move_time = 0  # Timestamp of last movement
        self._lock = threading.Lock()

    def connect(self):
        try:
            self.dashboard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dashboard.settimeout(5)
            self.dashboard.connect((self.ip, self.dashboard_port))
            print(f"  [ROBOT] Dashboard → {self.ip}:{self.dashboard_port}")
            self.connected = True
            return True
        except Exception as e:
            print(f"  [ROBOT ERROR] {e}")
            self.connected = False
            return False

    def _send_dashboard(self, cmd):
        if not self.dashboard: return None
        with self._lock:
            try:
                self.dashboard.sendall((cmd + "\n").encode("utf-8"))
                return self.dashboard.recv(1024).decode("utf-8").strip()
            except Exception as e:
                print(f"  [ROBOT ERROR] Dashboard: {e}")
                return None

    def initialize(self):
        if not self.connected: return False
        print("  [ROBOT] Initializing...")
        self._send_dashboard("RequestControl()")
        time.sleep(0.5)
        self._send_dashboard("ClearError()")
        time.sleep(0.5)
        resp = self._send_dashboard("EnableRobot()")
        print(f"  [ROBOT] EnableRobot → {resp}")
        print("  [ROBOT] Waiting for servos...")
        time.sleep(3)
        self.enabled = True
        print("  [ROBOT] ✓ Ready!")
        return True
    
    def move_to_preset(self, preset_num):
        """Move robot to preset position using JointMovJ"""
        if not self.connected or not self.enabled:
            return False
        
        if preset_num not in PRESETS:
            print(f"  [ROBOT ERROR] Preset {preset_num} not found")
            return False
        
        preset = PRESETS[preset_num]
        joints = preset["joints"]
        name = preset["name"]
        
        # Format: JointMovJ(j1,j2,j3,j4,j5,j6)
        joint_str = ",".join([f"{j:.2f}" for j in joints])
        cmd = f"JointMovJ({joint_str})"
        
        print(f"  [ROBOT] ► Moving to {name} (Preset {preset_num})")
        resp = self._send_dashboard(cmd)
        
        if resp:
            self.current_preset = preset_num
            self.last_move_time = time.time()
            print(f"  [ROBOT] Response: {resp}")
            return True
        else:
            print(f"  [ROBOT ERROR] Failed to move to preset {preset_num}")
            return False
    
    def stop(self):
        if self.connected:
            # No need to stop for preset movements
            pass
    
    def disable(self):
        if self.connected:
            time.sleep(0.3)
            self._send_dashboard("DisableRobot()")
            self.enabled = False
            print("  [ROBOT] Disabled")
        
        axis = movement_info["axis"]
        delta = movement_info["delta"]
        name = movement_info["name"]
        
        # Update pose first
        if not self.update_pose():
            print(f"  [ROBOT ERROR] Cannot get current pose for {name} movement")
            return
        
        # Calculate new position
        new_pose = self.current_pose.copy()
        axis_map = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}
        
        if axis in axis_map:
            idx = axis_map[axis]
            new_pose[idx] += delta
            
            # Add safety bounds check
            if self._check_pose_safety(new_pose):
                # Send MovL command with proper format
                pose_str = ",".join([f"{p:.2f}" for p in new_pose])
                cmd = f"MovL({pose_str})"
                resp = self._send_move(cmd)
                print(f"  [ROBOT] ► Moving {name} ({axis}{delta:+.1f}) → {resp}")
                self.current_movement = name
                self.current_pose = new_pose
            else:
                print(f"  [ROBOT WARNING] {name} movement blocked - would exceed safety limits")
                self.current_movement = None
    
    def _check_pose_safety(self, pose):
        """Basic safety check for pose limits"""
        x, y, z, rx, ry, rz = pose
        
        # Basic workspace limits for Dobot Nova 5 (adjust as needed)
        if not (-800 <= x <= 800):  # X limits
            return False
        if not (-800 <= y <= 800):  # Y limits  
            return False
        if not (-200 <= z <= 600):  # Z limits
            return False
        if not (-180 <= rx <= 180): # RX limits
            return False
        if not (-180 <= ry <= 180): # RY limits
            return False
        if not (-180 <= rz <= 180): # RZ limits
            return False
            
        return True

    def stop(self):
        if self.connected:
            self._send_dashboard("MoveJog()")
            self.current_jog = None
            print("  [ROBOT] ■ Stopped")

    def disable(self):
        if self.connected:
            self._send_dashboard("MoveJog()")  # Stop jogging first
            self.current_jog = None
            time.sleep(0.3)
            self._send_dashboard("DisableRobot()")
            self.enabled = False
            print("  [ROBOT] Disabled")

    def disconnect(self):
        if self.enabled:
            self.disable()
        if self.dashboard:
            self.dashboard.close()
        self.connected = False
        print("  [ROBOT] Disconnected")

    def get_status(self):
        return {
            "connected": self.connected,
            "enabled": self.enabled,
            "ip": self.ip,
            "current_preset": self.current_preset,
            "presets": {k: v["name"] for k, v in PRESETS.items()}
        }


# ═══════════════════════════════════════════════════════════════════
#  CANON EOS 6D CAMERA CONTROLLER (via digiCamControl)
# ═══════════════════════════════════════════════════════════════════

class CanonCamera:
    """
    Controls a Canon EOS 6D via digiCamControl's built-in HTTP webserver.
    
    digiCamControl is a free, open-source Windows app that handles the
    Canon EDSDK driver layer for you. Your Python code just makes simple
    HTTP requests to localhost:5513.
    
    ── SETUP (one-time) ──────────────────────────────────────────────
    
    1. Download & install digiCamControl:
       https://sourceforge.net/projects/digicamcontrol/files/latest/download
    
    2. CLOSE Canon EOS Utility (it locks the USB — only one app at a time!)
    
    3. Open digiCamControl, verify your Canon EOS 6D appears in the app.
    
    4. Enable the webserver:
       File → Settings → Webserver →
         ✓ Enable "Use web server"
         ✓ Enable "Allow interaction via webserver"
         Port: 5513
       → RESTART digiCamControl after changing this!
    
    5. Set your session folder:
       Session → Edit Current Session → Folder → (pick your save directory)
    
    6. Set camera to Manual (M) mode. Disable auto-sleep / auto-power-off.
    
    7. Leave digiCamControl running in the background while your script runs.
    
    ── HOW IT WORKS ──────────────────────────────────────────────────
    
    Capture:    GET http://localhost:5513/?CMD=Capture
    Last file:  GET http://localhost:5513/?slc=get&param1=lastcaptured&param2=
    Preview:    GET http://localhost:5513/preview.jpg
    Session:    GET http://localhost:5513/session.json
    Set folder: GET http://localhost:5513/?slc=set&param1=session.folder&param2=C:\\Photos
    """
    
    def __init__(self, save_dir=CANON_SAVE_DIR, digicam_url=DIGICAM_URL):
        self.save_dir = save_dir
        self.base_url = digicam_url.rstrip("/")
        self.connected = False
        self.last_capture_path = None
        self.last_capture_time = 0
        self.capture_count = 0
        self._lock = threading.Lock()
        self._flash_until = 0  # Timestamp until which "CAPTURED" overlay shows
    
    def _http_get(self, path, timeout=5):
        """Make a simple GET request to digiCamControl's webserver."""
        import urllib.request
        import urllib.error
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace").strip()
        except urllib.error.URLError as e:
            print(f"  [CANON ERROR] Cannot reach digiCamControl: {e.reason}")
            return None
        except Exception as e:
            print(f"  [CANON ERROR] HTTP request failed: {e}")
            return None
    
    def detect(self):
        """
        Check if digiCamControl is running and a camera is connected.
        Tests by fetching session.json from the webserver.
        """
        try:
            result = self._http_get("/session.json", timeout=3)
            if result is not None:
                print(f"  [CANON] digiCamControl webserver is responding")
                self.connected = True
                
                # Try to set the save folder
                escaped_dir = self.save_dir.replace("\\", "\\\\")
                self._http_get(f"/?slc=set&param1=session.folder&param2={self.save_dir}")
                
                return True
            else:
                print("  [CANON] digiCamControl webserver not responding")
                print("  [CANON]   1. Is digiCamControl running?")
                print("  [CANON]   2. Is the webserver enabled? (File → Settings → Webserver)")
                print("  [CANON]   3. Did you restart digiCamControl after enabling it?")
                self.connected = False
                return False
        except Exception as e:
            print(f"  [CANON ERROR] Detection failed: {e}")
            self.connected = False
            return False
    
    def capture(self):
        """
        Trigger a capture on the Canon EOS 6D via digiCamControl.
        Returns the file path of the captured image, or None on failure.
        """
        if not self.connected:
            print("  [CANON ERROR] Camera not connected")
            return None
        
        with self._lock:
            print("  [CANON] Triggering capture...")
            
            # Send capture command
            result = self._http_get("/?CMD=Capture", timeout=DIGICAM_CAPTURE_TIMEOUT)
            
            if result is None:
                print("  [CANON ERROR] Capture command failed — no response")
                return None
            
            # Wait briefly for the photo to be saved and transferred
            time.sleep(1.5)
            
            # Poll for the last captured filename
            for attempt in range(10):
                last_file = self._http_get(
                    "/?slc=get&param1=lastcaptured&param2=", timeout=5
                )
                if last_file and last_file != "-" and last_file.strip():
                    filepath = last_file.strip()
                    self.capture_count += 1
                    self.last_capture_path = filepath
                    self.last_capture_time = time.time()
                    self._flash_until = time.time() + 2.0  # Flash for 2 seconds
                    print(f"  [CANON] ✓ Photo #{self.capture_count}: {filepath}")
                    return filepath
                time.sleep(0.5)
            
            # If polling didn't return a filename, capture may still have worked
            self.capture_count += 1
            self.last_capture_time = time.time()
            self._flash_until = time.time() + 2.0
            print(f"  [CANON] ✓ Capture triggered (file path not confirmed)")
            return "captured"
    
    def is_flashing(self):
        """Returns True if the capture flash indicator should show."""
        return time.time() < self._flash_until
    
    def get_status(self):
        return {
            "connected": self.connected,
            "capture_count": self.capture_count,
            "last_capture": self.last_capture_path,
            "last_capture_time": self.last_capture_time,
            "save_dir": self.save_dir,
            "digicam_url": self.base_url,
        }


# ═══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════

camera = None
camera_lock = threading.Lock()
latest_frame = None
is_running = False
tracking_active = True  # Auto-start ON

yolo_model = None
hands_detector = None
sklearn_model = None
robot = None
canon = None  # Canon EOS 6D camera controller
dry_run = False

latest_detection = {
    "hand_detected": False, "gesture_id": None, "gesture_name": "None",
    "confidence": 0.0, "class_id": None, "bbox": None,
    "method": "none", "robot_preset": None,
}
detection_lock = threading.Lock()

_debounce_counter = 0
_debounce_last_gesture = None
_cooldown_counter = 0
_no_hand_counter = 0

_last_full_detection = {}
_full_detection_lock = threading.Lock()


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
        if robot and robot.connected and robot.enabled:
            robot.jog(None)
        print("  [TRACKING] ■ Stopped")


def toggle_tracking():
    if tracking_active:
        stop_tracking()
    else:
        start_tracking()


def emergency_stop():
    stop_tracking()
    if robot and robot.connected:
        robot.stop()
    print("  [EMERGENCY] ■ Robot stopped + tracking paused")


# ═══════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER (Thread 3)
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

    # Use keyboard library hotkeys (works globally, even when terminal not focused)
    keyboard.add_hotkey('space', toggle_tracking, suppress=False)
    keyboard.add_hotkey('s', emergency_stop, suppress=False)

    # ESC handled in main loop to allow clean shutdown
    while is_running:
        if keyboard.is_pressed('esc'):
            print("\n  [ESC] Exit requested...")
            break
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def init_camera():
    global camera, is_running
    if sys.platform == "win32":
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
    is_running = True
    print(f"  [CAM] Opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")

def init_yolo():
    global yolo_model
    if not YOLO_AVAILABLE or not os.path.exists(MODEL_PATH):
        if YOLO_AVAILABLE: print(f"  [WARN] YOLO model not found: {MODEL_PATH}")
        return False
    try:
        yolo_model = YOLO(MODEL_PATH)
        print(f"  [OK] YOLOv8 loaded"); return True
    except Exception as e:
        print(f"  [ERROR] YOLO: {e}"); return False

def init_mediapipe():
    global hands_detector
    try:
        hands_detector = mp_hands.Hands(
            static_image_mode=False, max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_DETECTION_CONF, min_tracking_confidence=MP_TRACKING_CONF)
        print("  [OK] MediaPipe Hands ready"); return True
    except Exception as e:
        print(f"  [ERROR] MediaPipe: {e}"); return False

def init_sklearn():
    global sklearn_model
    if not JOBLIB_AVAILABLE or not SKLEARN_MODEL_PATH: return False
    if not os.path.exists(SKLEARN_MODEL_PATH): return False
    try: sklearn_model = joblib.load(SKLEARN_MODEL_PATH); return True
    except: return False

def init_robot():
    global robot
    robot = DobotController(DOBOT_IP, DASHBOARD_PORT, MOVE_PORT)
    if robot.connect(): robot.initialize(); return True
    print("  [WARN] Robot not connected"); return False


# ═══════════════════════════════════════════════════════════════════
#  GESTURE DETECTION
# ═══════════════════════════════════════════════════════════════════

def is_finger_up(lm, tip, pip): return lm[tip].y < lm[pip].y
def is_thumb_up(lm, hand="Right"):
    return (lm[THUMB_TIP].x < lm[THUMB_IP].x) if hand == "Right" else (lm[THUMB_TIP].x > lm[THUMB_IP].x)

def classify_fingers(fingers):
    t,i,m,r,p = fingers
    if not any(fingers):          return 10, "Fist"
    if t and i and m and r and p: return 5,  "All Fingers"
    if t and i and m and r:       return 9,  "Thumb + Index + Middle + Ring"
    if t and i and m:             return 8,  "Thumb + Index + Middle"
    if t and i:                   return 7,  "Thumb + Index"
    if t:                         return 6,  "Thumb"
    if i and m and r and p:       return 4,  "Index + Middle + Ring + Pinky"
    if i and m and r:             return 3,  "Index + Middle + Ring"
    if i and m:                   return 2,  "Index + Middle"
    if i:                         return 1,  "Index"
    return min(sum(fingers),10), GESTURE_ID_TO_NAME.get(sum(fingers),"Unknown")

def detect_yolo(frame):
    if yolo_model is None: return None
    try:
        results = yolo_model(frame, verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes)==0:
            return {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                    "confidence":0.0,"class_id":None,"bbox":None,"method":"YOLOv8"}
        box = results[0].boxes[0]
        cid = int(box.cls[0].item()); conf = float(box.conf[0].item())
        bbox = box.xyxy[0].cpu().numpy().tolist()
        gid = CLASS_TO_GESTURE.get(cid, cid+1)
        return {"hand_detected":True,"gesture_id":gid,"gesture_name":GESTURE_LABELS.get(cid,f"Class_{cid}"),
                "confidence":conf,"class_id":cid,"bbox":bbox,"method":"YOLOv8"}
    except Exception as e:
        print(f"  [ERROR] YOLO: {e}"); return None

def detect_mediapipe(frame):
    if hands_detector is None: return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                "confidence":0.0,"class_id":None,"bbox":None,"method":"MediaPipe"}
    hlm = results.multi_hand_landmarks[0]; lm = hlm.landmark
    hand = "Right"
    if results.multi_handedness: hand = results.multi_handedness[0].classification[0].label
    h,w,_ = frame.shape
    xs=[l.x*w for l in lm]; ys=[l.y*h for l in lm]
    bbox=[max(0,int(min(xs))-20),max(0,int(min(ys))-20),min(w,int(max(xs))+20),min(h,int(max(ys))+20)]
    fingers = [is_thumb_up(lm,hand), is_finger_up(lm,INDEX_TIP,INDEX_PIP),
               is_finger_up(lm,MIDDLE_TIP,MIDDLE_PIP), is_finger_up(lm,RING_TIP,RING_PIP),
               is_finger_up(lm,PINKY_TIP,PINKY_PIP)]
    gid,gname = classify_fingers(fingers)
    return {"hand_detected":True,"gesture_id":gid,"gesture_name":gname,"confidence":0.85,
            "class_id":gid-1 if gid<10 else 0,"bbox":bbox,"method":"MediaPipe+Rules",
            "landmarks":hlm,"handedness":hand}


# ═══════════════════════════════════════════════════════════════════
#  ROBOT BRIDGE
# ═══════════════════════════════════════════════════════════════════

def process_robot(gesture_id):
    global _debounce_counter, _debounce_last_gesture, _cooldown_counter, _no_hand_counter
    if robot is None or not robot.connected or not robot.enabled:
        return None
    
    if _cooldown_counter > 0:
        _cooldown_counter -= 1
        return robot.current_preset
    
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
        
        # ── Camera capture gesture ──────────────────────────────
        if action == "camera":
            if canon and canon.connected:
                # Run capture in a separate thread so it doesn't block detection
                threading.Thread(target=canon.capture, daemon=True).start()
                print(f"  [GESTURE] Thumb + Index → SNAP PHOTO")
                _debounce_counter = 0
                _cooldown_counter = CANON_CAPTURE_COOLDOWN
                return "camera"
            else:
                print(f"  [GESTURE] Camera gesture detected but camera not connected")
                _debounce_counter = 0
                return robot.current_preset
        
        # ── Robot movement gesture ──────────────────────────────
        elif action and isinstance(action, int):
            robot.move_to_preset(action)
            _debounce_counter = 0
            _cooldown_counter = COOLDOWN_FRAMES
            return action
        
        _debounce_counter = 0
    
    return robot.current_preset


# ═══════════════════════════════════════════════════════════════════
#  FRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_frame(frame):
    global latest_detection
    if not tracking_active:
        idle = {"hand_detected":False,"gesture_id":None,"gesture_name":"Tracking Paused",
                "confidence":0.0,"class_id":None,"bbox":None,"method":"paused","robot_jog":None}
        with detection_lock: latest_detection = idle
        return idle

    det = None
    if DETECTION_METHOD == "yolo":       det = detect_yolo(frame)
    elif DETECTION_METHOD == "mediapipe": det = detect_mediapipe(frame)
    elif DETECTION_METHOD == "auto":
        det = detect_yolo(frame)
        if det is None: det = detect_mediapipe(frame)

    if det is None:
        det = {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
               "confidence":0.0,"class_id":None,"bbox":None,"method":"none"}

    active_preset = process_robot(det.get("gesture_id"))
    det["robot_preset"] = active_preset

    clean = {k:v for k,v in det.items() if k not in ("landmarks","handedness")}
    with detection_lock: latest_detection = clean
    return det


# ═══════════════════════════════════════════════════════════════════
#  OVERLAY
# ═══════════════════════════════════════════════════════════════════

def draw_overlay(frame):
    out = frame.copy()
    with detection_lock: det = latest_detection.copy()
    with _full_detection_lock: full = _last_full_detection.copy()

    # Paused banner
    if not tracking_active:
        overlay = out.copy()
        cv2.rectangle(overlay, (0,0), (FRAME_WIDTH, FRAME_HEIGHT), (0,0,0), -1)
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
        cv2.putText(out, "TRACKING PAUSED", (FRAME_WIDTH//2-180, FRAME_HEIGHT//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
        cv2.putText(out, "Press [SPACE] to resume",
                    (FRAME_WIDTH//2-140, FRAME_HEIGHT//2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        if robot and robot.connected:
            st = "ROBOT: ONLINE" if robot.enabled else "ROBOT: DISABLED"
            sc = (0,255,0) if robot.enabled else (0,200,255)
        elif dry_run: st,sc = "ROBOT: DRY RUN",(0,255,255)
        else: st,sc = "ROBOT: OFFLINE",(80,80,255)
        cv2.putText(out, st, (FRAME_WIDTH-230, FRAME_HEIGHT-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)
        return out

    # Hand skeleton
    if "landmarks" in full and full.get("hand_detected"):
        mp_drawing.draw_landmarks(out, full["landmarks"], mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    if det.get("hand_detected"):
        gid=det["gesture_id"]; gname=det["gesture_name"]; conf=det["confidence"]
        bbox=det["bbox"]; preset=det.get("robot_preset"); color=GESTURE_COLORS.get(gid,(255,255,255))
        if bbox:
            x1,y1,x2,y2=[int(v) for v in bbox]
            cv2.rectangle(out,(x1,y1),(x2,y2),color,3)
            label=f"{gid}: {gname}"
            (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(out,(x1,y1-lh-10),(x1+lw+10,y1),color,-1)
            cv2.putText(out,label,(x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(out,str(gid),(20,70),cv2.FONT_HERSHEY_SIMPLEX,2.5,color,5)
        cv2.putText(out,gname,(90,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        cv2.putText(out,f"{conf:.0%}",(90,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,255,150),1)
        pt=f"PRESET: {preset}" if preset else "READY"
        pc=(0,255,0) if preset else (80,80,255)
        
        # Camera capture flash indicator
        if preset == "camera" or (canon and canon.is_flashing()):
            pt = f"PHOTO #{canon.capture_count}" if canon else "SNAP!"
            pc = (0,255,255)  # Cyan for camera
            # White flash effect
            overlay = out.copy()
            cv2.rectangle(overlay, (0,0), (FRAME_WIDTH, FRAME_HEIGHT), (255,255,255), -1)
            alpha = max(0.0, (canon._flash_until - time.time()) / 2.0 * 0.3) if canon else 0.15
            out = cv2.addWeighted(overlay, alpha, out, 1-alpha, 0)
            # Camera icon text
            cv2.putText(out, "PHOTO CAPTURED", (FRAME_WIDTH//2-130, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.putText(out,pt,(FRAME_WIDTH-220,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,pc,2)
        cv2.putText(out,f"[{det.get('method','')}]",(10,FRAME_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1)
    else:
        cv2.putText(out,"No hand detected",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,255),2)
        cv2.putText(out,"READY",(FRAME_WIDTH-220,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(80,80,255),2)

    # Tracking active indicator
    cv2.circle(out,(20,FRAME_HEIGHT-20),8,(0,255,0),-1)
    cv2.putText(out,"TRACKING",(35,FRAME_HEIGHT-13),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

    if robot and robot.connected:
        st="ROBOT: ONLINE" if robot.enabled else "ROBOT: DISABLED"
        sc=(0,255,0) if robot.enabled else (0,200,255)
    elif dry_run: st,sc="ROBOT: DRY RUN",(0,255,255)
    else: st,sc="ROBOT: OFFLINE",(80,80,255)
    cv2.putText(out,st,(FRAME_WIDTH-230,FRAME_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.55,sc,2)
    
    # Canon camera status
    if CANON_ENABLED:
        if canon and canon.connected:
            cs = f"CANON: {canon.capture_count} shots"
            cc = (0, 255, 255)
        else:
            cs, cc = "CANON: OFFLINE", (80, 80, 255)
        cv2.putText(out, cs, (FRAME_WIDTH-230, FRAME_HEIGHT-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, 1)
    
    return out


# ═══════════════════════════════════════════════════════════════════
#  CAPTURE LOOP (Thread 1)
# ═══════════════════════════════════════════════════════════════════

def capture_loop():
    global latest_frame
    while is_running:
        with camera_lock:
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    latest_frame = frame
                    full = analyze_frame(frame)
                    with _full_detection_lock:
                        _last_full_detection.clear()
                        _last_full_detection.update(full)
        time.sleep(0.025)


# ═══════════════════════════════════════════════════════════════════
#  FLASK API + WEB DASHBOARD (Thread 2)
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
      <div style="height:8px"></div>
      <button class="btn btn-outline" onclick="manualCapture()" style="border-color:#d29922;color:#d29922">Manual Capture</button>
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
    document.getElementById('canonLast').textContent=d.path.split(/[\\/]/).pop()}
  }).catch(()=>{})
}
function poll(){fetch('/detection').then(r=>r.json()).then(d=>{
  const n=document.getElementById('gestureNum'),m=document.getElementById('gestureName'),j=document.getElementById('gestureJog');
  if(d.hand_detected){n.textContent=d.gesture_id;n.style.color='#58a6ff';m.textContent=d.gesture_name;
    if(d.robot_preset==='camera'){j.textContent='SNAP!';j.style.color='#d29922'}
    else{j.textContent=d.robot_preset?'PRESET: '+d.robot_preset:'READY';j.style.color=d.robot_preset?'#3fb950':'#f85149'}}
  else{n.textContent='—';n.style.color='#484f58';m.textContent=d.gesture_name||'No hand';j.textContent='—';j.style.color='#484f58'}
  if(d.method==='paused'&&isTracking)updateUI(false);
  if(d.method!=='paused'&&!isTracking)updateUI(true);
  if(d.robot){const r=d.robot;
    document.getElementById('robotConn').innerHTML=r.connected?'<span class="dot green"></span>Connected':'<span class="dot red"></span>Disconnected';
    document.getElementById('robotEnabled').innerHTML=r.enabled?'<span class="dot green"></span>Yes':'<span class="dot yellow"></span>No';
    document.getElementById('robotJog').textContent=r.current_preset?'Preset '+r.current_preset:'None';
    document.getElementById('robotIp').textContent=r.ip||'—'}}).catch(()=>{})}
function loadMap(){fetch('/config/gesture_map').then(r=>r.json()).then(m=>{
  const names={1:'Index',2:'Index+Mid',3:'Idx+Mid+Ring',4:'Idx+Mid+Rng+Pnk',5:'All Fingers',6:'Thumb',7:'Thumb+Idx',8:'Thm+Idx+Mid',9:'Thm+Idx+Mid+Rng',10:'Fist'};
  const presets={1:'Preset 1',2:'Preset 2',3:'-',4:'-',5:'-',6:'-',7:'CAMERA',8:'-',9:'-',10:'-'};
  const g=document.getElementById('gestureMap');g.innerHTML='';
  for(let i=1;i<=10;i++){const act=presets[i]||'-';const style=act==='CAMERA'?'color:#d29922;font-weight:700':'';
    g.innerHTML+='<div class="gid">'+i+'</div><div class="gname">'+(names[i]||'?')+'</div><div class="gjog" style="'+style+'">'+act+'</div>'}})}
function pollCamera(){fetch('/camera/status').then(r=>r.json()).then(d=>{
  document.getElementById('canonConn').innerHTML=d.connected?'<span class="dot green"></span>Connected':'<span class="dot red"></span>Offline';
  document.getElementById('canonCount').textContent=d.capture_count||0;
  if(d.last_capture){document.getElementById('canonLast').textContent=d.last_capture.split(/[\\/]/).pop()}
}).catch(()=>{})}
// Also listen for keyboard in the browser
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
    rs = robot.get_status() if robot and robot.connected else {"connected":False,"dry_run":dry_run}
    cs = canon.get_status() if canon else {"connected":False}
    return jsonify({"running":is_running,"tracking":tracking_active,"detection_method":DETECTION_METHOD,
                    "yolo_loaded":yolo_model is not None,"mediapipe_loaded":hands_detector is not None,
                    "robot":rs,"canon":cs,"gestures":GESTURE_ID_TO_NAME,
                    "gesture_to_preset":{str(k):v for k,v in GESTURE_TO_PRESET.items()},
                    "presets":{str(k):v["name"] for k,v in PRESETS.items()}})

@flask_app.route("/detection", methods=["GET"])
def api_detection():
    with detection_lock: d = latest_detection.copy()
    d["tracking"] = tracking_active
    if robot: d["robot"] = robot.get_status()
    return jsonify(d)

@flask_app.route("/frame", methods=["GET"])
def api_frame():
    if latest_frame is None: return jsonify({"error":"No frame"}),503
    _,buf = cv2.imencode(".jpg",latest_frame,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
    return jsonify({"frame":base64.b64encode(buf).decode()})

@flask_app.route("/frame/annotated", methods=["GET"])
def api_annotated():
    if latest_frame is None: return jsonify({"error":"No frame"}),503
    _,buf = cv2.imencode(".jpg",draw_overlay(latest_frame),[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
    return jsonify({"frame":base64.b64encode(buf).decode()})

@flask_app.route("/stream", methods=["GET"])
def api_stream():
    def gen():
        while is_running:
            if latest_frame is not None:
                _,buf = cv2.imencode(".jpg",draw_overlay(latest_frame),[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
            time.sleep(0.033)
    return Response(gen(),mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/robot/stop", methods=["POST"])
def api_stop():
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
        success = robot.move_to_preset(int(preset_num))
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
        # First try loading from digiCamControl's preview endpoint
        import urllib.request
        preview_url = f"{canon.base_url}/preview.jpg"
        try:
            with urllib.request.urlopen(preview_url, timeout=5) as resp:
                img_data = resp.read()
                return jsonify({
                    "frame": base64.b64encode(img_data).decode(),
                    "path": canon.last_capture_path or "unknown",
                    "count": canon.capture_count
                })
        except Exception:
            pass
        
        # Fallback: try loading the file directly if it exists locally
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


def run_flask():
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True, use_reloader=False)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    global DOBOT_IP, dry_run, robot, canon, is_running

    parser = argparse.ArgumentParser(description="Finger Gesture + Dobot Nova 5 + Canon EOS 6D (All-in-One)")
    parser.add_argument("--no-robot", action="store_true", help="Vision only, no robot")
    parser.add_argument("--no-camera", action="store_true", help="Disable Canon camera capture")
    parser.add_argument("--ip", default=DOBOT_IP, help=f"Dobot IP (default: {DOBOT_IP})")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help=f"Web port (default: {FLASK_PORT})")
    parser.add_argument("--photo-dir", default=CANON_SAVE_DIR, help=f"Photo save directory")
    args = parser.parse_args()
    DOBOT_IP = args.ip; dry_run = args.no_robot

    print()
    print("═"*64)
    print("  Finger Gesture → Dobot Nova 5 + Canon EOS 6D")
    print("  Auto-start │ Web dashboard │ Keyboard hotkeys")
    print("═"*64)
    print()

    init_camera()
    yolo_ok = init_yolo(); mp_ok = init_mediapipe(); init_sklearn()

    if DETECTION_METHOD=="auto":
        if yolo_ok:   print("  [OK] YOLOv8 primary + MediaPipe fallback")
        elif mp_ok:   print("  [OK] MediaPipe only")
        else:         print("  [ERROR] No detection!"); sys.exit(1)

    if dry_run:
        print("\n  [DRY RUN] --no-robot: commands printed, not sent")
        robot = DobotController(DOBOT_IP, DASHBOARD_PORT, MOVE_PORT)
        robot.connected=True; robot.enabled=True
        def fake_jog(axis):
            if axis != robot.current_jog:
                if axis: print(f"  [DRY RUN] MoveJog({axis})")
                else: print(f"  [DRY RUN] MoveJog() → STOP")
                robot.current_jog = axis
        robot.jog = fake_jog
        def fake_jog(axis):
            if axis != robot.current_jog:
                if axis: print(f"  [DRY RUN] MoveJog({axis})")
                else: print(f"  [DRY RUN] MoveJog() → STOP")
                robot.current_jog = axis
        robot.jog = fake_jog
    else:
        print(f"\n  Connecting to Dobot Nova at {DOBOT_IP}...")
        if not init_robot():
            print("  Running in vision-only mode (robot offline)")

    # ── Canon EOS 6D Setup (via digiCamControl) ───────────────────
    if CANON_ENABLED and not args.no_camera:
        print(f"\n  Connecting to Canon EOS 6D via digiCamControl...")
        canon = CanonCamera(
            save_dir=args.photo_dir,
            digicam_url=DIGICAM_URL
        )
        if canon.detect():
            print(f"  [CANON] ✓ Camera ready! Photos → {canon.save_dir}")
            print(f"  [CANON] Capture gesture: {CANON_CAPTURE_GESTURE} "
                  f"({GESTURE_ID_TO_NAME.get(CANON_CAPTURE_GESTURE, '?')})")
        else:
            print("  [CANON] Camera not available — capture gestures will be ignored")
    else:
        print("\n  [CANON] Camera disabled (--no-camera or CANON_ENABLED=False)")

    # Start threads
    cam_thread = threading.Thread(target=capture_loop, daemon=True)
    cam_thread.start()

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start keyboard listener thread
    key_thread = threading.Thread(target=keyboard_listener, daemon=True)
    key_thread.start()

    print()
    print(f"  ✓ Detection auto-started")
    print(f"  ✓ Dashboard → http://localhost:{args.port}")
    print(f"  ✓ Stream    → http://localhost:{args.port}/stream")
    if canon and canon.connected:
        print(f"  ✓ Canon EOS 6D → Gesture {CANON_CAPTURE_GESTURE} to snap")
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
            # Check if keyboard thread signaled exit
            if KEYBOARD_AVAILABLE and not key_thread.is_alive():
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  [!] Ctrl+C detected")
    finally:
        is_running = False; time.sleep(0.3)
        if robot and not dry_run: robot.disconnect()
        if camera: camera.release()
        if hands_detector: hands_detector.close()
        if KEYBOARD_AVAILABLE:
            try: keyboard.unhook_all()
            except: pass
        print("  ✓ Shutdown complete.")


if __name__ == "__main__":
    main()