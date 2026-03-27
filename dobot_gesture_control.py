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

TTS_AVAILABLE = False  # pyttsx3 removed — using Windows SAPI via PowerShell instead

# Windows beep fallback
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

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
CANON_CAPTURE_COOLDOWN = 90               # Frames to wait between captures (~3 sec at 30fps)
DIGICAM_URL = "http://localhost:5513"     # digiCamControl webserver address
DIGICAM_CAPTURE_TIMEOUT = 15             # Max seconds to wait for a capture
# digiCamControl command-line tool (fallback if HTTP webserver is not enabled)
DIGICAM_EXE = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
# digiCamControl main application — auto-launched if not running
DIGICAM_APP_EXE = r"C:\Program Files (x86)\digiCamControl\digiCamControl.exe"
DIGICAM_AUTOLAUNCH = True     # Auto-start digiCamControl if not running
DIGICAM_LAUNCH_WAIT = 8       # Seconds to wait for it to start up

# ── Audio Countdown Configuration ───────────────────────────────
COUNTDOWN_ENABLED = True          # Play audio countdown before every preset capture
AUTO_CAPTURE_AFTER_PRESET = True  # Auto-trigger camera after preset move + countdown
ROBOT_SETTLE_DELAY = 2.5          # Seconds to wait for robot to reach position
COUNTDOWN_SPEAK_INTRO = "Lihat ke kamera"  # Spoken intro phrase (Indonesian)
COUNTDOWN_NUMBERS = [3, 2, 1]     # Countdown sequence
COUNTDOWN_NUMBER_DELAY = 0.8      # Pause (seconds) between each number
TTS_SPEECH_RATE = 130             # Words per minute for TTS voice
# Total auto-capture cooldown frames (settle + ~4s countdown + buffer, at ~30fps)
COUNTDOWN_COOLDOWN_FRAMES = int((ROBOT_SETTLE_DELAY + len(COUNTDOWN_NUMBERS) * COUNTDOWN_NUMBER_DELAY + 3) * 30)

DEBOUNCE_FRAMES = 8
COOLDOWN_FRAMES = 15
NO_HAND_STOP_DELAY = 10

FLASK_PORT = 5001

# ── DobotStudio Pro presets.json path ───────────────────────────
# The file is written by DobotStudio Pro at this location.
# Change the path here if your installation is elsewhere.
DOBOTSTUDIO_PRESETS_JSON = r"C:\Program Files (x86)\DobotStudio Pro\presets.json"

# Fallback hardcoded values (used only when the JSON file cannot be read)
_FALLBACK_INITIAL_POSE = {
    "name": "InitialPose",
    "joints": [71.627, -5.6467, -119.3281, 112.4991, -18.3817, 118.9324]
}
_FALLBACK_PRESETS = {
    1: {"name": "P1", "joints": [-82.7149,  89.5943, -88.0344,  -0.2165,  89.7407,  78.4673]},
    2: {"name": "P2", "joints": [-178.2973, -20.7081,  -0.3736,  72.4075, -272.0576, 169.0177]},
    3: {"name": "P3", "joints": [-269.4157, -86.9001,   0.2984, 172.7563, -274.2545, 280.4257]},
    4: {"name": "P4", "joints": [  0.0659,   0.0711,   0.0776,   0.0523,   0.0063,   0.0913]},
}

def load_presets_from_json(path=DOBOTSTUDIO_PRESETS_JSON):
    """
    Load presets from DobotStudio Pro's presets.json.

    The JSON is a list of objects with shape:
      { "name": "P1", "joint": [j1, j2, j3, j4, j5, j6], "id": ..., ... }

    Entries named "InitialPose" are stored as INITIAL_POSE.
    Entries named P1..P4 (or any name starting with 'P' followed by a digit)
    are mapped to gesture slots 1..4 in order of appearance.

    Returns (initial_pose_dict, presets_dict).
    Falls back to hardcoded values on any error.
    """
    if not os.path.exists(path):
        print(f"  [PRESETS] File not found: {path}")
        print(f"  [PRESETS] Using hardcoded fallback presets.")
        return _FALLBACK_INITIAL_POSE, _FALLBACK_PRESETS

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        initial_pose = None
        presets = {}
        slot = 1  # gesture slot counter (1..4)

        for entry in data:
            name   = entry.get("name", "")
            joints = entry.get("joint", [])

            if len(joints) != 6:
                print(f"  [PRESETS] Skipping '{name}' — expected 6 joints, got {len(joints)}")
                continue

            joints = [round(float(j), 4) for j in joints]

            if name == "InitialPose":
                initial_pose = {"name": name, "joints": joints}
                print(f"  [PRESETS] InitialPose loaded: {joints}")
            else:
                # Map every non-base entry to the next gesture slot
                presets[slot] = {"name": name, "joints": joints}
                print(f"  [PRESETS] Gesture {slot} → {name}: {joints}")
                slot += 1
                if slot > 10:   # safety cap — we only have 10 gesture slots
                    break

        if not presets:
            raise ValueError("No usable presets found in JSON")

        if initial_pose is None:
            initial_pose = _FALLBACK_INITIAL_POSE

        print(f"  [PRESETS] ✓ Loaded {len(presets)} preset(s) from {path}")
        return initial_pose, presets

    except Exception as e:
        print(f"  [PRESETS] ERROR reading {path}: {e}")
        print(f"  [PRESETS] Using hardcoded fallback presets.")
        return _FALLBACK_INITIAL_POSE, _FALLBACK_PRESETS


# These are populated at startup by load_presets_from_json()
INITIAL_POSE = _FALLBACK_INITIAL_POSE
PRESETS      = _FALLBACK_PRESETS

# ── Continuous Scan Positions (Fist gesture) ─────────────────────
# Robot oscillates between these two joint targets when Fist is shown.
# Open palm (All Fingers) stops the scan.
SCAN_POSITION_1 = {
    "name": "Scan Pos 1",
    "joints": [178.93, 107.52, 15.69, 295.53, -92.90, 118.99]
}
SCAN_POSITION_2 = {
    "name": "Scan Pos 2",
    "joints": [178.82, 84.41, 55.71, 278.61, -92.88, 118.94]
}
SCAN_INITIAL_DELAY = 5.0    # Seconds to wait after first move to Pos 1
SCAN_LOOP_DELAY = 0.5       # Seconds between subsequent moves

# Gesture to action mapping — rebuilt at startup after presets are loaded
# Use "camera" for photo capture, "scan" to start scanning, "stop_scan" to stop
def build_gesture_map(presets):
    """Build GESTURE_TO_PRESET from however many presets were loaded.
    Gesture 7 (Thumb+Index) → direct camera shutter (instant capture with countdown).
    Auto-capture ALSO fires after every preset move (gestures 1-4) when robot is active.
    """
    mapping = {}
    for slot in range(1, 11):
        if slot in presets:
            mapping[slot] = slot           # gesture N → Preset N
        elif slot == 5:
            mapping[slot] = "stop_scan"    # All Fingers → STOP SCANNING
        elif slot == 7:
            mapping[slot] = None           # unused — shutter fires automatically after preset move
        elif slot == 10:
            mapping[slot] = "scan"         # Fist → START CONTINUOUS SCAN
        else:
            mapping[slot] = None
    return mapping

GESTURE_TO_PRESET = build_gesture_map(PRESETS)   # default; rebuilt in main()

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
    
    def move_to_joints(self, joints, name="Custom"):
        """Move robot to arbitrary joint angles using JointMovJ"""
        if not self.connected or not self.enabled:
            return False
        
        joint_str = ",".join([f"{j:.2f}" for j in joints])
        cmd = f"JointMovJ({joint_str})"
        
        print(f"  [ROBOT] ► Moving to {name}")
        resp = self._send_dashboard(cmd)
        
        if resp:
            self.last_move_time = time.time()
            print(f"  [ROBOT] Response: {resp}")
            return True
        else:
            print(f"  [ROBOT ERROR] Failed to move to {name}")
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
#  CANON EOS 6D CAMERA CONTROLLER
#  Three independent capture methods tried in order:
#    1. digiCamControl HTTP webserver  (best — full metadata)
#    2. digiCamControl CLI             (fallback if webserver off)
#    3. Windows WIA via PowerShell     (zero-install, always works)
# ═══════════════════════════════════════════════════════════════════

class CanonCamera:

    def __init__(self, save_dir=CANON_SAVE_DIR, digicam_url=DIGICAM_URL):
        self.save_dir       = save_dir
        self.base_url       = digicam_url.rstrip("/")
        self.connected      = False   # True once any method is confirmed working
        self.capture_method = None    # "http" | "cli" | "wia"
        self.last_capture_path  = None
        self.last_capture_time  = 0
        self.capture_count      = 0
        self._lock              = threading.Lock()
        self._flash_until       = 0
        self._capture_in_progress = False
        self.last_error         = ""
        self._last_reconnect_attempt = 0
        self._reconnect_interval     = 10
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────

    def _http_get(self, path, timeout=5):
        import urllib.request, urllib.error
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace").strip()
        except urllib.error.URLError as e:
            self.last_error = f"HTTP: {e.reason}"
            return None
        except Exception as e:
            self.last_error = f"HTTP: {e}"
            return None

    def _is_digicam_running(self):
        import subprocess
        try:
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq digiCamControl.exe", "/NH"],
                               capture_output=True, text=True, timeout=5)
            return "digiCamControl.exe" in r.stdout
        except Exception:
            return False

    def _find_digicam_exe(self, exe_name):
        """Search registry + common paths for a digiCamControl executable."""
        import subprocess
        candidates = [
            rf"C:\Program Files (x86)\digiCamControl\{exe_name}",
            rf"C:\Program Files\digiCamControl\{exe_name}",
            rf"C:\digiCamControl\{exe_name}",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "digiCamControl", exe_name),
            os.path.join(os.environ.get("APPDATA", ""), "digiCamControl", exe_name),
        ]
        # Registry search
        try:
            import winreg
            reg_roots = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            for hive, key_path in reg_roots:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        for i in range(winreg.QueryInfoKey(key)[0]):
                            try:
                                with winreg.OpenKey(key, winreg.EnumKey(key, i)) as sub:
                                    try:
                                        if "digicam" in winreg.QueryValueEx(sub, "DisplayName")[0].lower():
                                            loc = winreg.QueryValueEx(sub, "InstallLocation")[0]
                                            candidates.insert(0, os.path.join(loc, exe_name))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
        except ImportError:
            pass
        for p in candidates:
            if p and os.path.exists(p):
                print(f"  [CANON] Found {exe_name} at: {p}")
                return p
        try:
            r = subprocess.run(["where", exe_name], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                found = r.stdout.strip().splitlines()[0]
                print(f"  [CANON] Found {exe_name} in PATH: {found}")
                return found
        except Exception:
            pass
        return None

    # ── Method 1: digiCamControl HTTP ────────────────────────────

    def _detect_http(self):
        result = self._http_get("/session.json", timeout=3)
        if result is not None:
            self._http_get(f"/?slc=set&param1=session.folder&param2={self.save_dir}")
            return True
        return False

    def _capture_via_http(self):
        print("  [CANON] Triggering capture via HTTP (digiCamControl)...")
        result = self._http_get("/?CMD=Capture", timeout=DIGICAM_CAPTURE_TIMEOUT)
        if result is None:
            self.connected = False
            return None
        time.sleep(1.5)
        for _ in range(10):
            last = self._http_get("/?slc=get&param1=lastcaptured&param2=", timeout=5)
            if last and last.strip() and last.strip() != "-":
                return last.strip()
            time.sleep(0.5)
        return "captured_http"

    # ── Method 2: digiCamControl CLI ─────────────────────────────

    def _launch_digicam(self):
        import subprocess
        if self._is_digicam_running():
            return True
        app_exe = self._find_digicam_exe("digiCamControl.exe")
        if not app_exe:
            print("  [CANON] digiCamControl.exe not found on this system")
            self.last_error = "digiCamControl not installed"
            return False
        try:
            print(f"  [CANON] Launching {app_exe}...")
            subprocess.Popen([app_exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=0x00000008)
        except Exception as e:
            self.last_error = f"Launch failed: {e}"
            return False
        deadline = time.time() + DIGICAM_LAUNCH_WAIT
        while time.time() < deadline:
            time.sleep(1.5)
            if self._http_get("/session.json", timeout=2) is not None:
                print("  [CANON] digiCamControl webserver is up!")
                self._http_get(f"/?slc=set&param1=session.folder&param2={self.save_dir}")
                return True
        self.last_error = "digiCamControl launched but webserver not reachable — enable in settings"
        return False

    def _capture_via_cli(self):
        import subprocess
        if not self._is_digicam_running():
            if DIGICAM_AUTOLAUNCH:
                if not self._launch_digicam():
                    return None
            else:
                return None
        exe = self._find_digicam_exe("CameraControlCmd.exe")
        if not exe:
            return None
        print(f"  [CANON] Triggering capture via CLI: {exe} /capture")
        try:
            r = subprocess.run([exe, "/capture"], capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                print("  [CANON] CLI capture succeeded")
                return "captured_cli"
            msg = (r.stderr or r.stdout or "unknown").strip()
            print(f"  [CANON ERROR] CLI failed (code {r.returncode}): {msg}")
            self.last_error = f"CLI: {msg[:60]}"
        except subprocess.TimeoutExpired:
            print("  [CANON ERROR] CLI timed out")
            self.last_error = "CLI timed out"
        except Exception as e:
            self.last_error = str(e)[:60]
        return None

    # ── Method 3: Windows WIA (zero-install fallback) ────────────

    def _detect_wia(self):
        """Check if Windows WIA can see a Canon camera via PowerShell."""
        import subprocess
        ps = (
            "$wia = New-Object -ComObject WIA.DeviceManager;"
            "$found = $false;"
            "foreach ($d in $wia.DeviceInfos) {"
            "  try { $n = $d.Properties['Name'].Value } catch { $n = '' };"
            "  if ($n -like '*Canon*' -or $d.Type -eq 2) { $found = $true; Write-Output $n; break }"
            "};"
            "if (-not $found) { exit 1 }"
        )
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                print(f"  [CANON] WIA device found: {r.stdout.strip()}")
                return True
        except Exception as e:
            print(f"  [CANON] WIA check failed: {e}")
        return False

    def _find_eos_utility(self):
        """Find Canon EOS Utility executable."""
        candidates = [
            r"C:\Program Files (x86)\Canon\EOS Utility\EU3\EOS Utility 3.exe",
            r"C:\Program Files (x86)\Canon\EOS Utility\EOS Utility.exe",
            r"C:\Program Files\Canon\EOS Utility\EU3\EOS Utility 3.exe",
            r"C:\Program Files\Canon\EOS Utility\EOS Utility.exe",
            r"C:\Program Files (x86)\Canon\EOS Utility 3\EOS Utility 3.exe",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _find_edsdk_dll(self):
        """
        Find ANY Canon EDSDK.dll that can be loaded by the current Python process.
        Since EOS Utility is deleted, we use whatever EDSDK.dll Canon left on the system.
        We test-load each candidate — the one that loads without error is used.
        """
        import ctypes, sys
        python_is_64 = sys.maxsize > 2**32

        # Collect ALL EDSDK.dll files from Canon folders
        all_dlls = []
        for base in [r"C:\Program Files (x86)\Canon", r"C:\Program Files\Canon"]:
            if os.path.exists(base):
                for root, _dirs, files in os.walk(base):
                    for f in files:
                        if f.upper() == "EDSDK.DLL":
                            all_dlls.append(os.path.join(root, f))

        # Add system paths
        all_dlls += [
            r"C:\Windows\System32\EDSDK.dll",
            r"C:\Windows\SysWOW64\EDSDK.dll",
        ]

        # Sort: prefer 64-bit paths when Python is 64-bit (Program Files not x86)
        if python_is_64:
            all_dlls.sort(key=lambda p: (
                0 if ("Program Files" in p and "(x86)" not in p) else
                0 if "System32" in p else
                1 if "Network Setting" in p else
                1 if "runtimes" in p else
                2
            ))
        else:
            all_dlls.sort(key=lambda p: (
                0 if r"(x86)" in p else 1
            ))

        print(f"  [CANON] Python is {'64-bit' if python_is_64 else '32-bit'}. Testing EDSDK candidates:")
        for p in all_dlls:
            if not os.path.exists(p):
                continue
            try:
                sdk = ctypes.WinDLL(p)
                if hasattr(sdk, "EdsInitializeSDK"):
                    print(f"  [CANON] ✓ Loadable EDSDK.dll: {p}")
                    return p
                else:
                    print(f"  [CANON]   Skip (not EDSDK): {p}")
            except OSError as e:
                err = str(e)
                if "193" in err:
                    print(f"  [CANON]   Skip (wrong bitness): {p}")
                else:
                    print(f"  [CANON]   Skip ({err[:60]}): {p}")
        return None

    def _capture_via_edsdk(self):
        """
        Trigger Canon EOS shutter directly via EDSDK.dll loaded into Python.
        EOS Utility has been deleted so nothing holds the USB port — direct
        ctypes load now works correctly.
        """
        import ctypes

        # Brief wait to ensure any previous USB handle is released
        time.sleep(0.5)

        dll_path = self._find_edsdk_dll()
        if not dll_path:
            print("  [CANON] No loadable EDSDK.dll found on this system.")
            print("  [CANON] → Install digiCamControl: https://digicamcontrol.com/download")
            self.last_error = "No EDSDK.dll found"
            return None

        sdk = None
        camera_list = ctypes.c_void_p()
        camera     = ctypes.c_void_p()

        EDS_ERR_OK    = 0x00000000
        kTakePicture  = 0x00000000
        kPressShutter = 0x00000004
        kShutterFull  = 0x00000003
        kShutterOff   = 0x00000000

        try:
            print(f"  [CANON] Loading EDSDK: {dll_path}")
            sdk = ctypes.WinDLL(dll_path)

            # ── 1. Init ──────────────────────────────────────────
            err = sdk.EdsInitializeSDK()
            print(f"  [CANON] EdsInitializeSDK → 0x{err:08X}")
            if err != EDS_ERR_OK:
                self.last_error = f"EDSDK init: 0x{err:08X}"
                return None

            # ── 2. Camera list ───────────────────────────────────
            err = sdk.EdsGetCameraList(ctypes.byref(camera_list))
            print(f"  [CANON] EdsGetCameraList → 0x{err:08X}")
            if err != EDS_ERR_OK:
                self.last_error = f"GetCameraList: 0x{err:08X}"
                return None

            count = ctypes.c_uint32(0)
            sdk.EdsGetChildCount(camera_list, ctypes.byref(count))
            print(f"  [CANON] Cameras found: {count.value}")
            if count.value == 0:
                sdk.EdsRelease(camera_list)
                self.last_error = "No camera found — USB connected and camera ON?"
                print(f"  [CANON] {self.last_error}")
                return None

            # ── 3. Get camera ────────────────────────────────────
            err = sdk.EdsGetChildAtIndex(camera_list, 0, ctypes.byref(camera))
            sdk.EdsRelease(camera_list)
            camera_list = ctypes.c_void_p()  # already released
            print(f"  [CANON] EdsGetChildAtIndex → 0x{err:08X}")
            if err != EDS_ERR_OK:
                self.last_error = f"GetCamera: 0x{err:08X}"
                return None

            # ── 4. Open session ──────────────────────────────────
            err = sdk.EdsOpenSession(camera)
            print(f"  [CANON] EdsOpenSession → 0x{err:08X}")
            if err == 0x00000002:
                # 0x00000002 = EDS_ERR_COMM_DISCONNECTED
                # Camera might need a moment — retry once
                print("  [CANON] Comm disconnected — waiting 2s and retrying...")
                time.sleep(2)
                err = sdk.EdsOpenSession(camera)
                print(f"  [CANON] EdsOpenSession retry → 0x{err:08X}")
            if err != EDS_ERR_OK:
                self.last_error = f"OpenSession: 0x{err:08X}"
                print(f"  [CANON] {self.last_error}")
                hints = {
                    0x00000002: "USB comm error — try a different USB port or cable",
                    0x00000083: "Camera not ready — set mode dial to M/Av/Tv/P",
                    0x00000851: "Camera busy — another app holds the USB session",
                }
                hint = hints.get(err, "Check USB cable and camera power")
                print(f"  [CANON]   Hint: {hint}")
                return None
            print("  [CANON] Session opened ✓")

            # ── 5. Fire shutter ──────────────────────────────────
            print("  [CANON] Sending TakePicture...")
            err = sdk.EdsSendCommand(camera, kTakePicture, 0)
            print(f"  [CANON] TakePicture → 0x{err:08X}")

            if err != EDS_ERR_OK:
                print("  [CANON] TakePicture failed — trying PressShutterButton...")
                err2 = sdk.EdsSendCommand(camera, kPressShutter, kShutterFull)
                time.sleep(0.5)
                sdk.EdsSendCommand(camera, kPressShutter, kShutterOff)
                print(f"  [CANON] PressShutter(Full) → 0x{err2:08X}")
                if err2 != EDS_ERR_OK:
                    self.last_error = f"Shutter failed: 0x{err2:08X}"
                    return None

            time.sleep(2.0)  # wait for mechanical shutter + image transfer

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"photo_{timestamp}.jpg")
            print(f"  [CANON] ✓ Shutter fired! Photo on SD card (or {self.save_dir})")
            return save_path if os.path.exists(save_path) else f"captured_edsdk_{timestamp}"

        except Exception as e:
            print(f"  [CANON ERROR] EDSDK exception: {e}")
            self.last_error = str(e)[:80]
            return None

        finally:
            try:
                if camera.value:
                    sdk.EdsCloseSession(camera)
                    sdk.EdsRelease(camera)
                if sdk:
                    sdk.EdsTerminateSDK()
                print("  [CANON] EDSDK session closed")
            except Exception:
                pass

        """Try all methods to find a working capture path."""
        if self._detect_http():
            print("  [CANON] ✓ Connected via digiCamControl HTTP")
            self.connected = True; self.capture_method = "http"; self.last_error = ""; return True
        if self._find_digicam_exe("CameraControlCmd.exe"):
            print("  [CANON] ✓ digiCamControl CLI found")
            self.connected = True; self.capture_method = "cli"; self.last_error = ""; return True
        if self._find_edsdk_dll():
            print("  [CANON] ✓ Canon EDSDK.dll found — will use direct SDK capture")
            self.connected = True; self.capture_method = "edsdk"; self.last_error = ""; return True
        if self._detect_wia():
            print("  [CANON] ✓ Canon EOS 6D detected via USB/WIA")
            self.connected = True; self.capture_method = "edsdk"; self.last_error = ""; return True
        self.connected = False
        self.last_error = "Camera not found. Is USB connected and camera ON?"
        if not silent:
            print("  [CANON] ✗ Camera not found by any method.")
            print("  [CANON]   Make sure Canon EOS 6D is plugged in via USB and switched ON.")
            print("  [CANON]   EDSDK.dll from EOS Utility was not found. Is EOS Utility installed?")
        return False

    def detect(self, silent=False):
        """Try all methods to find a working capture path."""
        if self._detect_http():
            print("  [CANON] ✓ Connected via digiCamControl HTTP")
            self.connected = True; self.capture_method = "http"; self.last_error = ""; return True
        if self._find_digicam_exe("CameraControlCmd.exe"):
            print("  [CANON] ✓ digiCamControl CLI found")
            self.connected = True; self.capture_method = "cli"; self.last_error = ""; return True
        if self._find_edsdk_dll():
            print("  [CANON] ✓ EDSDK.dll found — will capture via Canon SDK")
            self.connected = True; self.capture_method = "edsdk"; self.last_error = ""; return True
        if self._detect_wia():
            print("  [CANON] ✓ Canon EOS 6D on USB — will attempt EDSDK capture")
            self.connected = True; self.capture_method = "edsdk"; self.last_error = ""; return True
        self.connected = False
        self.last_error = "Camera not found. Is USB connected and camera ON?"
        if not silent:
            print("  [CANON] ✗ No working capture method found.")
            print("  [CANON]   → Install digiCamControl: https://digicamcontrol.com/download")
            print("  [CANON]   → Enable webserver: File → Settings → Webserver → Enable → Restart")
        return False

    def try_reconnect(self):
        now = time.time()
        if now - self._last_reconnect_attempt < self._reconnect_interval:
            return self.connected
        self._last_reconnect_attempt = now
        print("  [CANON] Reconnecting...")
        return self.detect(silent=True)

    def capture(self):
        """
        Fire the shutter. Order:
          1. digiCamControl HTTP  (if running with webserver on)
          2. digiCamControl CLI   (if installed)
          3. Canon EDSDK.dll      (installed with EOS Utility — most reliable offline method)
        """
        if self._capture_in_progress:
            print("  [CANON] Capture already in progress — skipping")
            return None
        if not self.connected:
            self.try_reconnect()

        self._capture_in_progress = True
        try:
            with self._lock:
                filepath = None

                # Always try HTTP first — digiCamControl may have started after script launch
                if self._detect_http():
                    self.connected = True
                    self.capture_method = "http"
                    filepath = self._capture_via_http()

                # Method 2: digiCamControl CLI
                if filepath is None:
                    fp = self._capture_via_cli()
                    if fp:
                        filepath = fp; self.capture_method = "cli"; self.connected = True

                # Method 3: Canon EDSDK direct
                if filepath is None:
                    fp = self._capture_via_edsdk()
                    if fp:
                        filepath = fp; self.capture_method = "edsdk"; self.connected = True

                if filepath:
                    self.capture_count    += 1
                    self.last_capture_path = filepath
                    self.last_capture_time = time.time()
                    self._flash_until      = time.time() + 2.0
                    self.connected         = True
                    self.last_error        = ""
                    print(f"  [CANON] ✓ Photo #{self.capture_count} via [{self.capture_method}]: {filepath}")
                    return filepath

                print()
                print("  ╔══════════════════════════════════════════════════════╗")
                print("  ║  AUTO-SHOOT FAILED — action needed:                  ║")
                print("  ║                                                      ║")
                print("  ║  Option A — Install digiCamControl (easiest):        ║")
                print("  ║    1. digicamcontrol.com/download → install          ║")
                print("  ║    2. File → Settings → Webserver → Enable           ║")
                print("  ║    3. Restart digiCamControl                         ║")
                print("  ║                                                      ║")
                print("  ║  Option B — already done (EOS Utility installed):    ║")
                print("  ║    CLOSE EOS Utility — it locks the USB port         ║")
                print("  ║    Then retry — EDSDK will connect automatically     ║")
                print("  ╚══════════════════════════════════════════════════════╝")
                print()
                return None
        finally:
            self._capture_in_progress = False

    def is_flashing(self):
        return time.time() < self._flash_until

    def get_status(self):
        return {
            "connected":           self.connected,
            "capture_method":      self.capture_method,
            "capture_count":       self.capture_count,
            "last_capture":        self.last_capture_path,
            "last_capture_time":   self.last_capture_time,
            "save_dir":            self.save_dir,
            "digicam_url":         self.base_url,
            "last_error":          self.last_error,
            "capture_in_progress": self._capture_in_progress,
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

# ── Scan loop state ──────────────────────────────────────────────
_scan_active = False          # True while scan loop is running
_scan_thread = None           # Reference to scan worker thread
_scan_stop_event = threading.Event()  # Signal to stop the scan loop
_scan_current_pos = None      # "pos1" or "pos2" — which position we're at/heading to

# ── Countdown state ──────────────────────────────────────────────
_countdown_state = {
    "active": False,
    "phase": None,     # "settling" | "intro" | "counting" | "capturing" | None
    "number": None,    # 3, 2, 1 or None
    "preset": None,    # which preset triggered this countdown
}
_countdown_lock = threading.Lock()

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
        # Stop scan loop if running
        if _scan_active:
            stop_scan()
        abort_countdown()
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
    if _scan_active:
        stop_scan()
    abort_countdown()
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
#  AUDIO COUNTDOWN ENGINE
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  AUDIO COUNTDOWN ENGINE
# ═══════════════════════════════════════════════════════════════════

def _kill_eos_utility():
    """Kill EOS Utility so it releases the USB port before EDSDK capture."""
    import subprocess
    killed = False
    for proc in ["EOS Utility 3.exe", "EOS Utility.exe", "EOS Utility 2.exe",
                 "RemoteCapture.exe", "EOSDigital.exe"]:
        try:
            r = subprocess.run(["taskkill", "/F", "/IM", proc],
                               capture_output=True, timeout=5)
            if r.returncode == 0:
                print(f"  [CANON] Closed {proc} (freed USB port)")
                killed = True
        except Exception:
            pass
    if killed:
        time.sleep(2.0)   # Give Windows time to release the USB handle

def _speak(text, block=True):
    """
    Speak text using Windows SAPI via PowerShell.
    block=True  → wait for speech to finish (used during countdown).
    block=False → fire-and-forget (used when we need to shoot immediately after).
    """
    import subprocess
    ps = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 0; "
        f"$s.Speak('{text}');"
    )
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
             "-ExecutionPolicy", "Bypass", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if block:
            proc.wait(timeout=8)
    except Exception as e:
        print(f"  [AUDIO] SAPI error: {e}")
        _beep(880, 300)

def _beep(frequency=880, duration_ms=250):
    """Play a Windows beep."""
    if WINSOUND_AVAILABLE:
        try:
            winsound.Beep(int(frequency), int(duration_ms))
        except Exception:
            pass

def countdown_and_capture(preset_num):
    """
    Guaranteed flow every time a preset gesture is detected:
      1. Settle  — wait ROBOT_SETTLE_DELAY seconds for robot to reach position
      2. Intro   — say "Lihat ke kamera"
      3. Count   — 3 … 2 … 1  (beep + Indonesian spoken number)
      4. SHUTTER — fire digiCamControl capture immediately (no extra speech delay)
    """
    print(f"\n  [COUNTDOWN] ══ Preset {preset_num} ══ Starting auto-capture sequence")

    # ── 1. Settle ────────────────────────────────────────────────
    with _countdown_lock:
        _countdown_state.update({"active": True, "phase": "settling",
                                  "number": None, "preset": preset_num})
    print(f"  [COUNTDOWN] ⏳ Settling {ROBOT_SETTLE_DELAY}s for robot to reach position...")
    time.sleep(ROBOT_SETTLE_DELAY)

    # ── 2. "Lihat ke kamera" ─────────────────────────────────────
    with _countdown_lock:
        _countdown_state.update({"phase": "intro", "number": None})
    print(f"  [COUNTDOWN] 🔊 {COUNTDOWN_SPEAK_INTRO}")
    _speak(COUNTDOWN_SPEAK_INTRO, block=True)   # blocking — subject needs time to look

    # ── 3. 3 … 2 … 1 ─────────────────────────────────────────────
    with _countdown_lock:
        _countdown_state["phase"] = "counting"

    num_words  = {1: "satu", 2: "dua", 3: "tiga"}
    beep_freqs = {3: 700,   2: 850,   1: 1000}

    for num in COUNTDOWN_NUMBERS:
        with _countdown_lock:
            _countdown_state["number"] = num
        print(f"  [COUNTDOWN] 🔊 {num}...")
        _beep(beep_freqs.get(num, 880), 150)
        _speak(num_words.get(num, str(num)), block=True)
        time.sleep(COUNTDOWN_NUMBER_DELAY)

    # ── 4. FIRE SHUTTER — immediately, no extra speech ───────────
    with _countdown_lock:
        _countdown_state.update({"phase": "capturing", "number": None})

    print(f"  [COUNTDOWN] 📸 SHUTTER — firing now!")
    _beep(1200, 400)                             # camera-click beep only, no speech delay

    if canon is None:
        print("  [COUNTDOWN] ✗ Canon not initialized — set CANON_ENABLED=True and restart")
    else:
        # Wait for any overlapping capture to clear
        waited = 0
        while canon._capture_in_progress and waited < 5:
            time.sleep(0.2)
            waited += 0.2

        # Reconnect if digiCamControl was started after the script
        if not canon.connected:
            print("  [COUNTDOWN] 🔄 Camera offline — reconnecting to digiCamControl...")
            canon.detect(silent=False)

        result = canon.capture()
        if result:
            print(f"  [COUNTDOWN] ✓ Photo #{canon.capture_count} saved → {result}")
        else:
            print(f"  [COUNTDOWN] ✗ Capture FAILED — check:")
            print(f"  [COUNTDOWN]   1. digiCamControl is open")
            print(f"  [COUNTDOWN]   2. Webserver ON: File → Settings → Webserver → Enable → Restart")
            print(f"  [COUNTDOWN]   3. Camera USB connected and switched ON")
            print(f"  [COUNTDOWN]   4. Test URL: {DIGICAM_URL}/session.json")

    # ── Done ─────────────────────────────────────────────────────
    with _countdown_lock:
        _countdown_state.update({"active": False, "phase": None,
                                  "number": None, "preset": None})
    print(f"  [COUNTDOWN] ══ Sequence complete ══\n")


def abort_countdown():
    """Cancel any running countdown."""
    with _countdown_lock:
        _countdown_state["active"] = False
        _countdown_state["phase"]  = None
        _countdown_state["number"] = None
        _countdown_state["preset"] = None


# ═══════════════════════════════════════════════════════════════════
#  CONTINUOUS SCAN LOOP (Fist gesture)
# ═══════════════════════════════════════════════════════════════════

def _scan_loop_worker():
    """
    Background thread that oscillates the robot between two joint positions.
    
    Sequence:
      1. Move to Position 1, wait SCAN_INITIAL_DELAY (5s) for robot to arrive
      2. Move to Position 2, wait SCAN_LOOP_DELAY (0.5s)
      3. Move to Position 1, wait SCAN_LOOP_DELAY (0.5s)
      4. Repeat steps 2-3 until stopped by open palm gesture
    """
    global _scan_active, _scan_current_pos
    
    if not robot or not robot.connected or not robot.enabled:
        print("  [SCAN ERROR] Robot not available")
        _scan_active = False
        return
    
    print("  [SCAN] ═══ Starting continuous scan ═══")
    _scan_active = True
    
    # Step 1: Move to Position 1 with long initial delay
    _scan_current_pos = "pos1"
    print(f"  [SCAN] → Position 1 (initial move, waiting {SCAN_INITIAL_DELAY}s)")
    robot.move_to_joints(SCAN_POSITION_1["joints"], SCAN_POSITION_1["name"])
    robot.current_preset = "scan"
    
    # Wait for initial delay (check stop event periodically)
    if _scan_stop_event.wait(timeout=SCAN_INITIAL_DELAY):
        print("  [SCAN] ■ Stopped during initial move")
        _scan_active = False
        _scan_current_pos = None
        return
    
    # Step 2+: Oscillate between positions
    cycle = 0
    while not _scan_stop_event.is_set():
        cycle += 1
        
        # Move to Position 2
        _scan_current_pos = "pos2"
        print(f"  [SCAN] → Position 2 (cycle {cycle})")
        robot.move_to_joints(SCAN_POSITION_2["joints"], SCAN_POSITION_2["name"])
        
        if _scan_stop_event.wait(timeout=SCAN_LOOP_DELAY):
            break
        
        # Move to Position 1
        _scan_current_pos = "pos1"
        print(f"  [SCAN] → Position 1 (cycle {cycle})")
        robot.move_to_joints(SCAN_POSITION_1["joints"], SCAN_POSITION_1["name"])
        
        if _scan_stop_event.wait(timeout=SCAN_LOOP_DELAY):
            break
    
    print(f"  [SCAN] ■ Scan stopped after {cycle} cycles")
    _scan_active = False
    _scan_current_pos = None
    robot.current_preset = None


def start_scan():
    """Start the continuous scan loop in a background thread."""
    global _scan_thread, _scan_active
    
    if _scan_active:
        print("  [SCAN] Already scanning")
        return
    
    _scan_stop_event.clear()
    _scan_thread = threading.Thread(target=_scan_loop_worker, daemon=True)
    _scan_thread.start()


def stop_scan():
    """Stop the continuous scan loop."""
    global _scan_active
    
    if not _scan_active:
        return
    
    print("  [SCAN] ■ Stop requested")
    _scan_stop_event.set()
    # The worker thread will clean up _scan_active and _scan_current_pos


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
            if robot_ok and not _scan_active:
                start_scan()
                print(f"  [GESTURE] Fist → START CONTINUOUS SCAN")
            _debounce_counter = 0
            _cooldown_counter = COOLDOWN_FRAMES * 3
            return "scan"

        # ── Stop continuous scan (Open Palm) ────────────────────
        elif action == "stop_scan":
            if _scan_active:
                stop_scan()
                print(f"  [GESTURE] All Fingers → STOP SCAN")
            _debounce_counter = 0
            _cooldown_counter = COOLDOWN_FRAMES * 2
            return None

        # ── Robot movement + auto countdown + capture ───────────
        elif action and isinstance(action, int):
            if _scan_active:
                stop_scan()
                time.sleep(0.2)

            # Move robot (if connected)
            if robot_ok:
                robot.move_to_preset(action)
                print(f"  [GESTURE] Gesture {gesture_id} → Robot Preset {action}")
            else:
                print(f"  [GESTURE] Gesture {gesture_id} → Preset {action} (robot offline — camera only)")

            # ── Always start countdown + auto capture ──────────
            abort_countdown()
            threading.Thread(
                target=countdown_and_capture,
                args=(action,),
                daemon=True,
                name=f"countdown_preset_{action}"
            ).start()

            _debounce_counter = 0
            _cooldown_counter = COUNTDOWN_COOLDOWN_FRAMES
            return action

        _debounce_counter = 0

    return robot.current_preset if robot_ok else None


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
        
        # Scan loop indicator
        elif _scan_active:
            pos_label = "POS 1" if _scan_current_pos == "pos1" else "POS 2"
            pt = f"SCANNING: {pos_label}"
            pc = (0, 200, 255)  # Orange for scanning
            # Pulsing scan indicator
            pulse = int(127 + 128 * np.sin(time.time() * 4))
            cv2.putText(out, "SCANNING", (FRAME_WIDTH//2-80, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, pulse, 255), 2)
        
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
        elif canon and canon.last_error:
            cs = f"CANON: {canon.last_error[:28]}"
            cc = (0, 80, 255)
        else:
            cs, cc = "CANON: OFFLINE", (0, 80, 255)
        cv2.putText(out, cs, (FRAME_WIDTH-230, FRAME_HEIGHT-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, 1)
        # Blink "RECONNECTING..." if a capture was just attempted while offline
        if canon and not canon.connected and (time.time() - canon._last_reconnect_attempt) < 3:
            cv2.putText(out, "RECONNECTING...", (FRAME_WIDTH//2 - 90, FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # ── Countdown overlay (drawn on top of everything) ────────────
    with _countdown_lock:
        cd = _countdown_state.copy()

    if cd.get("active"):
        # Semi-transparent dark vignette
        vignette = out.copy()
        cv2.rectangle(vignette, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
        out = cv2.addWeighted(vignette, 0.45, out, 0.55, 0)

        cx = FRAME_WIDTH // 2
        cy = FRAME_HEIGHT // 2
        phase = cd.get("phase")

        if phase == "settling":
            # "Bersiap..." pulsing text
            pulse = 0.6 + 0.4 * abs(np.sin(time.time() * 3))
            color = (int(255 * pulse), int(220 * pulse), 0)
            cv2.putText(out, "Bersiap...", (cx - 110, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

        elif phase == "intro":
            txt = COUNTDOWN_SPEAK_INTRO
            (tw, _th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            cv2.putText(out, txt, (cx - tw // 2, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, f"Preset {cd.get('preset', '?')} terkunci", (cx - 90, cy + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        elif phase == "counting":
            num = cd.get("number")
            if num is not None:
                num_colors = {3: (0, 220, 255), 2: (0, 165, 255), 1: (0, 80, 255)}
                color = num_colors.get(num, (255, 255, 255))
                num_str = str(num)
                (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
                cv2.putText(out, num_str, (cx - tw // 2, cy + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10, cv2.LINE_AA)
                cv2.putText(out, COUNTDOWN_SPEAK_INTRO, (cx - 130, cy - th // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

        elif phase == "capturing":
            flash = out.copy()
            cv2.rectangle(flash, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255), -1)
            out = cv2.addWeighted(flash, 0.4, out, 0.6, 0)
            cv2.putText(out, "FOTO!", (cx - 90, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 220, 0), 6, cv2.LINE_AA)

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
    document.getElementById('canonLast').textContent=d.path.split(/[\\/]/).pop()}
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
  // Merge in names from the live API response
  if(m.presets){for(const[k,v] of Object.entries(m.presets)){const s=m.gesture_to_preset[k];if(s&&!isNaN(s))presets[parseInt(k)]=v;}}
  const specials={'AUTO':'color:#8b949e;font-style:italic','SCAN':'color:#ff6b35;font-weight:700','STOP':'color:#f85149;font-weight:700'};
  const g=document.getElementById('gestureMap');g.innerHTML='';
  for(let i=1;i<=10;i++){const act=presets[i]||'-';const style=specials[act]||'';
    g.innerHTML+='<div class="gid">'+i+'</div><div class="gname">'+(names[i]||'?')+'</div><div class="gjog" style="'+style+'">'+act+'</div>'}})}
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
  if(d.last_capture){document.getElementById('canonLast').textContent=d.last_capture.split(/[\\/]/).pop()}
  const errRow=document.getElementById('canonErrRow');
  if(d.last_error){document.getElementById('canonErr').textContent=d.last_error;errRow.style.display='flex'}
  else{errRow.style.display='none'}
}).catch(()={})}
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
    d["scan_active"] = _scan_active
    d["scan_position"] = _scan_current_pos
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
    if _scan_active: stop_scan()
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

@flask_app.route("/camera/reconnect", methods=["POST"])
def api_camera_reconnect():
    """Try to reconnect to digiCamControl — will auto-launch it if not running."""
    if not canon:
        return jsonify({"error": "Camera not initialized"}), 503
    # Reset both cooldown timers so detect/reconnect runs immediately
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
    global INITIAL_POSE, PRESETS, GESTURE_TO_PRESET

    parser = argparse.ArgumentParser(description="Finger Gesture + Dobot Nova 5 + Canon EOS 6D (All-in-One)")
    parser.add_argument("--no-robot", action="store_true", help="Vision only, no robot")
    parser.add_argument("--no-camera", action="store_true", help="Disable Canon camera capture")
    parser.add_argument("--ip", default=DOBOT_IP, help=f"Dobot IP (default: {DOBOT_IP})")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help=f"Web port (default: {FLASK_PORT})")
    parser.add_argument("--photo-dir", default=CANON_SAVE_DIR, help=f"Photo save directory")
    parser.add_argument("--presets-json", default=DOBOTSTUDIO_PRESETS_JSON,
                        help=f"Path to DobotStudio Pro presets.json (default: {DOBOTSTUDIO_PRESETS_JSON})")
    args = parser.parse_args()
    DOBOT_IP = args.ip; dry_run = args.no_robot

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
        def fake_move_to_joints(joints, name="Custom"):
            joint_str = ",".join([f"{j:.2f}" for j in joints])
            print(f"  [DRY RUN] JointMovJ({joint_str})  # {name}")
            robot.last_move_time = time.time()
            return True
        robot.move_to_joints = fake_move_to_joints
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
        # Auto-launch digiCamControl if not already running
        if DIGICAM_AUTOLAUNCH and not canon._is_digicam_running():
            print("  [CANON] digiCamControl not running — attempting auto-launch...")
            canon._launch_digicam()
        if canon.detect():
            print(f"  [CANON] ✓ Camera ready! Photos → {canon.save_dir}")
            print(f"  [CANON] Auto-shutter fires after every preset move (countdown timer)")
        else:
            print("  [CANON] ⚠ Camera not detected at startup — will retry before each capture")
            print(f"  [CANON]   Make sure digiCamControl is open and webserver is enabled")
            print(f"  [CANON]   File → Settings → Webserver → Enable → Restart digiCamControl")
            # Keep canon object alive so retries happen during capture
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
        print(f"  ✓ Canon EOS 6D → auto-capture after each preset countdown (method: {canon.capture_method})")
    if COUNTDOWN_ENABLED and AUTO_CAPTURE_AFTER_PRESET:
        print(f"  ✓ Audio countdown enabled: '{COUNTDOWN_SPEAK_INTRO}... 3... 2... 1...'")
        print(f"    Settle delay: {ROBOT_SETTLE_DELAY}s │ Voice: Windows SAPI (built-in)")
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
        if _scan_active: stop_scan()
        if robot and not dry_run: robot.disconnect()
        if camera: camera.release()
        if hands_detector: hands_detector.close()
        if KEYBOARD_AVAILABLE:
            try: keyboard.unhook_all()
            except: pass
        print("  ✓ Shutdown complete.")


if __name__ == "__main__":
    main()