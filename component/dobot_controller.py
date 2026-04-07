import socket
import time
import json
import os
import threading

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION — fallback presets
# ═══════════════════════════════════════════════════════════════════

# ── DobotStudio Pro presets.json path ───────────────────────────
# The file is written by DobotStudio Pro at this location.
# Change the path here if your installation is elsewhere.
DOBOTSTUDIO_PRESETS_JSON = r"E:\Magang\Jonas\ArmRobot\Dobot\v2\preset\presets.json"

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

# ── Scan Positions (Fist gesture oscillation) ────────────────────
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


# ═══════════════════════════════════════════════════════════════════
#  PRESET LOADER
# ═══════════════════════════════════════════════════════════════════

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
        slot = 1  # gesture slot counter (1..10)

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

    def move_to_preset(self, preset_num, presets):
        """Move robot to preset position using JointMovJ"""
        if not self.connected or not self.enabled:
            return False

        if preset_num not in presets:
            print(f"  [ROBOT ERROR] Preset {preset_num} not found")
            return False

        preset = presets[preset_num]
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

    def move_axis(self, movement_info):
        """Move robot along a single axis by a delta amount."""
        if not self.connected or not self.enabled:
            return

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

    def get_status(self, presets=None):
        return {
            "connected": self.connected,
            "enabled": self.enabled,
            "ip": self.ip,
            "current_preset": self.current_preset,
            "presets": {k: v["name"] for k, v in presets.items()} if presets else {}
        }


# ═══════════════════════════════════════════════════════════════════
#  CONTINUOUS SCAN LOOP
# ═══════════════════════════════════════════════════════════════════

class ScanController:
    """Manages the continuous scan oscillation loop (Fist gesture)."""

    def __init__(self):
        self._scan_active      = False
        self._scan_thread      = None
        self._scan_stop_event  = threading.Event()
        self._scan_current_pos = None  # "pos1" or "pos2"

    @property
    def active(self):
        return self._scan_active

    @property
    def current_pos(self):
        return self._scan_current_pos

    def start(self, robot):
        """Start the continuous scan loop in a background thread."""
        if self._scan_active:
            print("  [SCAN] Already scanning")
            return

        self._scan_stop_event.clear()
        self._scan_thread = threading.Thread(
            target=self._scan_loop_worker, args=(robot,), daemon=True)
        self._scan_thread.start()

    def stop(self):
        """Stop the continuous scan loop."""
        if not self._scan_active:
            return
        print("  [SCAN] ■ Stop requested")
        self._scan_stop_event.set()

    def _scan_loop_worker(self, robot):
        """
        Background thread that oscillates the robot between two joint positions.

        Sequence:
          1. Move to Position 1, wait SCAN_INITIAL_DELAY (5s) for robot to arrive
          2. Move to Position 2, wait SCAN_LOOP_DELAY (0.5s)
          3. Move to Position 1, wait SCAN_LOOP_DELAY (0.5s)
          4. Repeat steps 2-3 until stopped by open palm gesture
        """
        if not robot or not robot.connected or not robot.enabled:
            print("  [SCAN ERROR] Robot not available")
            self._scan_active = False
            return

        print("  [SCAN] ═══ Starting continuous scan ═══")
        self._scan_active = True

        # Step 1: Move to Position 1 with long initial delay
        self._scan_current_pos = "pos1"
        print(f"  [SCAN] → Position 1 (initial move, waiting {SCAN_INITIAL_DELAY}s)")
        robot.move_to_joints(SCAN_POSITION_1["joints"], SCAN_POSITION_1["name"])
        robot.current_preset = "scan"

        # Wait for initial delay (check stop event periodically)
        if self._scan_stop_event.wait(timeout=SCAN_INITIAL_DELAY):
            print("  [SCAN] ■ Stopped during initial move")
            self._scan_active = False
            self._scan_current_pos = None
            return

        # Step 2+: Oscillate between positions
        cycle = 0
        while not self._scan_stop_event.is_set():
            cycle += 1

            # Move to Position 2
            self._scan_current_pos = "pos2"
            print(f"  [SCAN] → Position 2 (cycle {cycle})")
            robot.move_to_joints(SCAN_POSITION_2["joints"], SCAN_POSITION_2["name"])

            if self._scan_stop_event.wait(timeout=SCAN_LOOP_DELAY):
                break

            # Move to Position 1
            self._scan_current_pos = "pos1"
            print(f"  [SCAN] → Position 1 (cycle {cycle})")
            robot.move_to_joints(SCAN_POSITION_1["joints"], SCAN_POSITION_1["name"])

            if self._scan_stop_event.wait(timeout=SCAN_LOOP_DELAY):
                break

        print(f"  [SCAN] ■ Scan stopped after {cycle} cycles")
        self._scan_active = False
        self._scan_current_pos = None
        robot.current_preset = None
