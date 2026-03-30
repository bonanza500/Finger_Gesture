import socket
import time
import json
import os

class DobotController:

    def __init__(self, ip: str, port: int = 29999):
        self.ip     = ip
        self.port   = port
        self.client = None
        self.current_preset_index = None  # Track current position
        self.reset_position_index = None  # Index for P4 (0 degree position)

    # ── Koneksi ───────────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(5)
            self.client.connect((self.ip, self.port))
            print(f"[DOBOT] ✓ Terhubung ke {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[DOBOT] ✗ Gagal terhubung: {e}")
            self.client = None
            return False

    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
            print("[DOBOT] Koneksi ditutus.")

    # ── Kirim perintah ────────────────────────────────────────────────
    def send_cmd(self, cmd: str, debug: bool = True) -> str | None:
        if not self.client:
            return None
        try:
            self.client.sendall((cmd + "\n").encode('utf-8'))
            response = self.client.recv(4096).decode('utf-8').strip()
            if debug:
                print(f"    >> {cmd}")
                print(f"    << {response}")
            return response
        except Exception as e:
            print(f"[DOBOT] Error saat kirim '{cmd}': {e}")
            return None

    # ── Inisialisasi ──────────────────────────────────────────────────
    def initialize(self):
        print("[DOBOT] Inisialisasi...")
        self.send_cmd("RequestControl()")
        time.sleep(1)
        self.send_cmd("ClearError()")
        time.sleep(1)
        self.send_cmd("EnableRobot()")
        print("[DOBOT] Menunggu servo aktif...")
        time.sleep(3)
        print("[DOBOT] ✓ Robot siap.")

    def shutdown(self):
        print("[DOBOT] Mematikan robot...")
        self.send_cmd("DisableRobot()")
        time.sleep(1)
        self.disconnect()

    # ── Cek status robot ──────────────────────────────────────────────
    def is_moving(self) -> bool:
        response = self.send_cmd("RobotMode()", debug=False)
        if response is None:
            return False
        try:
            # Format respons: "0,{mode},RobotMode();"
            mode = int(response.split(",")[1].strip())
            return mode == 5
        except Exception:
            return False

    def wait_until_done(self, timeout: float = 30.0, poll_interval: float = 0.3) -> bool:
        print("[DOBOT] ⏳ Menunggu robot selesai bergerak...")
        # Jeda singkat supaya robot sempat mulai bergerak sebelum dicek
        time.sleep(0.5)
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_moving():
                elapsed = time.time() - start + 0.5
                print(f"[DOBOT] ✓ Robot selesai. ({elapsed:.1f}s)")
                return True
            time.sleep(poll_interval)
        print(f"[DOBOT] ⚠ Timeout setelah {timeout}s.")
        return False

    # ── Set reset position (P4) ───────────────────────────────────────
    def set_reset_position(self, preset_index: int, preset_name: str):
        """Set which preset is the reset/home position (usually P4 at 0 degrees)"""
        self.reset_position_index = preset_index
        print(f"[DOBOT] Reset position set to index {preset_index}: {preset_name}")

    # ── Gerak preset dengan auto-reset ───────────────────────────────
    def move_to_preset(self, joint_str: str, name: str = "", preset_index: int = None) -> bool:
        """Kirim JointMovJ. Tidak memblokir — panggil wait_until_done() setelahnya."""
        label = f" ({name})" if name else ""
        print(f"[DOBOT] → Pindah ke{label}")
        response = self.send_cmd(f"JointMovJ({joint_str})")
        
        # Update current position after successful move
        if response is not None and preset_index is not None:
            self.current_preset_index = preset_index
        
        return response is not None

    def move_to_preset_with_reset(self, preset: dict, preset_index: int, 
                                   reset_preset: dict = None, timeout: float = 30.0) -> bool:
        """
        Move to preset position with automatic reset via P4.
        
        Flow:
        - If target is P4 (reset position) → move directly
        - If already at P4 → move directly to target
        - Otherwise → move to P4 first, then to target
        
        Args:
            preset: Target preset dict with 'name' and 'joint_str'
            preset_index: Index of target preset
            reset_preset: P4 reset preset dict (if None, moves directly)
            timeout: Timeout for each movement
        
        Returns:
            bool: True if all movements succeeded
        """
        target_name = preset['name']
        
        # Case 1: Target IS the reset position → move directly
        if preset_index == self.reset_position_index:
            print(f"[DOBOT] 🎯 Target adalah reset position ({target_name}) → langsung")
            success = self.move_to_preset(preset['joint_str'], target_name, preset_index)
            if success:
                self.wait_until_done(timeout)
            return success
        
        # Case 2: Already AT reset position → move directly to target
        if self.current_preset_index == self.reset_position_index:
            print(f"[DOBOT] 📍 Sudah di reset position → langsung ke {target_name}")
            success = self.move_to_preset(preset['joint_str'], target_name, preset_index)
            if success:
                self.wait_until_done(timeout)
            return success
        
        # Case 3: Need to go through reset position first
        if reset_preset is None:
            print(f"[DOBOT] ⚠️  No reset preset defined, moving directly to {target_name}")
            success = self.move_to_preset(preset['joint_str'], target_name, preset_index)
            if success:
                self.wait_until_done(timeout)
            return success
        
        # Move via reset position: Current → Reset → Target
        current_name = f"P{self.current_preset_index}" if self.current_preset_index is not None else "Unknown"
        reset_name = reset_preset['name']
        
        print(f"\n[DOBOT] 🔄 Perpindahan: {current_name} → {reset_name} (reset) → {target_name}")
        print(f"[DOBOT]     Step 1/2: {current_name} → {reset_name}")
        
        # Step 1: Move to reset position (P4)
        success_reset = self.move_to_preset(reset_preset['joint_str'], reset_name, self.reset_position_index)
        if not success_reset:
            print(f"[DOBOT] ❌ Gagal pindah ke reset position!")
            return False
        
        # Wait for reset movement to complete
        if not self.wait_until_done(timeout):
            print(f"[DOBOT] ❌ Timeout saat pindah ke reset position!")
            return False
        
        print(f"[DOBOT] ✓ Reset position tercapai")
        print(f"[DOBOT]     Step 2/2: {reset_name} → {target_name}")
        
        # Step 2: Move to target position
        success_target = self.move_to_preset(preset['joint_str'], target_name, preset_index)
        if not success_target:
            print(f"[DOBOT] ❌ Gagal pindah ke target position!")
            return False
        
        # Wait for target movement to complete
        if not self.wait_until_done(timeout):
            print(f"[DOBOT] ❌ Timeout saat pindah ke target position!")
            return False
        
        print(f"[DOBOT] ✅ Target position tercapai: {target_name}\n")
        return True

    def get_status(self):
        return {
            "connected": self.client is not None,
            "ip": self.ip,
            "current_preset_index": self.current_preset_index,
            "reset_position_index": self.reset_position_index,
            "is_moving": self.is_moving()
        }

# ── Helper: Load preset dari JSON ────────────────────────────────────
def load_presets(filename: str) -> list[dict]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File preset tidak ditemukan: '{filename}'")

    with open(filename, 'r') as f:
        data = json.load(f)

    presets = []
    for p in data:
        name   = p.get("name", "Unknown")
        joints = p.get("joint", [])
        if len(joints) == 6:
            joint_str = ",".join(str(round(j, 4)) for j in joints)
            presets.append({"name": name, "joint_str": joint_str})

    print(f"[DOBOT] {len(presets)} preset dimuat dari '{filename}'")
    return presets


def find_reset_position_index(presets: list[dict], reset_name: str = "P4") -> int | None:
    """Find index of reset position (usually P4) in presets list"""
    for idx, preset in enumerate(presets):
        if preset['name'] == reset_name:
            return idx
    return None
