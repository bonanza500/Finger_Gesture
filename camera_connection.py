import requests
import subprocess
import time
import os
import platform

class CanonCamera:
    def __init__(self, config):
        self.enabled = config.get("CANON_ENABLED", False)
        if not self.enabled:
            print("[CAMERA] Canon camera features are disabled.")
            return

        self.save_dir = config["CANON_SAVE_DIR"]
        self.capture_gesture = config["CANON_CAPTURE_GESTURE"]
        self.cooldown_frames = config["CANON_CAPTURE_COOLDOWN"]
        self.digicam_url = config["DIGICAM_URL"]
        self.digicam_exe = config["DIGICAM_EXE"]
        self.digicam_app_exe = config["DIGICAM_APP_EXE"]
        self.autolaunch = config["DIGICAM_AUTOLAUNCH"]
        self.launch_wait = config["DIGICAM_LAUNCH_WAIT"]
        self.capture_timeout = config["DIGICAM_CAPTURE_TIMEOUT"]
        
        self._last_capture_time = 0
        self._waiting_for_capture = False

        if self.autolaunch:
            self._ensure_digicam_running()

    def _is_process_running(self, process_name):
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output(f'tasklist /FI "IMAGENAME eq {process_name}"', shell=True).decode()
                return process_name in output
            else: # macOS / Linux
                output = subprocess.check_output(f"pgrep -f {process_name}", shell=True).decode()
                return len(output.strip()) > 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _ensure_digicam_running(self):
        if not os.path.exists(self.digicam_app_exe):
            print(f"[CAMERA] ❌ digiCamControl not found at: {self.digicam_app_exe}")
            self.enabled = False
            return

        if self._is_process_running(os.path.basename(self.digicam_app_exe)):
            print("[CAMERA] ✓ digiCamControl is already running.")
            return

        print("[CAMERA] digiCamControl is not running. Launching...")
        try:
            subprocess.Popen([self.digicam_app_exe])
            print(f"[CAMERA] Waiting {self.launch_wait}s for it to start...")
            time.sleep(self.launch_wait)
            print("[CAMERA] ✓ digiCamControl launched.")
        except Exception as e:
            print(f"[CAMERA] ❌ Failed to launch digiCamControl: {e}")
            self.enabled = False

    def _capture_http(self):
        print("[CAMERA] Attempting capture via HTTP webserver...")
        try:
            # Command to capture and transfer to PC
            url = f"{self.digicam_url}/?CMD=CaptureAll"
            response = requests.get(url, timeout=self.capture_timeout)
            if response.status_code == 200:
                print("[CAMERA] ✓ Capture command sent successfully via HTTP.")
                return True
            else:
                print(f"[CAMERA] ❌ HTTP capture failed. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[CAMERA] ❌ HTTP request failed: {e}")
            return False

    def _capture_cli(self):
        if not os.path.exists(self.digicam_exe):
            print(f"[CAMERA] ❌ CLI tool not found: {self.digicam_exe}")
            return False
        
        print("[CAMERA] Attempting capture via CLI...")
        try:
            command = [self.digicam_exe, "/capture", "/filename", os.path.join(self.save_dir, "capture.jpg")]
            subprocess.run(command, timeout=self.capture_timeout, check=True, shell=True)
            print("[CAMERA] ✓ Capture command sent successfully via CLI.")
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"[CAMERA] ❌ CLI capture failed: {e}")
            return False

    def capture_photo(self):
        if not self.enabled:
            return False
        
        if self._waiting_for_capture:
            print("[CAMERA] Already waiting for a capture to complete.")
            return False

        print("\n[CAMERA] --- TRIGGERING PHOTO CAPTURE ---")
        self._waiting_for_capture = True
        
        success = self._capture_http()
        if not success:
            print("[CAMERA] HTTP failed, falling back to CLI...")
            success = self._capture_cli()

        if success:
            self._last_capture_time = time.time()
            print("[CAMERA] ✓ Photo capture initiated.")
        else:
            print("[CAMERA] ❌ All capture methods failed.")

        self._waiting_for_capture = False
        return success

    def is_ready_for_capture(self):
        if not self.enabled or self._waiting_for_capture:
            return False
        
        return (time.time() - self._last_capture_time) * 30 > self.cooldown_frames
