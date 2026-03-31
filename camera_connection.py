import os
import subprocess
import threading
import time
import urllib.error
import urllib.request


DEFAULT_CONFIG = {
    "CANON_ENABLED": True,
    "CANON_SAVE_DIR": r"C:\CapturedPhotos",
    "CANON_CAPTURE_GESTURE": 7,
    "CANON_CAPTURE_COOLDOWN": 90,
    "DIGICAM_URL": "http://localhost:5513",
    "DIGICAM_CAPTURE_TIMEOUT": 15,
    "DIGICAM_EXE": r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe",
    "DIGICAM_APP_EXE": r"C:\Program Files (x86)\digiCamControl\CameraControl.exe",
    "DIGICAM_AUTOLAUNCH": True,
    "DIGICAM_LAUNCH_WAIT": 8,
}


class CanonCamera:
    """Simple camera bridge using digiCamControl (HTTP first, CLI fallback)."""

    def __init__(self, config=None):
        cfg = dict(DEFAULT_CONFIG)
        if isinstance(config, dict):
            cfg.update(config)

        self.enabled = cfg["CANON_ENABLED"]
        self.save_dir = cfg["CANON_SAVE_DIR"]
        self.capture_gesture = cfg["CANON_CAPTURE_GESTURE"]
        self.cooldown_frames = cfg["CANON_CAPTURE_COOLDOWN"]
        self.digicam_url = cfg["DIGICAM_URL"].rstrip("/")
        self.capture_timeout = cfg["DIGICAM_CAPTURE_TIMEOUT"]
        self.digicam_exe = cfg["DIGICAM_EXE"]
        self.digicam_app = cfg["DIGICAM_APP_EXE"]
        self.autolaunch = cfg["DIGICAM_AUTOLAUNCH"]
        self.launch_wait = cfg["DIGICAM_LAUNCH_WAIT"]

        self.connected = False
        self.capture_method = None
        self.last_capture_path = None
        self.last_capture_time = 0.0
        self.capture_count = 0
        self.last_error = ""

        self._lock = threading.Lock()
        self._capture_in_progress = False

        os.makedirs(self.save_dir, exist_ok=True)

    def _http_get(self, path, timeout=5):
        try:
            req = urllib.request.Request(f"{self.digicam_url}{path}")
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.read().decode("utf-8", errors="replace").strip()
        except urllib.error.URLError as e:
            self.last_error = f"HTTP: {e.reason}"
            return None
        except Exception as e:
            self.last_error = f"HTTP: {e}"
            return None

    def _is_app_running(self):
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq CameraControl.exe", "/NH"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return "CameraControl.exe" in result.stdout
        except Exception:
            return False

    def _launch_app(self):
        if self._is_app_running():
            return True
        if not self.autolaunch:
            self.last_error = "CameraControl is not running"
            return False

        candidates = [
            self.digicam_app,
            r"C:\Program Files (x86)\digiCamControl\CameraControl.exe",
            r"C:\Program Files (x86)\digiCamControl\CameraControl.exe",
            r"C:\Program Files\digiCamControl\CameraControl.exe",
            r"C:\Program Files\digiCamControl\digiCamControl.exe",
        ]
        app_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if not app_path:
            self.last_error = "digiCamControl app not found"
            return False

        try:
            print(f"[CAMERA] Launching digiCamControl: {app_path}")
            if app_path.lower().endswith(".exe"):
                os.startfile(app_path)
            else:
                subprocess.Popen([app_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(self.launch_wait)
            return self._is_app_running() or self._http_get("/session.json", timeout=2) is not None
        except Exception as e:
            self.last_error = f"Launch failed: {e}"
            return False

    def detect(self, silent=False):
        if not self.enabled:
            self.connected = False
            self.last_error = "Camera disabled"
            return False

        app_ok = self._is_app_running() or self._launch_app()
        web_ok = self._http_get("/session.json", timeout=2) is not None
        cli_ok = os.path.exists(self.digicam_exe)

        if web_ok:
            self.connected = True
            self.capture_method = "http"
            self.last_error = ""
            return True
        if app_ok and cli_ok:
            self.connected = True
            self.capture_method = "cli"
            self.last_error = ""
            return True

        self.connected = False
        self.capture_method = None
        if not self.last_error:
            self.last_error = "No HTTP webserver and no CLI tool"
        if not silent:
            print(f"[CAMERA] Not ready: {self.last_error}")
        return False

    def is_ready_for_capture(self):
        if not self.enabled or self._capture_in_progress:
            return False
        return (time.time() - self.last_capture_time) * 30 > self.cooldown_frames

    def _capture_http(self):
        result = self._http_get("/?CMD=Capture", timeout=self.capture_timeout)
        if result is not None:
            return "captured_http"
        return None

    def _capture_cli(self):
        if not os.path.exists(self.digicam_exe):
            self.last_error = f"CLI not found: {self.digicam_exe}"
            return None
        try:
            command = [self.digicam_exe, "/capture", "/filename", os.path.join(self.save_dir, "capture.jpg")]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=self.capture_timeout)
            if proc.returncode == 0:
                return "captured_cli"
            self.last_error = (proc.stderr or proc.stdout or "CLI capture failed").strip()
            return None
        except Exception as e:
            self.last_error = f"CLI: {e}"
            return None

    def capture_photo(self):
        return self.capture() is not None

    def capture(self):
        if not self.enabled:
            return None
        if not self.is_ready_for_capture():
            return None

        with self._lock:
            self._capture_in_progress = True
            try:
                # Requirement: verify app is open first; launch if not.
                if not self._is_app_running() and not self._launch_app():
                    self.connected = False
                    return None

                filepath = self._capture_http()
                method = "http"
                if filepath is None:
                    filepath = self._capture_cli()
                    method = "cli"

                if filepath is None:
                    self.connected = False
                    return None

                self.connected = True
                self.capture_method = method
                self.capture_count += 1
                self.last_capture_path = filepath
                self.last_capture_time = time.time()
                self.last_error = ""
                print(f"[CAMERA] Photo #{self.capture_count} via {method}")
                return filepath
            finally:
                self._capture_in_progress = False

    def get_status(self):
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "capture_method": self.capture_method,
            "capture_count": self.capture_count,
            "last_capture": self.last_capture_path,
            "last_capture_time": self.last_capture_time,
            "last_error": self.last_error,
            "capture_in_progress": self._capture_in_progress,
        }

