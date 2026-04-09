"""
Canon EOS 6D Camera Controller
Three independent capture methods tried in order:
  1. digiCamControl HTTP webserver  (best — full metadata)
  2. digiCamControl CLI             (fallback if webserver off)
  3. Canon EDSDK.dll               (zero-install, always works)
"""

import os
import subprocess
import threading
import time
import urllib.request
import urllib.error

# ── Default configuration ─────────────────────────────────────────

CANON_SAVE_DIR         = r"C:\CapturedPhotos"
CANON_CAPTURE_GESTURE  = 7
CANON_CAPTURE_COOLDOWN = 90
DIGICAM_URL            = "http://localhost:5513"
DIGICAM_CAPTURE_TIMEOUT = 15
DIGICAM_EXE            = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"
DIGICAM_APP_EXE        = r"C:\Program Files (x86)\digiCamControl\digiCamControl.exe"
DIGICAM_AUTOLAUNCH     = True
DIGICAM_LAUNCH_WAIT    = 8


# ── Audio helpers ─────────────────────────────────────────────────

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# ── Countdown configuration ───────────────────────────────────────

COUNTDOWN_ENABLED          = True
AUTO_CAPTURE_AFTER_PRESET  = True
ROBOT_SETTLE_DELAY         = 2.5
COUNTDOWN_SPEAK_INTRO      = "Lihat ke kamera"
COUNTDOWN_NUMBERS          = [3, 2, 1]
COUNTDOWN_NUMBER_DELAY     = 0.8
TTS_SPEECH_RATE            = 130
COUNTDOWN_COOLDOWN_FRAMES  = int((ROBOT_SETTLE_DELAY + len(COUNTDOWN_NUMBERS) * COUNTDOWN_NUMBER_DELAY + 3) * 30)

# ── Session timeout configuration ─────────────────────────────────

SESSION_TIMEOUT_ENABLED    = True
SESSION_TIMEOUT_SECONDS    = 60  # 5 minutes = 300 seconds
_session_start_time        = None  # Track when session started
_session_timeout_callback  = None  # Callback function when timeout occurs


def _speak(text, block=True):
    """
    Speak text using Windows SAPI via PowerShell.
    block=True  → wait for speech to finish (used during countdown).
    block=False → fire-and-forget (used when we need to shoot immediately after).
    """
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


# ═══════════════════════════════════════════════════════════════════
#  SESSION TIMEOUT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def start_session_timer():
    """Start tracking session time. Call this when system starts."""
    global _session_start_time
    _session_start_time = time.time()
    print(f"  [SESSION] ⏱️  Session timer started (timeout in {SESSION_TIMEOUT_SECONDS}s)")


def get_session_elapsed_time():
    """Get elapsed time in seconds since session started."""
    if _session_start_time is None:
        return 0.0
    return time.time() - _session_start_time


def get_session_remaining_time():
    """Get remaining time before timeout in seconds."""
    if not SESSION_TIMEOUT_ENABLED or _session_start_time is None:
        return SESSION_TIMEOUT_SECONDS
    elapsed = get_session_elapsed_time()
    remaining = SESSION_TIMEOUT_SECONDS - elapsed
    return max(0.0, remaining)


def is_session_timeout():
    """Check if 5-minute session timeout has been reached."""
    if not SESSION_TIMEOUT_ENABLED or _session_start_time is None:
        return False
    elapsed = get_session_elapsed_time()
    return elapsed >= SESSION_TIMEOUT_SECONDS


def set_session_timeout_callback(callback):
    """Register a callback function to be called when session times out."""
    global _session_timeout_callback
    _session_timeout_callback = callback


def check_session_timeout():
    """
    Check if timeout reached. If so, invoke callback and print warning.
    Call this periodically from main loop.
    Returns True if timeout reached.
    """
    if is_session_timeout():
        remaining = get_session_remaining_time()
        if remaining <= 0:
            print("  [SESSION] ⏱️  5-MINUTE SESSION TIMEOUT REACHED!")
            print("  [SESSION] Shutting down system...")
            if _session_timeout_callback:
                _session_timeout_callback()
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
#  CANON CAMERA CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class CanonCamera:

    def __init__(self, save_dir=CANON_SAVE_DIR, digicam_url=DIGICAM_URL):
        self.save_dir       = save_dir
        self.base_url       = digicam_url.rstrip("/")
        self.connected      = False   # True once any method is confirmed working
        self.capture_method = None    # "http" | "cli" | "edsdk"
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
        try:
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq digiCamControl.exe", "/NH"],
                               capture_output=True, text=True, timeout=5)
            return "digiCamControl.exe" in r.stdout
        except Exception:
            return False

    def _find_digicam_exe(self, exe_name):
        """Search registry + common paths for a digiCamControl executable."""
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

    # ── Detection ─────────────────────────────────────────────────

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
#  COUNTDOWN ENGINE
# ═══════════════════════════════════════════════════════════════════

def countdown_and_capture(preset_num, canon, countdown_state, countdown_lock):
    """
    Guaranteed flow every time a preset gesture is detected:
      1. Settle  — wait ROBOT_SETTLE_DELAY seconds for robot to reach position
      2. Intro   — say "Lihat ke kamera"
      3. Count   — 3 … 2 … 1  (beep + Indonesian spoken number)
      4. SHUTTER — fire digiCamControl capture immediately (no extra speech delay)
    """
    print(f"\n  [COUNTDOWN] ══ Preset {preset_num} ══ Starting auto-capture sequence")

    # ── 1. Settle ────────────────────────────────────────────────
    with countdown_lock:
        countdown_state.update({"active": True, "phase": "settling",
                                 "number": None, "preset": preset_num})
    print(f"  [COUNTDOWN] ⏳ Settling {ROBOT_SETTLE_DELAY}s for robot to reach position...")
    time.sleep(ROBOT_SETTLE_DELAY)

    # ── 2. "Lihat ke kamera" ─────────────────────────────────────
    with countdown_lock:
        countdown_state.update({"phase": "intro", "number": None})
    print(f"  [COUNTDOWN] 🔊 {COUNTDOWN_SPEAK_INTRO}")
    _speak(COUNTDOWN_SPEAK_INTRO, block=True)   # blocking — subject needs time to look

    # ── 3. 3 … 2 … 1 ─────────────────────────────────────────────
    with countdown_lock:
        countdown_state["phase"] = "counting"

    num_words  = {1: "satu", 2: "dua", 3: "tiga"}
    beep_freqs = {3: 700,   2: 850,   1: 1000}

    for num in COUNTDOWN_NUMBERS:
        with countdown_lock:
            countdown_state["number"] = num
        print(f"  [COUNTDOWN] 🔊 {num}...")
        _beep(beep_freqs.get(num, 880), 150)
        _speak(num_words.get(num, str(num)), block=True)
        time.sleep(COUNTDOWN_NUMBER_DELAY)

    # ── 4. FIRE SHUTTER — immediately, no extra speech ───────────
    with countdown_lock:
        countdown_state.update({"phase": "capturing", "number": None})

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
            print(f"  [COUNTDOWN]   4. Test URL: {canon.base_url}/session.json")

    # ── Done ─────────────────────────────────────────────────────
    with countdown_lock:
        countdown_state.update({"active": False, "phase": None,
                                 "number": None, "preset": None})
    print(f"  [COUNTDOWN] ══ Sequence complete ══\n")


def abort_countdown(countdown_state, countdown_lock):
    """Cancel any running countdown."""
    with countdown_lock:
        countdown_state["active"] = False
        countdown_state["phase"]  = None
        countdown_state["number"] = None
        countdown_state["preset"] = None
