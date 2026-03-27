"""
═══════════════════════════════════════════════════════
  Canon EOS 6D — Standalone Camera Test & Auto-Shoot
  Run this SEPARATELY from the main script to diagnose
  and test the camera shutter.

  Usage:
    python test_camera.py            # Run all tests
    python test_camera.py --shoot    # Just take a photo now

  Requirements: Canon EOS 6D plugged in via USB, camera ON.
  Close EOS Utility before running.
═══════════════════════════════════════════════════════
"""

import os, sys, time, ctypes, subprocess

SAVE_DIR = r"C:\CapturedPhotos"

# ─────────────────────────────────────────────────────
#  STEP 1: Find EDSDK.dll
# ─────────────────────────────────────────────────────

def find_edsdk():
    import ctypes, sys
    python_is_64 = sys.maxsize > 2**32
    arch_label = "64-bit" if python_is_64 else "32-bit"
    print(f"\n[1] Python is {arch_label}. Searching for matching EDSDK.dll...")

    candidates_64 = [
        r"C:\Program Files\Canon\EOS Utility\EU3\EDSDK.dll",
        r"C:\Program Files\Canon\EOS Utility\EDSDK.dll",
        r"C:\Program Files\Canon\EOS Utility 3\EDSDK.dll",
        r"C:\Windows\System32\EDSDK.dll",
    ]
    candidates_32 = [
        r"C:\Program Files (x86)\Canon\EOS Utility\EU3\EDSDK.dll",
        r"C:\Program Files (x86)\Canon\EOS Utility\EDSDK.dll",
        r"C:\Program Files (x86)\Canon\EOS Utility 3\EDSDK.dll",
        r"C:\Windows\SysWOW64\EDSDK.dll",
    ]

    # Prefer 32-bit (EOS Utility) DLLs since we use 32-bit PowerShell
    # Skip tools unrelated to camera control
    print("   All EDSDK.dll files on this system:")
    for base in [r"C:\Program Files (x86)\Canon", r"C:\Program Files\Canon"]:
        if os.path.exists(base):
            for root, _dirs, files in os.walk(base):
                for f in files:
                    if f.upper() == "EDSDK.DLL":
                        full = os.path.join(root, f)
                        size = os.path.getsize(full)
                        skip_tools = ["Web Service", "Network Setting", "ImageGateway"]
                        skip = any(t in full for t in skip_tools)
                        note = " [SKIP - unrelated tool]" if skip else ""
                        print(f"     {full}  ({size:,} bytes){note}")
                        if not skip:
                            if r"Program Files (x86)" in full:
                                candidates_32.insert(0, full)
                            else:
                                candidates_64.insert(0, full)

    # For the 32-bit PowerShell approach, we want a 32-bit DLL
    # Try 32-bit first regardless of Python bitness
    ordered = candidates_32 + candidates_64

    for p in ordered:
        if p and os.path.exists(p):
            print(f"   ✓ Using: {p}")
            return p

    print("   ✗ No suitable EDSDK.dll found")
    print("     → Is Canon EOS Utility installed?")
    return None


# ─────────────────────────────────────────────────────
#  STEP 2: List all files in EOS Utility folder
# ─────────────────────────────────────────────────────

def list_eos_files():
    print("\n[2] Canon EOS Utility folder contents:")
    for base in [r"C:\Program Files (x86)\Canon\EOS Utility",
                 r"C:\Program Files\Canon\EOS Utility"]:
        if os.path.exists(base):
            for root, _dirs, files in os.walk(base):
                for f in files:
                    full = os.path.join(root, f)
                    size = os.path.getsize(full)
                    print(f"   {full}  ({size:,} bytes)")
            return
    print("   EOS Utility folder not found")


# ─────────────────────────────────────────────────────
#  STEP 3: Check what Canon processes are running
# ─────────────────────────────────────────────────────

def check_processes():
    print("\n[3] Canon processes currently running:")
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq *EOS*", "/FO", "LIST"],
            capture_output=True, text=True, timeout=5
        )
        found = False
        for line in r.stdout.splitlines():
            if line.strip() and "No tasks" not in line:
                print(f"   {line.strip()}")
                found = True
        if not found:
            print("   No EOS processes running (good — USB port is free)")
    except Exception as e:
        print(f"   Error: {e}")


# ─────────────────────────────────────────────────────
#  STEP 4: Kill EOS Utility (frees USB port)
# ─────────────────────────────────────────────────────

def kill_eos_utility():
    print("\n[4] Closing EOS Utility (releases USB port)...")
    killed = False
    for name in ["EOS Utility 3.exe", "EOS Utility.exe", "EOS Utility 2.exe",
                 "EOSDigital.exe", "RemoteCapture.exe"]:
        r = subprocess.run(["taskkill", "/F", "/IM", name],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"   Closed: {name}")
            killed = True
    if not killed:
        print("   Nothing to close (EOS Utility was not running)")
    time.sleep(1.5)


# ─────────────────────────────────────────────────────
#  STEP 5: Take photo via EDSDK
# ─────────────────────────────────────────────────────

def shoot_via_edsdk(dll_path):
    """
    Trigger shutter via 32-bit PowerShell P/Invoke.
    This bypasses the 64-bit Python / 32-bit DLL mismatch completely.
    """
    import subprocess

    ps32 = r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
    if not os.path.exists(ps32):
        print("    ✗ 32-bit PowerShell not found at SysWOW64")
        return False

    dll_ps = dll_path.replace("\\", "\\\\")

    ps_script = f"""
$ErrorActionPreference = 'Stop'
$code = @"
using System;
using System.Runtime.InteropServices;
public class EDSDK {{
    [DllImport(@"{dll_ps}")] public static extern uint EdsInitializeSDK();
    [DllImport(@"{dll_ps}")] public static extern uint EdsGetCameraList(out IntPtr o);
    [DllImport(@"{dll_ps}")] public static extern uint EdsGetChildCount(IntPtr r, out uint c);
    [DllImport(@"{dll_ps}")] public static extern uint EdsGetChildAtIndex(IntPtr r, int i, out IntPtr o);
    [DllImport(@"{dll_ps}")] public static extern uint EdsOpenSession(IntPtr cam);
    [DllImport(@"{dll_ps}")] public static extern uint EdsSendCommand(IntPtr cam, uint cmd, int p);
    [DllImport(@"{dll_ps}")] public static extern uint EdsCloseSession(IntPtr cam);
    [DllImport(@"{dll_ps}")] public static extern uint EdsRelease(IntPtr r);
    [DllImport(@"{dll_ps}")] public static extern uint EdsTerminateSDK();
}}
"@
Add-Type -TypeDefinition $code -Language CSharp

$r = [EDSDK]::EdsInitializeSDK()
Write-Host "EdsInitializeSDK: 0x$($r.ToString('X8'))"
if ($r -ne 0) {{ throw "Init failed" }}

$list = [IntPtr]::Zero
[EDSDK]::EdsGetCameraList([ref]$list) | Out-Null
$count = 0
[EDSDK]::EdsGetChildCount($list, [ref]$count) | Out-Null
Write-Host "Cameras: $count"
if ($count -eq 0) {{ throw "No camera found" }}

$cam = [IntPtr]::Zero
[EDSDK]::EdsGetChildAtIndex($list, 0, [ref]$cam) | Out-Null
[EDSDK]::EdsRelease($list) | Out-Null

$r = [EDSDK]::EdsOpenSession($cam)
Write-Host "EdsOpenSession: 0x$($r.ToString('X8'))"
if ($r -ne 0) {{ throw "OpenSession failed: 0x$($r.ToString('X8'))" }}

$r = [EDSDK]::EdsSendCommand($cam, 0, 0)
Write-Host "TakePicture: 0x$($r.ToString('X8'))"
if ($r -ne 0) {{
    [EDSDK]::EdsSendCommand($cam, 4, 3) | Out-Null
    Start-Sleep -Milliseconds 600
    [EDSDK]::EdsSendCommand($cam, 4, 0) | Out-Null
    Write-Host "PressShutter fallback sent"
}}
Start-Sleep -Seconds 2
[EDSDK]::EdsCloseSession($cam) | Out-Null
[EDSDK]::EdsRelease($cam) | Out-Null
[EDSDK]::EdsTerminateSDK() | Out-Null
Write-Host "SHUTTER_OK"
"""

    print(f"\n[5] Triggering shutter via 32-bit PowerShell + EDSDK P/Invoke...")
    print(f"    DLL: {dll_path}")
    print(f"    PS32: {ps32}")

    try:
        r = subprocess.run(
            [ps32, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True, text=True, timeout=40
        )
        for line in (r.stdout + r.stderr).strip().splitlines():
            if line.strip():
                print(f"    {line.strip()}")

        if r.returncode == 0 and "SHUTTER_OK" in r.stdout:
            print("\n    ✓✓✓ SHUTTER FIRED SUCCESSFULLY!")
            print("        Photo saved to camera SD card.")
            return True
        else:
            print(f"\n    ✗ Script failed (exit code {r.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("    ✗ Timed out (40s)")
        return False
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        return False


# ─────────────────────────────────────────────────────
#  STEP 6: digiCamControl HTTP (if running)
# ─────────────────────────────────────────────────────

def shoot_via_digicam():
    print("\n[6] Trying digiCamControl HTTP (http://localhost:5513)...")
    try:
        import urllib.request
        r = urllib.request.urlopen("http://localhost:5513/session.json", timeout=3)
        print("    ✓ digiCamControl is running!")
        # Trigger capture
        urllib.request.urlopen("http://localhost:5513/?CMD=Capture", timeout=15)
        print("    ✓ Capture command sent!")
        return True
    except Exception as e:
        print(f"    ✗ digiCamControl not available: {e}")
        return False


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────

def main():
    print("═" * 55)
    print("  Canon EOS 6D — Camera Test Script")
    print("═" * 55)
    print("  Make sure:")
    print("  1. Canon EOS 6D is ON")
    print("  2. USB cable is plugged in")
    print("  3. Camera set to M, Av, Tv, or P mode")
    print("  4. EOS Utility is CLOSED (this script will close it)")
    print()

    # Show folder contents so user can see what DLLs exist
    list_eos_files()

    # Kill EOS Utility
    kill_eos_utility()

    # Check running processes
    check_processes()

    # Try digiCamControl first (easiest)
    if shoot_via_digicam():
        print("\n✓ SUCCESS via digiCamControl HTTP")
        print("  The main script will use this method automatically.")
        return

    # Find and use EDSDK
    dll = find_edsdk()
    if dll:
        if shoot_via_edsdk(dll):
            print("\n✓ SUCCESS via Canon EDSDK")
            print("  The main script will use this method automatically.")
            print(f"  Photos saved to: {SAVE_DIR}")
        else:
            print("\n✗ EDSDK method failed — see errors above")
            print("\n  NEXT STEPS:")
            print("  1. Install digiCamControl: https://digicamcontrol.com/download")
            print("  2. Open it → File → Settings → Webserver → Enable → Restart")
            print("  3. Run this script again — it will use digiCamControl")
    else:
        print("\n✗ EDSDK.dll not found")
        print("\n  NEXT STEPS:")
        print("  The EDSDK.dll that ships with EOS Utility was not found.")
        print("  Paste the full list above into the chat so we can see what IS installed.")


if __name__ == "__main__":
    main()
