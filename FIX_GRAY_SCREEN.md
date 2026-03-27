# Fix: Gray Screen in DobotStudio Pro Plugin

## The Problem
You opened the FingerGesture plugin in DobotStudio Pro and see only a gray/blank screen.

## The Cause
The plugin UI (`index.html`) is trying to connect to the Python server at `http://localhost:5001`, but the server isn't running yet.

## The Solution (3 Steps)

### Step 1: Start the Python Server FIRST

Open a terminal/command prompt in your project folder and run:

```bash
python finger_server.py
```

**Or double-click:** `start_finger_server.bat`

### Step 2: Wait for Server to Start

You should see this output:
```
═══════════════════════════════════════════════════════════════
  Finger Gesture Detection Server
  YOLOv8 + MediaPipe Hybrid — 10 Gestures (1-10)
═══════════════════════════════════════════════════════════════

  [CAM] Opened: 640x480
  [OK] MediaPipe Hands ready
  [OK] Server starting on http://localhost:5001
```

**Keep this terminal window open!** The server needs to keep running.

### Step 3: Test the Server (Optional but Recommended)

Open your web browser and go to:
```
http://localhost:5001/stream
```

You should see:
- Your camera feed
- Hand skeleton overlay when you show your hand
- Gesture number displayed

If this works, the server is ready!

### Step 4: Now Open the Plugin

1. Go back to DobotStudio Pro
2. Open the FingerGesture plugin
3. The gray screen should now show the UI with:
   - "Connected to finger gesture server" (green status)
   - Camera feed
   - Gesture detection display

## Still Gray Screen?

### Check 1: Is the server actually running?
Look at your terminal - it should still be showing the server output.
If it closed or shows an error, restart it.

### Check 2: Test the connection
In the plugin UI, click the "Test Connection" button.
It should show "Connected" in green.

### Check 3: Check the browser console
In DobotStudio Pro, if you can access developer tools:
- Press F12 to open developer console
- Look for errors mentioning "localhost:5001"
- Common error: "Failed to fetch" = server not running

### Check 4: Port conflict
Another program might be using port 5001.

**To check (Windows):**
```bash
netstat -ano | findstr :5001
```

**To fix:** Change the port in both files:
- In `finger_server.py`: Change `FLASK_PORT = 5001` to `5002`
- In `index.html`: Change `SERVER_URL = 'http://localhost:5001'` to `'http://localhost:5002'`

## Quick Checklist

- [ ] Terminal/command prompt is open with `finger_server.py` running
- [ ] Terminal shows "Server starting on http://localhost:5001"
- [ ] Browser test works: http://localhost:5001/stream shows camera
- [ ] Plugin opened AFTER server started
- [ ] No firewall blocking localhost:5001

## Common Mistakes

❌ **Opening plugin before starting server**
✅ Start server first, then open plugin

❌ **Closing the terminal window**
✅ Keep terminal open while using plugin

❌ **Wrong folder**
✅ Run `finger_server.py` from the project folder

❌ **Missing dependencies**
✅ Run: `pip install mediapipe opencv-python flask flask-cors numpy`

## Visual Guide

```
WRONG ORDER:
1. Open DobotStudio Pro plugin → Gray screen ❌
2. Start finger_server.py → Too late!

CORRECT ORDER:
1. Start finger_server.py → Server running ✅
2. Test in browser (optional) → http://localhost:5001/stream
3. Open DobotStudio Pro plugin → Works! ✅
```

## Need More Help?

1. Check `SETUP_INSTRUCTIONS.md` for detailed setup
2. Check `README.md` for full documentation
3. Look at terminal output for error messages
4. Test the server in browser first before using plugin

## Summary

**The plugin is just a UI - it needs the Python server to work!**

Think of it like:
- Python server = The brain (does all the work)
- Plugin UI = The display (shows what the brain sees)

Always start the brain before opening the display! 🧠 → 📺
