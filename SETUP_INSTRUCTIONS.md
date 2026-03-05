# Setup Instructions for Finger Gesture Plugin

## The plugin has TWO parts that work together:

1. **Python Server** (`finger_server.py`) - Runs the camera and gesture detection
2. **Plugin UI** (`index.html`) - Shows in DobotStudio Pro

## Step-by-Step Setup

### Step 1: Start the Python Server

The plugin UI needs the Python server running first!

**Option A: Double-click the batch file**
```
start_finger_server.bat
```

**Option B: Run manually**
```bash
python finger_server.py
```

You should see:
```
═══════════════════════════════════════════════════════════════
  Finger Gesture Detection Server
  YOLOv8 + MediaPipe Hybrid — 10 Gestures (1-10)
═══════════════════════════════════════════════════════════════

  [CAM] Opened: 640x480
  [OK] MediaPipe Hands ready
  [OK] Server starting on http://localhost:5001

  Open http://localhost:5001/stream in a browser to test.
```

**Test the server:**
Open your browser and go to: `http://localhost:5001/stream`
- You should see your camera feed with hand detection

### Step 2: Open the Plugin in DobotStudio Pro

Now that the server is running:

1. Open DobotStudio Pro
2. Go to the plugin section
3. Find "FingerGesture" plugin
4. Click to open it

The plugin UI should now show:
- ✓ Connected status (green)
- Live camera feed
- Gesture detection (show your hand!)

### Step 3: Use the Plugin

**In the Plugin UI:**
- Show hand gestures to your camera
- See the gesture number (1-10) displayed
- Click "Start Tracking" to begin
- Click "Stop Tracking" to pause

**In Blockly/Lua Scripts:**
```lua
-- Initialize
FingerInit()

-- Get current gesture
local gesture_id, gesture_name, confidence = GetGesture()
print("Gesture: " .. gesture_id)

-- Move robot by gesture
MoveByGesture()  -- Auto-detects current gesture

-- Or specify gesture
MoveByGesture(1)  -- Move UP

-- Continuous control for 30 seconds
StartGestureControl(30)
```

## Troubleshooting

### Gray Screen in Plugin

**Problem:** Plugin shows gray/blank screen
**Cause:** Python server is not running
**Solution:** 
1. Start `finger_server.py` first
2. Wait for "Server starting on http://localhost:5001"
3. Then open the plugin in DobotStudio Pro
4. Click "Test Connection" button in the plugin

### "Cannot connect" Error

**Problem:** Plugin says "Cannot connect to server"
**Solutions:**
1. Make sure `finger_server.py` is running
2. Check if port 5001 is available (not used by another program)
3. Test in browser: http://localhost:5001/status
4. Check firewall isn't blocking localhost:5001

### Camera Not Working

**Problem:** "Cannot open camera" error
**Solutions:**
1. Close other apps using the camera (Zoom, Teams, etc.)
2. Try different camera index in `finger_server.py`:
   ```python
   CAMERA_INDEX = 1  # Change from 0 to 1, 2, etc.
   ```
3. Check camera permissions in Windows Settings

### Missing Dependencies

**Problem:** "ModuleNotFoundError: No module named 'mediapipe'"
**Solution:** Install required packages:
```bash
pip install mediapipe opencv-python flask flask-cors numpy
```

## Quick Test Checklist

✅ Python server running? → Check terminal shows "Server starting"
✅ Server accessible? → Open http://localhost:5001/status in browser
✅ Camera working? → Open http://localhost:5001/stream in browser
✅ Plugin loaded? → Check DobotStudio Pro plugin list
✅ Plugin connected? → Should show green "Connected" status

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DobotStudio Pro                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Plugin UI (index.html)                            │    │
│  │  - Shows camera feed                               │    │
│  │  - Displays gesture detection                      │    │
│  │  - Connects to localhost:5001                      │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │ HTTP Requests                           │
│  ┌────────────────┴───────────────────────────────────┐    │
│  │  Lua API (api.lua)                                 │    │
│  │  - FingerInit(), GetGesture(), MoveByGesture()     │    │
│  │  - Connects to localhost:5001                      │    │
│  └────────────────┬───────────────────────────────────┘    │
└───────────────────┼─────────────────────────────────────────┘
                    │ HTTP Requests
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Python Server (finger_server.py)                          │
│  - Runs on localhost:5001                                   │
│  - Captures camera feed                                     │
│  - Detects hand gestures (MediaPipe/YOLO)                  │
│  - Provides REST API                                        │
└─────────────────────────────────────────────────────────────┘
```

## Alternative: Standalone Python Controller

If you want to control the robot directly from Python (without DobotStudio Pro):

```bash
# Run the all-in-one controller
python dobot_gesture_control.py

# Or vision-only mode (no robot)
python dobot_gesture_control.py --no-robot

# Or specify robot IP
python dobot_gesture_control.py --ip 192.168.5.1
```

This opens a web dashboard at http://localhost:5001 with full control.

## Summary

**For DobotStudio Pro Plugin:**
1. Start `finger_server.py` (keep it running)
2. Open plugin in DobotStudio Pro
3. Use Blockly blocks or Lua functions

**For Standalone Control:**
1. Run `dobot_gesture_control.py`
2. Open http://localhost:5001 in browser
3. Control robot with hand gestures

Both modes use the same gesture detection, just different control interfaces!
