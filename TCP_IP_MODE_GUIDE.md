# Using Finger Gesture Control in TCP/IP Mode

## Overview

Since you're using TCP/IP mode, you **cannot** use the Lua plugin. Instead, you use the standalone Python controller (`dobot_gesture_control.py`) which connects directly to the robot via TCP/IP.

## Setup Steps

### 1. Enable TCP/IP Mode in DobotStudio Pro

1. Open DobotStudio Pro
2. Go to robot connection settings
3. Enable TCP/IP mode
4. Note the robot's IP address (usually 192.168.1.6)

### 2. Start the Python Controller

```bash
python dobot_gesture_control.py
```

**Or with custom robot IP:**
```bash
python dobot_gesture_control.py --ip 192.168.5.1
```

**Or in dry-run mode (no robot):**
```bash
python dobot_gesture_control.py --no-robot
```

### 3. Open the Web Dashboard

Open your browser and go to: **http://localhost:5001**

You should see:
- Live camera feed with gesture overlay
- Current gesture display
- Robot status
- Control buttons

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Python Controller (dobot_gesture_control.py)              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Camera +      │  │  Web Dashboard  │  │   Robot     │ │
│  │   Gesture       │  │  (localhost:    │  │  Control    │ │
│  │   Detection     │  │   5001)         │  │  (TCP/IP)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

Everything runs in one Python process - no separate server needed!

## Controls

### Web Dashboard Controls:
- **Start/Stop Tracking** - Enable/disable gesture control
- **Emergency Stop** - Immediately stop robot
- **Enable/Disable Robot** - Control robot power

### Keyboard Controls (when terminal is focused):
- **SPACE** - Toggle tracking on/off
- **S** - Emergency stop
- **ESC** - Exit program

### Gesture Controls:
Show these gestures to your camera:

| Gesture | Fingers | Action |
|---------|---------|--------|
| 1 | Index | Move UP (+Z, 10mm) |
| 2 | Index + Middle | Move FORWARD (+Y, 10mm) |
| 3 | Index + Middle + Ring | Move RIGHT (+X, 10mm) |
| 4 | Index + Middle + Ring + Pinky | Move DOWN (-Z, 10mm) |
| 5 | All 5 fingers | STOP |
| 6 | Thumb | Move LEFT (-X, 10mm) |
| 7 | Thumb + Index | Move BACKWARD (-Y, 10mm) |
| 8 | Thumb + Index + Middle | Rotate CW (+5°) |
| 9 | Thumb + Index + Middle + Ring | Rotate CCW (-5°) |
| 10 | Fist | STOP |

## Testing

Run the test script to verify everything works:

```bash
python test_python_controller.py
```

This will check:
- ✓ Server connection
- ✓ Gesture detection
- ✓ Robot connection
- ✓ Movement commands

## Troubleshooting

### Issue: "Cannot connect to robot"

**Solutions:**
1. Check robot IP address:
   ```bash
   python dobot_gesture_control.py --ip YOUR_ROBOT_IP
   ```
2. Make sure robot is in TCP/IP mode
3. Check network connection to robot
4. Try pinging the robot: `ping 192.168.1.6`

### Issue: "Camera not found"

**Solutions:**
1. Close other apps using camera (Zoom, Teams, etc.)
2. Try different camera index in the code:
   ```python
   CAMERA_INDEX = 1  # Change from 0 to 1, 2, etc.
   ```

### Issue: Gestures detected but robot doesn't move

**Check:**
1. Robot status in web dashboard - should show "Connected" and "Enabled"
2. Tracking status - should show "TRACKING" (green)
3. Hold gestures steady for 1+ seconds
4. Check gesture mapping in web dashboard

### Issue: Robot moves wrong direction

**Solution:** The coordinate system might be different. You can adjust the movement mapping in the code:

```python
GESTURE_TO_MOVEMENT = {
    1: {"axis": "Z", "delta": MOVE_STEP, "name": "UP"},     # Change axis or delta
    2: {"axis": "Y", "delta": MOVE_STEP, "name": "FORWARD"}, # as needed
    # etc.
}
```

## Configuration

### Adjust Movement Size

Edit `dobot_gesture_control.py`:
```python
MOVE_STEP = 10  # Change to 5, 20, 50, etc. (mm)
```

### Adjust Robot IP

```python
DOBOT_IP = "192.168.1.6"  # Change to your robot's IP
```

### Adjust Detection Sensitivity

```python
DEBOUNCE_FRAMES = 8      # Gesture must be stable for 8 frames
COOLDOWN_FRAMES = 15     # Wait 15 frames between movements
NO_HAND_STOP_DELAY = 10  # Stop after 10 frames of no hand
```

## Advantages of TCP/IP Mode

✅ **Direct robot control** - No DobotStudio Pro dependency
✅ **Real-time feedback** - See robot status in web dashboard  
✅ **Keyboard shortcuts** - Quick control without mouse
✅ **Standalone operation** - Works independently
✅ **Better debugging** - See all status in terminal and web

## Disadvantages

❌ **No Lua scripting** - Can't use DobotStudio Pro's Lua console
❌ **No Blockly blocks** - Can't drag-and-drop programming
❌ **TCP/IP only** - Robot must support TCP/IP mode

## Summary

In TCP/IP mode:
1. **Don't use** the Lua plugin or DobotStudio Pro plugins
2. **Do use** `dobot_gesture_control.py` directly
3. **Control via** web dashboard at http://localhost:5001
4. **Monitor via** terminal output and web interface

This gives you direct, real-time control of the robot with hand gestures!