# Finger Gesture Plugin for DobotStudio Pro v2.x

Advanced 10-gesture finger control for Dobot Nova 5 using MediaPipe hand tracking.

## Features

- **10 Distinct Gestures**: Precise control using finger combinations (1-10)
- **Real-time Detection**: MediaPipe-based hand landmark tracking
- **Comprehensive Control**: 6-axis movement + rotation + gripper control
- **Visual Feedback**: Live camera feed with gesture overlay
- **High Accuracy**: Geometric finger pattern detection

## Gesture Mapping

| Gesture ID | Fingers | Robot Action |
|------------|---------|--------------|
| 1 | Index | Move UP (+Z, 10mm) |
| 2 | Index + Middle | Move FORWARD (+Y, 10mm) |
| 3 | Index + Middle + Ring | Move RIGHT (+X, 10mm) |
| 4 | Index + Middle + Ring + Pinky | Move DOWN (-Z, 10mm) |
| 5 | All Fingers (open hand) | STOP |
| 6 | Thumb | Move LEFT (-X, 10mm) |
| 7 | Thumb + Index | Move BACKWARD (-Y, 10mm) |
| 8 | Thumb + Index + Middle | Rotate CW (+5°) |
| 9 | Thumb + Index + Middle + Ring | Rotate CCW (-5°) |
| 10 | Fist (no fingers) | STOP |

## Installation

### 1. Install Dependencies

```bash
pip install mediapipe opencv-python flask flask-cors scikit-image joblib numpy
```

### 2. Start the Finger Gesture Server

```bash
cd hand_tracking
python FingerGesture/finger_server.py
```

The server will start on `http://localhost:5001`

Test it: Open `http://localhost:5001/stream` in your browser

### 3. Install the Plugin

Copy the `FingerGesture` folder to:
```
C:\Program Files (x86)\DobotStudio Pro\resources\dobot+\FingerGesture\
```

### 4. Restart DobotStudio Pro

The FingerGesture plugin should now appear in the plugin list.

## Usage

### From Plugin UI

1. Open the FingerGesture plugin in DobotStudio Pro
2. The UI will automatically connect to the server
3. You'll see:
   - Large gesture number display (1-10)
   - Gesture name
   - Raised fingers list
   - Live camera feed with hand skeleton
   - Gesture reference guide

### From Blockly

Available blocks:
- **Initialize Finger Gesture System** - Connect to server
- **Get detected finger gesture** - Returns gesture ID, name, confidence
- **Get finger states** - Returns individual finger states
- **Move robot by gesture** - Execute movement based on gesture
- **Start gesture control for X seconds** - Continuous tracking mode
- **Control gripper by finger count** - Open/close based on finger count
- **Test finger gesture system** - Run diagnostic test

### From Lua Script

```lua
-- Initialize
FingerInit()

-- Get gesture
local gesture_id, gesture_name, confidence = GetGesture()
print("Gesture " .. gesture_id .. ": " .. gesture_name .. " (" .. (confidence * 100) .. "%)")

-- Get finger states
local fingers = GetFingerStates()
print("Thumb: " .. tostring(fingers.thumb))
print("Index: " .. tostring(fingers.index))

-- Move by gesture
MoveByGesture(1)  -- or leave empty to auto-detect

-- Start continuous control for 30 seconds
StartGestureControl(30)

-- Control gripper
GripperByFingerCount()

-- Run test
TestFingerGesture()
```

## Advantages Over Previous Plugin

1. **More Gestures**: 10 gestures vs 7 (43% more control options)
2. **Better Precision**: Geometric finger detection vs named gesture recognition
3. **More Control**: Includes rotation control (gestures 8 & 9)
4. **Clearer Mapping**: Logical progression (1 finger → 2 fingers → 3 fingers, etc.)
5. **Standalone Server**: Runs independently, easier to debug
6. **Different Port**: Runs on 5001, won't conflict with other vision server

## Configuration

### Movement Parameters

Edit `dobot_gesture_control.py` to customize:

```python
MOVE_STEP = 10  # mm per gesture (default 10mm)
```

For rotation gestures (8 & 9), edit the delta value in `GESTURE_TO_MOVEMENT`:

```python
8: {"axis": "RZ", "delta": 5, "name": "ROTATE_CW"},  # Change 5 to desired degrees
9: {"axis": "RZ", "delta": -5, "name": "ROTATE_CCW"},
```

## Troubleshooting

### Server won't start

**Missing dependencies:**
```bash
pip install mediapipe opencv-python flask flask-cors scikit-image joblib
```

**Camera in use:**
- Close other apps using webcam
- Change camera index in `finger_server.py`:
  ```python
  CAMERA_INDEX = 1  # Try different numbers
  ```

### Plugin can't connect

1. Make sure `finger_server.py` is running
2. Check http://localhost:5001/status in browser
3. Look for firewall blocking localhost:5001
4. Check plugin UI log for error details

### Gestures not detected

1. Improve lighting
2. Keep hand clearly visible in frame
3. Make distinct finger positions
4. Check camera feed in browser: http://localhost:5001/stream

### Wrong movements

1. Check gesture mapping in `api.lua`
2. Adjust `MOVE_STEP` if movements too large/small
3. Verify robot coordinate system matches expectations

## Technical Details

### Detection Method

Uses MediaPipe Hands for landmark detection:
- 21 hand landmarks tracked in real-time
- Finger states determined by comparing tip vs joint positions
- Thumb detection accounts for left/right handedness
- Pattern matching against 10 predefined gesture patterns

### Server Architecture

- Flask REST API on port 5001
- Separate capture thread (~40 FPS)
- Thread-safe state management
- CORS enabled for browser access
- MJPEG streaming for live preview

### Performance

- Detection rate: ~30 FPS
- Latency: ~100ms (gesture → detection)
- UI update: 10 Hz (tracking), 5 Hz (video)
- Robot command rate: ~2 Hz (500ms polling)

## Comparison with CameraVision Plugin

| Feature | CameraVision | FingerGesture |
|---------|--------------|---------------|
| Gestures | 7 named gestures | 10 finger combinations |
| Port | 5000 | 5001 |
| Detection | Gesture recognizer model | Geometric finger detection |
| Rotation | No | Yes (gestures 8 & 9) |
| Precision | Good | Excellent |
| Setup | Requires gesture_recognizer.task | No model file needed |

## Next Steps

1. Calibrate camera-to-robot coordinates
2. Add safety boundaries
3. Implement smooth tracking with ServoP
4. Train custom SVM model for improved accuracy
5. Add gesture sequences for complex commands

## License

MIT
