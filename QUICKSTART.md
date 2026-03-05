# Quick Start - Finger Gesture Control

## 3-Minute Setup

### Step 1: Install Dependencies (1 min)

```bash
pip install mediapipe opencv-python flask flask-cors scikit-image joblib numpy
```

### Step 2: Start Server (30 seconds)

**Option A**: Double-click `start_finger_server.bat`

**Option B**: Manual start:
```bash
cd hand_tracking/FingerGesture
python finger_server.py
```

Wait for: `[OK] Server starting on http://localhost:5001`

**Test**: Open http://localhost:5001/stream in browser
- Should see webcam feed
- Make gestures → See number overlay

### Step 3: Install Plugin (1 min)

Copy `FingerGesture` folder to:
```
C:\Program Files (x86)\DobotStudio Pro\resources\dobot+\FingerGesture\
```

Restart DobotStudio Pro.

### Step 4: Test (30 seconds)

**In Plugin UI:**
1. Open FingerGesture plugin
2. Should show "Connected to finger gesture server"
3. Make gestures → See detection

**In Lua:**
```lua
FingerInit()
TestFingerGesture()
```

## Gesture Quick Reference

Show these gestures to the camera:

```
1️⃣  Index finger only          → Robot moves UP
2️⃣  Index + Middle              → Robot moves FORWARD  
3️⃣  Index + Middle + Ring       → Robot moves RIGHT
4️⃣  Index + Middle + Ring + Pinky → Robot moves DOWN
5️⃣  All 5 fingers (open hand)   → STOP
6️⃣  Thumb only                  → Robot moves LEFT
7️⃣  Thumb + Index (gun shape)   → Robot moves BACKWARD
8️⃣  Thumb + Index + Middle      → Rotate CLOCKWISE
9️⃣  Thumb + Index + Middle + Ring → Rotate COUNTER-CLOCKWISE
🔟  Fist (no fingers)            → STOP
```

## First Robot Test (CAREFUL!)

```lua
-- Initialize
FingerInit()

-- Test WITHOUT robot movement (just read gestures)
for i = 1, 10 do
    local id, name, conf = GetGesture()
    if id then
        print("Gesture " .. id .. ": " .. name)
    end
    Sleep(1000)
end

-- Test WITH robot movement (MAKE SURE WORKSPACE IS CLEAR!)
-- Start with short duration
StartGestureControl(5)  -- Only 5 seconds for first test
```

## Tips for Best Detection

1. **Lighting**: Good lighting is essential
2. **Distance**: Keep hand 30-60cm from camera
3. **Background**: Plain background works best
4. **Clarity**: Make distinct finger positions
5. **Stability**: Hold gesture for 0.5 seconds

## Troubleshooting

**"Cannot connect to server"**
- Start finger_server.py first
- Check http://localhost:5001/status

**"No hand detected"**
- Improve lighting
- Move hand closer to camera
- Check camera feed: http://localhost:5001/stream

**Gestures not recognized**
- Make clear finger positions
- Hold gesture steady
- Check which fingers are detected in plugin UI

**Robot doesn't move**
- Robot connected? (ping 192.168.1.6)
- Robot enabled?
- Check Lua console for errors

## What's Next?

1. Practice all 10 gestures
2. Adjust movement step size in `dobot_gesture_control.py` (MOVE_STEP variable)
3. Customize gesture mapping in GESTURE_TO_MOVEMENT
4. Add safety boundaries
5. Try continuous control mode

## Differences from CameraVision

- **Port**: 5001 (not 5000)
- **Gestures**: 10 finger combinations (not 7 named gestures)
- **Includes**: Rotation control (gestures 8 & 9)
- **Detection**: Geometric finger detection (more precise)
- **No model file needed**: Uses MediaPipe landmarks only
- **Movement**: Coordinate-based (X, Y, Z) instead of joint angles

Both plugins can run simultaneously on different ports!
