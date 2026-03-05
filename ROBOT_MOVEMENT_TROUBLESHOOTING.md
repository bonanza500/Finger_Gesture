# Robot Movement Troubleshooting Guide

## Problem: Hand gestures are detected but robot doesn't move

This guide will help you diagnose and fix robot movement issues.

## Quick Checklist

### 1. Is the Python server running?
```bash
python finger_server.py
```
Should show: `[OK] Server starting on http://localhost:5001`

### 2. Is the robot connected in DobotStudio Pro?
- Check robot status in DobotStudio Pro
- Robot should be "Connected" and "Enabled"
- Try manual movement first: `MovL({100,100,100,0,0,0}, 50)`

### 3. Can you detect gestures?
Run this in DobotStudio Pro Lua console:
```lua
FingerInit()
GetGesture()  -- Should return gesture info if hand is visible
```

## Step-by-Step Diagnosis

### Step 1: Test Server Connection

In DobotStudio Pro Lua console:
```lua
-- Test if server is reachable
local status = httpGet("http://localhost:5001/status")
if status then
    print("Server OK")
else
    print("Server not reachable - start finger_server.py")
end
```

### Step 2: Test Gesture Detection

```lua
-- Test gesture detection
FingerInit()
for i = 1, 10 do
    local gesture_id, name, conf = GetGesture()
    if gesture_id then
        print("Gesture:", gesture_id, name, conf)
    else
        print("No hand detected")
    end
    Sleep(1000)
end
```

### Step 3: Test Robot Connection

```lua
-- Test robot connection
local pose = GetPose()
if pose then
    print("Robot connected at:", pose[1], pose[2], pose[3])
else
    print("Robot not connected!")
end
```

### Step 4: Test Manual Movement

```lua
-- Test manual robot movement
local current = GetPose()
if current then
    local new_pose = {current[1], current[2], current[3] + 10, 
                      current[4], current[5], current[6]}
    MovL(new_pose, 50)  -- Move up 10mm
    print("Manual movement sent")
else
    print("Cannot get robot position")
end
```

### Step 5: Test Gesture Movement

```lua
-- Test gesture-based movement
MoveByGesture(1)  -- Should move UP
Sleep(2000)
MoveByGesture(6)  -- Should move LEFT
```

## Common Issues and Solutions

### Issue 1: "Failed to connect to server"

**Cause:** Python server not running
**Solution:**
```bash
cd your_project_folder
python finger_server.py
```
Keep the terminal open!

### Issue 2: "Cannot get current robot position"

**Cause:** Robot not connected or enabled
**Solution:**
1. Check robot connection in DobotStudio Pro
2. Enable robot if disabled
3. Check robot IP (default: 192.168.1.6)
4. Try connecting manually in DobotStudio Pro

### Issue 3: Gestures detected but no movement

**Possible causes:**
1. Robot in error state
2. Robot not enabled
3. Movement commands not reaching robot
4. Gesture detection too fast/unstable

**Solutions:**
```lua
-- Check robot status
local pose = GetPose()
print("Robot pose:", pose)

-- Try manual movement first
MovL({100, 100, 100, 0, 0, 0}, 50)

-- Test single gesture
MoveByGesture(1)  -- Force gesture 1 (UP)
```

### Issue 4: Robot moves but wrong direction

**Cause:** Coordinate system mismatch
**Solution:** Check the movement mapping in `api.lua`:
```lua
-- In MoveByGesture function:
if gesture_id == 1 then
    z = z + MOVE_STEP  -- UP
elseif gesture_id == 2 then
    y = y + MOVE_STEP  -- FORWARD
-- etc.
```

### Issue 5: Movement too fast/slow

**Solution:** Adjust parameters in `api.lua`:
```lua
local MOVE_STEP = 50    -- Change to 10, 20, 100, etc.
local MOVE_SPEED = 100  -- Change to 50, 200, etc.
```

## Advanced Debugging

### Enable Debug Logging

Add this to your Lua script:
```lua
-- Debug function
function debugGesture()
    local data = httpGet("http://localhost:5001/detection")
    if data then
        print("Raw data:", data)
        print("Hand detected:", data.hand_detected)
        print("Gesture ID:", data.gesture_id)
        print("Confidence:", data.confidence)
    end
end

-- Call it
debugGesture()
```

### Check Network Issues

Test server directly in browser:
- http://localhost:5001/status
- http://localhost:5001/detection
- http://localhost:5001/stream

### Check Robot Communication

```lua
-- Test robot commands directly
local result = MovL({100, 100, 100, 0, 0, 0}, 50)
print("MovL result:", result)

-- Check current position
local pose = GetPose()
print("Current pose:", pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])
```

## Test Scripts

I've created several test scripts for you:

1. **`simple_test.lua`** - Basic functionality test
2. **`diagnose_connection.lua`** - Comprehensive diagnostics
3. **`test_robot_movement.lua`** - Full movement test

Copy and paste these into DobotStudio Pro's Lua console.

## Expected Behavior

When working correctly:
1. `FingerInit()` returns 0
2. `GetGesture()` returns gesture ID when hand is shown
3. `MoveByGesture(1)` moves robot UP by 50mm
4. `StartGestureControl(10)` enables continuous control for 10 seconds

## Still Not Working?

### Check These:

1. **Python server output** - Any error messages?
2. **DobotStudio Pro console** - Any Lua errors?
3. **Robot status** - Connected, enabled, no errors?
4. **Camera** - Working in http://localhost:5001/stream?
5. **Gestures** - Clear, steady hand positions?

### Try This Minimal Test:

```lua
-- Minimal test
print("1. Init:", FingerInit())
print("2. Pose:", GetPose())
print("3. Gesture:", GetGesture())
MoveByGesture(1)
print("4. Movement sent")
```

If this doesn't work, the issue is in the basic setup.

## Get Help

If you're still stuck, run the diagnostic script and share:
1. Python server terminal output
2. DobotStudio Pro Lua console output
3. Robot connection status
4. Any error messages

The issue is likely one of:
- Server not running
- Robot not connected/enabled
- Network/firewall blocking localhost:5001
- Gesture detection not stable enough