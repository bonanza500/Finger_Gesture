-- Test script for robot movement with finger gestures
-- Run this in DobotStudio Pro's Lua console

print("=== Finger Gesture Robot Movement Test ===")

-- Step 1: Initialize the finger gesture system
print("1. Initializing finger gesture system...")
local init_result = FingerInit()
if init_result ~= 0 then
    print("[ERROR] Failed to initialize finger gesture system")
    print("Make sure finger_server.py is running on localhost:5001")
    return
end
print("[OK] Finger gesture system initialized")

-- Step 2: Test gesture detection (without movement)
print("\n2. Testing gesture detection for 10 seconds...")
print("Show different hand gestures to the camera:")
for i = 1, 20 do  -- 20 iterations = 10 seconds at 0.5s each
    local gesture_id, gesture_name, confidence = GetGesture()
    if gesture_id then
        print(string.format("  Detected: Gesture %d (%s) - Confidence: %.0f%%", 
              gesture_id, gesture_name, confidence * 100))
    else
        print("  No hand detected")
    end
    Sleep(500)  -- Wait 0.5 seconds
end

-- Step 3: Test robot movement (CAREFUL!)
print("\n3. Testing robot movement...")
print("IMPORTANT: Make sure robot workspace is clear!")
print("The robot will move based on your gestures for 15 seconds")
print("Gesture mapping:")
print("  1 (Index) = Move UP")
print("  2 (Index+Middle) = Move FORWARD") 
print("  3 (Index+Middle+Ring) = Move RIGHT")
print("  4 (4 fingers) = Move DOWN")
print("  5 (All fingers) = STOP")
print("  6 (Thumb) = Move LEFT")
print("  7 (Thumb+Index) = Move BACKWARD")
print("  8 (Thumb+Index+Middle) = Rotate CW")
print("  9 (4 fingers+Thumb) = Rotate CCW")
print("  10 (Fist) = STOP")

print("\nStarting movement control in 3 seconds...")
Sleep(1000)
print("3...")
Sleep(1000)
print("2...")
Sleep(1000)
print("1...")
Sleep(1000)
print("GO! Show gestures to move the robot!")

-- Run gesture control for 15 seconds
StartGestureControl(15)

print("\n=== Test Complete ===")
print("If gestures were detected but robot didn't move:")
print("1. Check robot is connected and enabled")
print("2. Check robot IP in api.lua (default: 192.168.1.6)")
print("3. Make sure robot is not in an error state")
print("4. Try manual movement first: MovL({100,100,100,0,0,0}, 50)")