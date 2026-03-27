-- Diagnostic script to check finger gesture system
-- Run this in DobotStudio Pro's Lua console

print("=== Finger Gesture System Diagnostics ===")

-- Test 1: Check if we can reach the Python server
print("\n1. Testing connection to Python server...")
local status = httpGet("http://localhost:5001/status")
if status then
    print("[OK] Server is running")
    print("  Camera: " .. status.width .. "x" .. status.height)
    print("  Model loaded: " .. tostring(status.model_loaded))
    print("  Available gestures: " .. tostring(#status.gestures or 0))
else
    print("[ERROR] Cannot connect to server at localhost:5001")
    print("Make sure finger_server.py is running!")
    return
end

-- Test 2: Check gesture detection
print("\n2. Testing gesture detection...")
local detection = httpGet("http://localhost:5001/detection")
if detection then
    print("[OK] Detection endpoint working")
    if detection.hand_detected then
        print("  Hand detected: YES")
        print("  Gesture ID: " .. tostring(detection.gesture_id))
        print("  Gesture name: " .. tostring(detection.gesture_name))
        print("  Confidence: " .. string.format("%.0f%%", detection.confidence * 100))
    else
        print("  Hand detected: NO")
        print("  Show your hand to the camera!")
    end
else
    print("[ERROR] Detection endpoint not working")
    return
end

-- Test 3: Check robot connection
print("\n3. Testing robot connection...")
local current_pose = GetPose()
if current_pose then
    print("[OK] Robot is connected")
    print(string.format("  Current position: X=%.1f, Y=%.1f, Z=%.1f", 
          current_pose[1], current_pose[2], current_pose[3]))
    print(string.format("  Current rotation: RX=%.1f, RY=%.1f, RZ=%.1f", 
          current_pose[4], current_pose[5], current_pose[6]))
else
    print("[ERROR] Cannot get robot position")
    print("Check if robot is connected and enabled")
    return
end

-- Test 4: Test a small movement (CAREFUL!)
print("\n4. Testing robot movement...")
print("WARNING: Robot will move UP by 10mm in 3 seconds!")
print("Press STOP if needed!")
Sleep(1000)
print("3...")
Sleep(1000)
print("2...")
Sleep(1000)
print("1...")
Sleep(1000)

local start_pose = GetPose()
if start_pose then
    local new_pose = {start_pose[1], start_pose[2], start_pose[3] + 10, 
                      start_pose[4], start_pose[5], start_pose[6]}
    print("Moving UP by 10mm...")
    MovL(new_pose, 50)
    Sleep(2000)
    
    -- Move back
    print("Moving back to original position...")
    MovL(start_pose, 50)
    Sleep(2000)
    print("[OK] Movement test complete")
else
    print("[ERROR] Cannot get robot position for movement test")
end

-- Test 5: Test the plugin functions
print("\n5. Testing plugin functions...")

-- Test FingerInit
local init_result = FingerInit()
if init_result == 0 then
    print("[OK] FingerInit() works")
else
    print("[ERROR] FingerInit() failed")
end

-- Test GetGesture
local gesture_id, gesture_name, confidence = GetGesture()
if gesture_id then
    print("[OK] GetGesture() works - Detected: " .. gesture_id .. " (" .. gesture_name .. ")")
else
    print("[INFO] GetGesture() works but no hand detected")
end

print("\n=== Diagnostics Complete ===")
print("\nIf everything shows [OK] but gestures don't move robot:")
print("1. Make sure you're showing clear gestures to the camera")
print("2. Hold gestures steady for at least 1 second")
print("3. Check the gesture mapping in api.lua")
print("4. Try: MoveByGesture(1) to test gesture 1 manually")
print("5. Try: StartGestureControl(10) for continuous control")