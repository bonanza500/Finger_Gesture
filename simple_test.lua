-- Simple test to check if finger gestures can move the robot
-- Copy and paste this into DobotStudio Pro's Lua console

print("=== Simple Finger Gesture Test ===")

-- Step 1: Check server connection
print("Checking server connection...")
local init_result = FingerInit()
if init_result ~= 0 then
    print("ERROR: Cannot connect to finger_server.py")
    print("Make sure finger_server.py is running!")
    return
end

-- Step 2: Check robot connection  
print("Checking robot connection...")
local pose = GetPose()
if not pose then
    print("ERROR: Robot not connected!")
    print("Make sure robot is connected and enabled in DobotStudio Pro")
    return
end
print("Robot connected at position:", pose[1], pose[2], pose[3])

-- Step 3: Test gesture detection
print("Testing gesture detection for 5 seconds...")
print("Show your hand to the camera!")
for i = 1, 10 do
    local gesture_id, gesture_name, confidence = GetGesture()
    if gesture_id then
        print("Detected gesture", gesture_id, ":", gesture_name)
    else
        print("No hand detected")
    end
    Sleep(500)
end

-- Step 4: Test ONE movement
print("Testing single movement...")
print("Show gesture 1 (index finger) to move UP by 50mm")
print("Waiting for gesture 1...")

local timeout = 0
while timeout < 20 do  -- 10 second timeout
    local gesture_id = GetGesture()
    if gesture_id == 1 then
        print("Gesture 1 detected! Moving robot UP...")
        MoveByGesture(1)
        print("Movement command sent!")
        break
    end
    Sleep(500)
    timeout = timeout + 1
end

if timeout >= 20 then
    print("Timeout - no gesture 1 detected")
    print("Try manually: MoveByGesture(1)")
end

print("Test complete!")