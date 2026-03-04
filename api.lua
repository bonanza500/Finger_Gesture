-- Finger Gesture API for Dobot Nova 5
-- Connects to Python finger gesture server running on localhost:5001
-- Detects 10 finger gestures (1-10)

local http = require("socket.http")
local json = require("cjson")
local ltn12 = require("ltn12")

-- Configuration
local SERVER_URL = "http://localhost:5001"
local ROBOT_IP = "192.168.1.6"
local ROBOT_PORT = 22001

-- Movement parameters
local MOVE_STEP = 50  -- mm per gesture command
local MOVE_SPEED = 100  -- movement speed percentage

-- HTTP helper function
local function httpGet(url)
    local response_body = {}
    local res, code, response_headers = http.request{
        url = url,
        sink = ltn12.sink.table(response_body)
    }
    
    if code == 200 then
        local body = table.concat(response_body)
        local success, data = pcall(json.decode, body)
        if success then
            return data
        else
            print("[ERROR] JSON decode failed: " .. body)
            return nil
        end
    else
        print("[ERROR] HTTP request failed: " .. tostring(code))
        return nil
    end
end

-- Initialize finger gesture system
function Plugin.FingerInit()
    print("[FingerGesture] Initializing...")
    local status = httpGet(SERVER_URL .. "/status")
    if status and status.running then
        print("[FingerGesture] Connected! Camera: " .. status.width .. "x" .. status.height)
        print("[FingerGesture] Model loaded: " .. tostring(status.model_loaded))
        print("[FingerGesture] Available gestures:")
        for id, name in pairs(status.gestures) do
            print("  " .. id .. " = " .. name)
        end
        return 0
    else
        print("[FingerGesture] Failed to connect to server")
        return -1
    end
end

-- Get current gesture detection
function Plugin.GetGesture()
    local data = httpGet(SERVER_URL .. "/detection")
    if data and data.hand_detected then
        return data.gesture_id, data.gesture_name, data.confidence
    else
        return nil, "None", 0
    end
end

-- Get finger states
function Plugin.GetFingerStates()
    local data = httpGet(SERVER_URL .. "/detection")
    if data and data.hand_detected then
        return data.finger_states
    else
        return {thumb=false, index=false, middle=false, ring=false, pinky=false}
    end
end

-- Gesture-based movement control
-- Maps gestures 1-10 to robot movements
function Plugin.MoveByGesture(gesture_id)
    if gesture_id == nil then
        gesture_id, _, _ = Plugin.GetGesture()
        if gesture_id == nil then
            print("[FingerGesture] No gesture detected")
            return -1
        end
    end
    
    print("[FingerGesture] Gesture detected: " .. gesture_id)
    
    -- Get current position
    local currentPos = GetPose()
    if currentPos == nil then
        print("[ERROR] Cannot get current robot position")
        return -1
    end
    
    local x, y, z = currentPos[1], currentPos[2], currentPos[3]
    local rx, ry, rz = currentPos[4], currentPos[5], currentPos[6]
    
    -- Map gestures to movements
    -- 1 = Index -> Move UP
    -- 2 = Index + Middle -> Move FORWARD
    -- 3 = Index + Middle + Ring -> Move RIGHT
    -- 4 = Index + Middle + Ring + Pinky -> Move DOWN
    -- 5 = All Fingers -> STOP/HOME
    -- 6 = Thumb -> Move LEFT
    -- 7 = Thumb + Index -> Move BACKWARD
    -- 8 = Thumb + Index + Middle -> Rotate CW
    -- 9 = Thumb + Index + Middle + Ring -> Rotate CCW
    -- 10 = Fist -> STOP
    
    if gesture_id == 1 then
        -- Index: Move UP
        z = z + MOVE_STEP
        print("[FingerGesture] Moving UP")
    elseif gesture_id == 2 then
        -- Index + Middle: Move FORWARD
        y = y + MOVE_STEP
        print("[FingerGesture] Moving FORWARD")
    elseif gesture_id == 3 then
        -- Index + Middle + Ring: Move RIGHT
        x = x + MOVE_STEP
        print("[FingerGesture] Moving RIGHT")
    elseif gesture_id == 4 then
        -- Index + Middle + Ring + Pinky: Move DOWN
        z = z - MOVE_STEP
        print("[FingerGesture] Moving DOWN")
    elseif gesture_id == 5 then
        -- All Fingers: STOP/HOME
        print("[FingerGesture] STOP (All fingers)")
        return 0
    elseif gesture_id == 6 then
        -- Thumb: Move LEFT
        x = x - MOVE_STEP
        print("[FingerGesture] Moving LEFT")
    elseif gesture_id == 7 then
        -- Thumb + Index: Move BACKWARD
        y = y - MOVE_STEP
        print("[FingerGesture] Moving BACKWARD")
    elseif gesture_id == 8 then
        -- Thumb + Index + Middle: Rotate CW
        rz = rz + 15  -- 15 degrees
        print("[FingerGesture] Rotating CW")
    elseif gesture_id == 9 then
        -- Thumb + Index + Middle + Ring: Rotate CCW
        rz = rz - 15  -- 15 degrees
        print("[FingerGesture] Rotating CCW")
    elseif gesture_id == 10 then
        -- Fist: STOP
        print("[FingerGesture] STOP (Fist)")
        return 0
    else
        print("[FingerGesture] Unknown gesture: " .. gesture_id)
        return 0
    end
    
    -- Execute movement
    MovL({x, y, z, rx, ry, rz}, MOVE_SPEED)
    return 0
end

-- Continuous gesture tracking mode
function Plugin.StartGestureControl(duration)
    if duration == nil then
        duration = 30  -- Default 30 seconds
    end
    
    print("[FingerGesture] Starting gesture control for " .. duration .. " seconds")
    print("[FingerGesture] Gesture mapping:")
    print("  1 (Index)                          -> Move UP")
    print("  2 (Index + Middle)                 -> Move FORWARD")
    print("  3 (Index + Middle + Ring)          -> Move RIGHT")
    print("  4 (Index + Middle + Ring + Pinky)  -> Move DOWN")
    print("  5 (All Fingers)                    -> STOP")
    print("  6 (Thumb)                          -> Move LEFT")
    print("  7 (Thumb + Index)                  -> Move BACKWARD")
    print("  8 (Thumb + Index + Middle)         -> Rotate CW")
    print("  9 (Thumb + Index + Middle + Ring)  -> Rotate CCW")
    print("  10 (Fist)                          -> STOP")
    
    local startTime = os.time()
    local lastGesture = nil
    
    while os.time() - startTime < duration do
        local gesture_id, gesture_name, confidence = Plugin.GetGesture()
        
        -- Only act on high-confidence gestures that changed
        if gesture_id ~= nil and gesture_id ~= lastGesture and confidence > 0.7 then
            Plugin.MoveByGesture(gesture_id)
            lastGesture = gesture_id
        end
        
        Sleep(500)  -- Check every 500ms
    end
    
    print("[FingerGesture] Gesture control ended")
    return 0
end

-- Gripper control by finger count
function Plugin.GripperByFingerCount()
    local data = httpGet(SERVER_URL .. "/detection")
    if not data or not data.hand_detected then
        return -1
    end
    
    local raised_count = #data.raised_fingers
    
    if raised_count >= 4 then
        -- 4-5 fingers -> open gripper
        print("[FingerGesture] Opening gripper (" .. raised_count .. " fingers)")
        SetDO(1, 0)  -- Adjust pin based on your gripper
    elseif raised_count <= 1 then
        -- 0-1 fingers -> close gripper
        print("[FingerGesture] Closing gripper (" .. raised_count .. " fingers)")
        SetDO(1, 1)  -- Adjust pin based on your gripper
    end
    
    return 0
end

-- Test function
function Plugin.TestFingerGesture()
    print("=== Finger Gesture Test ===")
    
    local init = Plugin.FingerInit()
    if init ~= 0 then
        print("[FAIL] Initialization failed")
        return -1
    end
    
    print("[OK] Server connected")
    
    for i = 1, 5 do
        local gesture_id, gesture_name, confidence = Plugin.GetGesture()
        if gesture_id then
            print(string.format("[%d] Gesture %d: %s (%.0f%%)", 
                  i, gesture_id, gesture_name, confidence * 100))
        else
            print(string.format("[%d] No hand detected", i))
        end
        Sleep(1000)
    end
    
    print("=== Test Complete ===")
    return 0
end

print("[FingerGesture] API loaded")
