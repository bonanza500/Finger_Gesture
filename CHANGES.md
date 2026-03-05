# Changes: Joint-based to Coordinate-based Movement

## Summary

Modified `dobot_gesture_control.py` to use coordinate-based movement (X, Y, Z, RX, RY, RZ) instead of joint-based jogging (J1+, J2-, etc.).

## Key Changes

### 1. Movement Mapping
**Before (Joint-based):**
```python
GESTURE_TO_JOG = {
    1: "J1+", 2: "J1-", 3: "J2+", 4: "J2-", 5: "J3+",
    6: "J3-", 7: "J4+", 8: "J4-", 9: "J5+", 10: None,
}
```

**After (Coordinate-based):**
```python
MOVE_STEP = 10  # mm per gesture command

GESTURE_TO_MOVEMENT = {
    1: {"axis": "Z", "delta": MOVE_STEP, "name": "UP"},           # Move UP
    2: {"axis": "Y", "delta": MOVE_STEP, "name": "FORWARD"},      # Move FORWARD
    3: {"axis": "X", "delta": MOVE_STEP, "name": "RIGHT"},        # Move RIGHT
    4: {"axis": "Z", "delta": -MOVE_STEP, "name": "DOWN"},        # Move DOWN
    5: {"axis": None, "delta": 0, "name": "STOP"},                # STOP
    6: {"axis": "X", "delta": -MOVE_STEP, "name": "LEFT"},        # Move LEFT
    7: {"axis": "Y", "delta": -MOVE_STEP, "name": "BACKWARD"},    # Move BACKWARD
    8: {"axis": "RZ", "delta": 5, "name": "ROTATE_CW"},           # Rotate CW
    9: {"axis": "RZ", "delta": -5, "name": "ROTATE_CCW"},         # Rotate CCW
    10: {"axis": None, "delta": 0, "name": "STOP"},               # STOP
}
```

### 2. DobotController Class

**Added:**
- `current_pose` attribute to track [X, Y, Z, RX, RY, RZ]
- `_move_speed` attribute (50 mm/s default)
- `update_pose()` method to get current robot position via GetPose()
- `move_coordinate()` method to move by coordinate offsets using MovL()

**Removed:**
- `current_jog` attribute
- `jog()` method (replaced with `move_coordinate()`)

**Modified:**
- `initialize()` now calls `update_pose()` to get initial position
- `stop()` uses `Stop()` command instead of `MoveJog()`
- `get_status()` returns `current_movement` and `current_pose`

### 3. Movement Logic

**Before:**
```python
robot.jog("J1+")  # Continuous jogging
```

**After:**
```python
robot.move_coordinate({
    "axis": "X",
    "delta": 10,
    "name": "RIGHT"
})  # Incremental coordinate movement
```

### 4. API Endpoints

**Changed:**
- `/robot/jog` → `/robot/move` (accepts gesture_id instead of axis)
- `/config/gesture_map` now returns movement names instead of jog commands

### 5. Web Dashboard

**Updated:**
- "Current Jog" → "Movement"
- "JOG: J1+" → "MOVE: RIGHT"
- Gesture map shows movement names (UP, DOWN, LEFT, RIGHT, etc.)

## Benefits

1. **More Intuitive**: X/Y/Z coordinates are easier to understand than joint angles
2. **Consistent with Plugin**: Matches the Lua plugin's MovL-based approach
3. **Precise Control**: Direct coordinate control for better positioning
4. **Safer**: Smaller incremental movements (10mm vs continuous jogging)
5. **Better Integration**: Works seamlessly with DobotStudio Pro's coordinate system

## Configuration

Adjust movement parameters in `dobot_gesture_control.py`:

```python
# Movement step size (mm)
MOVE_STEP = 10  # Change to 5, 20, etc.

# Rotation amount (degrees)
8: {"axis": "RZ", "delta": 5, "name": "ROTATE_CW"},  # Change 5 to desired degrees
```

## Testing

Run in dry-run mode to test without robot:
```bash
python dobot_gesture_control.py --no-robot
```

The system will print movement commands without sending them to the robot.

## Compatibility

- The Lua plugin (`api.lua`) already uses coordinate-based movement with MovL()
- Both Python and Lua implementations now use the same movement paradigm
- No changes needed to the Lua plugin
