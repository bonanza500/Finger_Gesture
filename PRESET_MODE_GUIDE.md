# Preset Mode - Hand Gesture Control

## Overview

The robot now moves to preset positions using `JointMovJ()` instead of continuous jogging. This is safer, more predictable, and matches how your example code works.

## How It Works

When you show a gesture, the robot moves to a predefined position (preset) with all 6 joints moving together smoothly.

## Preset Positions

### Preset 1 (Gesture 1 - Index Finger)
```
J1: -198.39°
J2: -85.32°
J3: 32.59°
J4: 22.86°
J5: -0.01°
J6: 1.99°
```

### Preset 2 (Gesture 2 - Index + Middle)
```
J1: -252.05°
J2: -22.05°
J3: -99.95°
J4: 22.86°
J5: -0.01°
J6: 1.99°
```

## Gesture Mapping

| Gesture | Fingers | Action |
|---------|---------|--------|
| 1 | Index | Move to Preset 1 |
| 2 | Index + Middle | Move to Preset 2 |
| 3-10 | (other gestures) | Not assigned yet |

## Usage

### 1. Start the Controller
```bash
python dobot_gesture_control.py
```

### 2. Open Web Dashboard
http://localhost:5001

### 3. Show Gestures
- **Gesture 1** (index finger) → Robot moves to Preset 1
- **Gesture 2** (index + middle) → Robot moves to Preset 2
- Hold gesture steady for ~0.5 seconds
- Robot will move smoothly to the position
- Wait 3 seconds before next gesture (cooldown period)

## Safety Features

1. **Debouncing**: Gesture must be held steady for 8 frames (~0.4s)
2. **Cooldown**: 3-second wait between movements (allows robot to complete)
3. **No hand detection**: Robot stays at current position
4. **Preset validation**: Only moves to defined presets

## Adding More Presets

Edit `dobot_gesture_control.py`:

```python
PRESETS = {
    1: {
        "name": "Preset 1",
        "joints": [-198.39, -85.32, 32.59, 22.86, -0.01, 1.99]
    },
    2: {
        "name": "Preset 2", 
        "joints": [-252.05, -22.05, -99.95, 22.86, -0.01, 1.99]
    },
    3: {  # Add new preset
        "name": "Home Position",
        "joints": [0, 0, 0, 0, 0, 0]
    }
}

# Map gesture to preset
GESTURE_TO_PRESET = {
    1: 1,    # Index -> Preset 1
    2: 2,    # Index + Middle -> Preset 2
    3: 3,    # Index + Middle + Ring -> Preset 3 (new!)
    # ... etc
}
```

## Getting Joint Angles from DobotStudio Pro

1. Move robot to desired position manually
2. In DobotStudio Pro, note the joint angles (J1-J6)
3. Add them to PRESETS dictionary
4. Restart the controller

## Advantages Over Jogging

✅ **Safer** - Robot moves to known safe positions
✅ **Predictable** - Always goes to same position
✅ **Smoother** - All joints move together optimally
✅ **No emergency stops** - Uses proper joint movement commands
✅ **Repeatable** - Perfect for pick-and-place tasks

## Troubleshooting

### Robot doesn't move
- Check if preset is defined in PRESETS
- Check if gesture is mapped in GESTURE_TO_PRESET
- Wait for cooldown period (3 seconds) between gestures
- Check terminal output for error messages

### Robot moves to wrong position
- Verify joint angles in PRESETS match your desired position
- Check gesture mapping in GESTURE_TO_PRESET
- Test preset manually: Show gesture and watch terminal output

### Movement too slow/fast
The robot uses its default speed settings. To adjust, you can add speed parameter to JointMovJ command in the code.

## Example Workflow

1. **Start**: `python dobot_gesture_control.py`
2. **Initialize**: Robot connects and enables
3. **Show Gesture 1**: Robot moves to Preset 1
4. **Wait 3 seconds**: Cooldown period
5. **Show Gesture 2**: Robot moves to Preset 2
6. **Repeat**: Continue showing gestures

## Web Dashboard Features

- Live camera feed with gesture detection
- Current preset display
- Robot status (connected, enabled)
- Start/stop tracking button
- Emergency stop button

## Next Steps

1. Test with your two presets
2. Add more presets for common positions
3. Map gestures 3-10 to additional presets
4. Create preset sequences for complex tasks

## Technical Details

- **Command**: `JointMovJ(j1,j2,j3,j4,j5,j6)`
- **Port**: Dashboard port 29999
- **Debounce**: 8 frames (~400ms)
- **Cooldown**: 60 frames (~3 seconds)
- **Detection rate**: ~20 FPS

This approach is much more reliable than coordinate-based movement and matches how professional robot programming works!