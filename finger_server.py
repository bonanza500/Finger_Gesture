"""
Finger Gesture Detection Server for Dobot Nova 5
Uses YOLOv8 trained model (best.pt) for primary gesture classification
Falls back to MediaPipe Hands + rule-based detection if YOLO unavailable

Detects 10 finger gestures:
  1  = Index
  2  = Index + Middle
  3  = Index + Middle + Ring
  4  = Index + Middle + Ring + Pinky
  5  = All Fingers (open hand)
  6  = Thumb
  7  = Thumb + Index
  8  = Thumb + Index + Middle
  9  = Thumb + Index + Middle + Ring
 10  = Fist (no fingers raised)

Requirements:
  pip install flask flask-cors ultralytics mediapipe opencv-python numpy joblib
  Python 3.10 recommended
"""

import cv2
import numpy as np
import base64
import json
import threading
import time
import os
import sys
import joblib

from flask import Flask, jsonify, Response
from flask_cors import CORS

# ─── MediaPipe Setup ─────────────────────────────────────────────
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand Landmark Constants
THUMB_TIP  = mp_hands.HandLandmark.THUMB_TIP
THUMB_IP   = mp_hands.HandLandmark.THUMB_IP
THUMB_MCP  = mp_hands.HandLandmark.THUMB_MCP
THUMB_CMC  = mp_hands.HandLandmark.THUMB_CMC

INDEX_TIP  = mp_hands.HandLandmark.INDEX_FINGER_TIP
INDEX_PIP  = mp_hands.HandLandmark.INDEX_FINGER_PIP
INDEX_MCP  = mp_hands.HandLandmark.INDEX_FINGER_MCP

MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
MIDDLE_PIP = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
MIDDLE_MCP = mp_hands.HandLandmark.MIDDLE_FINGER_MCP

RING_TIP   = mp_hands.HandLandmark.RING_FINGER_TIP
RING_PIP   = mp_hands.HandLandmark.RING_FINGER_PIP
RING_MCP   = mp_hands.HandLandmark.RING_FINGER_MCP

PINKY_TIP  = mp_hands.HandLandmark.PINKY_TIP
PINKY_PIP  = mp_hands.HandLandmark.PINKY_PIP
PINKY_MCP  = mp_hands.HandLandmark.PINKY_MCP

WRIST      = mp_hands.HandLandmark.WRIST

# ─── YOLO Setup ──────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not installed. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 75

# Detection method: "yolo", "mediapipe", or "auto" (tries YOLO first, falls back to MediaPipe)
DETECTION_METHOD = "auto"

# YOLO model path — update this to your actual model location
MODEL_PATH = r"C:\My Websites\hand_tracking\runs\train\finger_gestures\weights\best.pt"

# Optional: path to a joblib-saved sklearn classifier for MediaPipe landmark features
# Set to None if not using a secondary classifier
SKLEARN_MODEL_PATH = None  # e.g., r"C:\path\to\gesture_classifier.pkl"

# MediaPipe Hands configuration
MP_MAX_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.7
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ═══════════════════════════════════════════════════════════════════
#  GESTURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# Gesture definitions (class IDs from YOLO model)
GESTURE_LABELS = {
    0: "Fist",
    1: "Index",
    2: "Index + Middle",
    3: "Index + Middle + Ring",
    4: "Index + Middle + Ring + Pinky",
    5: "All Fingers",
    6: "Thumb",
    7: "Thumb + Index",
    8: "Thumb + Index + Middle",
    9: "Thumb + Index + Middle + Ring",
}

# Map YOLO class IDs to gesture IDs (1-10)
CLASS_TO_GESTURE = {
    0: 10,  # Fist -> 10
    1: 1,   # Index -> 1
    2: 2,   # Index + Middle -> 2
    3: 3,   # Index + Middle + Ring -> 3
    4: 4,   # Index + Middle + Ring + Pinky -> 4
    5: 5,   # All Fingers -> 5
    6: 6,   # Thumb -> 6
    7: 7,   # Thumb + Index -> 7
    8: 8,   # Thumb + Index + Middle -> 8
    9: 9,   # Thumb + Index + Middle + Ring -> 9
}

# Reverse map: gesture ID -> name
GESTURE_ID_TO_NAME = {
    1:  "Index",
    2:  "Index + Middle",
    3:  "Index + Middle + Ring",
    4:  "Index + Middle + Ring + Pinky",
    5:  "All Fingers",
    6:  "Thumb",
    7:  "Thumb + Index",
    8:  "Thumb + Index + Middle",
    9:  "Thumb + Index + Middle + Ring",
    10: "Fist",
}

GESTURE_COLORS = {
    1:  (0, 165, 255),    # Orange
    2:  (0, 255, 255),    # Yellow
    3:  (0, 255, 0),      # Green
    4:  (255, 165, 0),    # Light Blue
    5:  (255, 0, 255),    # Magenta
    6:  (0, 100, 255),    # Dark Orange
    7:  (100, 255, 255),  # Light Yellow
    8:  (100, 255, 100),  # Light Green
    9:  (255, 200, 100),  # Sky Blue
    10: (80, 80, 255),    # Red
}

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════

camera = None
camera_lock = threading.Lock()
latest_frame = None
is_running = False

yolo_model = None
hands_detector = None
sklearn_model = None

# Latest detection results
latest_detection = {
    "hand_detected": False,
    "gesture_id": None,
    "gesture_name": "None",
    "confidence": 0.0,
    "class_id": None,
    "bbox": None,
    "method": "none"
}
detection_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def init_camera():
    """Initialize webcam capture."""
    global camera, is_running

    # Use CAP_DSHOW on Windows for faster startup
    if sys.platform == "win32":
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 30)

    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")

    is_running = True
    print(f"[OK] Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")


def init_yolo_model():
    """Load YOLOv8 gesture detection model."""
    global yolo_model

    if not YOLO_AVAILABLE:
        print("[WARN] ultralytics not installed — YOLO detection unavailable")
        return False

    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] YOLO model not found at {MODEL_PATH}")
        return False

    try:
        yolo_model = YOLO(MODEL_PATH)
        names = yolo_model.names
        print(f"[OK] Loaded YOLOv8 model from {MODEL_PATH}")
        print(f"[OK] Model classes: {names}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return False


def init_mediapipe():
    """Initialize MediaPipe Hands detector."""
    global hands_detector

    try:
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        print("[OK] MediaPipe Hands initialized")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to init MediaPipe: {e}")
        return False


def init_sklearn_model():
    """Load optional sklearn classifier for landmark-based gesture recognition."""
    global sklearn_model

    if SKLEARN_MODEL_PATH is None:
        return False

    if not os.path.exists(SKLEARN_MODEL_PATH):
        print(f"[WARN] sklearn model not found at {SKLEARN_MODEL_PATH}")
        return False

    try:
        sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
        print(f"[OK] Loaded sklearn model from {SKLEARN_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load sklearn model: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE RULE-BASED GESTURE DETECTION
# ═══════════════════════════════════════════════════════════════════

def is_finger_up(landmarks, tip, pip_joint):
    """Check if a finger is extended (tip is above PIP joint in image coords)."""
    return landmarks[tip].y < landmarks[pip_joint].y


def is_thumb_up(landmarks, handedness="Right"):
    """Check if thumb is extended (uses x-axis comparison based on handedness)."""
    if handedness == "Right":
        return landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
    else:
        return landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x


def count_fingers(landmarks, handedness="Right"):
    """
    Count raised fingers and return (finger_count, finger_list).
    finger_list: [thumb, index, middle, ring, pinky] as booleans.
    """
    thumb = is_thumb_up(landmarks, handedness)
    index = is_finger_up(landmarks, INDEX_TIP, INDEX_PIP)
    middle = is_finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = is_finger_up(landmarks, RING_TIP, RING_PIP)
    pinky = is_finger_up(landmarks, PINKY_TIP, PINKY_PIP)

    fingers = [thumb, index, middle, ring, pinky]
    return sum(fingers), fingers


def classify_gesture_from_fingers(fingers):
    """
    Map finger states [thumb, index, middle, ring, pinky] to gesture ID (1-10).

    Gesture mapping:
      1  = Index only
      2  = Index + Middle
      3  = Index + Middle + Ring
      4  = Index + Middle + Ring + Pinky
      5  = All Fingers (thumb + all four)
      6  = Thumb only
      7  = Thumb + Index
      8  = Thumb + Index + Middle
      9  = Thumb + Index + Middle + Ring
     10  = Fist (nothing raised)
    """
    thumb, index, middle, ring, pinky = fingers

    # Exact pattern matching (order matters for specificity)
    if not any(fingers):
        return 10, "Fist"

    if thumb and index and middle and ring and pinky:
        return 5, "All Fingers"

    if thumb and index and middle and ring and not pinky:
        return 9, "Thumb + Index + Middle + Ring"

    if thumb and index and middle and not ring and not pinky:
        return 8, "Thumb + Index + Middle"

    if thumb and index and not middle and not ring and not pinky:
        return 7, "Thumb + Index"

    if thumb and not index and not middle and not ring and not pinky:
        return 6, "Thumb"

    if not thumb and index and middle and ring and pinky:
        return 4, "Index + Middle + Ring + Pinky"

    if not thumb and index and middle and ring and not pinky:
        return 3, "Index + Middle + Ring"

    if not thumb and index and middle and not ring and not pinky:
        return 2, "Index + Middle"

    if not thumb and index and not middle and not ring and not pinky:
        return 1, "Index"

    # Fallback: count total fingers
    total = sum(fingers)
    name = GESTURE_ID_TO_NAME.get(total, f"Unknown ({total})")
    return total if 1 <= total <= 10 else 10, name


def detect_with_mediapipe(frame):
    """
    Run MediaPipe Hands on frame and classify gesture using rule-based logic
    or an optional sklearn classifier.
    Returns detection dict.
    """
    if hands_detector is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_detector.process(rgb)
    rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        return {
            "hand_detected": False,
            "gesture_id": None,
            "gesture_name": "None",
            "confidence": 0.0,
            "class_id": None,
            "bbox": None,
            "method": "MediaPipe"
        }

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = hand_landmarks.landmark

    # Determine handedness
    handedness = "Right"
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label

    # Get bounding box from landmarks
    h, w, _ = frame.shape
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    x1, y1 = int(min(x_coords)) - 20, int(min(y_coords)) - 20
    x2, y2 = int(max(x_coords)) + 20, int(max(y_coords)) + 20
    bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

    # Classify gesture
    if sklearn_model is not None:
        # Use sklearn classifier on landmark features
        try:
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features).reshape(1, -1)
            prediction = sklearn_model.predict(features)[0]
            confidence = max(sklearn_model.predict_proba(features)[0])
            gesture_id = int(prediction)
            gesture_name = GESTURE_ID_TO_NAME.get(gesture_id, f"Gesture {gesture_id}")
            method = "MediaPipe + sklearn"
        except Exception as e:
            print(f"[WARN] sklearn prediction failed: {e}, falling back to rules")
            _, finger_states = count_fingers(landmarks, handedness)
            gesture_id, gesture_name = classify_gesture_from_fingers(finger_states)
            confidence = 0.85
            method = "MediaPipe + Rules"
    else:
        # Rule-based classification
        _, finger_states = count_fingers(landmarks, handedness)
        gesture_id, gesture_name = classify_gesture_from_fingers(finger_states)
        confidence = 0.85
        method = "MediaPipe + Rules"

    return {
        "hand_detected": True,
        "gesture_id": gesture_id,
        "gesture_name": gesture_name,
        "confidence": confidence,
        "class_id": gesture_id - 1 if gesture_id < 10 else 0,
        "bbox": bbox,
        "method": method,
        "landmarks": hand_landmarks,  # kept for drawing overlay
        "handedness": handedness
    }


# ═══════════════════════════════════════════════════════════════════
#  YOLO GESTURE DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_with_yolo(frame):
    """Run YOLO inference on frame. Returns detection dict."""
    if yolo_model is None:
        return None

    try:
        results = yolo_model(frame, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return {
                "hand_detected": False,
                "gesture_id": None,
                "gesture_name": "None",
                "confidence": 0.0,
                "class_id": None,
                "bbox": None,
                "method": "YOLOv8"
            }

        result = results[0]
        boxes = result.boxes

        if len(boxes) > 0:
            box = boxes[0]  # Highest confidence detection
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            bbox = box.xyxy[0].cpu().numpy().tolist()

            gesture_id = CLASS_TO_GESTURE.get(class_id, class_id + 1)
            gesture_name = GESTURE_LABELS.get(class_id, f"Class_{class_id}")

            return {
                "hand_detected": True,
                "gesture_id": gesture_id,
                "gesture_name": gesture_name,
                "confidence": confidence,
                "class_id": class_id,
                "bbox": bbox,
                "method": "YOLOv8"
            }

    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════
#  UNIFIED ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_frame(frame):
    """Analyze frame using configured detection method and update global state."""
    global latest_detection

    detection = None

    if DETECTION_METHOD == "yolo":
        detection = detect_with_yolo(frame)
    elif DETECTION_METHOD == "mediapipe":
        detection = detect_with_mediapipe(frame)
    elif DETECTION_METHOD == "auto":
        # Try YOLO first, fall back to MediaPipe
        detection = detect_with_yolo(frame)
        if detection is None:
            detection = detect_with_mediapipe(frame)

    if detection is None:
        detection = {
            "hand_detected": False,
            "gesture_id": None,
            "gesture_name": "None",
            "confidence": 0.0,
            "class_id": None,
            "bbox": None,
            "method": "none"
        }

    # Remove non-serializable fields before storing
    clean_detection = {k: v for k, v in detection.items() if k not in ("landmarks", "handedness")}

    with detection_lock:
        latest_detection = clean_detection

    # Store landmarks separately for drawing
    return detection


# ═══════════════════════════════════════════════════════════════════
#  DRAWING OVERLAY
# ═══════════════════════════════════════════════════════════════════

# Store last full detection (including landmarks) for overlay drawing
_last_full_detection = {}
_full_detection_lock = threading.Lock()


def draw_overlay(frame):
    """Draw detection results on frame."""
    annotated = frame.copy()

    with detection_lock:
        detection = latest_detection.copy()

    with _full_detection_lock:
        full = _last_full_detection.copy()

    # Draw MediaPipe hand landmarks if available
    if "landmarks" in full and full.get("hand_detected"):
        mp_drawing.draw_landmarks(
            annotated,
            full["landmarks"],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    if detection.get("hand_detected"):
        gesture_id = detection["gesture_id"]
        gesture_name = detection["gesture_name"]
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        method = detection.get("method", "")

        color = GESTURE_COLORS.get(gesture_id, (255, 255, 255))

        # Draw bounding box
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Label background
            label = f"{gesture_id}: {gesture_name}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)

            # Label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Big gesture number
        cv2.putText(annotated, str(gesture_id), (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)

        # Gesture name
        cv2.putText(annotated, gesture_name, (90, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Confidence
        conf_text = f"{confidence:.0%}"
        cv2.putText(annotated, conf_text, (90, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

        # Method indicator
        cv2.putText(annotated, f"[{method}]", (FRAME_WIDTH - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(annotated, "No hand detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    return annotated


# ═══════════════════════════════════════════════════════════════════
#  CAPTURE LOOP
# ═══════════════════════════════════════════════════════════════════

def capture_loop():
    """Continuously capture frames and run gesture detection."""
    global latest_frame

    while is_running:
        with camera_lock:
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
                    latest_frame = frame

                    full_detection = analyze_frame(frame)

                    # Store full detection (with landmarks) for overlay
                    with _full_detection_lock:
                        _last_full_detection.clear()
                        _last_full_detection.update(full_detection)

        time.sleep(0.025)  # ~40 FPS


def gesture_to_finger_states(gesture_id):
    """Convert gesture ID back to finger states for Lua plugin compatibility."""
    finger_map = {
        1:  {"thumb": False, "index": True,  "middle": False, "ring": False, "pinky": False},  # Index
        2:  {"thumb": False, "index": True,  "middle": True,  "ring": False, "pinky": False},  # Index + Middle
        3:  {"thumb": False, "index": True,  "middle": True,  "ring": True,  "pinky": False},  # Index + Middle + Ring
        4:  {"thumb": False, "index": True,  "middle": True,  "ring": True,  "pinky": True},   # Index + Middle + Ring + Pinky
        5:  {"thumb": True,  "index": True,  "middle": True,  "ring": True,  "pinky": True},   # All Fingers
        6:  {"thumb": True,  "index": False, "middle": False, "ring": False, "pinky": False}, # Thumb
        7:  {"thumb": True,  "index": True,  "middle": False, "ring": False, "pinky": False}, # Thumb + Index
        8:  {"thumb": True,  "index": True,  "middle": True,  "ring": False, "pinky": False}, # Thumb + Index + Middle
        9:  {"thumb": True,  "index": True,  "middle": True,  "ring": True,  "pinky": False}, # Thumb + Index + Middle + Ring
        10: {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}, # Fist
    }
    return finger_map.get(gesture_id, {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False})


# ═══════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/status", methods=["GET"])
def status():
    """Server health check - Compatible with Lua plugin."""
    return jsonify({
        "running": is_running,
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "detection_method": DETECTION_METHOD,
        "yolo_loaded": yolo_model is not None,
        "mediapipe_loaded": hands_detector is not None,
        "model_loaded": yolo_model is not None or hands_detector is not None,
        "sklearn_loaded": sklearn_model is not None,
        "model_path": MODEL_PATH,
        "gestures": GESTURE_ID_TO_NAME,
    })


@app.route("/detection", methods=["GET"])
def get_detection():
    """Get latest gesture detection data (JSON) - Compatible with Lua plugin."""
    with detection_lock:
        data = latest_detection.copy()
    
    # Add finger_states for Lua plugin compatibility
    if data.get("hand_detected"):
        # Convert gesture back to finger states for Lua plugin
        gesture_id = data.get("gesture_id")
        finger_states = gesture_to_finger_states(gesture_id)
        data["finger_states"] = finger_states
        data["raised_fingers"] = [name for name, raised in finger_states.items() if raised]
    else:
        data["finger_states"] = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
        data["raised_fingers"] = []
    
    return jsonify(data)


@app.route("/frame", methods=["GET"])
def get_frame():
    """Get latest raw frame as base64 JPEG."""
    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 503
    _, buf = cv2.imencode(".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jsonify({"frame": base64.b64encode(buf).decode("utf-8")})


@app.route("/frame/annotated", methods=["GET"])
def get_annotated_frame():
    """Get latest frame with detection overlay as base64 JPEG."""
    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 503
    annotated = draw_overlay(latest_frame)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jsonify({"frame": base64.b64encode(buf).decode("utf-8")})


@app.route("/stream", methods=["GET"])
def mjpeg_stream():
    """MJPEG live stream with overlay (open in browser)."""
    def generate():
        while is_running:
            if latest_frame is not None:
                annotated = draw_overlay(latest_frame)
                _, buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Gracefully shut down the server."""
    global is_running
    is_running = False
    if camera:
        camera.release()
    if hands_detector:
        hands_detector.close()
    return jsonify({"status": "shutting down"})


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Finger Gesture Detection Server")
    print("  YOLOv8 + MediaPipe Hybrid — 10 Gestures (1-10)")
    print("=" * 60)
    print()

    # 1. Camera
    init_camera()

    # 2. Detection models
    yolo_ok = init_yolo_model()
    mp_ok = init_mediapipe()
    sklearn_ok = init_sklearn_model()

    # Determine active method
    if DETECTION_METHOD == "auto":
        if yolo_ok:
            print("[OK] Auto mode: using YOLOv8 (primary) + MediaPipe (fallback)")
        elif mp_ok:
            print("[OK] Auto mode: using MediaPipe only (YOLO not available)")
        else:
            print("[ERROR] No detection model available!")
            sys.exit(1)
    elif DETECTION_METHOD == "yolo" and not yolo_ok:
        print("[ERROR] YOLO mode selected but model failed to load")
        print(f"  Make sure model exists at: {MODEL_PATH}")
        print("  Install ultralytics: pip install ultralytics")
        sys.exit(1)
    elif DETECTION_METHOD == "mediapipe" and not mp_ok:
        print("[ERROR] MediaPipe mode selected but failed to initialize")
        sys.exit(1)

    # 3. Start capture thread
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # 4. Print info
    print()
    print(f"[OK] Server starting on http://localhost:5001")
    print()
    print("  Endpoints:")
    print("    GET  /status            Server health & config")
    print("    GET  /detection         Current gesture data (JSON)")
    print("    GET  /frame             Raw frame (base64)")
    print("    GET  /frame/annotated   Frame with overlay (base64)")
    print("    GET  /stream            MJPEG live stream")
    print("    POST /shutdown          Stop server")
    print()
    print("  Open http://localhost:5001/stream in a browser to test.")
    print()
    print("  Gesture Map:")
    for gid, gname in sorted(GESTURE_ID_TO_NAME.items()):
        print(f"    {gid:>2}  =  {gname}")
    print()

    # 5. Run Flask
    app.run(host="0.0.0.0", port=5001, threaded=True)