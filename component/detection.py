import cv2
import time
import threading
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE — GestureRecognizer (Tasks API, from Finger_Gesture-ucup)
# ═══════════════════════════════════════════════════════════════════

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "model", "hand_gesture_recognizer_v2"
)

# ── Landmark index constants ──────────────────────────────────────
WRIST      = 0
THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_PIP   = 14; RING_TIP   = 16
PINKY_PIP  = 18; PINKY_TIP  = 20

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Mapping from model category_name string → gesture_id integer
# The model returns "1".."10"; our pipeline uses int keys 1..10
_CATEGORY_TO_ID = {str(i): i for i in range(1, 11)}

# ── Optional imports ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not installed")
    YOLO_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
#  GESTURE / LABEL MAPS
# ═══════════════════════════════════════════════════════════════════

GESTURE_LABELS = {
    0: "Fist", 1: "Index", 2: "Index + Middle",
    3: "Index + Middle + Ring", 4: "Index + Middle + Ring + Pinky",
    5: "All Fingers", 6: "Thumb", 7: "Thumb + Index",
    8: "Thumb + Index + Middle", 9: "Thumb + Index + Middle + Ring",
}
CLASS_TO_GESTURE = {0:10, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
GESTURE_ID_TO_NAME = {
    1: "Index", 2: "Index + Middle", 3: "Index + Middle + Ring",
    4: "Index + Middle + Ring + Pinky", 5: "All Fingers",
    6: "Thumb", 7: "Thumb + Index", 8: "Thumb + Index + Middle",
    9: "Thumb + Index + Middle + Ring", 10: "Fist",
}
GESTURE_COLORS = {
    1:(0,165,255), 2:(0,255,255), 3:(0,255,0), 4:(255,165,0), 5:(255,0,255),
    6:(0,100,255), 7:(100,255,255), 8:(100,255,100), 9:(255,200,100), 10:(80,80,255),
}

# ═══════════════════════════════════════════════════════════════════
#  FINGER HELPERS
# ═══════════════════════════════════════════════════════════════════

def _xy(lm, idx):
    item = lm[idx]
    return (item.x, item.y) if hasattr(item, 'x') else (float(item[0]), float(item[1]))

def is_finger_up(lm, tip, pip):
    return _xy(lm, tip)[1] < _xy(lm, pip)[1]

def is_thumb_up(lm, hand="Right"):
    tx = _xy(lm, THUMB_TIP)[0]; ix = _xy(lm, THUMB_IP)[0]
    return (tx < ix) if hand == "Right" else (tx > ix)

def classify_fingers(fingers):
    t,i,m,r,p = fingers
    if not any(fingers):          return 10, "Fist"
    if t and i and m and r and p: return 5,  "All Fingers"
    if t and i and m and r:       return 9,  "Thumb + Index + Middle + Ring"
    if t and i and m:             return 8,  "Thumb + Index + Middle"
    if t and i:                   return 7,  "Thumb + Index"
    if t:                         return 6,  "Thumb"
    if i and m and r and p:       return 4,  "Index + Middle + Ring + Pinky"
    if i and m and r:             return 3,  "Index + Middle + Ring"
    if i and m:                   return 2,  "Index + Middle"
    if i:                         return 1,  "Index"
    return min(sum(fingers), 10), GESTURE_ID_TO_NAME.get(sum(fingers), "Unknown")

def _landmarks_to_bbox(lm, w, h, pad=20):
    xs = [_xy(lm, i)[0] * w for i in range(21)]
    ys = [_xy(lm, i)[1] * h for i in range(21)]
    return [max(0,int(min(xs))-pad), max(0,int(min(ys))-pad),
            min(w,int(max(xs))+pad), min(h,int(max(ys))+pad)]

def _classify_from_landmarks(lm, hand="Right"):
    fingers = [
        is_thumb_up(lm, hand),
        is_finger_up(lm, INDEX_TIP,  INDEX_PIP),
        is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP),
        is_finger_up(lm, RING_TIP,   RING_PIP),
        is_finger_up(lm, PINKY_TIP,  PINKY_PIP),
    ]
    return classify_fingers(fingers)

def _draw_landmarks_manual(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(_xy(lm,i)[0]*w), int(_xy(lm,i)[1]*h)) for i in range(21)]
    for s, e in _HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 200, 80), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 80, 80), -1)

# ═══════════════════════════════════════════════════════════════════
#  DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def detect_yolo(frame, yolo_model):
    if yolo_model is None: return None
    try:
        results = yolo_model(frame, verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                    "confidence":0.0,"class_id":None,"bbox":None,"method":"YOLOv8"}
        box  = results[0].boxes[0]
        cid  = int(box.cls[0].item()); conf = float(box.conf[0].item())
        bbox = box.xyxy[0].cpu().numpy().tolist()
        gid  = CLASS_TO_GESTURE.get(cid, cid+1)
        return {"hand_detected":True,"gesture_id":gid,
                "gesture_name":GESTURE_LABELS.get(cid,f"Class_{cid}"),
                "confidence":conf,"class_id":cid,"bbox":bbox,"method":"YOLOv8"}
    except Exception as e:
        print(f"  [ERROR] YOLO: {e}"); return None

def _read_gesture_result(frame, result):
    """Convert a GestureRecognizer callback result into our standard detection dict."""
    if result is None or not result.gestures:
        return {"hand_detected": False, "gesture_id": None, "gesture_name": "None",
                "confidence": 0.0, "class_id": None, "bbox": None,
                "method": "GestureRecognizer"}
    top  = result.gestures[0][0]
    conf = top.score
    gid  = _CATEGORY_TO_ID.get(top.category_name)
    if gid is None:
        return {"hand_detected": False, "gesture_id": None, "gesture_name": "None",
                "confidence": 0.0, "class_id": None, "bbox": None,
                "method": "GestureRecognizer"}
    gname  = GESTURE_ID_TO_NAME.get(gid, top.category_name)
    bbox   = None
    lm_raw = None
    if result.hand_landmarks:
        lm_raw = result.hand_landmarks[0]
        h, w, _ = frame.shape
        bbox = _landmarks_to_bbox(lm_raw, w, h)
    return {"hand_detected": True, "gesture_id": gid, "gesture_name": gname,
            "confidence": conf, "class_id": gid - 1 if gid < 10 else 0,
            "bbox": bbox, "method": "GestureRecognizer", "landmarks_raw": lm_raw}

def detect_mediapipe(frame, hands_detector):
    if hands_detector is None: return None
    if isinstance(hands_detector, dict) and hands_detector.get("type") == "gesture_recognizer":
        # Read the latest result written by the async callback
        return _read_gesture_result(frame, hands_detector["last_result"][0])
    return None

# ═══════════════════════════════════════════════════════════════════
#  GESTURE DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════

class GestureDetector:

    def __init__(self, camera_index=1, frame_width=640, frame_height=480,
                 camera_source="laptop", detection_method="auto",
                 model_path=None, sklearn_model_path=None,
                 mp_max_hands=1, mp_detection_conf=0.7, mp_tracking_conf=0.5):

        self._camera_index       = camera_index
        self._frame_width        = frame_width
        self._frame_height       = frame_height
        self._camera_source      = camera_source
        self._detection_method   = detection_method
        self._model_path         = model_path
        self._sklearn_model_path = sklearn_model_path
        self._mp_max_hands       = mp_max_hands
        self._mp_detection_conf  = mp_detection_conf
        self._mp_tracking_conf   = mp_tracking_conf

        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

        self._camera      = None
        self._rs_pipeline = None
        self._rs_align    = None

        self._yolo_model     = None
        self._hands_detector = None
        self._sklearn_model  = None

        self._latest_frame = None
        self._latest_detection = {
            "hand_detected": False, "gesture_id": None, "gesture_name": "None",
            "confidence": 0.0, "class_id": None, "bbox": None,
            "method": "none", "robot_preset": None,
        }
        self._last_full_detection = {}

    def _init_camera(self):
        import sys
        if self._camera_source == "realsense":
            if not REALSENSE_AVAILABLE:
                raise RuntimeError("pyrealsense2 not installed")
            self._rs_pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self._frame_width, self._frame_height, rs.format.bgr8, 30)
            self._rs_pipeline.start(cfg)
            self._rs_align = rs.align(rs.stream.color)
            time.sleep(1)
            print(f"  [CAM] RealSense opened: {self._frame_width}x{self._frame_height}")
            
        elif self._camera_source == "laptop":
            self._camera = cv2.VideoCapture(0)
            if not self._camera.isOpened():
                self._camera.release()
                self._camera = None
                print("  [WARN] Cannot open camera index 0, trying other indices...")

        else:
            indices = [self._camera_index] + [i for i in range(4) if i != self._camera_index]
            for idx in indices:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform=="win32" else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self._camera = cap
                        self._camera_index = idx
                        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
                        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
                        self._camera.set(cv2.CAP_PROP_FPS, 30)
                        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # no stale frame queue
                        print(f"  [CAM] Opened camera index {idx}: {self._frame_width}x{self._frame_height}")
                        return
                cap.release()
            raise RuntimeError("Cannot open any camera (tried indices 0-3)")

    def _init_yolo(self):
        if not YOLO_AVAILABLE or not self._model_path or not os.path.exists(self._model_path):
            if YOLO_AVAILABLE and self._model_path:
                print(f"  [WARN] YOLO model not found: {self._model_path}")
            return False
        try:
            self._yolo_model = YOLO(self._model_path)
            print("  [OK] YOLOv8 loaded"); return True
        except Exception as e:
            print(f"  [ERROR] YOLO: {e}"); return False

    def _init_mediapipe(self):
        if not os.path.exists(MODEL_PATH):
            print(f"  [ERROR] GestureRecognizer model not found: {MODEL_PATH}")
            return False
        try:
            # Mutable container so the async callback can write into it
            last_result = [None]

            def _on_result(result, output_image, timestamp_ms):
                last_result[0] = result

            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=_on_result,
            )
            recognizer = vision.GestureRecognizer.create_from_options(options)
            self._hands_detector = {
                "type": "gesture_recognizer",
                "recognizer": recognizer,
                "last_result": last_result,
            }
            print(f"  [OK] GestureRecognizer (ucup model) ready — {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"  [ERROR] GestureRecognizer init: {e}"); return False

    def _init_sklearn(self):
        if not JOBLIB_AVAILABLE or not self._sklearn_model_path: return False
        if not os.path.exists(self._sklearn_model_path): return False
        try: self._sklearn_model = joblib.load(self._sklearn_model_path); return True
        except: return False

    def _analyze_frame(self, frame, tracking_active=True):
        if not tracking_active:
            idle = {"hand_detected":False,"gesture_id":None,"gesture_name":"Tracking Paused",
                    "confidence":0.0,"class_id":None,"bbox":None,"method":"paused","robot_jog":None}
            with self._lock: self._latest_detection = idle
            return idle

        det = None
        if self._detection_method == "yolo":
            det = detect_yolo(frame, self._yolo_model)
        elif self._detection_method == "mediapipe":
            det = detect_mediapipe(frame, self._hands_detector)
        elif self._detection_method == "auto":
            det = detect_yolo(frame, self._yolo_model)
            if det is None: det = detect_mediapipe(frame, self._hands_detector)

        if det is None:
            det = {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                   "confidence":0.0,"class_id":None,"bbox":None,"method":"none"}

        clean = {k:v for k,v in det.items() if k not in ("landmarks","landmarks_raw","handedness")}
        with self._lock:
            self._latest_detection = clean
            self._last_full_detection.clear()
            self._last_full_detection.update(det)
        return det

    def draw_overlay(self, frame, tracking_active=True, robot=None, canon=None,
                     scan_active=False, scan_current_pos=None, countdown_state=None,
                     dry_run=False, safety_state=None):
        out = frame.copy()
        with self._lock:
            det  = self._latest_detection.copy()
            full = self._last_full_detection.copy()
        fw, fh = frame.shape[1], frame.shape[0]

        if not tracking_active:
            ov = out.copy()
            cv2.rectangle(ov, (0,0), (fw,fh), (0,0,0), -1)
            out = cv2.addWeighted(ov, 0.3, out, 0.7, 0)
            cv2.putText(out, "TRACKING PAUSED", (fw//2-180, fh//2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
            cv2.putText(out, "Press [SPACE] to resume", (fw//2-140, fh//2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            return out

        # ── Safety gate counter (top-right corner) ────────────────────
        if safety_state is not None:
            unlocked   = safety_state.get("unlocked", False)
            hold_sec   = safety_state.get("hold_sec", 0.0)
            hold_total = safety_state.get("hold_total", 4.0)

            if unlocked:
                # Green box — UNLOCKED
                label    = "SAFETY: UNLOCKED"
                box_col  = (0, 200, 60)
                txt_col  = (0, 0, 0)
                bar_fill = 1.0
            else:
                # Red/amber box — LOCKED with hold progress bar
                label    = "SAFETY: LOCKED"
                box_col  = (0, 0, 180)
                txt_col  = (255, 255, 255)
                bar_fill = min(hold_sec / max(hold_total, 0.001), 1.0)

            # Background pill
            bx, by, bw, bh = fw - 240, 10, 230, 52
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), box_col, 2)

            # Label text
            cv2.putText(out, label, (bx + 8, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_col, 2, cv2.LINE_AA)

            if not unlocked:
                # Progress bar (shows gesture-5 hold progress)
                bar_x = bx + 8; bar_y = by + 30; bar_w = bw - 16; bar_h = 12
                cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * bar_fill)
                if fill_w > 0:
                    # Colour shifts red → amber → green as bar fills
                    r = int(200 * (1.0 - bar_fill))
                    g = int(200 * bar_fill)
                    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, g, r + 30), -1)
                # Countdown text inside bar
                secs_left = hold_total - hold_sec
                bar_label = f"Hold 5 ({secs_left:.1f}s)" if hold_sec > 0 else "Show all fingers"
                cv2.putText(out, bar_label, (bar_x + 4, bar_y + bar_h - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                # Unlocked: show auto-lock countdown
                time_left = safety_state.get("timeout_left", 0.0)
                cv2.putText(out, f"Auto-lock in {time_left:.1f}s", (bx + 8, by + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)
        # ── End safety gate counter ───────────────────────────────────

        # ── Preset picker (shown only when gate is UNLOCKED) ──────────
        if safety_state is not None and safety_state.get("unlocked"):
            presets      = safety_state.get("presets", {})
            gesture_map  = safety_state.get("gesture_map", {})
            timeout_left  = safety_state.get("timeout_left", 0.0)
            timeout_total = safety_state.get("timeout_total", 10.0)
            deb_ratio     = safety_state.get("debounce_ratio", 0.0)
            deb_gid       = safety_state.get("debounce_gid")

            # Build ordered list: gesture_id → preset name
            entries = []
            for gid in range(1, 11):
                action = gesture_map.get(gid)
                if isinstance(action, int) and action in presets:
                    entries.append((gid, presets[action]["name"]))
                elif action == "scan":
                    entries.append((gid, "SCAN"))
                elif action == "stop_scan":
                    entries.append((gid, "STOP SCAN"))

            if entries:
                row_h    = 28
                pad      = 10
                col_w    = 210
                cbar_h   = 26          # countdown bar height
                n_rows   = len(entries)
                box_h    = row_h + n_rows * row_h + cbar_h + pad * 2
                bx       = pad
                by       = fh // 2 - box_h // 2

                # Semi-transparent dark background
                overlay_bg = out.copy()
                cv2.rectangle(overlay_bg, (bx, by), (bx + col_w, by + box_h), (10, 10, 10), -1)
                out = cv2.addWeighted(overlay_bg, 0.75, out, 0.25, 0)

                # Border — pulses amber as time runs low
                border_col = (0, 200, 60) if timeout_left > 1.5 else (0, 140, 255)
                cv2.rectangle(out, (bx, by), (bx + col_w, by + box_h), border_col, 2)

                # Title
                cv2.putText(out, "SELECT PRESET", (bx + 8, by + pad + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 60), 1, cv2.LINE_AA)

                # Each preset row
                cur_gid = det.get("gesture_id")
                for i, (gid, name) in enumerate(entries):
                    ry = by + pad + row_h * (i + 1)
                    # Highlight row matching currently shown gesture
                    if cur_gid == gid:
                        cv2.rectangle(out, (bx + 4, ry - row_h + 6),
                                      (bx + col_w - 4, ry + 2), (0, 60, 20), -1)
                        # Debounce hold progress bar inside highlighted row
                        if deb_gid == gid and deb_ratio > 0:
                            bar_x = bx + 6; bar_y = ry - 8
                            bar_w = col_w - 14; bar_h = 5
                            cv2.rectangle(out, (bar_x, bar_y),
                                          (bar_x + bar_w, bar_y + bar_h), (30, 80, 30), -1)
                            fill = int(bar_w * min(deb_ratio, 1.0))
                            if fill > 0:
                                cv2.rectangle(out, (bar_x, bar_y),
                                              (bar_x + fill, bar_y + bar_h), (0, 255, 100), -1)
                    # Gesture number badge
                    cv2.rectangle(out, (bx + 6, ry - row_h + 8),
                                  (bx + 28, ry), (0, 150, 50), -1)
                    cv2.putText(out, str(gid), (bx + 10, ry - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                    # Preset name
                    txt_col = (0, 255, 100) if cur_gid == gid else (200, 220, 200)
                    cv2.putText(out, name, (bx + 34, ry - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, txt_col, 1, cv2.LINE_AA)

                # ── Countdown bar at the bottom of the picker ────────
                cbar_y  = by + box_h - cbar_h - pad // 2
                cbar_x  = bx + 6
                cbar_w  = col_w - 12
                ratio   = max(0.0, min(timeout_left / timeout_total, 1.0))
                fill_w  = int(cbar_w * ratio)

                # Track background
                cv2.rectangle(out, (cbar_x, cbar_y),
                              (cbar_x + cbar_w, cbar_y + cbar_h - 4), (40, 40, 40), -1)
                # Fill — green → amber → red as time runs out
                if ratio > 0.5:
                    bar_col = (0, 200, 60)
                elif ratio > 0.25:
                    bar_col = (0, 160, 255)
                else:
                    bar_col = (0, 60, 220)
                if fill_w > 0:
                    cv2.rectangle(out, (cbar_x, cbar_y),
                                  (cbar_x + fill_w, cbar_y + cbar_h - 4), bar_col, -1)
                # Countdown text centred inside bar
                timer_txt = f"{timeout_left:.1f}s to pick"
                (tw, th), _ = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.putText(out, timer_txt,
                            (cbar_x + (cbar_w - tw) // 2, cbar_y + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        # ── End preset picker ─────────────────────────────────────────

        # Draw hand skeleton
        if full.get("hand_detected"):
            if "landmarks" in full and _MP_OLD_API and _mp_drawing:
                try:
                    _mp_drawing.draw_landmarks(out, full["landmarks"], _mp_hands.HAND_CONNECTIONS,
                        _mp_drawing_styles.get_default_hand_landmarks_style(),
                        _mp_drawing_styles.get_default_hand_connections_style())
                except Exception: pass
            elif "landmarks_raw" in full:
                _draw_landmarks_manual(out, full["landmarks_raw"])

        if det.get("hand_detected"):
            gid = det["gesture_id"]; gname = det["gesture_name"]
            conf = det["confidence"]; bbox = det["bbox"]
            preset = det.get("robot_preset")
            color = GESTURE_COLORS.get(gid, (255,255,255))
            if bbox:
                x1,y1,x2,y2 = [int(v) for v in bbox]
                cv2.rectangle(out, (x1,y1), (x2,y2), color, 3)
                label = f"{gid}: {gname}"
                (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(out, (x1,y1-lh-10), (x1+lw+10,y1), color, -1)
                cv2.putText(out, label, (x1+5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(out, str(gid), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)
            cv2.putText(out, gname, (90,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(out, f"{conf:.0%}", (90,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 1)
            pt = f"PRESET: {preset}" if preset else "READY"
            pc = (0,255,0) if preset else (80,80,255)
            if preset == "camera" or (canon and canon.is_flashing()):
                pt = f"PHOTO #{canon.capture_count}" if canon else "SNAP!"
                pc = (0,255,255)
                ov = out.copy()
                cv2.rectangle(ov, (0,0), (fw,fh), (255,255,255), -1)
                alpha = max(0.0, (canon._flash_until - time.time())/2.0*0.3) if canon else 0.15
                out = cv2.addWeighted(ov, alpha, out, 1-alpha, 0)
                cv2.putText(out, "PHOTO CAPTURED", (fw//2-130,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            elif scan_active:
                pos_lbl = "POS 1" if scan_current_pos == "pos1" else "POS 2"
                pt = f"SCANNING: {pos_lbl}"; pc = (0,200,255)
                pulse = int(127 + 128*np.sin(time.time()*4))
                cv2.putText(out, "SCANNING", (fw//2-80,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,pulse,255), 2)
            cv2.putText(out, pt, (fw-220,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pc, 2)
            cv2.putText(out, f"[{det.get('method','')}]", (10,fh-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        else:
            cv2.putText(out, "No hand detected", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,255), 2)
            cv2.putText(out, "READY", (fw-220,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,255), 2)

        cv2.circle(out, (20,fh-20), 8, (0,255,0), -1)
        cv2.putText(out, "TRACKING", (35,fh-13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        if robot and robot.connected:
            st,sc = ("ROBOT: ONLINE",(0,255,0)) if robot.enabled else ("ROBOT: DISABLED",(0,200,255))
        elif dry_run: st,sc = "ROBOT: DRY RUN",(0,255,255)
        else: st,sc = "ROBOT: OFFLINE",(80,80,255)
        cv2.putText(out, st, (fw-230,fh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)

        if canon:
            if canon.connected: cs,cc = f"CANON: {canon.capture_count} shots",(0,255,255)
            elif canon.last_error: cs,cc = f"CANON: {canon.last_error[:28]}",(0,80,255)
            else: cs,cc = "CANON: OFFLINE",(0,80,255)
            cv2.putText(out, cs, (fw-230,fh-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, 1)

        if countdown_state and countdown_state.get("active"):
            vig = out.copy(); cv2.rectangle(vig,(0,0),(fw,fh),(0,0,0),-1)
            out = cv2.addWeighted(vig, 0.45, out, 0.55, 0)
            cx, cy = fw//2, fh//2; phase = countdown_state.get("phase")
            if phase == "settling":
                p = 0.6 + 0.4*abs(np.sin(time.time()*3)); col=(int(255*p),int(220*p),0)
                cv2.putText(out,"Bersiap...",(cx-110,cy+10),cv2.FONT_HERSHEY_SIMPLEX,1.4,col,3,cv2.LINE_AA)
            elif phase == "counting":
                num = countdown_state.get("number")
                if num:
                    nc={3:(0,220,255),2:(0,165,255),1:(0,80,255)}; col=nc.get(num,(255,255,255))
                    ns=str(num); (tw,th),_=cv2.getTextSize(ns,cv2.FONT_HERSHEY_SIMPLEX,5,10)
                    cv2.putText(out,ns,(cx-tw//2,cy+th//2),cv2.FONT_HERSHEY_SIMPLEX,5,col,10,cv2.LINE_AA)
            elif phase == "capturing":
                fl=out.copy(); cv2.rectangle(fl,(0,0),(fw,fh),(255,255,255),-1)
                out=cv2.addWeighted(fl,0.4,out,0.6,0)
                cv2.putText(out,"FOTO!",(cx-90,cy+20),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,220,0),6,cv2.LINE_AA)
        return out

    def _run(self, tracking_active_fn):
        self._init_camera()
        yolo_ok = self._init_yolo()
        mp_ok   = self._init_mediapipe()
        self._init_sklearn()
        if self._detection_method == "auto":
            if yolo_ok:   print("  [OK] YOLOv8 primary + MediaPipe fallback")
            elif mp_ok:   print("  [OK] MediaPipe only")
            else:         print("  [ERROR] No detection method available!")
        try:
            while self._running:
                frame = None
                if self._camera_source == "realsense" and self._rs_pipeline:
                    frames = self._rs_pipeline.wait_for_frames(timeout_ms=200)
                    cf = frames.get_color_frame()
                    if cf:
                        frame = np.asanyarray(cf.get_data())
                        frame = cv2.flip(frame, 1)
                elif self._camera and self._camera.isOpened():
                    # Flush any buffered frames — grab without decoding until buffer empty
                    self._camera.grab()
                    ret, frame = self._camera.read()
                    if ret: frame = cv2.flip(frame, 1)
                    else: frame = None
                if frame is not None:
                    with self._lock: self._latest_frame = frame.copy()
                    # Feed frame into GestureRecognizer (async, result comes via callback)
                    if isinstance(self._hands_detector, dict) and \
                            self._hands_detector.get("type") == "gesture_recognizer":
                        try:
                            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            self._hands_detector["recognizer"].recognize_async(
                                mp_img, int(time.time() * 1000))
                        except Exception as e:
                            print(f"  [WARN] recognize_async: {e}")
                    self._analyze_frame(frame, tracking_active=tracking_active_fn())
        finally:
            if self._camera: self._camera.release()
            if self._rs_pipeline: self._rs_pipeline.stop()
            if isinstance(self._hands_detector, dict):
                try: self._hands_detector["recognizer"].close()
                except: pass
            print("  [CAM] Camera released.")

    def start(self, tracking_active_fn=lambda: True):
        self._running = True
        self._thread  = threading.Thread(target=self._run, args=(tracking_active_fn,), daemon=True)
        self._thread.start()
        print("  [COMVIS] Gesture detector running in background.")

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=3)

    def is_running(self): return self._running

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_detection(self):
        with self._lock: return self._latest_detection.copy()
