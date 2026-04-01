import cv2
import time
import threading
import numpy as np

import mediapipe as mp

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
    print("[WARN] pyrealsense2 not installed. RealSense camera unavailable.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

THUMB_TIP  = mp_hands.HandLandmark.THUMB_TIP
THUMB_IP   = mp_hands.HandLandmark.THUMB_IP
INDEX_TIP  = mp_hands.HandLandmark.INDEX_FINGER_TIP
INDEX_PIP  = mp_hands.HandLandmark.INDEX_FINGER_PIP
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
MIDDLE_PIP = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
RING_TIP   = mp_hands.HandLandmark.RING_FINGER_TIP
RING_PIP   = mp_hands.HandLandmark.RING_FINGER_PIP
PINKY_TIP  = mp_hands.HandLandmark.PINKY_TIP
PINKY_PIP  = mp_hands.HandLandmark.PINKY_PIP
WRIST      = mp_hands.HandLandmark.WRIST

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

# ── Finger helpers ────────────────────────────────────────────────

def is_finger_up(lm, tip, pip): return lm[tip].y < lm[pip].y
def is_thumb_up(lm, hand="Right"):
    return (lm[THUMB_TIP].x < lm[THUMB_IP].x) if hand == "Right" else (lm[THUMB_TIP].x > lm[THUMB_IP].x)

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
    return min(sum(fingers),10), GESTURE_ID_TO_NAME.get(sum(fingers),"Unknown")


# ── YOLO detection ────────────────────────────────────────────────

def detect_yolo(frame, yolo_model):
    if yolo_model is None: return None
    try:
        results = yolo_model(frame, verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes)==0:
            return {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                    "confidence":0.0,"class_id":None,"bbox":None,"method":"YOLOv8"}
        box = results[0].boxes[0]
        cid = int(box.cls[0].item()); conf = float(box.conf[0].item())
        bbox = box.xyxy[0].cpu().numpy().tolist()
        gid = CLASS_TO_GESTURE.get(cid, cid+1)
        return {"hand_detected":True,"gesture_id":gid,"gesture_name":GESTURE_LABELS.get(cid,f"Class_{cid}"),
                "confidence":conf,"class_id":cid,"bbox":bbox,"method":"YOLOv8"}
    except Exception as e:
        print(f"  [ERROR] YOLO: {e}"); return None


# ── MediaPipe detection ───────────────────────────────────────────

def detect_mediapipe(frame, hands_detector):
    if hands_detector is None: return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                "confidence":0.0,"class_id":None,"bbox":None,"method":"MediaPipe"}
    hlm = results.multi_hand_landmarks[0]; lm = hlm.landmark
    hand = "Right"
    if results.multi_handedness: hand = results.multi_handedness[0].classification[0].label
    h,w,_ = frame.shape
    xs=[l.x*w for l in lm]; ys=[l.y*h for l in lm]
    bbox=[max(0,int(min(xs))-20),max(0,int(min(ys))-20),min(w,int(max(xs))+20),min(h,int(max(ys))+20)]
    fingers = [is_thumb_up(lm,hand), is_finger_up(lm,INDEX_TIP,INDEX_PIP),
               is_finger_up(lm,MIDDLE_TIP,MIDDLE_PIP), is_finger_up(lm,RING_TIP,RING_PIP),
               is_finger_up(lm,PINKY_TIP,PINKY_PIP)]
    gid,gname = classify_fingers(fingers)
    return {"hand_detected":True,"gesture_id":gid,"gesture_name":gname,"confidence":0.85,
            "class_id":gid-1 if gid<10 else 0,"bbox":bbox,"method":"MediaPipe+Rules",
            "landmarks":hlm,"handedness":hand}


# ── GestureDetector class ─────────────────────────────────────────

class GestureDetector:

    def __init__(self, camera_index=1, frame_width=640, frame_height=480,
                 camera_source="laptop", detection_method="auto",
                 model_path=None, sklearn_model_path=None,
                 mp_max_hands=1, mp_detection_conf=0.7, mp_tracking_conf=0.5):

        self._camera_index     = camera_index
        self._frame_width      = frame_width
        self._frame_height     = frame_height
        self._camera_source    = camera_source
        self._detection_method = detection_method
        self._model_path       = model_path
        self._sklearn_model_path = sklearn_model_path
        self._mp_max_hands     = mp_max_hands
        self._mp_detection_conf = mp_detection_conf
        self._mp_tracking_conf = mp_tracking_conf

        self._lock             = threading.Lock()
        self._running          = False
        self._thread           = None

        # Camera handles
        self._camera           = None
        self._rs_pipeline      = None
        self._rs_align         = None

        # Detection models
        self._yolo_model       = None
        self._hands_detector   = None
        self._sklearn_model    = None

        # State
        self._latest_frame     = None
        self._latest_detection = {
            "hand_detected": False, "gesture_id": None, "gesture_name": "None",
            "confidence": 0.0, "class_id": None, "bbox": None,
            "method": "none", "robot_preset": None,
        }
        self._last_full_detection = {}

    # ── Initialization ────────────────────────────────────────────

    def _init_camera(self):
        import sys
        if self._camera_source == "realsense":
            if not REALSENSE_AVAILABLE:
                raise RuntimeError("pyrealsense2 is not installed. Run: pip install pyrealsense2")
            self._rs_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self._frame_width, self._frame_height, rs.format.bgr8, 30)
            self._rs_pipeline.start(config)
            self._rs_align = rs.align(rs.stream.color)
            time.sleep(1)
            print(f"  [CAM] Intel RealSense SR305 opened: {self._frame_width}x{self._frame_height}")
        else:
            if sys.platform == "win32":
                self._camera = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
            else:
                self._camera = cv2.VideoCapture(self._camera_index)
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self._camera.set(cv2.CAP_PROP_FPS, 30)
            if not self._camera.isOpened():
                raise RuntimeError(f"Cannot open camera index {self._camera_index}")
            print(f"  [CAM] Laptop camera opened: {self._frame_width}x{self._frame_height}")

    def _init_yolo(self):
        import os
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
        try:
            self._hands_detector = mp_hands.Hands(
                static_image_mode=False, max_num_hands=self._mp_max_hands,
                min_detection_confidence=self._mp_detection_conf,
                min_tracking_confidence=self._mp_tracking_conf)
            print("  [OK] MediaPipe Hands ready"); return True
        except Exception as e:
            print(f"  [ERROR] MediaPipe: {e}"); return False

    def _init_sklearn(self):
        import os
        if not JOBLIB_AVAILABLE or not self._sklearn_model_path: return False
        if not os.path.exists(self._sklearn_model_path): return False
        try:
            self._sklearn_model = joblib.load(self._sklearn_model_path); return True
        except:
            return False

    # ── Frame analysis ────────────────────────────────────────────

    def _analyze_frame(self, frame, tracking_active=True):
        if not tracking_active:
            idle = {"hand_detected":False,"gesture_id":None,"gesture_name":"Tracking Paused",
                    "confidence":0.0,"class_id":None,"bbox":None,"method":"paused","robot_jog":None}
            with self._lock:
                self._latest_detection = idle
            return idle

        det = None
        if self._detection_method == "yolo":
            det = detect_yolo(frame, self._yolo_model)
        elif self._detection_method == "mediapipe":
            det = detect_mediapipe(frame, self._hands_detector)
        elif self._detection_method == "auto":
            det = detect_yolo(frame, self._yolo_model)
            if det is None:
                det = detect_mediapipe(frame, self._hands_detector)

        if det is None:
            det = {"hand_detected":False,"gesture_id":None,"gesture_name":"None",
                   "confidence":0.0,"class_id":None,"bbox":None,"method":"none"}

        clean = {k:v for k,v in det.items() if k not in ("landmarks","handedness")}
        with self._lock:
            self._latest_detection = clean
            self._last_full_detection.clear()
            self._last_full_detection.update(det)

        return det

    # ── Overlay ───────────────────────────────────────────────────

    def draw_overlay(self, frame, tracking_active=True, robot=None, canon=None,
                     scan_active=False, scan_current_pos=None, countdown_state=None,
                     dry_run=False):
        out = frame.copy()
        with self._lock:
            det = self._latest_detection.copy()
            full = self._last_full_detection.copy()

        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        # Paused banner
        if not tracking_active:
            overlay = out.copy()
            cv2.rectangle(overlay, (0,0), (frame_w, frame_h), (0,0,0), -1)
            out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
            cv2.putText(out, "TRACKING PAUSED", (frame_w//2-180, frame_h//2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
            cv2.putText(out, "Press [SPACE] to resume",
                        (frame_w//2-140, frame_h//2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            if robot and robot.connected:
                st = "ROBOT: ONLINE" if robot.enabled else "ROBOT: DISABLED"
                sc = (0,255,0) if robot.enabled else (0,200,255)
            elif dry_run: st,sc = "ROBOT: DRY RUN",(0,255,255)
            else: st,sc = "ROBOT: OFFLINE",(80,80,255)
            cv2.putText(out, st, (frame_w-230, frame_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)
            return out

        # Hand skeleton
        if "landmarks" in full and full.get("hand_detected"):
            mp_drawing.draw_landmarks(out, full["landmarks"], mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        if det.get("hand_detected"):
            gid=det["gesture_id"]; gname=det["gesture_name"]; conf=det["confidence"]
            bbox=det["bbox"]; preset=det.get("robot_preset"); color=GESTURE_COLORS.get(gid,(255,255,255))
            if bbox:
                x1,y1,x2,y2=[int(v) for v in bbox]
                cv2.rectangle(out,(x1,y1),(x2,y2),color,3)
                label=f"{gid}: {gname}"
                (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
                cv2.rectangle(out,(x1,y1-lh-10),(x1+lw+10,y1),color,-1)
                cv2.putText(out,label,(x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(out,str(gid),(20,70),cv2.FONT_HERSHEY_SIMPLEX,2.5,color,5)
            cv2.putText(out,gname,(90,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            cv2.putText(out,f"{conf:.0%}",(90,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,255,150),1)
            pt=f"PRESET: {preset}" if preset else "READY"
            pc=(0,255,0) if preset else (80,80,255)

            # Camera capture flash indicator
            if preset == "camera" or (canon and canon.is_flashing()):
                pt = f"PHOTO #{canon.capture_count}" if canon else "SNAP!"
                pc = (0,255,255)
                overlay = out.copy()
                cv2.rectangle(overlay, (0,0), (frame_w, frame_h), (255,255,255), -1)
                alpha = max(0.0, (canon._flash_until - time.time()) / 2.0 * 0.3) if canon else 0.15
                out = cv2.addWeighted(overlay, alpha, out, 1-alpha, 0)
                cv2.putText(out, "PHOTO CAPTURED", (frame_w//2-130, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Scan loop indicator
            elif scan_active:
                pos_label = "POS 1" if scan_current_pos == "pos1" else "POS 2"
                pt = f"SCANNING: {pos_label}"
                pc = (0, 200, 255)
                pulse = int(127 + 128 * np.sin(time.time() * 4))
                cv2.putText(out, "SCANNING", (frame_w//2-80, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, pulse, 255), 2)

            cv2.putText(out,pt,(frame_w-220,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,pc,2)
            cv2.putText(out,f"[{det.get('method','')}]",(10,frame_h-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1)
        else:
            cv2.putText(out,"No hand detected",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,255),2)
            cv2.putText(out,"READY",(frame_w-220,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(80,80,255),2)

        # Tracking active indicator
        cv2.circle(out,(20,frame_h-20),8,(0,255,0),-1)
        cv2.putText(out,"TRACKING",(35,frame_h-13),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

        if robot and robot.connected:
            st="ROBOT: ONLINE" if robot.enabled else "ROBOT: DISABLED"
            sc=(0,255,0) if robot.enabled else (0,200,255)
        elif dry_run: st,sc="ROBOT: DRY RUN",(0,255,255)
        else: st,sc="ROBOT: OFFLINE",(80,80,255)
        cv2.putText(out,st,(frame_w-230,frame_h-15),cv2.FONT_HERSHEY_SIMPLEX,0.55,sc,2)

        # Canon camera status
        if canon is not None:
            if canon.connected:
                cs = f"CANON: {canon.capture_count} shots"
                cc = (0, 255, 255)
            elif canon.last_error:
                cs = f"CANON: {canon.last_error[:28]}"
                cc = (0, 80, 255)
            else:
                cs, cc = "CANON: OFFLINE", (0, 80, 255)
            cv2.putText(out, cs, (frame_w-230, frame_h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cc, 1)
            if not canon.connected and (time.time() - canon._last_reconnect_attempt) < 3:
                cv2.putText(out, "RECONNECTING...", (frame_w//2 - 90, frame_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Countdown overlay
        if countdown_state and countdown_state.get("active"):
            vignette = out.copy()
            cv2.rectangle(vignette, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
            out = cv2.addWeighted(vignette, 0.45, out, 0.55, 0)

            cx = frame_w // 2
            cy = frame_h // 2
            phase = countdown_state.get("phase")
            COUNTDOWN_SPEAK_INTRO = "Lihat ke kamera"

            if phase == "settling":
                pulse = 0.6 + 0.4 * abs(np.sin(time.time() * 3))
                color = (int(255 * pulse), int(220 * pulse), 0)
                cv2.putText(out, "Bersiap...", (cx - 110, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

            elif phase == "intro":
                txt = COUNTDOWN_SPEAK_INTRO
                (tw, _th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                cv2.putText(out, txt, (cx - tw // 2, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(out, f"Preset {countdown_state.get('preset', '?')} terkunci", (cx - 90, cy + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            elif phase == "counting":
                num = countdown_state.get("number")
                if num is not None:
                    num_colors = {3: (0, 220, 255), 2: (0, 165, 255), 1: (0, 80, 255)}
                    color = num_colors.get(num, (255, 255, 255))
                    num_str = str(num)
                    (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
                    cv2.putText(out, num_str, (cx - tw // 2, cy + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10, cv2.LINE_AA)
                    cv2.putText(out, COUNTDOWN_SPEAK_INTRO, (cx - 130, cy - th // 2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

            elif phase == "capturing":
                flash = out.copy()
                cv2.rectangle(flash, (0, 0), (frame_w, frame_h), (255, 255, 255), -1)
                out = cv2.addWeighted(flash, 0.4, out, 0.6, 0)
                cv2.putText(out, "FOTO!", (cx - 90, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 220, 0), 6, cv2.LINE_AA)

        return out

    # ── Capture loop (runs in background thread) ──────────────────

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
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        frame = np.asanyarray(color_frame.get_data())
                        frame = cv2.flip(frame, 1)
                elif self._camera_source == "laptop" and self._camera and self._camera.isOpened():
                    ret, frame = self._camera.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                    else:
                        frame = None

                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame.copy()
                    self._analyze_frame(frame, tracking_active=tracking_active_fn())

                time.sleep(0.025)
        finally:
            if self._camera:
                self._camera.release()
            if self._rs_pipeline:
                self._rs_pipeline.stop()
            if self._hands_detector:
                self._hands_detector.close()
            print("  [CAM] Camera released.")

    # ── Public API ────────────────────────────────────────────────

    def start(self, tracking_active_fn=lambda: True):
        self._running = True
        self._thread  = threading.Thread(target=self._run, args=(tracking_active_fn,), daemon=True)
        self._thread.start()
        print("  [COMVIS] Gesture detector running in background.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self):
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def get_latest_detection(self):
        with self._lock:
            return self._latest_detection.copy()
