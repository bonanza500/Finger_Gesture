import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r'model/hand_gesture_recognizer_v2'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def _draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for s, e in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[s].x * w), int(hand_landmarks[s].y * h)
        x2, y2 = int(hand_landmarks[e].x * w), int(hand_landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

class GestureDetector:

    def __init__(self, camera_index: int = 0, min_confidence: float = 0.55):
        self._lock            = threading.Lock()
        self._running         = False
        self._thread          = None
        self._camera_index    = camera_index
        self._min_confidence  = min_confidence

        # State yang dibaca oleh main.py
        self._current_gesture : str | None = None   # gesture yang terdeteksi sekarang
        self._gesture_since   : float      = 0.0    # kapan gesture ini pertama kali muncul

        # State untuk overlay UI
        self._latest_result   = None
        self._is_robot_busy   = False  # untuk tampilkan status di frame
        self._countdown_end_time = 0      # Waktu kapan countdown selesai

    # ── Internal MediaPipe callback ───────────────────────────────────
    def _on_result(self, result, output_image, timestamp_ms):
        gesture_name = None
        if result.gestures:
            top = result.gestures[0][0]
            if top.score >= self._min_confidence:
                gesture_name = top.category_name

        with self._lock:
            if gesture_name != self._current_gesture:
                # Gesture berubah — reset timer stabilitas
                self._current_gesture = gesture_name
                self._gesture_since   = time.time()
            self._latest_result = result

    def start_countdown(self, duration_sec: int):
        """Memulai countdown visual pada video feed."""
        with self._lock:
            self._countdown_end_time = time.time() + duration_sec
        print(f"[COMVIS] Countdown {duration_sec}s dimulai.")

    def get_latest_frame(self):
        """Returns the latest processed frame for web streaming."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def get_status(self):
        with self._lock:
            countdown_remaining = self._countdown_end_time - time.time()
            return {
                "current_gesture": self._current_gesture,
                "gesture_since": self._gesture_since,
                "is_robot_busy": self._is_robot_busy,
                "confidence": self._latest_result.gestures[0][0].score if (self._latest_result and self._latest_result.gestures) else 0,
                "countdown_active": countdown_remaining > 0,
                "countdown_remaining": round(countdown_remaining, 1)
            }

    # ── Camera loop ───────────────────────────────────────────────────
    def _run(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result
        )
        recognizer = vision.GestureRecognizer.create_from_options(options)
        cap = cv2.VideoCapture(self._camera_index)
        print(f"[COMVIS] Kamera {self._camera_index} aktif.")

        try:
            while self._running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    continue

                frame     = cv2.flip(frame, 1)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, int(time.time() * 1000))

                # ── Ambil state untuk overlay (thread-safe) ───────────
                with self._lock:
                    result       = self._latest_result
                    cur_gesture  = self._current_gesture
                    since        = self._gesture_since
                    robot_busy   = self._is_robot_busy
                    countdown_rem = self._countdown_end_time - time.time()

                # ── Gambar landmark ───────────────────────────────────
                if result and result.hand_landmarks:
                    for hand_lm in result.hand_landmarks:
                        _draw_landmarks(frame, hand_lm)

                # ── Overlay status robot ──────────────────────────────
                if robot_busy:
                    cv2.putText(frame, "ROBOT BERGERAK — menunggu...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 100, 255), 2, cv2.LINE_AA)
                elif countdown_rem > 0:
                    # ── Overlay countdown ─────────────────────────────
                    countdown_text = str(int(countdown_rem) + 1)
                    (w, h), _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 10, 20)
                    x = (frame.shape[1] - w) // 2
                    y = (frame.shape[0] + h) // 2
                    cv2.putText(frame, countdown_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 255), 20, cv2.LINE_AA)
                    cv2.putText(frame, "LIHAT KE KAMERA", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "SIAP menerima gesture",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 220, 0), 2, cv2.LINE_AA)

                # ── Overlay gesture + timer stabilitas ────────────────
                if cur_gesture and not (robot_busy or countdown_rem > 0):
                    held_for = time.time() - since
                    label    = f"Gesture: {cur_gesture}  ({held_for:.1f}s)"
                    cv2.putText(frame, label, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 2, cv2.LINE_AA)
                
                with self._lock:
                    self._latest_frame = frame.copy()

                # cv2.imshow("Gesture Detector", frame) # We stream it now

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False
                    break

        finally:
            recognizer.close()
            cap.release()
            cv2.destroyAllWindows()
            print("[COMVIS] Kamera ditutup.")

    # ── Public API ────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[COMVIS] Gesture detector berjalan di background.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def is_running(self) -> bool:
        return self._running

    def get_stable_gesture(self, stable_sec: float) -> str | None:

        with self._lock:
            if self._current_gesture is None:
                return None
            held_for = time.time() - self._gesture_since
            if held_for >= stable_sec:
                return self._current_gesture
            return None

    def set_robot_busy(self, busy: bool):

        with self._lock:
            self._is_robot_busy = busy
