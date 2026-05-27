"""
Face Presence Detection Module
Phase 3/4 — AI Interview Monitoring System

Detection backend:
  Primary  : OpenCV Haar Cascade (bundled — zero extra downloads)
  Optional : MediaPipe Tasks FaceDetector (if .tflite model file is present)

Handles: poor lighting, partial visibility, camera blur, temporary occlusion.
"""

import cv2
import time
import json
import threading
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import os


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class DetectorConfig:
    """Configurable thresholds for the detection module."""

    # Haar: minimum neighbours (higher = fewer false positives, misses partial faces)
    haar_min_neighbors: int = 4

    # Haar: scale factor for image pyramid
    haar_scale_factor: float = 1.1

    # Minimum face size as fraction of frame height
    min_face_size_ratio: float = 0.08

    # How many consecutive "absent" frames before absence is confirmed
    # At ~30fps, threshold=8 gives a ~0.25s grace period for occlusion/blink
    absent_frame_threshold: int = 8

    # Smoothing window size (last N frames used to decide presence)
    smoothing_window: int = 10

    # Fraction of smoothing window that must show a face to be "present"
    presence_ratio_threshold: float = 0.35

    # Apply CLAHE + sharpening before detection (helps poor lighting / blur)
    enhance_frame: bool = True

    # CLAHE clip limit
    clahe_clip_limit: float = 2.5

    # Optional: path to MediaPipe blaze_face .tflite model for higher accuracy
    mediapipe_model_path: Optional[str] = None


# ─────────────────────────────────────────────
# Frame Preprocessor
# ─────────────────────────────────────────────

class FramePreprocessor:
    def __init__(self, clip_limit: float = 2.5):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        self._kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")

    def enhance(self, bgr):
        """CLAHE on L channel + gentle sharpening."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        out = cv2.merge([l, a, b])
        out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
        out = cv2.filter2D(out, -1, self._kernel)
        return cv2.convertScaleAbs(out)


# ─────────────────────────────────────────────
# Detection Result
# ─────────────────────────────────────────────

@dataclass
class PresenceResult:
    face_present: bool
    duration_absent_sec: float
    confidence: float          # 0–1; Haar gives 1.0 when detected (no score)
    timestamp: str
    frame_index: int
    face_bbox: Optional[dict]  # {"x","y","w","h"} pixels, or None


# ─────────────────────────────────────────────
# Face Presence Detector
# ─────────────────────────────────────────────

class FacePresenceDetector:
    """
    Real-time face presence detector with dual backend.

    Usage:
        with FacePresenceDetector() as det:
            result = det.process_frame(bgr_frame, frame_idx)
            print(result.face_present, result.duration_absent_sec)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._preprocessor = FramePreprocessor(self.config.clahe_clip_limit)
        self._lock = threading.Lock()

        # ── Backend selection ──────────────────
        self._backend = "haar"
        self._mp_detector = None

        if self.config.mediapipe_model_path and os.path.exists(self.config.mediapipe_model_path):
            self._backend = self._try_init_mediapipe(self.config.mediapipe_model_path)

        # Haar cascade (always loaded as fallback)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        if self._haar.empty():
            raise RuntimeError("Could not load Haar cascade. Check OpenCV installation.")

        # ── State ─────────────────────────────
        self._smoothing_buf: deque = deque(maxlen=self.config.smoothing_window)
        self._absent_since: Optional[float] = None
        self._consecutive_absent: int = 0

        # ── Stats ─────────────────────────────
        self.total_frames: int = 0
        self.absent_frames: int = 0

    def _try_init_mediapipe(self, model_path: str) -> str:
        try:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            opts = mp_vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                min_detection_confidence=0.4
            )
            self._mp_detector = mp_vision.FaceDetector.create_from_options(opts)
            return "mediapipe"
        except Exception as e:
            print(f"[FaceDetector] MediaPipe init failed ({e}), falling back to Haar.")
            return "haar"

    # ── Core ──────────────────────────────────

    def process_frame(self, bgr_frame, frame_index: int = -1) -> PresenceResult:
        """Process one BGR frame. Thread-safe."""
        with self._lock:
            self.total_frames += 1
            h, w = bgr_frame.shape[:2]
            min_face_px = max(20, int(h * self.config.min_face_size_ratio))

            # 1. Preprocess
            enhanced = (self._preprocessor.enhance(bgr_frame)
                        if self.config.enhance_frame else bgr_frame)

            # 2. Detect
            if self._backend == "mediapipe" and self._mp_detector:
                raw_present, confidence, bbox = self._detect_mediapipe(enhanced, w, h)
            else:
                raw_present, confidence, bbox = self._detect_haar(enhanced, min_face_px)

            # 3. Smooth (reduces flicker from single-frame misses)
            self._smoothing_buf.append(1 if raw_present else 0)
            ratio = sum(self._smoothing_buf) / len(self._smoothing_buf)
            face_present = ratio >= self.config.presence_ratio_threshold

            # 4. Consecutive absent counter
            if not face_present:
                self._consecutive_absent += 1
            else:
                self._consecutive_absent = 0

            # 5. Absence timing with grace period
            confirmed_absent = (
                not face_present and
                self._consecutive_absent >= self.config.absent_frame_threshold
            )

            now = time.time()
            if confirmed_absent:
                if self._absent_since is None:
                    self._absent_since = now
                duration_absent = round(now - self._absent_since, 2)
                self.absent_frames += 1
            else:
                if face_present:
                    self._absent_since = None
                duration_absent = 0.0

            return PresenceResult(
                face_present=face_present,
                duration_absent_sec=duration_absent,
                confidence=round(confidence, 3),
                timestamp=datetime.utcnow().isoformat() + "Z",
                frame_index=frame_index,
                face_bbox=bbox,
            )

    def _detect_haar(self, bgr, min_face_px: int):
        """Haar cascade detection. Returns (present, confidence, bbox)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=self.config.haar_scale_factor,
            minNeighbors=self.config.haar_min_neighbors,
            minSize=(min_face_px, min_face_px),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return False, 0.0, None
        # Pick largest face
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        return True, 1.0, {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)}

    def _detect_mediapipe(self, bgr, frame_w: int, frame_h: int):
        """MediaPipe Tasks detection. Returns (present, confidence, bbox)."""
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._mp_detector.detect(mp_image)
        if not result.detections:
            return False, 0.0, None
        best = max(result.detections,
                   key=lambda d: d.categories[0].score if d.categories else 0)
        score = best.categories[0].score if best.categories else 1.0
        bb = best.bounding_box
        return True, score, {
            "x": bb.origin_x, "y": bb.origin_y,
            "w": bb.width,    "h": bb.height
        }

    # ── Utilities ─────────────────────────────

    def reset(self):
        """Reset all state (call between interview sessions)."""
        with self._lock:
            self._smoothing_buf.clear()
            self._absent_since = None
            self._consecutive_absent = 0
            self.total_frames = 0
            self.absent_frames = 0

    def get_stats(self) -> dict:
        with self._lock:
            pct = (round((self.absent_frames / self.total_frames) * 100, 1)
                   if self.total_frames > 0 else 0.0)
            return {
                "total_frames": self.total_frames,
                "absent_frames": self.absent_frames,
                "absent_percent": pct,
                "backend": self._backend,
            }

    def draw_overlay(self, bgr_frame, result: PresenceResult):
        """Draw detection overlay onto a frame for debugging."""
        frame = bgr_frame.copy()
        h, w = frame.shape[:2]
        if result.face_bbox:
            bb = result.face_bbox
            colour = (0, 220, 80) if result.face_present else (30, 30, 210)
            cv2.rectangle(frame,
                          (bb["x"], bb["y"]),
                          (bb["x"] + bb["w"], bb["y"] + bb["h"]),
                          colour, 2)
        status = "PRESENT" if result.face_present else "ABSENT"
        banner = (20, 160, 50) if result.face_present else (25, 25, 200)
        cv2.rectangle(frame, (0, 0), (w, 34), banner, -1)
        cv2.putText(
            frame,
            f"{status}  conf:{result.confidence:.2f}  absent:{result.duration_absent_sec}s",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2
        )
        return frame

    def close(self):
        if self._mp_detector:
            try:
                self._mp_detector.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
