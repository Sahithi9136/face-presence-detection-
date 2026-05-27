"""
Flask API — Face Presence Detection
Phase 3/4 — AI Interview Monitoring System

Endpoints:
  GET  /status          → latest presence JSON
  GET  /stream          → MJPEG annotated video stream
  GET  /stats           → session statistics
  POST /reset           → reset session state
  POST /config          → update detector thresholds at runtime
"""

import cv2
import json
import threading
import time
from flask import Flask, Response, jsonify, request
from face_presence_detector import FacePresenceDetector, DetectorConfig, PresenceResult
from dataclasses import asdict

app = Flask(__name__)

# ─────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────

class MonitoringSession:
    def __init__(self):
        self.config = DetectorConfig()
        self.detector = FacePresenceDetector(self.config)
        self.latest_result: PresenceResult = None
        self.frame_index = 0
        self.lock = threading.Lock()
        self._cap = None
        self._running = False
        self._thread = None

    def start_camera(self, source=0):
        """Start background capture thread."""
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {source}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # End of video file — loop back
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            result = self.detector.process_frame(frame, self.frame_index)
            with self.lock:
                self.latest_result = result
                self.frame_index += 1
            time.sleep(0.033)   # ~30 fps cap

    def get_annotated_frame(self):
        """Get the latest annotated frame (for MJPEG stream)."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        with self.lock:
            result = self.detector.process_frame(frame, self.frame_index)
            self.latest_result = result
            self.frame_index += 1
        return self.detector.draw_overlay(frame, result)

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()


session = MonitoringSession()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """
    Returns the latest presence detection result.

    Example response:
    {
        "face_present": false,
        "duration_absent_sec": 4.0,
        "confidence": 0.0,
        "timestamp": "2025-01-01T10:00:00Z",
        "frame_index": 120,
        "face_bbox": null
    }
    """
    with session.lock:
        result = session.latest_result

    if result is None:
        return jsonify({"error": "No frames processed yet"}), 503

    return jsonify(asdict(result))


@app.route("/stream", methods=["GET"])
def stream():
    """
    MJPEG stream with detection overlay.
    Open in browser or consume with OpenCV.
    """
    source = request.args.get("source", 0)
    try:
        source = int(source)
    except ValueError:
        pass  # keep as string (file path)

    def generate():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                result = session.detector.process_frame(frame, idx)
                with session.lock:
                    session.latest_result = result
                    session.frame_index = idx
                idx += 1
                annotated = session.detector.draw_overlay(frame, result)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        finally:
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats", methods=["GET"])
def stats():
    """Session-level statistics (absent %, total frames, etc.)."""
    return jsonify(session.detector.get_stats())


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the detector state for a new interview session."""
    session.detector.reset()
    with session.lock:
        session.latest_result = None
        session.frame_index = 0
    return jsonify({"status": "reset", "timestamp": time.time()})


@app.route("/config", methods=["POST"])
def update_config():
    """
    Update detector thresholds at runtime without restarting.

    Body (JSON, all fields optional):
    {
        "min_detection_confidence": 0.4,
        "absent_frame_threshold": 8,
        "smoothing_window": 10,
        "presence_ratio_threshold": 0.4,
        "min_face_area_ratio": 0.005,
        "enhance_frame": true
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    cfg = session.config
    allowed = {
        "min_detection_confidence", "absent_frame_threshold",
        "smoothing_window", "presence_ratio_threshold",
        "min_face_area_ratio", "clahe_clip_limit", "enhance_frame"
    }
    updated = {}
    for key, val in data.items():
        if key in allowed:
            setattr(cfg, key, val)
            updated[key] = val

    # Re-create detector with new config (preserves session stats)
    session.detector = FacePresenceDetector(cfg)
    return jsonify({"updated": updated, "current_config": cfg.__dict__})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Face Presence Detection API")
    print("  GET  /status   → latest detection result")
    print("  GET  /stream   → MJPEG stream (add ?source=0)")
    print("  GET  /stats    → session statistics")
    print("  POST /reset    → reset session")
    print("  POST /config   → update thresholds")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
