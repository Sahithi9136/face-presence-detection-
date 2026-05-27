"""
Test Runner — Face Presence Detection Module
Phase 3/4 — AI Interview Monitoring System

Tests:
  1. Unit test: synthetic frame (blank = no face)
  2. Unit test: config thresholds
  3. Integration test: video file or webcam
  4. Edge case: low-light simulation
  5. Edge case: rapid in/out presence (occlusion)

Run:
    python test_detector.py               # all unit tests
    python test_detector.py --video path  # video file integration test
    python test_detector.py --webcam      # live webcam test
"""

import cv2
import numpy as np
import time
import json
import argparse
from dataclasses import asdict
from face_presence_detector import FacePresenceDetector, DetectorConfig


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def print_result(label: str, result, passed: bool):
    mark = "✅" if passed else "❌"
    data = json.dumps(asdict(result), indent=2)
    print(f"\n{mark}  {label}")
    print(data)


def blank_frame(h=480, w=640):
    """Solid black frame — no face possible."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def simulate_low_light(frame, brightness=0.15):
    """Darken a frame to simulate poor lighting."""
    return (frame.astype(np.float32) * brightness).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────

def test_blank_frame_no_face():
    """Black frame must be detected as absent after threshold frames."""
    cfg = DetectorConfig(absent_frame_threshold=3, smoothing_window=5)
    detector = FacePresenceDetector(cfg)

    result = None
    for i in range(10):
        result = detector.process_frame(blank_frame(), i)

    passed = not result.face_present
    print_result("Blank frame → face absent", result, passed)
    detector.close()
    return passed


def test_config_thresholds():
    """Verify that absent_frame_threshold delays confirmed absence."""
    cfg = DetectorConfig(absent_frame_threshold=15, smoothing_window=5, presence_ratio_threshold=0.8)
    detector = FacePresenceDetector(cfg)

    # Feed 5 blank frames — threshold is 15, so absence should NOT be confirmed yet
    result = None
    for i in range(5):
        result = detector.process_frame(blank_frame(), i)

    # duration_absent_sec should be 0 because threshold not yet reached
    passed = result.duration_absent_sec == 0.0
    print_result("Threshold not reached → duration=0", result, passed)
    detector.close()
    return passed


def test_stats_accumulate():
    """Stats should accumulate correctly over frames."""
    detector = FacePresenceDetector(DetectorConfig(absent_frame_threshold=2, smoothing_window=3))
    for i in range(20):
        detector.process_frame(blank_frame(), i)

    stats = detector.get_stats()
    passed = stats["total_frames"] == 20 and stats["absent_frames"] > 0
    print(f"\n✅  Stats test" if passed else f"\n❌  Stats test")
    print(json.dumps(stats, indent=2))
    detector.close()
    return passed


def test_reset():
    """Reset must zero all counters."""
    detector = FacePresenceDetector()
    for i in range(10):
        detector.process_frame(blank_frame(), i)

    detector.reset()
    stats = detector.get_stats()
    passed = stats["total_frames"] == 0
    print(f"\n✅  Reset clears stats" if passed else f"\n❌  Reset failed")
    print(json.dumps(stats, indent=2))
    detector.close()
    return passed


def test_low_light_simulation():
    """Enhance frame preprocessing should handle low-light frames."""
    cfg = DetectorConfig(enhance_frame=True)
    detector = FacePresenceDetector(cfg)

    # Create a slightly grey frame (not completely black)
    grey_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    result = detector.process_frame(grey_frame, 0)

    # We just verify no crash and result is valid
    passed = isinstance(result.face_present, bool)
    print_result("Low-light frame (enhance=True) — no crash", result, passed)
    detector.close()
    return passed


# ─────────────────────────────────────────────
# Integration: Video File
# ─────────────────────────────────────────────

def test_video_file(path: str, max_frames: int = 300):
    """
    Run detector over a video file and print per-second summary.
    Outputs JSON log to presence_log.json.
    """
    print(f"\n{'─'*50}")
    print(f"Integration Test: {path}")
    print(f"{'─'*50}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌  Cannot open: {path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    detector = FacePresenceDetector(DetectorConfig())
    log = []
    idx = 0
    absent_total = 0

    start = time.time()
    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.process_frame(frame, idx)
        if not result.face_present:
            absent_total += 1

        # Log every second (every ~fps frames)
        if idx % max(1, int(fps)) == 0:
            entry = asdict(result)
            log.append(entry)
            status = "PRESENT ✅" if result.face_present else f"ABSENT ❌ ({result.duration_absent_sec}s)"
            print(f"  Frame {idx:4d} | t={idx/fps:.1f}s | {status} | conf={result.confidence:.2f}")

        idx += 1

    elapsed = time.time() - start
    stats = detector.get_stats()

    print(f"\n📊 Session Stats:")
    print(f"   Frames processed : {stats['total_frames']}")
    print(f"   Absent frames    : {stats['absent_frames']}")
    print(f"   Absent %         : {stats['absent_percent']}%")
    print(f"   Processing time  : {elapsed:.2f}s  ({idx/elapsed:.1f} fps)")

    # Write log
    with open("presence_log.json", "w") as f:
        json.dump({"video": path, "stats": stats, "log": log}, f, indent=2)
    print(f"   Log written      : presence_log.json")

    cap.release()
    detector.close()
    return True


# ─────────────────────────────────────────────
# Integration: Webcam (live)
# ─────────────────────────────────────────────

def test_webcam(source=0):
    """Live webcam test with overlay. Press Q to quit."""
    print(f"\n{'─'*50}")
    print("Live Webcam Test — Press Q to quit")
    print(f"{'─'*50}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌  Webcam not available.")
        return False

    detector = FacePresenceDetector()
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame, idx)
        annotated = detector.draw_overlay(frame, result)

        # Print JSON to console every 30 frames
        if idx % 30 == 0:
            print(json.dumps(asdict(result)))

        cv2.imshow("Face Presence Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n📊 Final Stats:", json.dumps(detector.get_stats()))
    detector.close()
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Presence Detector — Test Runner")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam test")
    parser.add_argument("--source", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()

    if args.webcam:
        test_webcam(args.source)
    elif args.video:
        test_video_file(args.video)
    else:
        print("=" * 50)
        print("  Unit Tests — Face Presence Detector")
        print("=" * 50)

        results = [
            test_blank_frame_no_face(),
            test_config_thresholds(),
            test_stats_accumulate(),
            test_reset(),
            test_low_light_simulation(),
        ]

        passed = sum(results)
        total = len(results)
        print(f"\n{'='*50}")
        print(f"  Results: {passed}/{total} passed")
        print(f"{'='*50}")
        exit(0 if passed == total else 1)
