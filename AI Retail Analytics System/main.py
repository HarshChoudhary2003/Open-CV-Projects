"""
main.py  ─  AI Retail Analytics System ─ Core Pipeline
=======================================================
Usage:
    python main.py                      # Live webcam (cam 0)
    python main.py --source store.mp4   # Video file
    python main.py --demo               # Generate & play synthetic demo
    python main.py --source 0 --no-display   # Headless (dashboard-only)

Press  ESC / Q  to quit the live window.
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from detection.yolo import PersonDetector
from tracking.tracker import RetailTracker
from analytics.footfall import FootfallCounter
from analytics.dwell_time import DwellTimeTracker
from analytics.heatmap import HeatmapEngine
from database.db import init_db, create_session, log_footfall, log_dwell, log_heatmap, log_frame_metrics
from utils.helpers import (resize_frame, overlay_text, draw_fps,
                            generate_demo_video, generate_insights)


# ── Constants ──────────────────────────────────────────────────────────────
DB_LOG_INTERVAL      = 30   # frames between DB writes
HEATMAP_LOG_INTERVAL = 150  # frames between heatmap snapshots
DWELL_ALERT_SEC      = 120  # flag customers > 2 min
DISPLAY_WIDTH        = 960


def parse_args():
    p = argparse.ArgumentParser(description="AI Retail Analytics System")
    p.add_argument("--source",      default="0",        help="Video source (int for webcam, or file path)")
    p.add_argument("--model",       default="yolov8n.pt", help="YOLOv8 model weights")
    p.add_argument("--conf",        type=float, default=0.4, help="Detection confidence threshold")
    p.add_argument("--demo",        action="store_true", help="Generate & use synthetic demo video")
    p.add_argument("--no-display",  action="store_true", help="Run headless (no OpenCV window)")
    p.add_argument("--heatmap",     action="store_true", help="Show heatmap overlay in live view")
    p.add_argument("--output",      default=None,        help="Optional output video file path")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Database init ───────────────────────────────────────────────────
    init_db()
    source_label = args.source

    # ── 2. Demo video ──────────────────────────────────────────────────────
    if args.demo:
        demo_path = str(Path("assets") / "demo.mp4")
        generate_demo_video(output_path=demo_path, duration_sec=60, n_people=8)
        args.source = demo_path
        source_label = demo_path

    # ── 3. Open video source ───────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source!r}")
        sys.exit(1)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"[Pipeline] Source: {args.source}  |  {frame_w}×{frame_h} @ {fps_src:.1f} fps")

    # ── 4. Build pipeline components ──────────────────────────────────────
    detector    = PersonDetector(model_path=args.model, confidence=args.conf)
    tracker     = RetailTracker()
    footfall    = FootfallCounter()
    dwell       = DwellTimeTracker(alert_threshold_sec=DWELL_ALERT_SEC)
    heatmap_eng = HeatmapEngine(frame_h, frame_w)
    session_id  = create_session(source=source_label)

    print(f"[DB] Session ID: {session_id}  |  Tracker: {tracker.backend.upper()}")

    # ── 5. Optional video writer ───────────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (DISPLAY_WIDTH, int(frame_h * DISPLAY_WIDTH / frame_w)))
        print(f"[Output] Writing to: {args.output}")

    # ── 6. Main loop ───────────────────────────────────────────────────────
    frame_no    = 0
    fps_timer   = time.time()
    fps_display = 0.0

    print("[Pipeline] Running — press ESC or Q to stop...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Pipeline] End of stream.")
            break

        frame_no += 1
        loop_start = time.time()

        # ── Detection ──────────────────────────────────────────────────────
        raw_dets    = detector.detect(frame)
        dets_array  = detector.detections_to_array(raw_dets)

        # ── Tracking ──────────────────────────────────────────────────────
        tracks = tracker.update(dets_array)        # (M,5) [x1,y1,x2,y2,tid]
        track_ids = [int(t[4]) for t in tracks]

        # ── Analytics ─────────────────────────────────────────────────────
        footfall.update(track_ids)
        dwell.update(track_ids)
        heatmap_eng.update(tracks)

        dwell_times = {tid: dwell.get_dwell(tid) for tid in track_ids}

        # ── DB logging (batched) ───────────────────────────────────────────
        if frame_no % DB_LOG_INTERVAL == 0 and track_ids:
            log_footfall(session_id, track_ids)
            log_dwell(session_id, dwell.get_all_dwells())
            log_frame_metrics(session_id, frame_no,
                               len(track_ids), footfall.total, dwell.average_dwell())

        if frame_no % HEATMAP_LOG_INTERVAL == 0:
            zones = heatmap_eng.hot_zones(n=3)
            log_heatmap(session_id, zones)

        # ── Visual overlay ─────────────────────────────────────────────────
        if not args.no_display or writer:
            display = resize_frame(frame.copy(), DISPLAY_WIDTH)

            if args.heatmap:
                scale_x = DISPLAY_WIDTH / frame_w
                scale_y = display.shape[0] / frame_h
                # Resize heatmap to display size
                hm_h, hm_w = display.shape[:2]
                hm_img = heatmap_eng.get_heatmap_image()
                hm_img = cv2.resize(hm_img, (hm_w, hm_h))
                display = cv2.addWeighted(display, 0.55, hm_img, 0.45, 0)

            # Scale tracks to display
            sx = DISPLAY_WIDTH / frame_w
            sy = display.shape[0] / frame_h
            scaled_tracks = tracks.copy()
            if len(scaled_tracks):
                scaled_tracks[:, 0] *= sx
                scaled_tracks[:, 2] *= sx
                scaled_tracks[:, 1] *= sy
                scaled_tracks[:, 3] *= sy
            tracker.draw_tracks(display, scaled_tracks, dwell_times)

            # Stats HUD
            avg_dwell  = dwell.average_dwell()
            peak_h, _  = footfall.peak_hour()
            hud_lines  = [
                f"Visitors : {footfall.total}",
                f"On-screen: {len(track_ids)}",
                f"Avg Dwell: {avg_dwell:.0f}s",
                f"Peak Hour: {peak_h}",
                f"Session  : {session_id}",
            ]
            overlay_text(display, hud_lines, origin=(10, 10))
            draw_fps(display, fps_display)

            if writer:
                writer.write(display)

            if not args.no_display:
                cv2.imshow("🛒  AI Retail Analytics  —  ESC / Q to quit", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("[Pipeline] User requested stop.")
                    break

        # ── FPS calc ───────────────────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = frame_no / elapsed
            frame_no = 0
            fps_timer = time.time()

        # ── Terminal status ────────────────────────────────────────────────
        if int(time.time()) % 5 == 0 and frame_no == 1:
            insights = generate_insights(
                footfall.total, dwell.average_dwell(),
                footfall.peak_hour()[0], heatmap_eng.hot_zones(2), frame_w
            )
            print("\n[Insights]")
            for insight in insights:
                print(" ", insight)

    # ── Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final DB flush
    log_footfall(session_id, list(dwell.get_all_dwells().keys()))
    log_dwell(session_id, dwell.get_all_dwells())
    log_heatmap(session_id, heatmap_eng.hot_zones(3))

    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total Visitors : {footfall.total}")
    print(f"  Avg Dwell Time : {dwell.average_dwell():.1f}s")
    if dwell.get_all_dwells():
        mtid, msec = dwell.max_dwell()
        print(f"  Longest Stay   : ID {mtid}  →  {msec:.1f}s")
    ph, pc = footfall.peak_hour()
    print(f"  Peak Hour      : {ph}  ({pc} visitors)")
    print(f"  DB Session     : {session_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
