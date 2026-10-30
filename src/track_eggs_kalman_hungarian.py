#!/usr/bin/env python3
"""Classical egg detection + Kalman Filter + Hungarian assignment tracking.

Example:
python src/track_eggs_kalman_hungarian.py \
  --video data/assignment_video.mp4 \
  --output outputs/tracking/tracked_eggs.mp4 \
  --csv outputs/tracking/tracks.csv

This is a classical baseline. Tune detector parameters for the video and inspect
for missed detections and ID switches; it is not a trained egg detector.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def bbox(self):
        return (self.x, self.y, self.w, self.h)


@dataclass
class Track:
    track_id: int
    kf: cv2.KalmanFilter
    bbox: tuple[int, int, int, int]
    missed: int = 0
    age: int = 1
    hits: int = 1
    confirmed: bool = False
    trail: list[tuple[int, int]] = field(default_factory=list)
    pending_rows: list[dict] = field(default_factory=list)

    @property
    def predicted_center(self):
        # Called after predict(); statePre contains the current prediction.
        s = self.kf.statePre
        return float(s[0, 0]), float(s[1, 0])


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_eggs(
    frame: np.ndarray,
    roi_top: float = 0.38,
    roi_bottom: float = 0.95,
    min_area: int = 850,
    max_area_ratio: float = 0.08,
    min_circularity: float = 0.32,
    max_axis_ratio: float = 1.65,
    max_detections: int = 4,
) -> list[Detection]:
    """Find pale, oval egg candidates in the tabletop region."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    roi = np.zeros((h, w), dtype=np.uint8)
    y0 = max(0, min(h - 1, int(h * roi_top)))
    y1 = max(y0 + 1, min(h, int(h * roi_bottom)))
    roi[y0:y1, :] = 255

    # The visible eggs are near-white; the green background and printed shirt
    # are more saturated. Keep this as a color check as well as a brightness one.
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180], dtype=np.uint8),
        np.array([179, 35, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(mask, roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = h * w * max_area_ratio
    detections: list[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area or len(contour) < 5:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        axis_ratio = max(bw, bh) / max(1, min(bw, bh))
        if min(bw, bh) < 38 or axis_ratio > max_axis_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0.0

        if circularity < min_circularity or solidity < 0.65:
            continue

        # Require a substantial share of the contour interior to be genuinely
        # pale, so reflected highlights/text do not pass on shape alone.
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        pixels = hsv[contour_mask > 0]
        pale_fraction = float(np.mean((pixels[:, 1] <= 40) & (pixels[:, 2] >= 180)))
        if pale_fraction < 0.82:
            continue

        score = (0.40 * min(circularity, 1.0) +
                 0.30 * min(solidity, 1.0) + 0.30 * pale_fraction)
        detections.append(Detection(x, y, bw, bh, score))

    detections.sort(key=lambda d: d.score * d.w * d.h, reverse=True)

    # Remove overlapping duplicates.
    kept: list[Detection] = []
    for det in detections:
        if all(box_iou(det.bbox, other.bbox) < 0.45 for other in kept):
            kept.append(det)
        if len(kept) >= max_detections:
            break
    return kept


def create_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    """Constant-velocity state: [x, y, vx, vy]."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    return kf


def associate(
    tracks: list[Track], detections: list[Detection], max_distance: float,
):
    """Hungarian assignment with hard motion and size gates."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.full((len(tracks), len(detections)), 1e6, dtype=np.float32)
    for i, track in enumerate(tracks):
        px, py = track.predicted_center
        _, _, tw, th = track.bbox
        # A hand can carry an egg a substantial distance while it is hidden.
        # The gate expands with missed time, but global assignment, shape/size
        # cost, and peer-box overlap still decide whether reconnection fits.
        gate = min(max_distance * 6.0, max_distance + 8.0 * track.missed)
        for j, det in enumerate(detections):
            distance = math.hypot(px - det.cx, py - det.cy)
            if distance > gate:
                continue
            overlap = box_iou(track.bbox, det.bbox)
            # Penalize large scale changes; retain room for partial occlusion.
            size_change = abs(math.log(max(1, det.w * det.h) / max(1, tw * th)))
            if overlap < 0.01 and track.missed <= 3 and distance > gate * 0.60:
                continue
            peer_overlap = max(
                (box_iou(other.bbox, det.bbox)
                 for k, other in enumerate(tracks) if k != i),
                default=0.0,
            )
            # If a detection lies on another track's last observed box, avoid
            # stealing it for a track that has been carried elsewhere.
            cost[i, j] = (
                distance + 12.0 * size_change - min(12.0, overlap * 12.0)
                + 100.0 * peer_overlap
            )

    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e5]
    used_tracks = {i for i, _ in pairs}
    used_dets = {j for _, j in pairs}
    return pairs, [i for i in range(len(tracks)) if i not in used_tracks], [j for j in range(len(detections)) if j not in used_dets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="outputs/tracking/tracked_eggs.mp4")
    parser.add_argument("--csv", default="outputs/tracking/tracks.csv")
    parser.add_argument("--max-missed", type=int, default=240,
                        help="Keep unmatched tracks for this many frames")
    parser.add_argument("--max-distance", type=float, default=70.0,
                        help="Base maximum track/detection center distance in pixels")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="Detections required within a track lifetime to confirm it")
    parser.add_argument("--trail-length", type=int, default=12,
                        help="Number of recent observed centers to draw (0 disables trails)")
    parser.add_argument("--roi-top", type=float, default=0.38,
                        help="Top of detector ROI as fraction of frame height")
    parser.add_argument("--roi-bottom", type=float, default=0.95,
                        help="Bottom of detector ROI as fraction of frame height")
    parser.add_argument("--min-area", type=int, default=850)
    parser.add_argument("--min-circularity", type=float, default=0.32)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Optional short test; 0 processes the whole video")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(args.output)
    csv_path = Path(args.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output_path}")

    tracks: list[Track] = []
    next_id = 1
    frame_idx = 0
    csv_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break
        frame_idx += 1

        detections = detect_eggs(
            frame,
            roi_top=args.roi_top,
            roi_bottom=args.roi_bottom,
            min_area=args.min_area,
            min_circularity=args.min_circularity,
        )

        # Predict all live tracks before assignment.
        for track in tracks:
            prediction = track.kf.predict()
            track.age += 1
            px, py = float(prediction[0, 0]), float(prediction[1, 0])
            _, _, bw, bh = track.bbox
            track.bbox = (
                int(px - bw / 2), int(py - bh / 2), bw, bh
            )

        pairs, unmatched_tracks, unmatched_dets = associate(
            tracks, detections, args.max_distance
        )

        # Correct matched tracks with observed centers.
        for ti, di in pairs:
            track = tracks[ti]
            det = detections[di]
            measurement = np.array([[det.cx], [det.cy]], dtype=np.float32)
            track.kf.correct(measurement)
            track.bbox = det.bbox
            track.missed = 0
            track.hits += 1
            track.trail.append((int(det.cx), int(det.cy)))
            track.trail = track.trail[-max(1, args.trail_length):]
            row = {
                "frame": frame_idx,
                "time_s": round((frame_idx - 1) / fps, 3),
                "track_id": track.track_id,
                "x": det.x, "y": det.y, "w": det.w, "h": det.h,
                "center_x": round(det.cx, 2), "center_y": round(det.cy, 2),
                "detection_confidence": round(det.score, 4),
                "track_state": "confirmed",
            }
            if not track.confirmed:
                track.pending_rows.append(row)
                if track.hits >= args.min_hits and next_id <= 2:
                    track.track_id = next_id
                    next_id += 1
                    track.confirmed = True
                    for pending in track.pending_rows:
                        pending["track_id"] = track.track_id
                        csv_rows.append(pending)
                    track.pending_rows.clear()
            else:
                csv_rows.append(row)

        for ti in unmatched_tracks:
            tracks[ti].missed += 1

        for di in unmatched_dets:
            # Only two eggs are present. Once two tracks have been confirmed,
            # unmatched highlights cannot create more egg IDs.
            tentative_count = sum(not t.confirmed for t in tracks)
            if next_id > 2 or tentative_count >= 2:
                continue
            det = detections[di]
            det_axis_ratio = max(det.w, det.h) / max(1, min(det.w, det.h))
            # New IDs require a full, high-confidence egg contour. Smaller or
            # irregular partial contours can only maintain an existing track.
            if (det.score < 0.85 or min(det.w, det.h) < 38 or
                    det_axis_ratio > 1.65 or det.w * det.h < 1800):
                continue
            kf = create_kalman(det.cx, det.cy)
            new_track = Track(
                track_id=0, kf=kf, bbox=det.bbox,
                trail=[(int(det.cx), int(det.cy))],
            )
            new_track.pending_rows.append({
                "frame": frame_idx,
                "time_s": round((frame_idx - 1) / fps, 3),
                "track_id": 0,
                "x": det.x, "y": det.y, "w": det.w, "h": det.h,
                "center_x": round(det.cx, 2), "center_y": round(det.cy, 2),
                "detection_confidence": round(det.score, 4),
                "track_state": "confirmed",
            })
            tracks.append(new_track)

        # A tentative candidate must match in consecutive frames; confirmed
        # tracks coast through occlusions and retain their IDs.
        tracks = [t for t in tracks if
                  (t.confirmed and t.missed <= args.max_missed) or
                  (not t.confirmed and t.missed == 0)]

        # Draw current detections/tracks and short trails.
        for track in tracks:
            # Hide long-term predictions when an egg is fully occluded or moved;
            # keep its internal track alive for a later evidence-based match.
            if not track.confirmed or track.missed > min(args.max_missed, 12):
                continue
            x, y, bw, bh = track.bbox
            color = (0, 220, 0) if track.missed == 0 else (0, 165, 255)
            x, y = max(0, x), max(0, y)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame, f"Egg ID {track.track_id}" + (" (pred)" if track.missed else ""),
                (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA
            )
            if args.trail_length > 0 and len(track.trail) > 1:
                points = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], False, (0, 180, 255), 2)

        cv2.putText(
            frame, f"Frame {frame_idx} | Active tracks: {sum(t.missed == 0 and t.confirmed for t in tracks)}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 255), 2, cv2.LINE_AA
        )
        writer.write(frame)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames or '?'} frames")

    cap.release()
    writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame", "time_s", "track_id", "x", "y", "w", "h",
            "center_x", "center_y", "detection_confidence", "track_state"
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print(f"Done. Frames processed: {frame_idx}")
    print(f"Unique IDs created: {next_id - 1}")
    print(f"Detection rows written: {len(csv_rows)}")
    print(f"Video output: {output_path}")
    print(f"CSV output: {csv_path}")
    print("Inspect the video for missed detections and ID switches before submission.")


if __name__ == "__main__":
    main()
