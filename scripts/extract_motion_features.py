import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_tracks(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["track_id"] == "":
                continue
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time_seconds": float(row["time_seconds"]),
                    "track_id": int(row["track_id"]),
                    "confidence": float(row["confidence"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                }
            )
    return rows


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def track_motion_rows(track_rows):
    by_track = defaultdict(list)
    for row in track_rows:
        by_track[row["track_id"]].append(row)

    motion_rows = []
    for track_id, rows in sorted(by_track.items()):
        rows.sort(key=lambda item: (item["time_seconds"], item["frame"]))
        total_distance = 0.0
        previous = None

        for row in rows:
            dx = 0.0
            dy = 0.0
            dt = 0.0
            distance = 0.0
            speed = 0.0
            direction_degrees = ""

            if previous is not None:
                dt = row["time_seconds"] - previous["time_seconds"]
                dx = row["center_x"] - previous["center_x"]
                dy = row["center_y"] - previous["center_y"]
                distance = math.hypot(dx, dy)
                total_distance += distance
                if dt > 0:
                    speed = distance / dt
                    direction_degrees = math.degrees(math.atan2(dy, dx))

            area = row["width"] * row["height"]
            motion_rows.append(
                {
                    "frame": row["frame"],
                    "time_seconds": f"{row['time_seconds']:.6f}",
                    "track_id": track_id,
                    "confidence": f"{row['confidence']:.6f}",
                    "center_x": f"{row['center_x']:.3f}",
                    "center_y": f"{row['center_y']:.3f}",
                    "width": f"{row['width']:.3f}",
                    "height": f"{row['height']:.3f}",
                    "area": f"{area:.3f}",
                    "dt": f"{dt:.6f}",
                    "dx": f"{dx:.3f}",
                    "dy": f"{dy:.3f}",
                    "distance_px": f"{distance:.3f}",
                    "speed_px_per_second": f"{speed:.3f}",
                    "direction_degrees": (
                        f"{direction_degrees:.3f}"
                        if direction_degrees != ""
                        else ""
                    ),
                    "total_distance_px": f"{total_distance:.3f}",
                    "track_length_frames": len(rows),
                }
            )
            previous = row

    return motion_rows


def frame_summary_rows(motion_rows):
    by_frame = defaultdict(list)
    for row in motion_rows:
        by_frame[int(row["frame"])].append(row)

    summaries = []
    for frame, rows in sorted(by_frame.items()):
        speeds = [float(row["speed_px_per_second"]) for row in rows]
        moving = [speed for speed in speeds if speed > 0]
        areas = [float(row["area"]) for row in rows]
        xs = [float(row["center_x"]) for row in rows]
        ys = [float(row["center_y"]) for row in rows]

        summaries.append(
            {
                "frame": frame,
                "time_seconds": rows[0]["time_seconds"],
                "tracked_players": len(rows),
                "moving_players": len(moving),
                "mean_speed_px_per_second": f"{mean(speeds):.3f}",
                "max_speed_px_per_second": f"{max(speeds) if speeds else 0.0:.3f}",
                "mean_box_area": f"{mean(areas):.3f}",
                "min_center_x": f"{min(xs) if xs else 0.0:.3f}",
                "max_center_x": f"{max(xs) if xs else 0.0:.3f}",
                "min_center_y": f"{min(ys) if ys else 0.0:.3f}",
                "max_center_y": f"{max(ys) if ys else 0.0:.3f}",
            }
        )
    return summaries


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(track_rows, motion_rows, frame_rows):
    track_ids = sorted({row["track_id"] for row in track_rows})
    speeds = [float(row["speed_px_per_second"]) for row in motion_rows]
    active_speeds = [speed for speed in speeds if speed > 0]
    track_lengths = defaultdict(int)
    for row in track_rows:
        track_lengths[row["track_id"]] += 1

    return {
        "detections": len(track_rows),
        "tracks": len(track_ids),
        "frames_with_tracks": len(frame_rows),
        "mean_players_per_frame": mean(
            [row["tracked_players"] for row in frame_rows]
        ),
        "mean_active_speed_px_per_second": mean(active_speeds),
        "max_speed_px_per_second": max(speeds) if speeds else 0.0,
        "longest_track_frames": max(track_lengths.values()) if track_lengths else 0,
        "track_ids": track_ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO player tracks into motion feature CSVs."
    )
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    track_rows = read_tracks(args.tracks)
    motion_rows = track_motion_rows(track_rows)
    frame_rows = frame_summary_rows(motion_rows)

    track_motion_csv = args.output_dir / "track_motion_features.csv"
    frame_motion_csv = args.output_dir / "frame_motion_summary.csv"
    summary_json = args.output_dir / "motion_summary.json"

    write_csv(
        track_motion_csv,
        motion_rows,
        [
            "frame",
            "time_seconds",
            "track_id",
            "confidence",
            "center_x",
            "center_y",
            "width",
            "height",
            "area",
            "dt",
            "dx",
            "dy",
            "distance_px",
            "speed_px_per_second",
            "direction_degrees",
            "total_distance_px",
            "track_length_frames",
        ],
    )
    write_csv(
        frame_motion_csv,
        frame_rows,
        [
            "frame",
            "time_seconds",
            "tracked_players",
            "moving_players",
            "mean_speed_px_per_second",
            "max_speed_px_per_second",
            "mean_box_area",
            "min_center_x",
            "max_center_x",
            "min_center_y",
            "max_center_y",
        ],
    )
    summary = summarize(track_rows, motion_rows, frame_rows)
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Saved track motion: {track_motion_csv}")
    print(f"Saved frame summary: {frame_motion_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Tracks: {summary['tracks']}")
    print(f"Frames with tracks: {summary['frames_with_tracks']}")
    print(f"Mean players/frame: {summary['mean_players_per_frame']:.2f}")
    print(f"Mean active speed: {summary['mean_active_speed_px_per_second']:.2f} px/s")


if __name__ == "__main__":
    main()
