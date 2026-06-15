import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def window_start(time_seconds, window_seconds, stride_seconds):
    if stride_seconds <= 0:
        raise ValueError("--stride-seconds must be greater than 0")
    bucket = int(time_seconds // stride_seconds)
    start = bucket * stride_seconds
    if start + window_seconds <= time_seconds:
        start += stride_seconds
    return float(start)


def make_windows(frame_rows, track_rows, window_seconds, stride_seconds):
    max_time = 0.0
    for row in frame_rows:
        max_time = max(max_time, as_float(row, "time_seconds"))
    for row in track_rows:
        max_time = max(max_time, as_float(row, "time_seconds"))

    starts = []
    current = 0.0
    while current <= max_time:
        starts.append(round(current, 6))
        current += stride_seconds

    windows = []
    for start in starts:
        end = start + window_seconds
        frame_subset = [
            row
            for row in frame_rows
            if start <= as_float(row, "time_seconds") < end
        ]
        track_subset = [
            row
            for row in track_rows
            if start <= as_float(row, "time_seconds") < end
        ]
        windows.append(summarize_window(start, end, frame_subset, track_subset))
    return windows


def summarize_window(start, end, frame_rows, track_rows):
    frame_speeds = [as_float(row, "mean_speed_px_per_second") for row in frame_rows]
    frame_max_speeds = [as_float(row, "max_speed_px_per_second") for row in frame_rows]
    tracked_players = [as_float(row, "tracked_players") for row in frame_rows]
    moving_players = [as_float(row, "moving_players") for row in frame_rows]

    track_speeds = [as_float(row, "speed_px_per_second") for row in track_rows]
    track_distances = [as_float(row, "distance_px") for row in track_rows]
    active_track_speeds = [speed for speed in track_speeds if speed > 0]
    track_ids = {
        int(row["track_id"])
        for row in track_rows
        if row.get("track_id", "") != ""
    }

    return {
        "start": f"{start:.3f}",
        "end": f"{end:.3f}",
        "duration": f"{end - start:.3f}",
        "frames": len(frame_rows),
        "detections": len(track_rows),
        "unique_tracks": len(track_ids),
        "mean_players": f"{mean(tracked_players):.3f}",
        "max_players": f"{max(tracked_players) if tracked_players else 0.0:.3f}",
        "mean_moving_players": f"{mean(moving_players):.3f}",
        "mean_frame_speed": f"{mean(frame_speeds):.3f}",
        "max_frame_speed": f"{max(frame_max_speeds) if frame_max_speeds else 0.0:.3f}",
        "mean_track_speed": f"{mean(active_track_speeds):.3f}",
        "max_track_speed": f"{max(track_speeds) if track_speeds else 0.0:.3f}",
        "total_track_distance": f"{sum(track_distances):.3f}",
        "motion_burst_score": f"{motion_burst_score(frame_max_speeds, active_track_speeds):.3f}",
    }


def motion_burst_score(frame_max_speeds, active_track_speeds):
    if not frame_max_speeds and not active_track_speeds:
        return 0.0
    return max(frame_max_speeds or [0.0]) + mean(active_track_speeds)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def write_csv(path, rows):
    columns = [
        "start",
        "end",
        "duration",
        "frames",
        "detections",
        "unique_tracks",
        "mean_players",
        "max_players",
        "mean_moving_players",
        "mean_frame_speed",
        "max_frame_speed",
        "mean_track_speed",
        "max_track_speed",
        "total_track_distance",
        "motion_burst_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def top_windows(rows, key, limit):
    return sorted(rows, key=lambda row: float(row[key]), reverse=True)[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize motion features into fixed-time event windows."
    )
    parser.add_argument("--frame-summary", type=Path, required=True)
    parser.add_argument("--track-motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = read_csv(args.frame_summary)
    track_rows = read_csv(args.track_motion)
    windows = make_windows(
        frame_rows,
        track_rows,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    windows_csv = args.output_dir / "motion_windows.csv"
    summary_json = args.output_dir / "motion_windows_summary.json"
    write_csv(windows_csv, windows)

    summary = {
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "windows": len(windows),
        "frame_rows": len(frame_rows),
        "track_motion_rows": len(track_rows),
        "top_motion_bursts": top_windows(windows, "motion_burst_score", args.top_k),
        "top_total_distance": top_windows(windows, "total_track_distance", args.top_k),
        "top_unique_tracks": top_windows(windows, "unique_tracks", args.top_k),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Saved motion windows: {windows_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Windows: {len(windows)}")
    print("Top motion burst windows:")
    for row in summary["top_motion_bursts"][:5]:
        print(
            f"  {row['start']} - {row['end']} "
            f"burst={row['motion_burst_score']} "
            f"tracks={row['unique_tracks']} "
            f"distance={row['total_track_distance']}"
        )


if __name__ == "__main__":
    main()
