import argparse
import csv
import json
from pathlib import Path


def rows_from_tracks(tracks, role_filter):
    rows = []
    for track in tracks:
        track_id = str(track.get("track_id"))
        evidence = track.get("evidence") or {}
        role = evidence.get("role") or ""
        original_role = evidence.get("original_role") or ""
        if role_filter and role not in role_filter:
            continue

        for observation in track.get("observations") or []:
            x1, y1, x2, y2 = [float(value) for value in observation["bbox"]]
            width = x2 - x1
            height = y2 - y1
            rows.append(
                {
                    "frame": int(observation["frame_index"]),
                    "time_seconds": float(observation["timestamp"]),
                    "track_id": int(track_id),
                    "class_id": 0,
                    "class_name": "person",
                    "role": role,
                    "original_role": original_role,
                    "role_source": evidence.get("role_source") or "",
                    "confidence": float(observation.get("detection_confidence") or 0.0),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": x1 + width / 2.0,
                    "center_y": y1 + height / 2.0,
                    "width": width,
                    "height": height,
                }
            )
    rows.sort(key=lambda row: (row["frame"], row["track_id"]))
    return rows


def write_csv(path, rows):
    columns = [
        "frame",
        "time_seconds",
        "track_id",
        "class_id",
        "class_name",
        "role",
        "original_role",
        "role_source",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    roles = {}
    track_ids = set()
    for row in rows:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
        track_ids.add(row["track_id"])
    return {
        "detections": len(rows),
        "tracks": len(track_ids),
        "detections_by_role": roles,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export corrected sports_reID debug tracks to motion-friendly CSVs."
    )
    parser.add_argument("--tracks-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--roles",
        default="player",
        help="Comma-separated corrected roles to export, or 'all'. Default: player",
    )
    parser.add_argument("--output-name", default="player_tracks.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(args.tracks_json.read_text())
    role_filter = None if args.roles == "all" else set(args.roles.split(","))
    rows = rows_from_tracks(tracks, role_filter)

    csv_path = args.output_dir / args.output_name
    summary_path = args.output_dir / "export_summary.json"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    summary.update(
        {
            "source_tracks_json": str(args.tracks_json),
            "roles": "all" if role_filter is None else sorted(role_filter),
            "tracks_csv": str(csv_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved tracks CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Tracks: {summary['tracks']}")
    print(f"Detections: {summary['detections']}")


if __name__ == "__main__":
    main()
