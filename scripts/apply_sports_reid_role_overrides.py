import argparse
import csv
import json
from pathlib import Path


def parse_ids(value):
    if not value:
        return set()
    ids = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            ids.add(item)
    return ids


def load_review_rows(path):
    if not path:
        return {}
    with path.open(newline="") as handle:
        return {str(row["track_id"]): row for row in csv.DictReader(handle)}


def choose_role(track_id, original_role, ref_ids, goalie_ids, non_player_ids):
    if track_id in non_player_ids:
        return "non_player"
    if track_id in goalie_ids:
        return "goalkeeper"
    if track_id in ref_ids:
        return "referee"
    return "player"


def main():
    parser = argparse.ArgumentParser(
        description="Apply manually reviewed role labels to sports_reID track output."
    )
    parser.add_argument("--debug-tracks", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refs", default="")
    parser.add_argument("--goalies", default="")
    parser.add_argument("--non-players", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(args.debug_tracks.read_text())
    review_rows = load_review_rows(args.review_csv)
    ref_ids = parse_ids(args.refs)
    goalie_ids = parse_ids(args.goalies)
    non_player_ids = parse_ids(args.non_players)

    rows = []
    corrected_tracks = []
    counts = {"player": 0, "referee": 0, "goalkeeper": 0, "non_player": 0}

    for track in tracks:
        track_id = str(track.get("track_id"))
        evidence = track.get("evidence") or {}
        original_role = evidence.get("role") or ""
        corrected_role = choose_role(
            track_id, original_role, ref_ids, goalie_ids, non_player_ids
        )
        counts[corrected_role] += 1

        corrected = dict(track)
        corrected_evidence = dict(evidence)
        corrected_evidence["original_role"] = original_role
        corrected_evidence["role"] = corrected_role
        corrected_evidence["role_source"] = (
            "manual_override"
            if track_id in ref_ids or track_id in goalie_ids or track_id in non_player_ids
            else "manual_default_player"
        )
        corrected["evidence"] = corrected_evidence
        corrected["is_player"] = corrected_role == "player"
        corrected_tracks.append(corrected)

        review = review_rows.get(track_id, {})
        rows.append(
            {
                "track_id": track_id,
                "corrected_role": corrected_role,
                "original_role": original_role,
                "role_source": corrected_evidence["role_source"],
                "start_time": track.get("start_time"),
                "end_time": track.get("end_time"),
                "duration": track.get("duration"),
                "observations": track.get("num_observations"),
                "player_likelihood": track.get("player_likelihood"),
                "ref_score": review.get("ref_score", ""),
                "frame_index": review.get("frame_index", ""),
            }
        )

    rows.sort(key=lambda row: (row["corrected_role"], float(row["start_time"] or 0)))

    csv_path = args.output_dir / "roles_corrected.csv"
    json_path = args.output_dir / "debug_tracks_roles_corrected.json"
    summary_path = args.output_dir / "role_override_summary.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(corrected_tracks, indent=2))
    summary_path.write_text(
        json.dumps(
            {
                "source_debug_tracks": str(args.debug_tracks),
                "source_review_csv": str(args.review_csv) if args.review_csv else None,
                "counts": counts,
                "manual_refs": sorted(ref_ids, key=int),
                "manual_goalies": sorted(goalie_ids, key=int),
                "manual_non_players": sorted(non_player_ids, key=int),
                "default_for_unlisted_tracks": "player",
            },
            indent=2,
        )
    )

    print(f"Saved corrected role CSV: {csv_path}")
    print(f"Saved corrected debug tracks: {json_path}")
    print(f"Saved summary: {summary_path}")
    print("Counts:")
    for role, count in counts.items():
        print(f"  {role}: {count}")


if __name__ == "__main__":
    main()
