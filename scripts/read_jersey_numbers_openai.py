import argparse
import base64
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


JERSEY_SCHEMA = {
    "name": "jersey_number_read",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "number": {
                "type": "string",
                "description": "Full jersey number, or unknown.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence from 0.0 to 1.0.",
            },
            "visible_digits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Digits that are visibly readable.",
            },
            "possible_numbers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Roster-supported possible numbers if ambiguous.",
            },
            "reason": {
                "type": "string",
                "description": "Brief visual reason for the read.",
            },
        },
        "required": [
            "number",
            "confidence",
            "visible_digits",
            "possible_numbers",
            "reason",
        ],
    },
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    columns = [
        "track_id",
        "team_candidate",
        "number",
        "confidence",
        "visible_digits",
        "possible_numbers",
        "matched_player",
        "matched_team",
        "matched_position",
        "reason",
        "crop_count",
        "crop_paths",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_roster(path):
    metadata = json.loads(path.read_text())
    by_team = {}
    by_number = defaultdict(list)
    for team, players in (metadata.get("rosters") or {}).items():
        by_team[team] = {}
        for player in players:
            number = str(player.get("jersey_number"))
            by_team[team][number] = player
            by_number[number].append(player)
    return metadata, by_team, by_number


def group_manifest_rows(rows, max_images):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    for track_id in grouped:
        grouped[track_id].sort(
            key=lambda row: float(row.get("quality_score") or 0.0),
            reverse=True,
        )
        grouped[track_id] = grouped[track_id][:max_images]
    return grouped


def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def roster_candidates(team_candidate, by_team, by_number):
    if team_candidate in by_team:
        numbers = sorted(by_team[team_candidate], key=lambda value: int(value))
        return numbers
    numbers = sorted(by_number, key=lambda value: int(value))
    return numbers


def prompt_text(team_candidate, candidate_numbers):
    return (
        "You are reading lacrosse jersey numbers from broadcast video crops.\n\n"
        "I am giving you multiple crops of the same tracked player from nearby frames.\n"
        "Return JSON only.\n\n"
        "Rules:\n"
        "- Read only the jersey number visible on the player.\n"
        "- If uncertain, use \"unknown\".\n"
        "- Do not guess a full number unless all digits are visible.\n"
        "- If one digit is visible, return it under visible_digits.\n"
        "- Use roster candidates to resolve ambiguity only when visual evidence supports it.\n"
        "- If the crop is blurry, side-facing, blocked, or the number is not visible, say unknown.\n\n"
        f"Team color/roster candidate from tracking: {team_candidate or 'unknown'}\n"
        f"Roster candidate numbers: {candidate_numbers}\n"
    )


def build_content(rows, team_candidate, candidate_numbers):
    content = [{"type": "input_text", "text": prompt_text(team_candidate, candidate_numbers)}]
    for row in rows:
        image_data = encode_image(row["crop_path"])
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_data}",
            }
        )
    return content


def call_openai(client, model, rows, team_candidate, candidate_numbers):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": build_content(rows, team_candidate, candidate_numbers),
            }
        ],
        text={"format": {"type": "json_schema", **JERSEY_SCHEMA}},
    )
    return json.loads(response.output_text)


def match_player(number, team_candidate, by_team, by_number):
    if number == "unknown":
        return None
    if team_candidate in by_team and number in by_team[team_candidate]:
        return by_team[team_candidate][number]
    candidates = by_number.get(number) or []
    return candidates[0] if len(candidates) == 1 else None


def result_row(track_id, rows, result, by_team, by_number):
    team_candidate = rows[0].get("team_candidate") or ""
    number = str(result.get("number") or "unknown")
    player = match_player(number, team_candidate, by_team, by_number)
    return {
        "track_id": track_id,
        "team_candidate": team_candidate,
        "number": number,
        "confidence": f"{float(result.get('confidence') or 0.0):.3f}",
        "visible_digits": ",".join(result.get("visible_digits") or []),
        "possible_numbers": ",".join(result.get("possible_numbers") or []),
        "matched_player": player.get("player_name", "") if player else "",
        "matched_team": player.get("team_name", "") if player else "",
        "matched_position": player.get("position", "") if player else "",
        "reason": result.get("reason", ""),
        "crop_count": len(rows),
        "crop_paths": "|".join(row["crop_path"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Use OpenAI vision to read jersey numbers from per-track crop banks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-images-per-track", type=int, default=8)
    parser.add_argument("--limit-tracks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(args.manifest)
    _, by_team, by_number = load_roster(args.metadata)
    grouped = group_manifest_rows(manifest_rows, args.max_images_per_track)
    track_ids = sorted(grouped, key=lambda value: int(value))
    if args.limit_tracks is not None:
        track_ids = track_ids[: args.limit_tracks]

    preview_path = args.output_dir / "openai_jersey_request_preview.json"
    results_path = args.output_dir / "jersey_number_reads.json"
    csv_path = args.output_dir / "jersey_number_reads.csv"

    preview = []
    for track_id in track_ids:
        rows = grouped[track_id]
        team_candidate = rows[0].get("team_candidate") or ""
        candidates = roster_candidates(team_candidate, by_team, by_number)
        preview.append(
            {
                "track_id": track_id,
                "team_candidate": team_candidate,
                "candidate_numbers": candidates,
                "crop_paths": [row["crop_path"] for row in rows],
            }
        )

    preview_path.write_text(json.dumps(preview, indent=2))
    if args.dry_run:
        print(f"Saved dry-run request preview: {preview_path}")
        print(f"Tracks prepared: {len(preview)}")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run with --dry-run or export it.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI Python SDK first: pip install openai") from exc

    client = OpenAI()
    json_results = []
    csv_rows = []
    for index, track_id in enumerate(track_ids, start=1):
        rows = grouped[track_id]
        team_candidate = rows[0].get("team_candidate") or ""
        candidates = roster_candidates(team_candidate, by_team, by_number)
        print(f"[{index}/{len(track_ids)}] Reading track {track_id}", flush=True)
        result = call_openai(client, args.model, rows, team_candidate, candidates)
        json_results.append(
            {
                "track_id": track_id,
                "team_candidate": team_candidate,
                "candidate_numbers": candidates,
                "crop_paths": [row["crop_path"] for row in rows],
                "result": result,
            }
        )
        csv_rows.append(result_row(track_id, rows, result, by_team, by_number))

    results_path.write_text(json.dumps(json_results, indent=2))
    write_csv(csv_path, csv_rows)
    print(f"Saved JSON results: {results_path}")
    print(f"Saved CSV results: {csv_path}")


if __name__ == "__main__":
    main()
