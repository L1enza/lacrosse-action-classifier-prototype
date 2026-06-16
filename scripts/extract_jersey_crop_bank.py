import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path):
    return json.loads(path.read_text())


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def crop_box(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return frame[y1:y2, x1:x2].copy()


def jersey_box_from_player_bbox(bbox, mode):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = x2 - x1
    height = y2 - y1
    if mode == "upper":
        return [
            x1 + 0.08 * width,
            y1 + 0.16 * height,
            x2 - 0.08 * width,
            y1 + 0.66 * height,
        ]
    if mode == "full":
        return [x1, y1, x2, y2]
    raise ValueError(f"Unknown crop mode: {mode}")


def crop_quality(track, observation, min_height):
    bbox = observation.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if height < min_height or width < 18:
        return 0.0

    detection = float(observation.get("detection_confidence") or 0.0)
    jersey_quality = float(observation.get("jersey_quality") or 0.0)
    crop_quality_value = float(observation.get("crop_quality") or 0.0)
    occlusion = float(observation.get("occlusion_score") or 0.0)
    likelihood = float(track.get("player_likelihood") or 0.0)
    aspect = width / height
    aspect_score = 1.0 - min(1.0, abs(aspect - 0.55) / 0.55)
    height_score = min(1.0, height / 190.0)

    return (
        0.30 * detection
        + 0.25 * jersey_quality
        + 0.20 * crop_quality_value
        + 0.10 * likelihood
        + 0.10 * height_score
        + 0.05 * aspect_score
    ) * max(0.0, 1.0 - occlusion)


def resize_upscale(image, min_height):
    h, w = image.shape[:2]
    if h >= min_height:
        return image
    scale = min_height / max(h, 1)
    return cv2.resize(
        image,
        (max(1, int(w * scale)), min_height),
        interpolation=cv2.INTER_CUBIC,
    )


def save_contact_sheet(path, rows, columns=6, tile_height=180, tile_width=160):
    tiles = []
    for row in rows:
        image = cv2.imread(row["crop_path"])
        if image is None:
            continue
        tile = letterbox(image, tile_width, tile_height)
        label = f"id {row['track_id']} q={float(row['quality_score']):.2f}"
        cv2.rectangle(tile, (0, 0), (tile_width, 22), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label[:26],
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
    sheet_rows = [
        np.concatenate(tiles[index : index + columns], axis=1)
        for index in range(0, len(tiles), columns)
    ]
    cv2.imwrite(str(path), np.concatenate(sheet_rows, axis=0))


def letterbox(image, target_w, target_h):
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def write_csv(path, rows):
    columns = [
        "track_id",
        "crop_index",
        "crop_path",
        "frame_index",
        "timestamp",
        "quality_score",
        "team_candidate",
        "detection_confidence",
        "jersey_quality",
        "crop_quality",
        "player_likelihood",
        "source_video",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality jersey crops from corrected sports_ReID tracks."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", default="player")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--crops-per-track", type=int, default=8)
    parser.add_argument("--min-player-height", type=float, default=70.0)
    parser.add_argument("--upscale-height", type=int, default=320)
    parser.add_argument("--crop-mode", choices=["upper", "full"], default="upper")
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument(
        "--exclude-track-ids",
        default="",
        help="Comma-separated track IDs to skip after visual review.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_json(args.tracks_json)
    excluded_track_ids = {
        value.strip()
        for value in args.exclude_track_ids.split(",")
        if value.strip()
    }
    selected_tracks = []
    for track in tracks:
        if str(track.get("track_id")) in excluded_track_ids:
            continue
        role = (track.get("evidence") or {}).get("role") or ""
        if role != args.role:
            continue
        scored = []
        for observation in track.get("observations") or []:
            quality = crop_quality(track, observation, args.min_player_height)
            if quality > 0:
                scored.append((quality, observation))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            selected_tracks.append((track, scored[: args.crops_per_track]))

    selected_tracks.sort(
        key=lambda item: max(score for score, _ in item[1]),
        reverse=True,
    )
    if args.max_tracks is not None:
        selected_tracks = selected_tracks[: args.max_tracks]

    cap = open_video(args.video)
    rows = []
    for track, scored_observations in selected_tracks:
        track_id = str(track.get("track_id"))
        evidence = track.get("evidence") or {}
        team_candidate = evidence.get("team_argmax") or ""
        for crop_index, (quality, observation) in enumerate(scored_observations, start=1):
            frame_index = int(observation["frame_index"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            jersey_bbox = jersey_box_from_player_bbox(
                observation["bbox"], args.crop_mode
            )
            crop = crop_box(frame, jersey_bbox)
            crop = resize_upscale(crop, args.upscale_height)
            crop_path = crop_dir / f"track_{track_id}_crop_{crop_index:02d}.jpg"
            cv2.imwrite(
                str(crop_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            rows.append(
                {
                    "track_id": track_id,
                    "crop_index": crop_index,
                    "crop_path": str(crop_path),
                    "frame_index": frame_index,
                    "timestamp": f"{float(observation['timestamp']):.6f}",
                    "quality_score": f"{quality:.6f}",
                    "team_candidate": team_candidate,
                    "detection_confidence": f"{float(observation.get('detection_confidence') or 0.0):.6f}",
                    "jersey_quality": f"{float(observation.get('jersey_quality') or 0.0):.6f}",
                    "crop_quality": f"{float(observation.get('crop_quality') or 0.0):.6f}",
                    "player_likelihood": f"{float(track.get('player_likelihood') or 0.0):.6f}",
                    "source_video": str(args.video),
                }
            )
    cap.release()

    manifest_path = args.output_dir / "jersey_crop_manifest.csv"
    summary_path = args.output_dir / "jersey_crop_summary.json"
    sheet_path = args.output_dir / "jersey_crop_sheet.jpg"
    write_csv(manifest_path, rows)
    save_contact_sheet(sheet_path, rows[:120])
    summary_path.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "tracks_json": str(args.tracks_json),
                "tracks_with_crops": len({row["track_id"] for row in rows}),
                "crops": len(rows),
                "crops_per_track": args.crops_per_track,
                "crop_mode": args.crop_mode,
                "manifest": str(manifest_path),
                "contact_sheet": str(sheet_path),
            },
            indent=2,
        )
    )

    print(f"Saved crop manifest: {manifest_path}")
    print(f"Saved crop sheet: {sheet_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Tracks with crops: {len({row['track_id'] for row in rows})}")
    print(f"Crops: {len(rows)}")


if __name__ == "__main__":
    main()
