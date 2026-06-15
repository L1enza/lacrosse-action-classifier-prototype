import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path):
    return json.loads(path.read_text())


def crop_frame(video_path, frame_index, bbox):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return frame[y1:y2, x1:x2].copy()


def representative_observation(track):
    observations = track.get("observations") or []
    if not observations:
        return None
    return max(
        observations,
        key=lambda obs: (
            float(obs.get("detection_confidence") or 0.0),
            box_area(obs.get("bbox") or [0, 0, 0, 0]),
        ),
    )


def box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def lacrosse_ref_scores(crop):
    if crop is None or crop.size == 0:
        return empty_ref_scores()

    h, w = crop.shape[:2]
    center_x1 = int(w * 0.12)
    center_x2 = max(center_x1 + 1, int(w * 0.88))
    shirt = crop[
        int(h * 0.16) : max(int(h * 0.66), int(h * 0.16) + 1),
        center_x1:center_x2,
    ]
    head = crop[: max(1, int(h * 0.24)), center_x1:center_x2]
    lower = crop[
        int(h * 0.56) : max(int(h * 0.94), int(h * 0.56) + 1),
        center_x1:center_x2,
    ]
    if shirt.size == 0 or head.size == 0 or lower.size == 0:
        return empty_ref_scores()

    shirt_gray = cv2.cvtColor(shirt, cv2.COLOR_BGR2GRAY)
    shirt_hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
    head_hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    lower_gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)

    black = shirt_gray < 85
    white = (shirt_gray > 145) & (shirt_hsv[:, :, 1] < 125)
    black_ratio = float(np.mean(black))
    white_ratio = float(np.mean(white))
    hue = shirt_hsv[:, :, 0]
    sat = shirt_hsv[:, :, 1]
    val = shirt_hsv[:, :, 2]
    red_orange = (((hue <= 25) | (hue >= 165)) & (sat > 70) & (val > 70))
    red_orange_ratio = float(np.mean(red_orange))
    head_red_orange_ratio = red_orange_mask(head_hsv)

    col_profile = shirt_gray.mean(axis=0)
    row_profile = shirt_gray.mean(axis=1)
    column_std = float(np.std(col_profile))
    row_std = float(np.std(row_profile))
    vertical_orientation = column_std / max(column_std + row_std, 1e-6)
    dark_lower_ratio = float(np.mean(lower_gray < 95))

    black_columns = np.mean(black, axis=0)
    white_columns = np.mean(white, axis=0)
    column_state = np.zeros(len(black_columns), dtype=np.int8)
    column_state[black_columns > 0.22] = -1
    column_state[white_columns > 0.18] = 1
    active_columns = column_state[column_state != 0]
    transitions = int(np.sum(active_columns[1:] != active_columns[:-1])) if len(active_columns) > 1 else 0
    transition_score = min(1.0, transitions / 5.0)
    active_column_ratio = float(np.mean(column_state != 0)) if len(column_state) else 0.0

    stripe_balance = min(black_ratio / 0.20, white_ratio / 0.10, 1.0)
    vertical_texture = min(1.0, column_std / 34.0) * min(1.0, vertical_orientation / 0.46)
    red_orange_penalty = max(0.35, 1.0 - max(0.0, red_orange_ratio - 0.20) / 0.35)
    stripe_score = (
        0.40 * stripe_balance
        + 0.30 * transition_score
        + 0.20 * vertical_texture
        + 0.10 * min(1.0, active_column_ratio / 0.65)
    )
    stripe_score *= red_orange_penalty

    dark_lower_score = min(1.0, dark_lower_ratio / 0.58)
    player_like_penalty = 1.0
    if white_ratio > 0.32 and black_ratio < 0.08:
        player_like_penalty *= 0.45
    if black_ratio > 0.30 and white_ratio < 0.06:
        player_like_penalty *= 0.55
    if dark_lower_ratio < 0.18:
        player_like_penalty *= 0.45
    if head_red_orange_ratio > 0.10:
        player_like_penalty *= max(
            0.12, 1.0 - (head_red_orange_ratio - 0.10) / 0.22
        )

    ref_score = (0.78 * stripe_score + 0.22 * dark_lower_score) * player_like_penalty

    return {
        "stripe_score": stripe_score,
        "shirt_black_ratio": black_ratio,
        "shirt_white_ratio": white_ratio,
        "shirt_column_std": column_std,
        "shirt_row_std": row_std,
        "vertical_orientation": vertical_orientation,
        "stripe_transitions": transitions,
        "active_column_ratio": active_column_ratio,
        "red_orange_ratio": red_orange_ratio,
        "head_red_orange_ratio": head_red_orange_ratio,
        "dark_lower_ratio": dark_lower_ratio,
        "ref_score": ref_score,
    }


def red_orange_mask(hsv_image):
    hue = hsv_image[:, :, 0]
    sat = hsv_image[:, :, 1]
    val = hsv_image[:, :, 2]
    return float(np.mean(((hue <= 25) | (hue >= 165)) & (sat > 70) & (val > 70)))


def track_ref_likelihood(track):
    likelihood = float(track.get("player_likelihood") or 0.0)
    observations = float(track.get("num_observations") or 0.0)
    duration = float(track.get("duration") or 0.0)

    person_score = np.clip((likelihood - 0.18) / 0.42, 0.0, 1.0)
    if track.get("is_player") is True:
        person_score = max(person_score, 0.85)
    observation_score = min(1.0, observations / 15.0)
    duration_score = min(1.0, duration / 1.0)

    # Crowd/board artifacts can look striped in a tiny crop. A real official
    # should still behave like a person track across multiple frames.
    return float(
        (0.75 * person_score + 0.25)
        * (0.55 + 0.45 * observation_score)
        * (0.55 + 0.45 * duration_score)
    )


def empty_ref_scores():
    return {
        "stripe_score": 0.0,
        "shirt_black_ratio": 0.0,
        "shirt_white_ratio": 0.0,
        "shirt_column_std": 0.0,
        "shirt_row_std": 0.0,
        "vertical_orientation": 0.0,
        "stripe_transitions": 0,
        "active_column_ratio": 0.0,
        "red_orange_ratio": 0.0,
        "head_red_orange_ratio": 0.0,
        "dark_lower_ratio": 0.0,
        "ref_score": 0.0,
    }


def draw_tile(crop, row, tile_size):
    tile_w, tile_h = tile_size
    if crop is None or crop.size == 0:
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    else:
        tile = resize_letterbox(crop, tile_w, tile_h)
    label = (
        f"id {row['track_id']} {row['role']} "
        f"ref={float(row['ref_score']):.2f}"
    )
    cv2.rectangle(tile, (0, 0), (tile_w, 24), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label[:38],
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def resize_letterbox(image, target_w, target_h):
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


def write_contact_sheet(path, tiles, columns):
    if not tiles:
        return
    tile_h, tile_w = tiles[0].shape[:2]
    rows = []
    for index in range(0, len(tiles), columns):
        row_tiles = tiles[index : index + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), sheet)


def main():
    parser = argparse.ArgumentParser(
        description="Create a crop review sheet for sports_reID roles."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--debug-tracks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=80)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_json(args.debug_tracks)
    rows = []
    tiles = []

    for track in tracks:
        obs = representative_observation(track)
        if obs is None:
            continue
        crop = crop_frame(args.video, obs["frame_index"], obs["bbox"])
        scores = lacrosse_ref_scores(crop)
        visual_ref_score = scores["ref_score"]
        track_score = track_ref_likelihood(track)
        scores["visual_ref_score"] = visual_ref_score
        scores["track_ref_likelihood"] = track_score
        scores["ref_score"] = visual_ref_score * track_score
        row = {
            "track_id": track.get("track_id"),
            "start_time": track.get("start_time"),
            "end_time": track.get("end_time"),
            "duration": track.get("duration"),
            "observations": track.get("num_observations"),
            "is_player": track.get("is_player"),
            "player_likelihood": track.get("player_likelihood"),
            "role": (track.get("evidence") or {}).get("role") or "",
            "frame_index": obs["frame_index"],
            "detection_confidence": obs.get("detection_confidence"),
            **{key: f"{value:.6f}" for key, value in scores.items()},
        }
        rows.append(row)

    rows.sort(key=lambda row: float(row["ref_score"]), reverse=True)
    for row in rows[: args.max_tracks]:
        obs_track = next(t for t in tracks if str(t.get("track_id")) == str(row["track_id"]))
        obs = representative_observation(obs_track)
        crop = crop_frame(args.video, obs["frame_index"], obs["bbox"])
        tiles.append(draw_tile(crop, row, (220, 260)))

    csv_path = args.output_dir / "role_review.csv"
    sheet_path = args.output_dir / "role_review_sheet.jpg"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_contact_sheet(sheet_path, tiles, args.columns)

    print(f"Saved role CSV: {csv_path}")
    print(f"Saved contact sheet: {sheet_path}")
    print("Top referee-like tracks:")
    for row in rows[:10]:
        print(
            f"  id={row['track_id']} role={row['role'] or 'none'} "
            f"ref_score={row['ref_score']} stripe={row['stripe_score']}"
        )


if __name__ == "__main__":
    main()
