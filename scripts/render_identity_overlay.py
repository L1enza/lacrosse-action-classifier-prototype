import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    metadata = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0,
    }
    cap.release()
    return metadata


def group_tracks_by_frame(rows):
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    return by_frame


def load_identity_reads(path, confidence_threshold):
    reads = {}
    if path is None:
        return reads
    for row in read_csv(path):
        number = row.get("number") or "unknown"
        confidence = float(row.get("confidence") or 0.0)
        if number == "unknown" or confidence < confidence_threshold:
            continue
        reads[int(row["track_id"])] = row
    return reads


def label_for_track(row, reads, show_unknown_ids):
    track_id = int(row["track_id"])
    role = row.get("role") or "player"
    if role == "referee":
        return f"REF id {track_id}"
    if role == "goalkeeper":
        return f"GK id {track_id}"
    if role == "non_player":
        return f"NON id {track_id}"

    read = reads.get(track_id)
    if not read:
        return f"id {track_id}" if show_unknown_ids else ""
    number = read.get("number") or "?"
    player = read.get("matched_player") or ""
    confidence = float(read.get("confidence") or 0.0)
    if player:
        return f"#{number} {short_name(player)} {confidence:.2f}"
    return f"#{number} id {track_id} {confidence:.2f}"


def short_name(name):
    parts = name.split()
    if len(parts) <= 1:
        return name
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def color_for_track(track_id, has_identity, role):
    if role == "referee":
        return (235, 235, 235)
    if role == "goalkeeper":
        return (255, 70, 70)
    if role == "non_player":
        return (100, 100, 100)
    if has_identity:
        return (40, 220, 80)
    palette = [
        (50, 180, 255),
        (255, 180, 50),
        (220, 90, 255),
        (80, 220, 220),
        (255, 100, 100),
    ]
    return palette[track_id % len(palette)]


def draw_detection(frame, row, reads, show_unknown_ids):
    role = row.get("role") or "player"
    if role == "non_player":
        return

    track_id = int(row["track_id"])
    has_identity = track_id in reads
    color = color_for_track(track_id, has_identity, role)
    x1 = int(float(row["x1"]))
    y1 = int(float(row["y1"]))
    x2 = int(float(row["x2"]))
    y2 = int(float(row["y2"]))
    label = label_for_track(row, reads, show_unknown_ids)
    if not label:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_w, text_h = text_size
    label_y1 = max(0, y1 - text_h - 10)
    label_y2 = max(text_h + 8, y1)
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (min(frame.shape[1] - 1, x1 + text_w + 8), label_y2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, label_y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render bounding boxes plus jersey/name labels onto a video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--identity-reads", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--hide-unknown-ids",
        action="store_true",
        help="Only draw refs/goalies and tracks with accepted jersey reads.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = video_metadata(args.video)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    tracks = group_tracks_by_frame(read_csv(args.tracks))
    reads = load_identity_reads(args.identity_reads, args.confidence_threshold)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        metadata["fps"],
        (metadata["width"], metadata["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {args.output}")

    max_frames = metadata["frame_count"]
    if args.max_seconds is not None:
        max_frames = min(max_frames, int(args.max_seconds * metadata["fps"]))

    frame_index = 0
    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        for row in tracks.get(frame_index, []):
            draw_detection(frame, row, reads, show_unknown_ids=not args.hide_unknown_ids)
        writer.write(frame)
        frame_index += 1
        if frame_index % 150 == 0:
            print(f"Rendered {frame_index} frames", flush=True)

    cap.release()
    writer.release()
    print(f"Saved overlay video: {args.output}")
    print(f"Frames rendered: {frame_index}")
    print(f"Identity labels used: {len(reads)}")


if __name__ == "__main__":
    main()
