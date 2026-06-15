import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


PERSON_CLASS_ID = 0


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def open_preview_writer(path, metadata, fps_stride):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = max(metadata["fps"] / max(fps_stride, 1), 1.0)
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        output_fps,
        (metadata["width"], metadata["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open preview writer: {path}")
    return writer


def draw_tracks(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["xyxy"]]
        track_id = detection["track_id"]
        confidence = detection["confidence"]

        color = (30, 180, 255) if track_id is not None else (180, 180, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id {track_id}" if track_id is not None else "person"
        label = f"{label} {confidence:.2f}"
        cv2.rectangle(frame, (x1, max(y1 - 24, 0)), (x1 + 120, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(y1 - 7, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def result_detections(result, frame_index, fps):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    track_ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

    for box, confidence, class_id, track_id in zip(xyxy, confidences, classes, track_ids):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        detections.append(
            {
                "frame": frame_index,
                "time_seconds": frame_index / fps,
                "track_id": int(track_id) if track_id is not None else "",
                "class_id": int(class_id),
                "class_name": result.names.get(int(class_id), str(int(class_id))),
                "confidence": float(confidence),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "center_x": float(x1 + width / 2),
                "center_y": float(y1 + height / 2),
                "width": float(width),
                "height": float(height),
                "xyxy": box,
            }
        )
    return detections


def write_tracks_csv(path, rows):
    columns = [
        "frame",
        "time_seconds",
        "track_id",
        "class_id",
        "class_name",
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
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track players in lacrosse footage with YOLO."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = video_metadata(args.video)

    model = YOLO(args.model)
    classes = None if args.all_classes else [PERSON_CLASS_ID]

    tracks_csv = args.output_dir / "player_tracks.csv"
    metadata_json = args.output_dir / "metadata.json"
    preview_path = args.output_dir / "tracking_preview.mp4"
    preview_writer = (
        open_preview_writer(preview_path, metadata, args.frame_stride)
        if args.save_preview
        else None
    )

    rows = []
    processed_frames = 0
    stream = model.track(
        source=str(args.video),
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.confidence,
        imgsz=args.image_size,
        classes=classes,
        verbose=False,
    )

    for frame_index, result in enumerate(stream):
        if frame_index % args.frame_stride != 0:
            continue
        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

        detections = result_detections(result, frame_index, metadata["fps"])
        rows.extend(detections)

        if preview_writer is not None:
            frame = draw_tracks(result.orig_img.copy(), detections)
            preview_writer.write(frame)

        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"Processed {processed_frames} sampled frames", flush=True)

    if preview_writer is not None:
        preview_writer.release()

    write_tracks_csv(tracks_csv, rows)
    metadata_json.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "model": args.model,
                "tracker": args.tracker,
                "confidence": args.confidence,
                "image_size": args.image_size,
                "frame_stride": args.frame_stride,
                "all_classes": args.all_classes,
                "video_metadata": metadata,
                "sampled_frames": processed_frames,
                "detections": len(rows),
                "tracks_csv": str(tracks_csv),
                "preview_video": str(preview_path) if args.save_preview else None,
            },
            indent=2,
        )
    )

    print(f"Saved tracks: {tracks_csv}")
    print(f"Saved metadata: {metadata_json}")
    if args.save_preview:
        print(f"Saved preview: {preview_path}")
    print(f"Sampled frames: {processed_frames}")
    print(f"Detections: {len(rows)}")


if __name__ == "__main__":
    main()
