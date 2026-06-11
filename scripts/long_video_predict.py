import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from predict_action import TinyVideoClassifier, get_device


def seconds_to_timestamp(seconds):
    total_seconds = max(float(seconds), 0.0)
    minutes = int(total_seconds // 60)
    secs = total_seconds - (minutes * 60)
    return f"{minutes:02d}:{secs:05.2f}"


def read_video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return cap, fps, frame_count, duration


def make_windows(duration, window_seconds, stride_seconds):
    if duration <= 0:
        raise RuntimeError("Video duration could not be detected.")
    if window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0")
    if stride_seconds <= 0:
        raise ValueError("--stride-seconds must be greater than 0")

    if duration <= window_seconds:
        return [(0.0, duration)]

    last_start = max(duration - window_seconds, 0.0)
    starts = list(np.arange(0.0, last_start + 1e-6, stride_seconds))
    if not starts or abs(starts[-1] - last_start) > 1e-3:
        starts.append(last_start)

    windows = []
    for start in starts:
        end = min(start + window_seconds, duration)
        windows.append((float(start), float(end)))
    return windows


def load_video_window_tensor(cap, start, end, fps, frame_count, target_frames, size):
    if frame_count <= 0:
        raise RuntimeError("Video frame count could not be detected.")

    end = max(end, start + (1.0 / fps))
    sample_times = np.linspace(start, end, target_frames, endpoint=False)
    frame_indices = np.clip(np.rint(sample_times * fps), 0, frame_count - 1).astype(int)
    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].copy())
                continue
            frame = np.zeros((size, size, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).reshape(target_frames * 3, size, size)
    mean = torch.tensor([0.45] * (target_frames * 3)).view(target_frames * 3, 1, 1)
    std = torch.tensor([0.225] * (target_frames * 3)).view(target_frames * 3, 1, 1)
    return (tensor - mean) / std


def moving_average(probabilities, smooth_windows):
    if smooth_windows <= 1:
        return probabilities

    radius = smooth_windows // 2
    smoothed = []
    for index in range(len(probabilities)):
        start = max(0, index - radius)
        end = min(len(probabilities), index + radius + 1)
        smoothed.append(probabilities[start:end].mean(axis=0))
    return np.stack(smoothed)


def merge_events(window_rows, other_label, threshold, merge_gap_seconds, include_other):
    events = []

    for row in window_rows:
        label = row["smoothed_label"]
        confidence = row["smoothed_confidence"]
        if not include_other and label == other_label:
            continue
        if confidence < threshold:
            continue

        if (
            events
            and events[-1]["label"] == label
            and row["start"] <= events[-1]["end"] + merge_gap_seconds
        ):
            event = events[-1]
            event["end"] = max(event["end"], row["end"])
            event["window_count"] += 1
            event["confidences"].append(confidence)
            event["confidence"] = float(np.mean(event["confidences"]))
            event["max_confidence"] = max(event["max_confidence"], confidence)
            event["timestamp"] = (
                f"{seconds_to_timestamp(event['start'])} - "
                f"{seconds_to_timestamp(event['end'])}"
            )
            continue

        events.append(
            {
                "start": row["start"],
                "end": row["end"],
                "timestamp": (
                    f"{seconds_to_timestamp(row['start'])} - "
                    f"{seconds_to_timestamp(row['end'])}"
                ),
                "label": label,
                "confidence": confidence,
                "max_confidence": confidence,
                "window_count": 1,
                "confidences": [confidence],
            }
        )

    for event in events:
        event.pop("confidences", None)
    return events


def write_windows_csv(path, labels, rows):
    fieldnames = [
        "start",
        "end",
        "timestamp",
        "raw_label",
        "raw_confidence",
        "smoothed_label",
        "smoothed_confidence",
    ] + [f"prob_{label}" for label in labels]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {key: row[key] for key in fieldnames if key in row}
            for label, score in zip(labels, row["smoothed_probs"]):
                output_row[f"prob_{label}"] = float(score)
            writer.writerow(output_row)


def write_events_csv(path, events):
    fieldnames = [
        "start",
        "end",
        "timestamp",
        "label",
        "confidence",
        "max_confidence",
        "window_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def main():
    parser = argparse.ArgumentParser(
        description="Run a clip classifier over a longer video with sliding windows."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/action_classifier_no_leakage_v1/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/long_video_predictions"),
    )
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--smooth-windows", type=int, default=3)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--other-label", default="other")
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    labels = checkpoint["labels"]
    metadata = checkpoint.get("metadata", {})
    frames = int(metadata.get("frames", 8))
    size = int(metadata.get("size", 96))

    device = get_device()
    model = TinyVideoClassifier(num_classes=len(labels), frames=frames).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cap, fps, frame_count, duration = read_video_metadata(args.video)
    windows = make_windows(duration, args.window_seconds, args.stride_seconds)
    if args.max_windows:
        windows = windows[: args.max_windows]

    raw_rows = []
    all_probs = []

    print(f"Video: {args.video}")
    print(f"Duration: {duration:.2f}s, fps={fps:.2f}, frames={frame_count}")
    print(f"Windows: {len(windows)}")
    print(f"Device: {device}")

    with torch.no_grad():
        for index, (start, end) in enumerate(windows, start=1):
            video = load_video_window_tensor(
                cap, start, end, fps, frame_count, frames, size
            ).unsqueeze(0)
            probs = torch.softmax(model(video.to(device)), dim=1)[0].cpu().numpy()
            raw_index = int(np.argmax(probs))
            all_probs.append(probs)
            raw_rows.append(
                {
                    "start": start,
                    "end": end,
                    "timestamp": (
                        f"{seconds_to_timestamp(start)} - "
                        f"{seconds_to_timestamp(end)}"
                    ),
                    "raw_label": labels[raw_index],
                    "raw_confidence": float(probs[raw_index]),
                }
            )
            if index % 25 == 0 or index == len(windows):
                print(f"Processed {index}/{len(windows)} windows", flush=True)

    cap.release()

    probabilities = np.stack(all_probs)
    smoothed_probs = moving_average(probabilities, args.smooth_windows)
    window_rows = []

    for row, probs in zip(raw_rows, smoothed_probs):
        smooth_index = int(np.argmax(probs))
        window_rows.append(
            {
                **row,
                "smoothed_label": labels[smooth_index],
                "smoothed_confidence": float(probs[smooth_index]),
                "smoothed_probs": probs,
            }
        )

    events = merge_events(
        window_rows,
        other_label=args.other_label,
        threshold=args.threshold,
        merge_gap_seconds=args.merge_gap_seconds,
        include_other=args.include_other,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows_csv = args.output_dir / "predictions_windows.csv"
    events_csv = args.output_dir / "predictions_events.csv"
    events_json = args.output_dir / "predictions_events.json"

    write_windows_csv(windows_csv, labels, window_rows)
    write_events_csv(events_csv, events)
    events_json.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "model": str(args.model),
                "labels": labels,
                "duration_seconds": duration,
                "fps": fps,
                "frame_count": frame_count,
                "window_seconds": args.window_seconds,
                "stride_seconds": args.stride_seconds,
                "threshold": args.threshold,
                "smooth_windows": args.smooth_windows,
                "merge_gap_seconds": args.merge_gap_seconds,
                "events": events,
            },
            indent=2,
        )
    )

    print(f"\nSaved window predictions: {windows_csv}")
    print(f"Saved event CSV: {events_csv}")
    print(f"Saved event JSON: {events_json}")
    print("\nDetected events:")
    if not events:
        print("  None above threshold")
    for event in events:
        print(
            f"  {event['timestamp']} {event['label']} "
            f"confidence={event['confidence']:.3f} "
            f"windows={event['window_count']}"
        )


if __name__ == "__main__":
    main()
