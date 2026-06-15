import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def overlap_seconds(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def label_for_window(window, labels, min_overlap):
    start = float(window["start"])
    end = float(window["end"])
    duration = max(end - start, 1e-6)

    best = None
    for label in labels:
        overlap = overlap_seconds(
            start,
            end,
            float(label["start"]),
            float(label["end"]),
        )
        overlap_ratio = overlap / duration
        if overlap < min_overlap:
            continue
        candidate = {
            "label": label["label"],
            "label_overlap_seconds": overlap,
            "label_overlap_ratio": overlap_ratio,
            "label_source": label.get("source", ""),
            "label_notes": label.get("notes", ""),
        }
        if best is None or overlap > best["label_overlap_seconds"]:
            best = candidate

    if best is None:
        return {
            "label": "unlabeled",
            "label_overlap_seconds": 0.0,
            "label_overlap_ratio": 0.0,
            "label_source": "",
            "label_notes": "",
        }
    return best


def labels_for_video(rows, video):
    return [row for row in rows if row["video"] == video]


def main():
    parser = argparse.ArgumentParser(
        description="Join timestamp event labels onto motion-window feature rows."
    )
    parser.add_argument("--motion-windows", type=Path, required=True)
    parser.add_argument("--event-labels", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-overlap", type=float, default=0.25)
    args = parser.parse_args()

    windows = read_csv(args.motion_windows)
    labels = labels_for_video(read_csv(args.event_labels), args.video)
    if not labels:
        raise RuntimeError(f"No labels found for video: {args.video}")

    rows = []
    counts = {}
    for window in windows:
        label = label_for_window(window, labels, args.min_overlap)
        merged = dict(window)
        merged.update(
            {
                "label": label["label"],
                "label_overlap_seconds": f"{label['label_overlap_seconds']:.3f}",
                "label_overlap_ratio": f"{label['label_overlap_ratio']:.3f}",
                "label_source": label["label_source"],
                "label_notes": label["label_notes"],
            }
        )
        counts[merged["label"]] = counts.get(merged["label"], 0) + 1
        rows.append(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)

    print(f"Saved labeled windows: {args.output}")
    print(f"Windows: {len(rows)}")
    print("Label counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
