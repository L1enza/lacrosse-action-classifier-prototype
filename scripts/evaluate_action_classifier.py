import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from predict_action import TinyVideoClassifier, get_device, load_video_tensor


EXPECTED_LABELS = {
    "pass": "pass",
    "goal": "shot_goal",
    "save": "shot_save",
    "other": "other",
    "shot": None,
}


def predict_clip(model, clip_path, labels, frames, size, device):
    video = load_video_tensor(clip_path, frames, size).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(video), dim=1)[0].cpu()

    ranked = sorted(
        ((labels[index], float(score)) for index, score in enumerate(probs)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0], ranked[0][1], {label: float(probs[index]) for index, label in enumerate(labels)}


def collect_clips(data_dir):
    clips = []
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for clip_path in sorted(class_dir.glob("*.mp4")):
                clips.append((split_dir.name, class_dir.name, clip_path))
    return clips


def write_csv(records, labels, path):
    columns = [
        "split",
        "folder",
        "file",
        "expected",
        "predicted",
        "confidence",
        "correct",
        "path",
    ] + [f"score_{label}" for label in labels]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def make_chart(records, labels, path):
    known_records = [record for record in records if record["expected"]]
    folders = ["pass", "goal", "save", "other"]

    accuracy = []
    counts = []
    correct_counts = []
    for folder in folders:
        folder_records = [record for record in known_records if record["folder"] == folder]
        total = len(folder_records)
        correct = sum(1 for record in folder_records if record["correct"] == "1")
        counts.append(total)
        correct_counts.append(correct)
        accuracy.append((correct / total * 100) if total else 0)

    shot_records = [record for record in records if record["folder"] == "shot"]
    shot_distribution = Counter(record["predicted"] for record in shot_records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("Prototype 2 Lacrosse Action Classifier", fontsize=16, fontweight="bold")

    colors = ["#3b82f6", "#22c55e", "#f97316", "#64748b"]
    bars = axes[0].bar(folders, accuracy, color=colors)
    axes[0].set_title("Accuracy By Labeled Folder")
    axes[0].set_ylabel("Correct Predictions (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)

    for bar, correct, total in zip(bars, correct_counts, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{bar.get_height():.0f}%\n{correct}/{total}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if shot_records:
        shot_values = [shot_distribution.get(label, 0) for label in labels]
        axes[1].bar(labels, shot_values, color=["#64748b", "#3b82f6", "#22c55e", "#f97316"])
        axes[1].set_title("Current Shot Folder: Predicted As")
        axes[1].set_ylabel("Clip Count")
        axes[1].grid(axis="y", alpha=0.25)
        for index, value in enumerate(shot_values):
            axes[1].text(index, value + 0.2, str(value), ha="center", va="bottom", fontsize=10)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No clips found in shot folders", ha="center", va="center")

    note = (
        "Note: pass/goal/save/other folders have known expected labels. "
        "Shot folder is reported as prediction distribution because this model has no plain shot class."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(records):
    known_records = [record for record in records if record["expected"]]
    total = len(known_records)
    correct = sum(1 for record in known_records if record["correct"] == "1")
    print(f"Evaluated clips with known labels: {correct}/{total} correct ({correct / max(total, 1):.1%})")

    by_folder = defaultdict(list)
    for record in known_records:
        by_folder[record["folder"]].append(record)

    for folder in ["pass", "goal", "save", "other"]:
        folder_records = by_folder[folder]
        folder_total = len(folder_records)
        folder_correct = sum(1 for record in folder_records if record["correct"] == "1")
        print(
            f"{folder}: {folder_correct}/{folder_total} correct "
            f"({folder_correct / max(folder_total, 1):.1%})"
        )

    shot_records = [record for record in records if record["folder"] == "shot"]
    if shot_records:
        print("shot folder prediction distribution:")
        for label, count in Counter(record["predicted"] for record in shot_records).most_common():
            print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/action_classifier_v1/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/action_clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation_v1"))
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for split, folder, clip_path in collect_clips(args.data):
        expected = EXPECTED_LABELS.get(folder)
        predicted, confidence, scores = predict_clip(model, clip_path, labels, frames, size, device)
        correct = "" if expected is None else str(int(predicted == expected))
        record = {
            "split": split,
            "folder": folder,
            "file": clip_path.name,
            "expected": expected or "",
            "predicted": predicted,
            "confidence": f"{confidence:.6f}",
            "correct": correct,
            "path": str(clip_path),
        }
        for label in labels:
            record[f"score_{label}"] = f"{scores[label]:.6f}"
        records.append(record)

    csv_path = args.output_dir / "clip_predictions.csv"
    chart_path = args.output_dir / "action_classifier_summary.png"
    write_csv(records, labels, csv_path)
    make_chart(records, labels, chart_path)
    print_summary(records)
    print(f"CSV: {csv_path}")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
