import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("outputs/openai_vlm_benchmark/openai_predictions.csv"))
    args = parser.parse_args()

    with args.csv.open(newline="") as file:
        rows = list(csv.DictReader(file))

    known = [row for row in rows if row["expected"]]
    total = len(known)
    correct = sum(1 for row in known if row["correct"] == "1")
    print(f"Known-label accuracy: {correct}/{total} ({correct / max(total, 1):.1%})")

    by_folder = defaultdict(list)
    for row in known:
        by_folder[row["folder"]].append(row)

    for folder in ["pass", "goal", "save", "other"]:
        folder_rows = by_folder[folder]
        folder_total = len(folder_rows)
        folder_correct = sum(1 for row in folder_rows if row["correct"] == "1")
        print(f"{folder}: {folder_correct}/{folder_total} ({folder_correct / max(folder_total, 1):.1%})")

    shot_rows = [row for row in rows if row["folder"] == "shot"]
    if shot_rows:
        print("shot folder prediction distribution:")
        for label, count in Counter(row["predicted"] for row in shot_rows).most_common():
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
