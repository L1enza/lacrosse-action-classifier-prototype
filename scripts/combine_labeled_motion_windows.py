import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write.")
    columns = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Combine labeled motion-window CSVs into one dataset."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Inputs formatted as video_name=path/to/labeled_motion_windows.csv",
    )
    args = parser.parse_args()

    combined = []
    counts = {}
    for item in args.inputs:
        if "=" not in item:
            raise RuntimeError(
                "Each input must be formatted as video_name=path/to/file.csv"
            )
        video_name, path_text = item.split("=", 1)
        for row in read_csv(Path(path_text)):
            merged = {"video": video_name, **row}
            combined.append(merged)
            key = merged["label"]
            counts[key] = counts.get(key, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, combined)

    print(f"Saved combined dataset: {args.output}")
    print(f"Rows: {len(combined)}")
    print("Label counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
