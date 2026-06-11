import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path("data/action_clips")
TARGET = Path("data/shot_outcome_clips")
OUT = Path("outputs/game_split")

# Split by visible scorebox/game family. This intentionally keeps each broad
# game family entirely in one split to reduce broadcast/scorebug leakage.
GAME_TO_SPLIT = {
    "TOR_HFX": "train",
    "TOR_ROC": "train",
    "HFX_GA": "val",
    "COL_PC": "val",
}

HFX_GA_GOALS = {
    "aaa.mp4",
    "al.mp4",
    "ddo.mp4",
    "gaol.mp4",
    "gil.mp4",
    "goaaals.mp4",
    "goalase.mp4",
    "goalww.mp4",
    "goasol.mp4",
    "op.mp4",
    "pol.mp4",
    "shotgoalllp.mp4",
}

TOR_ROC_SAVES = {
    "5r.mp4",
    "lo.mp4",
    "p.mp4",
    "poiu.mp4",
    "pol.mp4",
    "qw.mp4",
    "qwe.mp4",
    "rrui.mp4",
    "wwe.mp4",
    "you.mp4",
}

HFX_GA_SAVES = {
    "aa.mp4",
    "po.mp4",
    "saal.mp4",
    "sapaps.mp4",
}

COL_PC_SAVES = {
    "polo.mp4",
    "saveyy.mp4",
    "tyu.mp4",
    "wer.mp4",
    "yoiu.mp4",
}


def game_for_clip(label, path):
    name = path.name
    if label == "shot_goal":
        return "HFX_GA" if name in HFX_GA_GOALS else "TOR_HFX"
    if label == "shot_save":
        if name in TOR_ROC_SAVES:
            return "TOR_ROC"
        if name in HFX_GA_SAVES:
            return "HFX_GA"
        if name in COL_PC_SAVES:
            return "COL_PC"
        return "TOR_HFX"
    raise ValueError(label)


def unique_target_path(path):
    split = GAME_TO_SPLIT[path["game"]]
    label = path["target_label"]
    source_path = path["source_path"]
    target_dir = TARGET / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    suffix = source_path.suffix
    candidate = target_dir / f"{path['source_split']}_{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = target_dir / f"{path['source_split']}_{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    if TARGET.exists():
        shutil.rmtree(TARGET)
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for source_folder, target_label in [("goal", "shot_goal"), ("save", "shot_save")]:
        for source_path in sorted(SOURCE.glob(f"*/{source_folder}/*.mp4")):
            source_split = source_path.parts[-3]
            game = game_for_clip(target_label, source_path)
            split = GAME_TO_SPLIT[game]
            record = {
                "source_split": source_split,
                "source_folder": source_folder,
                "source_file": source_path.name,
                "source_path": source_path,
                "target_label": target_label,
                "game": game,
                "target_split": split,
            }
            target_path = unique_target_path(record)
            link_or_copy(source_path, target_path)
            record["target_path"] = target_path
            records.append(record)

    manifest_path = OUT / "shot_outcome_game_split_manifest.csv"
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_split",
                "source_folder",
                "source_file",
                "source_path",
                "target_label",
                "game",
                "target_split",
                "target_path",
            ],
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["source_path"] = str(row["source_path"])
            row["target_path"] = str(row["target_path"])
            writer.writerow(row)

    counts_path = OUT / "shot_outcome_game_counts.csv"
    game_counts = Counter((r["target_label"], r["game"], r["target_split"]) for r in records)
    with counts_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target_label", "game", "target_split", "count"])
        for (label, game, split), count in sorted(game_counts.items()):
            writer.writerow([label, game, split, count])

    split_counts = defaultdict(Counter)
    for record in records:
        split_counts[record["target_split"]][record["target_label"]] += 1

    print(f"Created {TARGET}")
    print(f"Manifest: {manifest_path}")
    print(f"Game counts: {counts_path}")
    for split in ["train", "val"]:
        print(split, dict(split_counts[split]))


if __name__ == "__main__":
    main()
