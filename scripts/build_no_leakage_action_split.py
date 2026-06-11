import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path("data/action_clips")
TARGET = Path("data/action_clips/no_leakage")
OUT = Path("outputs/no_leakage_split")


# Keep shot outcomes from the previous game-family split.
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


# Pass clips are almost all TOR/HFX. These are the less common high-score/finals
# game-state clips visible in the scorebox; keeping them together in val reduces
# same-game-state leakage while preserving pass coverage in both splits.
PASS_VAL_PATHS = {
    "data/action_clips/train/pass/paaas.mp4",
    "data/action_clips/train/pass/passaa.mp4",
    "data/action_clips/train/pass/passssy.mp4",
    "data/action_clips/train/pass/peas.mp4",
    "data/action_clips/train/pass/pp.mp4",
    "data/action_clips/val/pass/pass.mp4",
    "data/action_clips/val/pass/passe.mp4",
    "data/action_clips/val/pass/passsss.mp4",
}


# Other is noisier, but these are either HFX/GA or high-score/finals TOR/HFX
# game-state clips, separated from the common low-score TOR/HFX train examples.
OTHER_VAL_PATHS = {
    "data/action_clips/train/other/othee.mp4",
    "data/action_clips/train/other/other.mp4",
    "data/action_clips/train/other/otherre.mp4",
    "data/action_clips/train/other/othtjer.mp4",
    "data/action_clips/train/other/ototot.mp4",
    "data/action_clips/train/other/oyot.mp4",
    "data/action_clips/val/other/qw.mp4",
}


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def unique_target_path(split, label, source_path):
    target_dir = TARGET / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / f"{source_path.parts[-3]}_{source_path.stem}{source_path.suffix}"
    count = 2
    while candidate.exists():
        candidate = target_dir / f"{source_path.parts[-3]}_{source_path.stem}_{count}{source_path.suffix}"
        count += 1
    return candidate


def classify_goal(path):
    if path.name in HFX_GA_GOALS:
        return "val", "HFX_GA", "visible HFX/GA scorebox"
    return "train", "TOR_HFX", "visible TOR/HFX scorebox"


def classify_save(path):
    if path.name in TOR_ROC_SAVES:
        return "train", "TOR_ROC", "visible TOR/ROC scorebox"
    if path.name in HFX_GA_SAVES:
        return "val", "HFX_GA", "visible HFX/GA scorebox"
    if path.name in COL_PC_SAVES:
        return "val", "COL_PC", "visible COL/PC scorebox"
    return "train", "TOR_HFX", "visible TOR/HFX scorebox"


def classify_pass(path):
    key = str(path)
    if key in PASS_VAL_PATHS:
        return "val", "TOR_HFX_HIGH_OR_FINALS", "less common high-score/finals TOR/HFX scorebox"
    return "train", "TOR_HFX_COMMON", "common low-score TOR/HFX scorebox"


def classify_other(path):
    key = str(path)
    if key in OTHER_VAL_PATHS:
        if path.name == "qw.mp4":
            return "val", "HFX_GA", "visible HFX/GA scorebox"
        return "val", "TOR_HFX_HIGH_OR_FINALS", "less common high-score/finals TOR/HFX scorebox"
    return "train", "TOR_HFX_COMMON", "common low-score TOR/HFX scorebox or non-play graphic"


def add_record(records, source_path, target_label, split, game_group, reason):
    target_path = unique_target_path(split, target_label, source_path)
    link_or_copy(source_path, target_path)
    records.append(
        {
            "source_split": source_path.parts[-3],
            "source_folder": source_path.parts[-2],
            "source_file": source_path.name,
            "source_path": str(source_path),
            "target_label": target_label,
            "game_group": game_group,
            "target_split": split,
            "reason": reason,
            "target_path": str(target_path),
        }
    )


def main():
    if TARGET.exists():
        shutil.rmtree(TARGET)
    OUT.mkdir(parents=True, exist_ok=True)

    records = []

    for path in sorted(SOURCE.glob("*/goal/*.mp4")):
        split, game, reason = classify_goal(path)
        add_record(records, path, "shot_goal", split, game, reason)

    for path in sorted(SOURCE.glob("*/save/*.mp4")):
        split, game, reason = classify_save(path)
        add_record(records, path, "shot_save", split, game, reason)

    for path in sorted(SOURCE.glob("*/pass/*.mp4")):
        split, game, reason = classify_pass(path)
        add_record(records, path, "pass", split, game, reason)

    for path in sorted(SOURCE.glob("*/other/*.mp4")):
        split, game, reason = classify_other(path)
        add_record(records, path, "other", split, game, reason)

    manifest = OUT / "no_leakage_manifest.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    counts_path = OUT / "no_leakage_counts.csv"
    counts = Counter((r["target_split"], r["target_label"], r["game_group"]) for r in records)
    with counts_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target_split", "target_label", "game_group", "count"])
        for (split, label, game), count in sorted(counts.items()):
            writer.writerow([split, label, game, count])

    split_counts = defaultdict(Counter)
    for record in records:
        split_counts[record["target_split"]][record["target_label"]] += 1

    print(f"Created {TARGET}")
    print(f"Manifest: {manifest}")
    print(f"Counts: {counts_path}")
    for split in ["train", "val"]:
        print(split, dict(split_counts[split]))


if __name__ == "__main__":
    main()
