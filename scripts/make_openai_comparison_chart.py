import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PYTORCH_CSV = Path("outputs/evaluation_v1/clip_predictions.csv")
OPENAI_CSV = Path("outputs/openai_vlm_benchmark/openai_val_batched.csv")
OUT_DIR = Path("outputs/openai_vlm_benchmark")
PNG_PATH = OUT_DIR / "openai_vs_pytorch_validation.png"
JSON_PATH = OUT_DIR / "openai_vs_pytorch_validation_summary.json"


def font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def summarize(rows, split=None):
    known = [row for row in rows if row["expected"]]
    if split:
        known = [row for row in known if row["split"] == split]

    by_folder = defaultdict(list)
    for row in known:
        by_folder[row["folder"]].append(row)

    summary = {}
    for folder in ["pass", "goal", "save", "other"]:
        folder_rows = by_folder[folder]
        correct = sum(1 for row in folder_rows if row["correct"] == "1")
        total = len(folder_rows)
        summary[folder] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0,
        }

    correct = sum(1 for row in known if row["correct"] == "1")
    total = len(known)
    summary["overall"] = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
    }
    return summary


def center_text(draw, x, y, text, text_font, fill):
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=text_font, fill=fill)


def main():
    pytorch = summarize(load_rows(PYTORCH_CSV), split="val")
    openai = summarize(load_rows(OPENAI_CSV))
    data = {"pytorch_validation": pytorch, "openai_validation": openai}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2))

    image = Image.new("RGB", (1500, 850), "#f8fafc")
    draw = ImageDraw.Draw(image)

    title = font(40, True)
    subtitle = font(22)
    label = font(20, True)
    small = font(17)
    tiny = font(14)

    draw.text((70, 48), "OpenAI VLM vs PyTorch Prototype", font=title, fill="#0f172a")
    draw.text(
        (72, 102),
        "Validation clips only, blind OpenAI prompt, 2 sampled frames per clip",
        font=subtitle,
        fill="#334155",
    )

    chart = (95, 190, 1405, 665)
    draw.rounded_rectangle(chart, radius=18, fill="#ffffff", outline="#dbe3ec", width=2)
    x0, y0, x1, y1 = 170, 275, 1340, 590
    draw.line((x0, y1, x1, y1), fill="#94a3b8", width=2)
    draw.line((x0, y0, x0, y1), fill="#94a3b8", width=2)

    for pct in [0, 25, 50, 75, 100]:
        y = y1 - (pct / 100) * (y1 - y0)
        draw.line((x0 - 8, y, x1, y), fill="#e2e8f0", width=1)
        draw.text((x0 - 58, y - 10), f"{pct}%", font=tiny, fill="#64748b")

    groups = ["overall", "pass", "goal", "save", "other"]
    group_labels = ["overall", "pass", "goal", "save", "other"]
    group_width = (x1 - x0) / len(groups)
    bar_width = 58
    colors = {"PyTorch": "#2563eb", "OpenAI": "#16a34a"}

    for i, key in enumerate(groups):
        gx = x0 + i * group_width + group_width / 2
        for j, (name, summary) in enumerate([("PyTorch", pytorch), ("OpenAI", openai)]):
            stats = summary[key]
            pct = stats["accuracy"] * 100
            bar_h = (pct / 100) * (y1 - y0)
            x_left = gx + (-bar_width - 7 if j == 0 else 7)
            y_top = y1 - bar_h
            draw.rounded_rectangle(
                (x_left, y_top, x_left + bar_width, y1),
                radius=7,
                fill=colors[name],
            )
            center_text(draw, x_left + bar_width / 2, y_top - 42, f"{pct:.0f}%", small, "#0f172a")
            center_text(
                draw,
                x_left + bar_width / 2,
                y_top - 21,
                f"{stats['correct']}/{stats['total']}",
                tiny,
                "#475569",
            )
        center_text(draw, gx, y1 + 22, group_labels[i], small, "#334155")

    draw.rectangle((1015, 215, 1040, 240), fill=colors["PyTorch"])
    draw.text((1050, 215), "PyTorch prototype", font=small, fill="#334155")
    draw.rectangle((1015, 250, 1040, 275), fill=colors["OpenAI"])
    draw.text((1050, 250), "OpenAI VLM", font=small, fill="#334155")

    note = (
        "OpenAI was run as a faster validation benchmark: 25 validation clips total; "
        "19 have known labels and 6 are the mixed shot folder."
    )
    draw.text((90, 725), note, font=small, fill="#475569")
    draw.text((90, 760), "The mixed shot folder is excluded from accuracy for both systems.", font=tiny, fill="#64748b")

    image.save(PNG_PATH)
    print(json.dumps(data, indent=2))
    print(f"Chart: {PNG_PATH}")


if __name__ == "__main__":
    main()
