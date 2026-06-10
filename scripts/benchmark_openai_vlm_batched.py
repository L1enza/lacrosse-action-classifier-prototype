import argparse
import base64
import csv
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

try:
    import certifi
except ImportError:
    certifi = None


EXPECTED_LABELS = {
    "pass": "pass",
    "goal": "shot_goal",
    "save": "shot_save",
    "other": "other",
    "shot": "",
}

LABELS = ["pass", "shot_goal", "shot_save", "other"]


def sample_frames(video_path, frame_count, width):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = frame_count

    indices = [
        round(i * max(total_frames - 1, 0) / max(frame_count - 1, 1))
        for i in range(frame_count)
    ]
    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue

        height, original_width = frame.shape[:2]
        if original_width > width:
            scale = width / original_width
            frame = cv2.resize(
                frame,
                (width, int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            encoded = base64.b64encode(buffer).decode("utf-8")
            frames.append(f"data:image/jpeg;base64,{encoded}")

    cap.release()
    if not frames:
        raise RuntimeError(f"No readable frames found in video: {video_path}")
    return frames


def collect_clips(data_dir, split_filter):
    clips = []
    clip_index = 1
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        if split_filter != "all" and split_dir.name != split_filter:
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for video_path in sorted(class_dir.glob("*.mp4")):
                clips.append(
                    {
                        "clip_id": f"clip_{clip_index:04d}",
                        "split": split_dir.name,
                        "folder": class_dir.name,
                        "path": video_path,
                    }
                )
                clip_index += 1
    return clips


def load_completed(output_csv):
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="") as file:
        return {row["path"] for row in csv.DictReader(file)}


def make_prompt(batch):
    clip_ids = ", ".join(item["clip_id"] for item in batch)
    return f"""
You are evaluating multiple short lacrosse video clips from sampled frames.

Each clip has an ID. Frames for each clip appear immediately after that clip's ID.
Use only the visual content of the sampled images. You are not given filenames, folders, or ground-truth labels.

Choose exactly one label for each clip:
- pass: a player passes the ball to a teammate
- shot_goal: a shot is taken and results in a goal
- shot_save: a shot is taken and the goalie or defense saves/stops it
- other: none of the above is clearly happening

Some clips may show a shot that misses or does not clearly result in a goal/save; label those as other unless a goal or save is clearly visible.

Return one prediction for each of these clip IDs: {clip_ids}.
Return only valid JSON matching the requested schema.
""".strip()


def build_content(batch, frames_per_clip, width):
    content = [{"type": "input_text", "text": make_prompt(batch)}]
    for item in batch:
        content.append({"type": "input_text", "text": f"{item['clip_id']} frames:"})
        for image_url in sample_frames(item["path"], frames_per_clip, width):
            content.append({"type": "input_image", "image_url": image_url, "detail": "low"})
    return content


def call_openai(api_key, model, content, max_retries):
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "lacrosse_action_batch_predictions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "clip_id": {"type": "string"},
                                    "predicted_label": {"type": "string", "enum": LABELS},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "clip_id",
                                    "predicted_label",
                                    "confidence",
                                    "reason",
                                ],
                            },
                        }
                    },
                    "required": ["predictions"],
                },
            }
        },
    }

    body = json.dumps(payload).encode("utf-8")
    ssl_context = None
    if certifi is not None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180, context=ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI API error {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            if attempt < max_retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI API network error: {exc}") from exc


def extract_output_text(response):
    if "output_text" in response:
        return response["output_text"]

    chunks = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


def parse_predictions(response):
    text = extract_output_text(response)
    data = json.loads(text)
    return data.get("predictions", [])


def batched(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/action_clips"))
    parser.add_argument("--output", type=Path, default=Path("outputs/openai_vlm_benchmark/openai_predictions_batched.csv"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output)
    clips = collect_clips(args.data, args.split)
    if args.limit:
        clips = clips[: args.limit]

    pending = [item for item in clips if str(item["path"]) not in completed]
    fieldnames = [
        "split",
        "folder",
        "file",
        "expected",
        "predicted",
        "confidence",
        "correct",
        "reason",
        "model",
        "frames_sent",
        "clip_id",
        "path",
    ]

    write_header = not args.output.exists()
    with args.output.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        batches = list(batched(pending, args.batch_size))
        for batch_number, batch in enumerate(batches, start=1):
            print(f"Batch {batch_number}/{len(batches)} ({len(batch)} clips)", flush=True)
            response = call_openai(
                api_key=api_key,
                model=args.model,
                content=build_content(batch, args.frames, args.width),
                max_retries=args.max_retries,
            )
            predictions = {item["clip_id"]: item for item in parse_predictions(response)}

            for item in batch:
                prediction = predictions.get(item["clip_id"], {})
                predicted = prediction.get("predicted_label", "other")
                if predicted not in LABELS:
                    predicted = "other"
                confidence = float(prediction.get("confidence", 0.0))
                reason = str(prediction.get("reason", "")).replace("\n", " ").strip()
                expected = EXPECTED_LABELS.get(item["folder"], "")
                correct = "" if not expected else str(int(predicted == expected))
                writer.writerow(
                    {
                        "split": item["split"],
                        "folder": item["folder"],
                        "file": item["path"].name,
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": f"{confidence:.4f}",
                        "correct": correct,
                        "reason": reason,
                        "model": args.model,
                        "frames_sent": args.frames,
                        "clip_id": item["clip_id"],
                        "path": str(item["path"]),
                    }
                )
            file.flush()
            time.sleep(args.sleep)

    print(f"Saved batched OpenAI benchmark CSV to {args.output}")


if __name__ == "__main__":
    main()
