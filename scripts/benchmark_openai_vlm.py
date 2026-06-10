import argparse
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2


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
    encoded_frames = []

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

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            encoded = base64.b64encode(buffer).decode("utf-8")
            encoded_frames.append(f"data:image/jpeg;base64,{encoded}")

    cap.release()

    if not encoded_frames:
        raise RuntimeError(f"No readable frames found in video: {video_path}")
    return encoded_frames


def collect_clips(data_dir):
    clips = []
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for video_path in sorted(class_dir.glob("*.mp4")):
                clips.append((split_dir.name, class_dir.name, video_path))
    return clips


def make_prompt(folder):
    expected = EXPECTED_LABELS.get(folder, "")
    return f"""
You are evaluating a short lacrosse video clip from sampled frames.

Choose exactly one label:
- pass: a player passes the ball to a teammate
- shot_goal: a shot is taken and results in a goal
- shot_save: a shot is taken and the goalie or defense saves/stops it
- other: none of the above is clearly happening

The folder label for this clip is "{folder}". Use the images, not the folder name, to decide.
For folder label "shot", there may be a shot but the current benchmark has no plain shot class.

Return only valid JSON with this schema:
{{
  "predicted_label": "one of: pass, shot_goal, shot_save, other",
  "confidence": 0.0,
  "reason": "brief visual reason"
}}

Known expected label used for scoring, if any: "{expected}".
""".strip()


def call_openai(api_key, model, prompt, image_urls, max_retries):
    content = [{"type": "input_text", "text": prompt}]
    for image_url in image_urls:
        content.append({"type": "input_image", "image_url": image_url})

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_object"}},
    }
    body = json.dumps(payload).encode("utf-8")

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
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI API error {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
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


def normalize_prediction(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        data = json.loads(text[start : end + 1])

    predicted = str(data.get("predicted_label", "")).strip()
    if predicted not in LABELS:
        predicted = "other"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))
    reason = str(data.get("reason", "")).replace("\n", " ").strip()
    return predicted, confidence, reason


def load_completed(output_csv):
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="") as file:
        return {row["path"] for row in csv.DictReader(file)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/action_clips"))
    parser.add_argument("--output", type=Path, default=Path("outputs/openai_vlm_benchmark/openai_predictions.csv"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output)
    clips = collect_clips(args.data)
    if args.limit:
        clips = clips[: args.limit]

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
        "path",
    ]

    write_header = not args.output.exists()
    with args.output.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for index, (split, folder, video_path) in enumerate(clips, start=1):
            path_key = str(video_path)
            if path_key in completed:
                continue

            expected = EXPECTED_LABELS.get(folder, "")
            print(f"[{index}/{len(clips)}] {split}/{folder}/{video_path.name}", flush=True)

            frames = sample_frames(video_path, args.frames, args.width)
            response = call_openai(
                api_key=api_key,
                model=args.model,
                prompt=make_prompt(folder),
                image_urls=frames,
                max_retries=args.max_retries,
            )
            text = extract_output_text(response)
            predicted, confidence, reason = normalize_prediction(text)
            correct = "" if not expected else str(int(predicted == expected))

            writer.writerow(
                {
                    "split": split,
                    "folder": folder,
                    "file": video_path.name,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": f"{confidence:.4f}",
                    "correct": correct,
                    "reason": reason,
                    "model": args.model,
                    "frames_sent": len(frames),
                    "path": path_key,
                }
            )
            file.flush()
            time.sleep(args.sleep)

    print(f"Saved OpenAI benchmark CSV to {args.output}")


if __name__ == "__main__":
    main()
