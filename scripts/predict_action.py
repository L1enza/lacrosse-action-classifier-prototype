import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def load_video_tensor(path, target_frames, size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = target_frames

    frame_indices = torch.linspace(0, max(total - 1, 0), target_frames).long().tolist()
    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].copy())
                continue
            frame = np.zeros((size, size, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()

    tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).reshape(target_frames * 3, size, size)
    mean = torch.tensor([0.45] * (target_frames * 3)).view(target_frames * 3, 1, 1)
    std = torch.tensor([0.225] * (target_frames * 3)).view(target_frames * 3, 1, 1)
    return (tensor - mean) / std


class TinyVideoClassifier(nn.Module):
    def __init__(self, num_classes, frames):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(frames * 3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/action_classifier_v1/best.pt"))
    parser.add_argument("--clip", type=Path, required=True)
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

    video = load_video_tensor(args.clip, frames, size).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(video), dim=1)[0].cpu()

    ranked = sorted(
        ((labels[index], float(score)) for index, score in enumerate(probs)),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"clip: {args.clip}")
    print(f"prediction: {ranked[0][0]} ({ranked[0][1]:.3f})")
    print("scores:")
    for label, score in ranked:
        print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
