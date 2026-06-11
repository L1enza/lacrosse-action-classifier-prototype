import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_MAPS = {
    "all_actions": {
    "pass": "pass",
    "goal": "shot_goal",
    "save": "shot_save",
    "other": "other",
    },
    "shot_outcome": {
        "shot_goal": "shot_goal",
        "shot_save": "shot_save",
    },
}


class LacrosseClipDataset(Dataset):
    def __init__(self, split_dir, label_to_idx, label_map, frames=16, size=112):
        self.split_dir = Path(split_dir)
        self.label_to_idx = label_to_idx
        self.label_map = label_map
        self.frames = frames
        self.size = size
        self.samples = []
        self.tensors = []

        for source_label, target_label in sorted(label_map.items()):
            class_dir = self.split_dir / source_label
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.mp4")):
                label_idx = self.label_to_idx[target_label]
                try:
                    tensor = load_video_tensor(path, self.frames, self.size)
                except RuntimeError as exc:
                    print(f"Skipping {path}: {exc}", flush=True)
                    continue
                self.samples.append((path, label_idx))
                self.tensors.append(tensor)

        if not self.samples:
            raise RuntimeError(f"No .mp4 clips found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, label = self.samples[index]
        video = self.tensors[index]
        return video, torch.tensor(label, dtype=torch.long)


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
            frame = torch.zeros(size, size, 3, dtype=torch.uint8).numpy()

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


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            logits = model(videos)
            loss = criterion(logits, labels)
            loss_total += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return loss_total / max(total, 1), correct / max(total, 1)


def count_by_label(dataset, labels):
    counts = {label: 0 for label in labels}
    for _, class_idx in dataset.samples:
        counts[labels[class_idx]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-map", choices=sorted(LABEL_MAPS), default="all_actions")
    args = parser.parse_args()

    label_map = LABEL_MAPS[args.label_map]
    labels = sorted(set(label_map.values()))
    label_to_idx = {label: index for index, label in enumerate(labels)}

    train_data = LacrosseClipDataset(
        args.data / "train", label_to_idx, label_map, args.frames, args.size
    )
    val_data = LacrosseClipDataset(
        args.data / "val", label_to_idx, label_map, args.frames, args.size
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    model = TinyVideoClassifier(num_classes=len(labels), frames=args.frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "labels": labels,
        "label_map_name": args.label_map,
        "label_map": label_map,
        "train_counts": count_by_label(train_data, labels),
        "val_counts": count_by_label(val_data, labels),
        "frames": args.frames,
        "size": args.size,
    }
    (args.output / "labels.json").write_text(json.dumps(metadata, indent=2))

    print(f"Device: {device}", flush=True)
    print(f"Labels: {labels}", flush=True)
    print(f"Train counts: {metadata['train_counts']}", flush=True)
    print(f"Val counts: {metadata['val_counts']}", flush=True)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_total = 0.0

        for videos, labels_tensor in train_loader:
            videos = videos.to(device)
            labels_tensor = labels_tensor.to(device)

            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * labels_tensor.size(0)
            correct += (logits.argmax(dim=1) == labels_tensor).sum().item()
            total += labels_tensor.size(0)

        train_loss = loss_total / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": labels,
                    "metadata": metadata,
                },
                args.output / "best.pt",
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "metadata": metadata,
        },
        args.output / "last.pt",
    )
    print(f"Saved best model to {args.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
