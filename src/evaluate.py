import argparse
import json
from pathlib import Path
from typing import Any

import albumentations as albu
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from tqdm import tqdm

from src.utils import load_rgb

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TARGET_SIZE = 224


class Dataset:
    def __init__(self, file_names: list):
        self.file_names = file_names

        self.transform = albu.Compose(
            [
                albu.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_CUBIC),
                albu.CenterCrop(height=224, width=224),
                albu.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = load_rgb(self.file_names[index])
        return {"image": self.transform(image=image)["image"]}


class RealLabelsImagenet:
    """
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/real_labels.py
    """

    def __init__(self, filenames: list[Path], real_json: Path | str = "real.json", topk: tuple[int, ...] = (1, 5)):
        with open(real_json, encoding="utf-8") as file:
            real_labels = {f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels for i, labels in enumerate(json.load(file))}

        self.real_labels = real_labels
        self.filenames = filenames
        assert len(self.filenames) == len(self.real_labels)
        self.topk = topk
        self.losses: list[float] = []
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.is_correct: dict[int, list[float]] = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()

        for pred in pred_batch:
            filename = self.filenames[self.sample_idx].name
            if self.real_labels[filename]:
                for k in self.topk:
                    self.is_correct[k].append(any(p in self.real_labels[filename] for p in pred[:k]))

                temp = []
                with torch.inference_mode():
                    for ground_truth_id, ground_truth in enumerate(self.real_labels[filename]):
                        temp.append(
                            self.loss(output[ground_truth_id], torch.Tensor([ground_truth])[0].long().cuda()).item()
                        )

                self.losses.append({"filename": filename, "loss": min(temp)})

            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            return {k: float(np.mean(self.is_correct[k])) * 100 for k in self.topk}

        return float(np.mean(self.is_correct[k])) * 100


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with images.", required=True)
    arg("-r", "--real_list", type=Path, help="Path with the list of Real Imagenet validation set.", required=True)
    arg("-c", "--checkpoint", type=Path, help="Path to the model checkpoint.", required=True)
    arg("-b", "--batch_size", type=int, help="Batch size.", default=256)
    arg("-o", "--output_path", type=Path, help="Path to save loss_file.", required=True)
    arg("-j", "--num_workers", type=int, help="The number of workers.", default=4)
    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    model = torch.jit.load(args.checkpoint).cuda()
    model.eval()

    file_names = sorted(args.input_path.glob("*.JPEG"))
    real_labels = RealLabelsImagenet(file_names, args.real_list)

    dataloader = torch.utils.data.DataLoader(
        Dataset(file_names),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            predictions = model(batch["image"].cuda())
            real_labels.add_result(predictions)

    print(real_labels.get_accuracy())

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump({"accuracy": real_labels.get_accuracy(), "losses": real_labels.losses}, file, indent=4)


if __name__ == "__main__":
    main()
