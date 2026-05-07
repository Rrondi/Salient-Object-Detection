import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt

from src.dataset import list_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr_images_dir", type=str, required=True)
    parser.add_argument("--tr_masks_dir", type=str, required=True)
    parser.add_argument("--te_images_dir", type=str, required=True)
    parser.add_argument("--te_masks_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="outputs_dataset_inspection")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_samples(pairs, prefix: str, out_dir: Path, count: int):
    picks = random.sample(pairs, min(count, len(pairs)))
    for i, (img_path, mask_path) in enumerate(picks):
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].imshow(plt.imread(img_path))
        ax[0].set_title("Image")
        ax[0].axis("off")
        ax[1].imshow(plt.imread(mask_path), cmap="gray")
        ax[1].set_title("Mask")
        ax[1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_sample_{i}_{img_path.stem}.png")
        plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_pairs = list_pairs(args.tr_images_dir, args.tr_masks_dir)
    te_pairs = list_pairs(args.te_images_dir, args.te_masks_dir)

    report_path = out_dir / "dataset_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DUTS Dataset Inspection\n")
        f.write("=======================\n")
        f.write(f"TR pairs: {len(tr_pairs)}\n")
        f.write(f"TE pairs: {len(te_pairs)}\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Pairs are matched by filename stem (jpg image with png mask is supported).\n")
        f.write("- Training uses DUTS-TR only; DUTS-TE is held out for final test.\n")
        f.write("- Images and masks are resized in the data loader to 128 or 224.\n")
        f.write("- Pixel values are normalized to [0, 1].\n")

    save_samples(tr_pairs, "tr", out_dir, args.num_samples)
    save_samples(te_pairs, "te", out_dir, args.num_samples)
    print(f"Dataset report saved: {report_path}")
    print(f"Sample visualizations saved in: {out_dir}")


if __name__ == "__main__":
    main()
