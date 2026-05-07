import argparse
import csv
import json
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def train_variant(args, name: str, image_size: int, lr: float, epochs: int):
    ckpt_dir = Path(args.experiments_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        "python",
        "train.py",
        "--train_images_dir",
        args.tr_images_dir,
        "--train_masks_dir",
        args.tr_masks_dir,
        "--image_size",
        str(image_size),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--patience",
        str(args.patience),
        "--checkpoint_dir",
        str(ckpt_dir),
    ]
    run_cmd(train_cmd)
    return ckpt_dir / "best.pt"


def eval_variant(args, name: str, image_size: int, checkpoint_path: Path):
    out_dir = Path(args.experiments_dir) / f"{name}_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = Path(args.experiments_dir) / f"{name}_metrics.json"
    eval_cmd = [
        "python",
        "evaluate.py",
        "--test_images_dir",
        args.te_images_dir,
        "--test_masks_dir",
        args.te_masks_dir,
        "--checkpoint",
        str(checkpoint_path),
        "--image_size",
        str(image_size),
        "--batch_size",
        str(args.batch_size),
        "--num_visuals",
        str(args.num_visuals),
        "--out_dir",
        str(out_dir),
        "--metrics_out",
        str(metrics_out),
    ]
    run_cmd(eval_cmd)
    with open(metrics_out, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr_images_dir", type=str, required=True)
    parser.add_argument("--tr_masks_dir", type=str, required=True)
    parser.add_argument("--te_images_dir", type=str, required=True)
    parser.add_argument("--te_masks_dir", type=str, required=True)
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_visuals", type=int, default=6)
    args = parser.parse_args()

    variants = [
        {"name": "baseline_128_lr1e3", "image_size": 128, "lr": 1e-3, "epochs": args.epochs},
        {"name": "improved_224_lr1e3", "image_size": 224, "lr": 1e-3, "epochs": args.epochs},
        {"name": "improved_128_lr5e4", "image_size": 128, "lr": 5e-4, "epochs": args.epochs},
    ]

    rows = []
    for variant in variants:
        best_ckpt = train_variant(
            args,
            name=variant["name"],
            image_size=variant["image_size"],
            lr=variant["lr"],
            epochs=variant["epochs"],
        )
        metrics = eval_variant(
            args,
            name=variant["name"],
            image_size=variant["image_size"],
            checkpoint_path=best_ckpt,
        )
        rows.append(
            {
                "variant": variant["name"],
                "image_size": variant["image_size"],
                "lr": variant["lr"],
                "iou": round(metrics["iou"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "mae": round(metrics["mae"], 4),
            }
        )

    results_csv = Path(args.experiments_dir) / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "image_size", "lr", "iou", "precision", "recall", "f1", "mae"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults table written to: {results_csv}")


if __name__ == "__main__":
    main()
