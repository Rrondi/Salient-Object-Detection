import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SODDataset, list_pairs
from src.metrics import iou_score, mae, precision_recall_f1
from src.model import SaliencyNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images_dir", type=str, required=True)
    parser.add_argument("--test_masks_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--image_size", type=int, default=128, choices=[128, 224])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_visuals", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


def overlay(image, pred):
    pred_rgb = pred.repeat(3, 1, 1)
    return (0.65 * image + 0.35 * pred_rgb).clamp(0, 1)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_pairs = list_pairs(args.test_images_dir, args.test_masks_dir)
    test_ds = SODDataset(test_pairs, image_size=args.image_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SaliencyNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ious, ps, rs, f1s, maes = [], [], [], [], []
    visuals_saved = 0

    with torch.no_grad():
        for images, masks, names in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            preds_bin = (preds > 0.5).float()

            ious.append(iou_score(preds_bin, masks).item())
            p, r, f1 = precision_recall_f1(preds_bin, masks)
            ps.append(p.item())
            rs.append(r.item())
            f1s.append(f1.item())
            maes.append(mae(preds, masks).item())

            if visuals_saved < args.num_visuals:
                b = images.size(0)
                for i in range(b):
                    if visuals_saved >= args.num_visuals:
                        break
                    img = images[i].cpu()
                    gt = masks[i].cpu()
                    pr = preds[i].cpu()
                    ov = overlay(img, pr)

                    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
                    ax[0].imshow(img.permute(1, 2, 0))
                    ax[0].set_title("Input")
                    ax[1].imshow(gt.squeeze(0), cmap="gray")
                    ax[1].set_title("Ground Truth")
                    ax[2].imshow(pr.squeeze(0), cmap="gray")
                    ax[2].set_title("Prediction")
                    ax[3].imshow(ov.permute(1, 2, 0))
                    ax[3].set_title("Overlay")
                    for a in ax:
                        a.axis("off")
                    fig.tight_layout()
                    fig.savefig(out_dir / f"sample_{visuals_saved}_{names[i]}.png")
                    plt.close(fig)
                    visuals_saved += 1

    metrics = {
        "iou": sum(ious) / len(ious),
        "precision": sum(ps) / len(ps),
        "recall": sum(rs) / len(rs),
        "f1": sum(f1s) / len(f1s),
        "mae": sum(maes) / len(maes),
    }
    print(f"Test IoU: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Visualizations saved in {out_dir}")
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics JSON saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
