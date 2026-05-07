import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SODDataset, list_pairs, split_train_val
from src.losses import BCESoftIoULoss
from src.metrics import iou_score
from src.model import SaliencyNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images_dir", type=str, required=True)
    parser.add_argument("--train_masks_dir", type=str, required=True)
    parser.add_argument("--val_images_dir", type=str, default="")
    parser.add_argument("--val_masks_dir", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=128, choices=[128, 224])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_checkpoint(path, epoch, model, optimizer, best_val_loss):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)
    print(f"[checkpoint] Saved: {path}")


def load_checkpoint(path, model, optimizer, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    print(f"[checkpoint] Resumed from: {path}")
    return state["epoch"] + 1, state["best_val_loss"]


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_iou = 0.0
    loop = tqdm(loader, leave=False)

    for images, masks, _ in loop:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train_mode):
            preds = model(images)
            loss = criterion(preds, masks)
            iou = iou_score((preds > 0.5).float(), masks)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        running_iou += iou.item()
        loop.set_postfix(loss=loss.item(), iou=iou.item())

    n = max(1, len(loader))
    return running_loss / n, running_iou / n


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pt"
    best_ckpt = checkpoint_dir / "best.pt"

    all_train_pairs = list_pairs(args.train_images_dir, args.train_masks_dir)
    if args.val_images_dir and args.val_masks_dir:
        train_pairs = all_train_pairs
        val_pairs = list_pairs(args.val_images_dir, args.val_masks_dir)
        print("Validation mode: explicit validation set provided.")
    else:
        train_pairs, val_pairs = split_train_val(all_train_pairs, val_ratio=args.val_ratio, seed=args.seed)
        print("Validation mode: random split from DUTS-TR.")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    train_ds = SODDataset(train_pairs, image_size=args.image_size, augment=True)
    val_ds = SODDataset(val_pairs, image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SaliencyNet().to(device)
    criterion = BCESoftIoULoss(iou_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and latest_ckpt.exists():
        start_epoch, best_val_loss = load_checkpoint(latest_ckpt, model, optimizer, device)

    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_iou = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
        print(f"train_loss={train_loss:.4f} train_iou={train_iou:.4f}")
        print(f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        save_checkpoint(latest_ckpt, epoch, model, optimizer, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(best_ckpt, epoch, model, optimizer, best_val_loss)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()
