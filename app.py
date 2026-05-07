import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from src.model import SaliencyNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "128"))
CHECKPOINT = os.getenv("CHECKPOINT_PATH", "checkpoints/best.pt")

model = SaliencyNet().to(DEVICE)
checkpoint_path = Path(CHECKPOINT)
if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at '{CHECKPOINT}'. "
        "Set CHECKPOINT_PATH env var or include checkpoints/best.pt."
    )
ckpt = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()


def preprocess(image_np: np.ndarray):
    if image_np is None:
        raise ValueError("Empty image.")
    # Gradio RGB; ensure float tensor NCHW
    img = np.asarray(image_np, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected image shape: {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.max() > 1.0:
        img = img / 255.0
    chw = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    resized = F.interpolate(
        chw, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    image_rgb = resized[0].permute(1, 2, 0).cpu().numpy()
    return image_rgb, resized


def predict(image_np):
    if image_np is None:
        return None, None, "No image provided."
    image_rgb, x = preprocess(image_np)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(x)[0, 0].cpu().numpy()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    mask = (pred * 255).astype(np.uint8)
    overlay = (
        0.7 * image_rgb + 0.3 * np.stack([pred, pred, pred], axis=-1)
    ).clip(0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    return mask, overlay, f"Inference time: {dt_ms:.2f} ms"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs=[
        gr.Image(type="numpy", label="Saliency Mask"),
        gr.Image(type="numpy", label="Overlay"),
        gr.Textbox(label="Latency"),
    ],
    title="Salient Object Detection Demo",
    description="Upload an image to generate saliency mask, overlay, and inference time.",
)

if __name__ == "__main__":
    demo.launch()
