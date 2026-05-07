import os
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from src.model import SaliencyNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "128"))
CHECKPOINT = os.getenv("CHECKPOINT_PATH", "checkpoints/best.pt")

# Model load runs in a background thread so the UI can bind before ~90 MB weights load
# (avoids sluggish startup / gateway timeouts).
_model: Optional[SaliencyNet] = None
_load_started = threading.Event()
_load_done = threading.Event()
_load_error: str | None = None


def _load_weights_sync() -> None:
    """Load checkpoint on DEVICE; must run once before inference."""
    global _model, _load_error
    try:
        checkpoint_path = Path(CHECKPOINT)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{CHECKPOINT}'. "
                "Set CHECKPOINT_PATH or add checkpoints/best.pt."
            )
        m = SaliencyNet().to(DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _model = m
    except Exception as e:
        _load_error = repr(e)
    finally:
        _load_done.set()


def _ensure_model_background() -> None:
    """Start loading once if not started."""
    if _load_started.is_set():
        return
    _load_started.set()
    threading.Thread(target=_load_weights_sync, daemon=True).start()


def _wait_for_model(timeout_sec: float = 900.0) -> tuple[bool, str]:
    """
    Blocks until weights are ready or error/timeout.
    Returns (ok, status_message_for_user).
    """
    _ensure_model_background()
    if _load_done.wait(timeout=timeout_sec):
        if _load_error:
            return False, f"Model load failed: {_load_error}"
        return True, "ready"
    return False, (
        "Model is still loading (large checkpoint). Wait a minute and retry, "
        "or reload the page in ~2 minutes."
    )


# Start loading as soon as the module is imported so cold start overlaps with Spaces boot.
_ensure_model_background()


def preprocess(image_np: np.ndarray):
    if image_np is None:
        raise ValueError("Empty image.")
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

    ok, msg = _wait_for_model()
    if not ok:
        return None, None, msg

    assert _model is not None

    image_rgb, x = preprocess(image_np)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = _model(x)[0, 0].cpu().numpy()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    mask = (pred * 255).astype(np.uint8)
    overlay = (
        0.7 * image_rgb + 0.3 * np.stack([pred, pred, pred], axis=-1)
    ).clip(0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    if DEVICE.type == "cuda":
        note = f"Inference time: {dt_ms:.2f} ms (GPU)"
    else:
        note = f"Inference time: {dt_ms:.2f} ms (CPU)"
    return mask, overlay, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs=[
        gr.Image(type="numpy", label="Saliency Mask"),
        gr.Image(type="numpy", label="Overlay"),
        gr.Textbox(label="Latency / status"),
    ],
    title="Salient Object Detection Demo",
    description=(
        "First prediction may take longer while ~90 MB of weights finish loading "
        "(especially on free CPU tiers). Subsequent runs are faster."
    ),
)


if __name__ == "__main__":
    demo.launch()
