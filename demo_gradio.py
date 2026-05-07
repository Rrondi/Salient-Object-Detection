import time

import cv2
import gradio as gr
import numpy as np
import torch

from src.model import SaliencyNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
CHECKPOINT = "checkpoints/best.pt"

model = SaliencyNet().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()


def preprocess(image_np):
    image = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return image, tensor


def predict(image_np):
    if image_np is None:
        return None, None, "No image provided."
    image_rgb, x = preprocess(image_np)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(x)[0, 0].cpu().numpy()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    mask = (pred * 255).astype(np.uint8)

    overlay = (0.7 * image_rgb + 0.3 * np.stack([pred, pred, pred], axis=-1)).clip(0, 1)
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
    description="Upload an image to predict saliency mask and overlay.",
)

if __name__ == "__main__":
    demo.launch()
