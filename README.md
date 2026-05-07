# Salient Object Detection (PyTorch, From Scratch)

End-to-end SOD project using a custom encoder-decoder CNN (no pretrained backbone), built for DUTS-style saliency masks.

## 1) Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) What This Repo Contains

- custom model architecture (`src/model.py`)
- data pipeline + augmentation (`src/dataset.py`)
- custom loss (`BCE + 0.5 * (1 - IoU)`) (`src/losses.py`)
- training loop with checkpoint resume (`train.py`)
- evaluation metrics + visual outputs (`evaluate.py`)
- dataset inspection utility (`inspect_dataset.py`)
- experiment runner for baseline vs improvements (`experiments.py`)
- interactive demo app (`demo_gradio.py`)

## 2) Dataset Structure (DUTS recommended)

Use one dataset (DUTS, ECSSD, MSRA10K, SALICON). Arrange files like:

```text
data/
  DUTS-TR/
    images/
      xxx.jpg
    masks/
      xxx.png
  DUTS-TE/
    images/
      yyy.jpg
    masks/
      yyy.png
```

Important: image and mask filenames should have matching base names (`xxx.jpg` with `xxx.png` is supported).

## 3) Train (TR -> train/val, with checkpoint resume)

```bash
python train.py --train_images_dir data/DUTS-TR/images --train_masks_dir data/DUTS-TR/masks --image_size 128 --epochs 20 --resume
```

Includes:

- train/val split from DUTS-TR (default val ratio = 15%)
- normalization to 0-1
- augmentations (flip, random crop, brightness)
- custom BCE + 0.5 * (1 - IoU) loss
- early stopping
- checkpoint save/resume (`checkpoints/latest.pt`)
- best model save (`checkpoints/best.pt`)

## 4) Evaluate on official test split (DUTS-TE) + Visualize

```bash
python evaluate.py --test_images_dir data/DUTS-TE/images --test_masks_dir data/DUTS-TE/masks --checkpoint checkpoints/best.pt --num_visuals 8
```

Outputs:

- IoU, Precision, Recall, F1, MAE
- saved sample visualizations in `outputs/`

## 5) Demo (Gradio)

```bash
python demo_gradio.py
```

For deployment (e.g., Hugging Face Spaces), use:

```bash
python app.py
```

`app.py` uses **PyTorch** for resizing only (no OpenCV), which avoids broken `cv2` wheels on some cloud runtimes (e.g. Python 3.14).

- On **Hugging Face Spaces**: set the Space dependency file to **`requirements_app.txt`** (or copy it to `requirements.txt` in the Space repo) so the build stays minimal and skips OpenCV.
- **Streamlit Cloud** runs **Streamlit** apps (`streamlit run ...`), not Gradio. Use **Hugging Face Spaces (Gradio)** for this demo as-is, or add a separate Streamlit wrapper if your course requires Streamlit specifically.

**If the Space “loads forever”:** free tiers often wake from cold sleep; PyTorch plus a ~90 MB `best.pt` can take **1–3 minutes** on CPU. This repo loads weights in a **background thread** after import so the page can appear sooner—first inference may still wait until weights are ready.

`app.py` supports:
- `CHECKPOINT_PATH` env var (default: `checkpoints/best.pt`)
- `IMAGE_SIZE` env var (default: `128`)

Shows:

- input image
- predicted saliency mask
- overlay
- inference time per image

## 6) Suggested Improvements for Report

Run baseline first, then create at least two improved variants:

1. Baseline: current architecture and defaults.
2. Improvement A: increase depth/channels.
3. Improvement B: stronger augmentations + tuned learning rate.

Create a result table in your report:

- Model variant
- IoU
- Precision
- Recall
- F1
- MAE
- Notes

## 7) Dataset Inspection Evidence

Generate dataset counts + sample image/mask pairs:

```bash
python inspect_dataset.py --tr_images_dir data/DUTS-TR/images --tr_masks_dir data/DUTS-TR/masks --te_images_dir data/DUTS-TE/images --te_masks_dir data/DUTS-TE/masks
```

This writes:

- `outputs_dataset_inspection/dataset_report.txt`
- sample visualizations in `outputs_dataset_inspection/`

## 8) Run Baseline + Two Improvements Automatically

```bash
python experiments.py --tr_images_dir data/DUTS-TR/images --tr_masks_dir data/DUTS-TR/masks --te_images_dir data/DUTS-TE/images --te_masks_dir data/DUTS-TE/masks --epochs 15
```

This writes a comparison table to:

- `experiments/results.csv`

