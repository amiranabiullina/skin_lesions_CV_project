# Live demo

A small local web app that runs both trained models side by side on dermoscopic images.
It works offline: the server uses only the Python standard library plus the packages already in
`requirements.txt`, and the page loads nothing from the internet.

```bash
source .venv/bin/activate
python demo/app.py              # opens http://127.0.0.1:7860
python demo/app.py --device cpu --port 8000 --no-browser   # other options
```

The checkpoints are read from `checkpoints/` (the paths in `configs/*.yaml`, e.g.
`checkpoints/unet_r34_best.pt` for seed 42). Seeds whose checkpoint is missing are simply not offered.

## What it shows

- **Live demo**: 13 validation images (never seen in training) in four groups: typical cases, cases where
  U-Net wins in every seed, cases where DeepLabV3+ wins in every seed, and the four failure cases from
  `results/failures/failure_analysis.md`. You can also upload your own image (no ground truth, so no metrics).
  - Views: overlay, contours, probability map, and an error map against the ground truth
    (correct / extra / missed pixels).
  - Threshold slider: masks redraw instantly, Dice / IoU / HD95 are recomputed on the server.
  - Seed selector for each model, inference time of the forward pass, agreement between the two models,
    and the Dice of the same image in all six training runs.
- **Results**: the final table (mean ± std over 3 seeds), Dice of every run, Dice by lesion size and
  diagnosis, and the figures from `results/`. Ablation results appear automatically if
  `results/ablation/*.json` / `*.png` exist.
- **How it works**: pipeline, data split, training setup, metrics, team.

The pipeline is the same as `evaluate.py`: resize to 256×256 (INTER_AREA), ImageNet normalisation,
model, sigmoid, threshold 0.5. Metrics are computed at 256×256 with `src/metrics.py`, so for the example
images the live Dice equals `results/*/per_image_val.csv` (checked: largest difference 5e-5).

Keyboard: ← / → switch image, 1–4 switch view.

## Rebuilding the examples

`demo/examples/` is committed. To rebuild it (needs the official training zip unpacked in `data/raw/` or `~/Downloads/`):

```bash
python demo/make_examples.py
```

## Suggested 2-minute demo

1. Typical melanoma (first image): both models agree with the ground truth, Dice ≈ 0.98.
2. Switch to **Errors vs GT** (key 4) and open a *U-Net wins* and a *DeepLabV3+ wins* case: the difference is
   at the border, and it holds in all three seeds (table at the bottom).
3. A failure case, e.g. the immersion-fluid bubble: both models segment the bubble instead of the lesion.
   Move the threshold slider: even at 0.9 the mask is still far too large (DeepLabV3+ seed 1: Dice 0.26 → 0.52),
   so a different threshold does not fix it.
4. Open **Results** for the final numbers.
