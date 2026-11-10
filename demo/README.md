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
`checkpoints/unet_r34_best.pt` for seed 42). Each model uses its seed-42 checkpoint, or the first available seed if that one is missing.

## What it shows

One page that you scroll top to bottom:

1. **Try it**: drop an image (or click a sample). The U-Net baseline and DeepLabV3+ masks are drawn next to the
   image, with Dice / IoU / HD95, the predicted lesion area and the inference time. *Errors* colours the
   correct, extra and missed pixels.
2. **Results**: the main numbers on the 389 validation images (mean over 3 seeds), Dice by lesion size and
   the Dice of every training run.
3. **Failures**: four images that fail in all six runs; click one to run it.
4. **Method**: the pipeline in one line.

`demo/test_images/` holds the 13 example images (all from our validation split, never used for training).
Dropped into the page they are recognised, so their ground truth and scores appear. Any other dermoscopic
image works too, but then there is no ground truth and only the masks and areas are shown.

The pipeline is the same as `evaluate.py`: resize to 256×256 (INTER_AREA), ImageNet normalisation,
model, sigmoid, threshold 0.5. Metrics are computed at 256×256 with `src/metrics.py`, so for the example
images the live Dice equals `results/*/per_image_val.csv` (checked: largest difference 5e-5).

## Rebuilding the examples

`demo/examples/` is committed. To rebuild it (needs the official training zip unpacked in `data/raw/` or `~/Downloads/`):

```bash
python demo/make_examples.py
```

## Suggested 2-minute demo

1. Drop `01_typical_melanoma_...jpg`: both models match the expert mask (Dice ≈ 0.98).
2. Drop `09_deeplab_melanoma_...jpg` and switch to *Errors*: on this image DeepLabV3+ wins in all three seeds.
3. Scroll to **Results**: overall the two models are equally accurate (Dice 0.903 vs 0.901). DeepLabV3+ makes fewer
   large border errors, and U-Net is better on small lesions and 1.6x faster.
4. Scroll to **Failures** and click the fluid-bubble case: both models outline the bubble instead of the lesion.
