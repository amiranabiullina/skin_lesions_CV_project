# Failure analysis (Andre, seed 42)

Figure: `results/failures/failure_cases.png` (GT=green, U-Net=blue, DLV3+=orange).

## Case 1 — ISIC_0013567 (melanoma in situ, < 5% area)
Dice: U-Net 0.010, DLV3+ 0.010. Both models miss the lesion entirely. The lesion is tiny (about 4% of the image) and sits at the image border with very low contrast against the surrounding skin. DLV3+ decodes at 1/4 input resolution, so a sub-10-pixel lesion is smoothed into nothing; U-Net at full resolution still fails, suggesting ambiguous annotation or a lesion indistinguishable from skin texture.

## Case 2 — ISIC_0011117 (nevus, 20–50% area)
Dice: U-Net 0.394, DLV3+ 0.378. Both models mis-segment a medium-size nevus. Visual inspection suggests strong plate glare / specular reflection and low border contrast — the nevus bleeds into the surrounding skin, and both decoders either under-segment or shift the mask. Hair shafts and a possible ruler artifact further confuse the encoders.

## Case 3 — ISIC_0012127 (nevus, < 5% area, 1.8%)
Dice: U-Net 0.833, DLV3+ 0.295. U-Net captures the tiny nevus well; DLV3+ almost completely loses it (Dice drops to 0.29). Cause: DLV3+ produces its decoder features at 1/4 resolution (16×16 on a 256 input), so a lesion spanning only a few pixels disappears under average pooling. U-Net keeps full-resolution skip connections and preserves small objects.

## Case 4 — ISIC_0015102 (nevus, > 50% area)
Dice: U-Net 0.554, DLV3+ 0.985. DLV3+ segments the large nevus nearly perfectly; U-Net breaks it into fragments (Dice 0.55). Likely cause: U-Net skip connections from early encoder stages inject high-frequency texture (skin patterns, vignette) into the decoder, fragmenting large homogeneous regions. DLV3+'s ASPP aggregates multi-scale context and smooths over local texture.

## Boundaries (section 7)
HD95 p95: 52.4 px (U-Net) vs 41.6 px (DLV3+) — DLV3+ has more stable borders on the worst 5% of images.
IoU < 0.65: 20 images fail for both models; 12 only U-Net; 15 only DLV3+. Neither model is free of gross boundary errors.

## Stability across seeds (section 6)
Seeds 1 and 2 are not yet in the repository (only seed 42 is available). Once `results/unet_r34_seed1|2/` and `results/deeplabv3plus_r34_seed1|2/` are pushed, per-image `per_image_val.csv` should be checked to confirm that these failures (low Dice across all three seeds) are data problems, not training noise.