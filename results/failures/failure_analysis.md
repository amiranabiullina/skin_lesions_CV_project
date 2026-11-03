# Failure analysis

Figure: `results/failures/failure_cases.png`, made by `scripts/plot_failures.py` (ground truth = green,
U-Net seed 42 = blue, DeepLabV3+ seed 1 = orange). All numbers are on the 389-image validation split.
All four cases fail in every run, so the choice of the shown runs does not change the picture.

## How the cases were chosen
An image counts as a failure if IoU < 0.65 (the ISIC threshold). 19 images fail in all three U-Net seeds, 18 in all
three DeepLabV3+ seeds, and 13 fail in all six runs. We picked four of these 13 with different causes, so that each case is a
property of the image and not an accident of one training run. Two images that looked like strong architecture
effects in seed 42 turned out to be seed noise and were dropped: ISIC_0015102 (U-Net Dice 0.55 in seed 42, but 0.99 and
0.94 in seeds 1 and 2) and ISIC_0012127 (U-Net 0.83 / 0.20 / 0.71, DeepLabV3+ 0.30 / 0.87 / 0.26).

Dice of the four cases in every run:

| image | diagnosis, area | U-Net s42 / s1 / s2 | DeepLabV3+ s42 / s1 / s2 |
|---|---|---|---|
| (a) ISIC_0011117 | nevus, 46% | 0.39 / 0.39 / 0.43 | 0.38 / 0.44 / 0.42 |
| (b) ISIC_0013567 | melanoma in situ, 4.1% | 0.01 / 0.60 / 0.03 | 0.01 / 0.04 / 0.00 |
| (c) ISIC_0013966 | nevus, 2.7% | 0.26 / 0.33 / 0.13 | 0.25 / 0.26 / 0.18 |
| (d) ISIC_0012160 | nevus, 1.8% | 0.61 / 0.66 / 0.67 | 0.53 / 0.55 / 0.58 |

## (a) ISIC_0011117: the annotation includes a pale halo
Both models outline the dark pigmented centre of the nevus almost identically, while the ground truth is a coarse
polygon that also covers a wide, barely coloured zone around it. The models follow the visible pigment border, so
they under-segment by about half of the area in all six runs. This is an annotation ambiguity rather than a model error:
where the lesion ends is not visible in the image, and the ISIC masks were drawn by different annotators with different
habits (some masks are polygons, some follow the border closely).

## (b) ISIC_0013567: the model picks the wrong object
The annotated lesion is a faint pinkish area with almost no pigment. Both models instead segment a small brown spot next
to it, which has much higher contrast; the pigmented ruler mark at the bottom is correctly ignored. The training data are
dominated by brown, well-pigmented lesions, so the models learned "darkest compact blob = lesion". A weakly pigmented
(amelanotic-looking) melanoma in situ does not fit this pattern. Only one of six runs finds part of the lesion
(U-Net seed 1, Dice 0.60).

## (c) ISIC_0013966: an artefact edge is taken as the lesion border
The image is taken through a contact plate with immersion fluid, and a fold of the fluid forms a large round region
around the small lesion; the skin inside it is also covered with purple marker dots. The models segment most of this
region instead of the small lesion, so the prediction is several times too large (Dice 0.13 to 0.33). The fluid edge is
the strongest closed contour in the image, and the dots look like pigment network. Light, low-contrast lesions inside
such bubbles or folds are a frequent failure of both models (another example: ISIC_0013360).

## (d) ISIC_0012160: small light lesion with a diffuse border
A small (1.8% of the image), light-brown nevus with a gradual border on similarly coloured skin. Both models
over-segment it, and DeepLabV3+ is worse in every seed (Dice 0.53 to 0.58 vs 0.61 to 0.67 for U-Net). This is the
systematic small-lesion effect seen in the whole validation set (lesions under 5% of the image: DeepLabV3+ -0.008 Dice
over three seeds): DeepLabV3+ predicts the mask at 1/4 of the input resolution (64x64 for a 256x256 input) and upsamples it
bilinearly, with only one skip connection from an early encoder stage, so on a lesion about 30 pixels wide it cannot
place the border as precisely as U-Net, which has skip connections at every scale.

## Boundary errors
Over three seeds, DeepLabV3+ has fewer extreme boundary errors: the 95th percentile of HD95 is 43.1 ± 1.7 px against
47.5 ± 4.2 px for U-Net (lower in two of three seeds). It does not have fewer failed images (IoU < 0.65: 33.3 ± 2.1 vs
32.3 ± 0.6). So the global context of DeepLabV3+ reduces the size of the worst border mistakes but does not fix the
images where the wrong region is segmented, as in cases (b) and (c).

## Limitations
- The causes above are read from the images; we did not test them (for example, by removing the fluid edge).
- All results are on our validation split; the official ISIC test set was not used.
- The four cases are a small, hand-picked sample of the 13 shared failures.
