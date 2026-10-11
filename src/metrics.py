"""Per-image segmentation metrics on binary masks (numpy bool arrays, H x W).

Conventions (documented in the report):
  * Dice / IoU: if both prediction and ground truth are empty the score is 1, otherwise
    the usual formula (an empty prediction on a non-empty mask scores 0).
  * HD95 (pixels, at the evaluation resolution): 95th percentile of the symmetric
    boundary-to-boundary distances. Both empty -> 0. Exactly one empty -> the image
    diagonal, i.e. the worst possible distance, so failures are penalised, not skipped.
"""
import numpy as np
from scipy import ndimage


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2.0 * inter / total


def iou_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def _boundary(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def hd95(pred, gt):
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.hypot(*gt.shape))
    bp, bg = _boundary(pred), _boundary(gt)
    # distance from every pixel to the nearest boundary pixel of the other mask
    dt_gt = ndimage.distance_transform_edt(~bg)
    dt_pred = ndimage.distance_transform_edt(~bp)
    dists = np.concatenate([dt_gt[bp], dt_pred[bg]])
    return float(np.percentile(dists, 95))


def all_metrics(pred, gt):
    return {"dice": dice_score(pred, gt), "iou": iou_score(pred, gt), "hd95": hd95(pred, gt)}
