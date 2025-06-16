import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure



def compute_segmentation_metrics(pred: np.ndarray,
                                 gt: np.ndarray,
                                 spacing=(1.0, 1.0, 1.0)) -> dict:
    """
    Compute a suite of similarity metrics between a predicted mask and ground truth mask.

    Parameters
    ----------
    pred : np.ndarray
        Binary or probabilistic predicted mask (same shape as gt).
    gt : np.ndarray
        Binary ground truth mask.
    spacing : tuple of floats
        Physical spacing of voxels along each dimension.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
          - Dice
          - Jaccard (IoU)
          - Precision
          - Recall
          - F1
          - RVD (Relative Volume Difference)
          - Hausdorff (max)
          - HD95 (95th percentile hausdorff)
          - ASSD (Average Symmetric Surface Distance)
    """

    # Binarize inputs
    pred_bin = (pred > 0).astype(bool)
    gt_bin   = (gt > 0).astype(bool)

    # Confusion counts
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()

    # Overlap metrics
    dice      = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 1.0
    jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    rvd       = (pred_bin.sum() - gt_bin.sum()) / gt_bin.sum() if gt_bin.sum() > 0 else 0.0

    # Surface extraction
    struct = generate_binary_structure(rank=pred_bin.ndim, connectivity=1)
    pred_border = np.logical_xor(pred_bin, binary_erosion(pred_bin, structure=struct))
    gt_border   = np.logical_xor(gt_bin,   binary_erosion(gt_bin,   structure=struct))

    # Distance transforms
    dt_gt   = distance_transform_edt(~gt_border,   sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_gt = dt_gt[pred_border]
    d_gt_pred = dt_pred[gt_border]

    # Hausdorff distances
    hausdorff = max(d_pred_gt.max() if d_pred_gt.size else 0.0,
                    d_gt_pred.max() if d_gt_pred.size else 0.0)
    hd95      = max(np.percentile(d_pred_gt, 95) if d_pred_gt.size else 0.0,
                    np.percentile(d_gt_pred,   95) if d_gt_pred.size else 0.0)

    # Average Symmetric Surface Distance
    total_dist = (d_pred_gt.sum() + d_gt_pred.sum())
    count_dist = (d_pred_gt.size + d_gt_pred.size)
    assd = total_dist / count_dist if count_dist > 0 else 0.0

    return {
        "Dice": dice,
        "Jaccard": jaccard,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "RVD": rvd,
        "Hausdorff": hausdorff,
        "HD95": hd95,
        "ASSD": assd
    }
