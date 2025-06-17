import os
import SimpleITK as sitk
from glob import glob
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

import warnings
warnings.filterwarnings('ignore')

#---------------------------------------------------------------------------------------
def Split4DNIFTY(inDir:str, outDir:str):

    for modality in modalities:
        os.makedirs(os.path.join(outDir, modality), exist_ok=True) #True not raise exception if exists

    #The glob function is used to find all the file paths matching a specified pattern
    nifti_files = glob(os.path.join(inDir, '4D', '*.nii.gz'))

    #print(nifti_files)

    #process each file
    for file_path in nifti_files:
        img_4d = sitk.ReadImage(file_path)
        size = list(img_4d.GetSize()) # Expected shape: [X, Y, Z, 4]
        #extract base name
        base_name = os.path.basename(file_path).replace('.nii.gz', '')

        for i, modality in enumerate(modalities):
            extract_size = list(img_4d.GetSize())[:3] + [0]
            extract_index = [0, 0, 0, i]
            img_3d = sitk.Extract(img_4d, extract_size, extract_index)
            output_path = os.path.join(outDir, modality, f'{base_name}_{modality}.nii.gz')
            sitk.WriteImage(img_3d, output_path)
            print(f'Saved: {output_path}')

#---------------------------------------------------------------------------------------
def call_Split4DNIFTY():
    dataset_path = 'D:/DS18/data/BrainTumour'
    modalities = ['T1', 'T1Gd', 'T2', 'FLAIR']
    Split4DNIFTY(f'{dataset_path}/imagesTs', f'{dataset_path}/imagesTs')

# call_Split4DNIFTY()


#---------------------------------------------------------------------------------------
def resample_image(image, new_size):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [
        original_spacing[i] * (original_size[i] / new_size[i])
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)

#---------------------------------------------------------------------------------------
def call_resample():
    # Input directories
    input_dirs = {
        'inference_set': r'D:\pyProjects\DS18\DS18_FinalProject1\3D_UNET Segmentation\inference_test'
        # "Tumor_Labels": r'D:\DS18\data\BrainTumour\imagesTs\T1'
        # "Tumor_T1": r'D:\DS18\data\BrainTumour\imagesTr\T1'
        #"Tumor_T1Gd": r'D:\DS18\data\BrainTumour\imagesTr\T1Gd'
        # "IXI-T1": r'D:/DS18/data/IXI-T1',
        # "IXI-T2": r'D:/DS18/data/IXI-T2'
    }

    # Output directories
    output_dirs = {
        'inference_set': r'D:\pyProjects\DS18\DS18_FinalProject1\3D_UNET Segmentation\inference_test1'
        # "Tumor_Labels": r'D:\DS18\data\BrainTumour\imagesTs\T1_resampled'
        # "Tumor_T1": r'D:\DS18\data\BrainTumour\imagesTr\T1_resampled'
        #"Tumor_T1Gd": r'D:\DS18\data\BrainTumour\imagesTr\T1Gd_resampled'
        # "IXI-T1": r'D:/DS18/data/IXI-T1_resampled',
        # "IXI-T2": r'D:/DS18/data/IXI-T2_resampled'
    }

    # Target shape
    target_size = (128, 128, 128)  # z, y, x

    # Create output directories
    for out_dir in output_dirs.values():
        os.makedirs(out_dir, exist_ok=True)

    # Resample all images in both folders
    for modality, input_dir in input_dirs.items():
        for file_path in glob(os.path.join(input_dir, "*.nii.gz")):
            img = sitk.ReadImage(file_path)
            resampled_img = resample_image(img, target_size)

            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dirs[modality], filename)
            sitk.WriteImage(resampled_img, output_path)
            print(f"✅ Resampled {filename} to {target_size}")

#call_resample()

#---------------------------------------------------------------------------------------
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


def call_compute_segmentation_metrics():

    base_dir = r'D:\pyProjects\DS18\DS18_FinalProject1\3D_UNET Segmentation\inference_test'
    gt_vol = sitk.ReadImage(os.path.join(base_dir, r'BRATS_463.nii.gz'))
    pred_vol = sitk.ReadImage(os.path.join(base_dir, r'BRATS_463_predict_seg.nii.gz'))

    gt_np = sitk.GetArrayFromImage(gt_vol)
    pred_np = sitk.GetArrayFromImage(pred_vol)

    stats = compute_segmentation_metrics(pred_np, gt_np)
    print(stats)

#call_compute_segmentation_metrics()


import openai

# 1) Install the OpenAI client if you haven’t already:
#    pip install openai

# 2) Set your API key in the environment:
#    export OPENAI_API_KEY="sk-…"

# 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

#openai.api_key = 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
client = openai.OpenAI(api_key="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")

def analyze_metrics_with_gpt(metrics: dict, model="gpt-4o-mini") -> str:
    """
    Send the metrics dict to ChatGPT and return its analysis.
    """
    # Build a system + user message pair
    messages = [
        {
            "role": "system",
            "content": (
                "You are a radiology/ML expert. "
                "When given segmentation metrics, you provide a concise, "
                "clear evaluation of strengths, weaknesses, and suggested next steps."
            )
        },
        {
            "role": "user",
            "content": (
                "Here are my segmentation results:\n\n"
                f"{metrics}\n\n"
                "Please interpret these numbers and suggest improvements."
            )
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=300,
    )

    '''
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=300,
    )
    '''
    return response.choices[0].message.content.strip()

'''
if __name__ == "__main__":
    metrics = {
        "Dice": 0.9388,
        "Jaccard": 0.8847,
        "Precision": 0.9689,
        "Recall": 0.9105,
        "F1": 0.9388,
        "RVD": -0.0603,
        "Hausdorff": 6.0,
        "HD95": 1.7321,
        "ASSD": 0.5029
    }
    analysis = analyze_metrics_with_gpt(metrics)
    print("ChatGPT Analysis:\n", analysis)
'''
