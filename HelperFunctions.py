import os
import SimpleITK as sitk
from glob import glob
import numpy as np

import warnings
warnings.filterwarnings('ignore')



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

def call_Split4DNIFTY():
    dataset_path = 'D:/DS18/data/BrainTumour'
    modalities = ['T1', 'T1Gd', 'T2', 'FLAIR']
    Split4DNIFTY(f'{dataset_path}/imagesTs', f'{dataset_path}/imagesTs')

# call_Split4DNIFTY()


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

call_resample()