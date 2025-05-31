import os
import SimpleITK as sitk
from glob import glob


dataset_path = 'D:/DS18/data/BrainTumour'
modalities = ['T1', 'T1Gd', 'T2', 'FLAIR']

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





# Split4DNIFTY(f'{dataset_path}/imagesTs', f'{dataset_path}/imagesTs')
