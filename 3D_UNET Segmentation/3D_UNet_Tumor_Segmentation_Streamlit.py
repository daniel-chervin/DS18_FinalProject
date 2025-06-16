import os
from git import Repo
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import streamlit as st


#Clone to local data repo on streamlit cloud

REPO_URL    = "https://github.com/daniel-chervin/DS18_FinalProject.git"
CLONE_DIR   = "repo_clone"      # temporary local clone
DATA_SUBDIR = os.path.join("3D_UNET Segmentation", "inference_test")            # the folder inside your repo with the data

@st.cache_resource(show_spinner=False)
def init_data_folder():
    """Clone the repo once per session and return the full path to the data folder."""
    if not os.path.isdir(CLONE_DIR):
        # Shallow clone just the tip of the default branch
        Repo.clone_from(REPO_URL, CLONE_DIR, multi_options=["--depth=1"])
    data_path = os.path.join(CLONE_DIR, DATA_SUBDIR)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Expected data folder at {data_path}")
    return data_path

# trigger clone (or pull) once
data_path = init_data_folder()
st.write(f"🔍 Loaded data from `{data_path}`")

# test: list your files
#for fname in sorted(os.listdir(data_path)):
#    st.write("- ", fname)


# Directories
INPUT_DIR = data_path # '/content/data/T1_tumor_eval_resampled/'
PRED_DIR  = data_path # '/content/drive/MyDrive/00-DataScience_BIU/Final Project/3D_UNet_Segmentation/'
GT_DIR    = data_path # '/content/data/T1_labels_resampled/'  # update if needed

# Collect case basenames
cases = [f.replace('_T1.nii.gz', '') for f in os.listdir(INPUT_DIR) if f.endswith('_T1.nii.gz')]
cases = sorted(cases)

st.title('Interactive Brain Tumor Segmentation Viewer')

# Cache loading of all volumes into memory
@st.cache_data
def load_all_volumes(input_dir, pred_dir, gt_dir):
    cases = [f.replace('_T1.nii.gz', '') for f in os.listdir(input_dir) if f.endswith('_T1.nii.gz')]
    cases = sorted(cases)
    mri_dict = {}
    pred_dict = {}
    gt_dict = {}
    for case in cases:
        # Load once and store arrays of shape [Z, Y, X]
        img_path  = os.path.join(input_dir, f"{case}_T1.nii.gz")
        pred_path = os.path.join(pred_dir,  f"{case}_predict_seg.nii.gz")
        gt_path   = os.path.join(gt_dir,    f"{case}.nii.gz")
        mri = sitk.ReadImage(img_path)
        pred = sitk.ReadImage(pred_path)
        gt = sitk.ReadImage(gt_path)
        mri_dict[case]  = sitk.GetArrayFromImage(mri)
        pred_dict[case] = sitk.GetArrayFromImage(pred)
        gt_dict[case]   = sitk.GetArrayFromImage(gt)
    return cases, mri_dict, pred_dict, gt_dict

# Load all volumes once
cases, mri_vols, pred_vols, gt_vols = load_all_volumes(INPUT_DIR, PRED_DIR, GT_DIR)

# Sidebar controls
st.sidebar.header('Controls')
selected_case = st.sidebar.selectbox('Case', cases)
plane = st.sidebar.selectbox('Plane', ['axial', 'sagittal', 'coronal'])

# Determine slice range based on plane
vol_shape = mri_vols[selected_case].shape
if plane == 'axial':
    max_idx = vol_shape[0]
elif plane == 'sagittal':
    max_idx = vol_shape[2]
else:  # coronal
    max_idx = vol_shape[1]
slice_idx = st.sidebar.slider('Slice', 0, max_idx - 1, max_idx // 2)

# Extract selected slices
img_sl  = (mri_vols[selected_case][slice_idx, :, :] if plane == 'axial'
           else mri_vols[selected_case][:, :, slice_idx] if plane == 'sagittal'
           else mri_vols[selected_case][:, slice_idx, :])
pred_sl = (pred_vols[selected_case][slice_idx, :, :] if plane == 'axial'
           else pred_vols[selected_case][:, :, slice_idx] if plane == 'sagittal'
           else pred_vols[selected_case][:, slice_idx, :])
gt_sl   = (gt_vols[selected_case][slice_idx, :, :] if plane == 'axial'
           else gt_vols[selected_case][:, :, slice_idx] if plane == 'sagittal'
           else gt_vols[selected_case][:, slice_idx, :])

# Plot three views
fig, axes = plt.subplots(2, 2, figsize=(6, 12))
axes[0, 0].imshow(img_sl, cmap='gray')
axes[0, 0].set_title('MRI')
axes[0, 0].axis('off')

# Top-right: MRI
axes[0, 1].imshow(img_sl, cmap='gray')
axes[0, 1].set_title('MRI')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_sl, cmap='gray')
axes[1, 0].imshow(pred_sl, cmap='jet', alpha=0.5)
axes[1, 0].set_title('Prediction')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_sl, cmap='gray')
axes[1, 1].imshow(gt_sl, cmap='jet', alpha=0.5)
axes[1, 1].set_title('Ground Truth')
axes[1, 1].axis('off')

plt.tight_layout()

st.pyplot(fig)
