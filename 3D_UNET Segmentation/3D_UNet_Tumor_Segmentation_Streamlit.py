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

# Example: list your files
for fname in sorted(os.listdir(data_path)):
    st.write("- ", fname)


# Directories
INPUT_DIR = '/content/data/T1_tumor_eval_resampled/'
PRED_DIR  = '/content/drive/MyDrive/00-DataScience_BIU/Final Project/3D_UNet_Segmentation/'
GT_DIR    = '/content/data/T1_labels_resampled/'  # update if needed

# Collect case basenames
cases = [f.replace('_T1.nii.gz', '') for f in os.listdir(INPUT_DIR) if f.endswith('_T1.nii.gz')]
cases = sorted(cases)

st.title('Interactive Brain Tumor Segmentation Viewer')

# Sidebar controls
st.sidebar.header('Controls')
selected_case = st.sidebar.selectbox('Case', cases)
plane = st.sidebar.selectbox('Plane', ['axial', 'sagittal', 'coronal'])

# Load volumes
@st.cache(allow_output_mutation=True)
def load_volume(dir_path, base, suffix):
    path = os.path.join(dir_path, f"{base}{suffix}")
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

img_vol = load_volume(INPUT_DIR, selected_case, '_T1.nii.gz')
pred_vol = load_volume(PRED_DIR,  selected_case, '_predict_seg.nii.gz')
gt_vol   = load_volume(GT_DIR,    selected_case, '.nii.gz')

# Determine slice max index
if plane == 'axial':
    max_idx = img_vol.shape[0]
elif plane == 'sagittal':
    max_idx = img_vol.shape[2]
else:
    max_idx = img_vol.shape[1]

slice_idx = st.sidebar.slider('Slice', min_value=0, max_value=max_idx-1, value=max_idx//2)

# Extract slice
if plane == 'axial':
    img_sl = img_vol[slice_idx, :, :]
    pred_sl= pred_vol[slice_idx, :, :]
    gt_sl  = gt_vol[slice_idx, :, :]
elif plane == 'sagittal':
    img_sl = img_vol[:, :, slice_idx]
    pred_sl= pred_vol[:, :, slice_idx]
    gt_sl  = gt_vol[:, :, slice_idx]
else:  # coronal
    img_sl = img_vol[:, slice_idx, :]
    pred_sl= pred_vol[:, slice_idx, :]
    gt_sl  = gt_vol[:, slice_idx, :]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(6, 12))
axes[0].imshow(img_sl, cmap='gray'); axes[0].set_title('MRI'); axes[0].axis('off')
axes[1].imshow(img_sl, cmap='gray'); axes[1].imshow(pred_sl, cmap='jet', alpha=0.5);
axes[1].set_title('Prediction'); axes[1].axis('off')
axes[2].imshow(img_sl, cmap='gray'); axes[2].imshow(gt_sl, cmap='jet', alpha=0.5);
axes[2].set_title('Ground Truth'); axes[2].axis('off')
plt.tight_layout()

st.pyplot(fig)
