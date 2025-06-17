import os

#from IPython.core.pylabtools import figsize
from git import Repo
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure


from HelperFunctions import compute_segmentation_metrics, analyze_metrics_with_gpt

import warnings
warnings.filterwarnings('ignore')

# Hide Streamlit pyplot global use deprecation warning
#st.set_option('deprecation.showPyplotGlobalUse', False)

# Suppress Streamlit deprecation warnings by raising the logger level
#st.set_option('logger.level', 'error')


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
#@st.cache(allow_output_mutation=True)
@st.cache_data
def load_all_volumes(input_dir, pred_dir, gt_dir):
    cases = [f.replace('_T1.nii.gz', '') for f in os.listdir(input_dir) if f.endswith('_T1.nii.gz')]
    cases = sorted(cases)
    mri_dict, pred_dict, gt_dict = {}, {}, {}
    for case in cases:
        mri_dict[case] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, f"{case}_T1.nii.gz")))
        pred_dict[case] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_dir,  f"{case}_predict_seg.nii.gz")))
        gt_dict[case]   = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir,    f"{case}.nii.gz")))
    return cases, mri_dict, pred_dict, gt_dict

cases, mri_vols, pred_vols, gt_vols = load_all_volumes(INPUT_DIR, PRED_DIR, GT_DIR)

# Sidebar controls
st.sidebar.header('Controls')
selected_case = st.sidebar.selectbox('Case', cases)
plane = st.sidebar.selectbox('Plane', ['axial', 'sagittal', 'coronal'])

# Determine slice range
shape = mri_vols[selected_case].shape
if plane == 'axial':
    max_idx = shape[0]
elif plane == 'sagittal':
    max_idx = shape[2]
else:
    max_idx = shape[1]
slice_idx = st.sidebar.slider('Slice', 0, max_idx-1, max_idx//2)

# Extract the slice based on selected plane
def get_slice(vol, plane, idx):
    if plane == 'axial':
        return vol[idx, :, :]
    if plane == 'sagittal':
        return vol[:, :, idx]
    return vol[:, idx, :]

img_sl  = get_slice(mri_vols[selected_case], plane, slice_idx)
pred_sl = get_slice(pred_vols[selected_case], plane, slice_idx)
gt_sl   = get_slice(gt_vols[selected_case], plane, slice_idx)

# Plot 2x2 views
fig, axes = plt.subplots(2,2, figsize=(10,10))

# Helper to plot with equal aspect ratio
def plot_img(ax, base_img, overlay=None, cmap='gray', overlay_cmap='jet'):
    im = ax.imshow(base_img, cmap=cmap, origin='lower', aspect='equal')
    if overlay is not None:
        ax.imshow(overlay, cmap=overlay_cmap, alpha=0.5, origin='lower', aspect='equal')


# Top-left: MRI + Ground Truth
plot_img(axes[0, 0], img_sl, overlay=gt_sl)
#axes[0, 0].imshow(img_sl, cmap='gray', origin='lower')
#axes[0, 0].imshow(gt_sl, cmap='jet', alpha=0.5, origin='lower')
axes[0, 0].set_title('Ground Truth')
axes[0, 0].axis('off')

# Top-right: MRI + Prediction
plot_img(axes[0, 1], img_sl, overlay=pred_sl)
#axes[0, 1].imshow(img_sl, cmap='gray', origin='lower')
#axes[0, 1].imshow(pred_sl, cmap='jet', alpha=0.5, origin='lower')
axes[0, 1].set_title('Prediction')
axes[0, 1].axis('off')

# Bottom-left: MRI
plot_img(axes[1, 0], img_sl)
#axes[1, 0].imshow(img_sl, cmap='gray', origin='lower')
axes[1, 0].set_title('MRI')
axes[1, 0].axis('off')

# Bottom-right: Combined Overlay (Prediction vs Ground Truth)
#diff = np.zeros_like(img_sl)
#diff[(pred_sl > 0) & (gt_sl == 0)] = 1  # false positives
plot_img(axes[1, 1], img_sl)
#axes[1, 1].imshow(img_sl, cmap='gray', origin='lower')
axes[1, 1].imshow(pred_sl, cmap='Reds', alpha=0.5, origin='lower', aspect='equal')
axes[1, 1].imshow(gt_sl, cmap='Blues', alpha=0.5, origin='lower', aspect='equal')
axes[1, 1].set_title('Pred (red) vs GT (blue)')
axes[1, 1].axis('off')

plt.tight_layout()

st.pyplot(fig)

# Compute and display metrics
metrics = compute_segmentation_metrics(pred_vols[selected_case], gt_vols[selected_case])
st.subheader("Segmentation Metrics Summary (Whole Volume)")
#print(metrics)
st.table(metrics)

import openai
test_key = st.secrets["openai"]["test_key"]
#openai_key = st.secrets["openai"]["api_key"]
#analysis = analyze_metrics_with_gpt(openai_key, metrics, "gpt-4o-mini")
st.subheader(f"Metrics analysis: {openai.__version__}")



