import os
from git import Repo
import numpy as np
import SimpleITK as sitk
import streamlit as st
import random
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

from HelperFunctions import compute_segmentation_metrics, analyze_metrics_with_gpt

import warnings
warnings.filterwarnings('ignore')

# Add Plotly for interactive pan & zoom
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Configuration for cloning data repository
REPO_URL    = "https://github.com/daniel-chervin/DS18_FinalProject.git"
CLONE_DIR   = "repo_clone"
DATA_SUBDIR = os.path.join("3D_UNET Segmentation", "inference_test")

@st.cache_resource(show_spinner=False)
def init_data_folder():
    """Clone the repo once per session and return the full path to the data folder."""
    if not os.path.isdir(CLONE_DIR):
        Repo.clone_from(REPO_URL, CLONE_DIR, multi_options=["--depth=1"])
    data_path = os.path.join(CLONE_DIR, DATA_SUBDIR)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Expected data folder at {data_path}")
    return data_path

# Initialize and load data
data_path = init_data_folder()
st.write(f"🔍 Loaded data from `{data_path}`")

INPUT_DIR = data_path
PRED_DIR  = os.path.join(data_path, 'pred_seg')
GT_DIR    = data_path

# Load all volumes into cache
@st.cache_data
def load_all_volumes(input_dir, pred_dir, gt_dir):
    cases = [f.replace('_T1.nii.gz', '') for f in os.listdir(input_dir) if f.endswith('_T1.nii.gz')]
    cases = sorted(cases)
    mri_dict, pred_dict, gt_dict = {}, {}, {}
    for case in cases:
        mri_dict[case]  = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir,      f"{case}_T1.nii.gz")))
        pred_dict[case] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_dir,       f"{case}_predict_seg.nii.gz")))
        gt_dict[case]   = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_dir,         f"{case}.nii.gz")))
    return cases, mri_dict, pred_dict, gt_dict

cases, mri_vols, pred_vols, gt_vols = load_all_volumes(INPUT_DIR, PRED_DIR, GT_DIR)

# Sidebar controls
st.sidebar.header('Controls')
selected_case = st.sidebar.selectbox('Case', cases)
plane         = st.sidebar.selectbox('Plane', ['axial', 'sagittal', 'coronal'])

# Track case changes
if "selected_case" not in st.session_state:
    st.session_state.selected_case = selected_case
if st.session_state.selected_case != selected_case:
    st.session_state.analysis_md = None
    st.session_state.selected_case = selected_case

# Determine slice index
shape = mri_vols[selected_case].shape
if plane == 'axial':   max_idx = shape[0]
elif plane == 'sagittal': max_idx = shape[2]
else: max_idx = shape[1]
slice_idx = st.sidebar.slider('Slice', 0, max_idx-1, max_idx//2)

# Extract 2D slices

def get_slice(vol, plane, idx):
    if plane == 'axial':    return vol[idx, :, :]
    if plane == 'sagittal': return vol[:, :, idx]
    return vol[:, idx, :]

img_sl  = get_slice(mri_vols[selected_case], plane, slice_idx)
pred_sl = get_slice(pred_vols[selected_case], plane, slice_idx)
gt_sl   = get_slice(gt_vols[selected_case], plane, slice_idx)

# Prepare interactive Plotly viewer
# Normalize grayscale for RGB base
img_norm = ((img_sl - img_sl.min()) / np.ptp(img_sl) * 255).astype(np.uint8)
base_rgb = np.stack([img_norm]*3, axis=-1)

# RGBA overlay for Ground Truth: red(1), green(2), blue(3)
overlay_gt_rgba = np.stack([
    np.where(gt_sl == 1, 255, 0),  # Red channel
    np.where(gt_sl == 2, 255, 0),  # Green channel
    np.where(gt_sl == 3, 255, 0),  # Blue channel
    np.where(gt_sl > 0, int(0.5*255), 0)  # Alpha channel
], axis=-1).astype(np.uint8)

# RGBA overlay for Prediction: red(1), green(2), blue(3)
overlay_pred_rgba = np.stack([
    np.where(pred_sl == 1, 255, 0),
    np.where(pred_sl == 2, 255, 0),
    np.where(pred_sl == 3, 255, 0),
    np.where(pred_sl > 0, int(0.5*255), 0)
], axis=-1).astype(np.uint8)

# Combined Pred vs GT overlay: Pred in red, GT in blue
overlay_pred_only_rgba = np.stack([
    np.where(pred_sl > 0, 255, 0),
    np.zeros_like(pred_sl),
    np.zeros_like(pred_sl),
    np.where(pred_sl > 0, int(0.5*255), 0)
], axis=-1).astype(np.uint8)
overlay_gt_only_rgba = np.stack([
    np.zeros_like(gt_sl),
    np.zeros_like(gt_sl),
    np.where(gt_sl > 0, 255, 0),
    np.where(gt_sl > 0, int(0.5*255), 0)
], axis=-1).astype(np.uint8)

# Build 2x2 Plotly figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Ground Truth', 'Prediction', 'MRI', 'Pred (red) vs GT (blue)')
)
# Layer base RGB then overlay RGBA in each subplot
def add_layer(row, col, overlay_rgba=None):
    fig.add_trace(go.Image(z=base_rgb), row=row, col=col)
    if overlay_rgba is not None:
        fig.add_trace(go.Image(z=overlay_rgba), row=row, col=col)

add_layer(1, 1, overlay_gt_rgba)
add_layer(1, 2, overlay_pred_rgba)
add_layer(2, 1, None)
add_layer(2, 2, None)
# On bottom-right, add combined overlays
fig.add_trace(go.Image(z=base_rgb), row=2, col=2)
fig.add_trace(go.Image(z=overlay_pred_only_rgba), row=2, col=2)
fig.add_trace(go.Image(z=overlay_gt_only_rgba),   row=2, col=2)

# Enable pan & zoom and size
fig.update_layout(
    dragmode='pan',
    autosize=False,
    width=1400, height=900,
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False
)

# Render interactive chart
st.plotly_chart(fig, use_container_width=True, height=900)

# Compute and display metrics
metrics = compute_segmentation_metrics(pred_vols[selected_case], gt_vols[selected_case])
st.subheader('Segmentation Metrics Summary (Whole Volume)')
st.table(metrics)

# GPT analysis
if st.session_state.get('analysis_md') is None:
    st.session_state.analysis_md = random.randint(1, 1000)
if st.session_state.analysis_md:
    st.markdown(st.session_state.analysis_md, unsafe_allow_html=True)
