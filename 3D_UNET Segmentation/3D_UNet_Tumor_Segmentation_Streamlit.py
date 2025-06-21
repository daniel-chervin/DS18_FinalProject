import os
from git import Repo
import numpy as np
import SimpleITK as sitk
import streamlit as st
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
# Normalize grayscale to uint8
img_norm = ((img_sl - img_sl.min()) / img_sl.ptp() * 255).astype(np.uint8)
base_rgb = np.stack([img_norm]*3, axis=-1)

# Ground Truth overlay (multicolor)
overlay_gt = np.zeros_like(base_rgb)
overlay_gt[gt_sl==1] = [255, 0, 0]
overlay_gt[gt_sl==2] = [0, 255, 0]
overlay_gt[gt_sl==3] = [0, 0, 255]
alpha_gt = (gt_sl > 0).astype(np.float32) * 0.5
overlay_gt_rgba = np.dstack([
    overlay_gt[...,0], overlay_gt[...,1], overlay_gt[...,2],
    (alpha_gt * 255).astype(np.uint8)
])

# Prediction overlay (multicolor)
overlay_pred = np.zeros_like(base_rgb)
overlay_pred[pred_sl==1] = [255, 0, 0]
overlay_pred[pred_sl==2] = [0, 255, 0]
overlay_pred[pred_sl==3] = [0, 0, 255]
alpha_pred = (pred_sl > 0).astype(np.float32) * 0.5
overlay_pred_rgba = np.dstack([
    overlay_pred[...,0], overlay_pred[...,1], overlay_pred[...,2],
    (alpha_pred * 255).astype(np.uint8)
])

# Combined overlay: Pred in red vs GT in blue
overlay_pred_only = np.zeros_like(base_rgb)
overlay_pred_only[pred_sl > 0] = [255, 0, 0]
overlay_gt_only   = np.zeros_like(base_rgb)
overlay_gt_only[gt_sl > 0]   = [0, 0, 255]
alpha_comb_pred = (pred_sl > 0).astype(np.float32) * 0.5
alpha_comb_gt   = (gt_sl   > 0).astype(np.float32) * 0.5
overlay_pred_only_rgba = np.dstack([
    overlay_pred_only[...,0], overlay_pred_only[...,1], overlay_pred_only[...,2],
    (alpha_comb_pred * 255).astype(np.uint8)
])
overlay_gt_only_rgba = np.dstack([
    overlay_gt_only[...,0], overlay_gt_only[...,1], overlay_gt_only[...,2],
    (alpha_comb_gt   * 255).astype(np.uint8)
])

# Create 2x2 subplot figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Ground Truth', 'Prediction', 'MRI', 'Pred (red) vs GT (blue)'
    )
)
# Top-left: GT overlay
fig.add_trace(go.Image(z=base_rgb),           row=1, col=1)
fig.add_trace(go.Image(z=overlay_gt_rgba),    row=1, col=1)
# Top-right: Prediction overlay
fig.add_trace(go.Image(z=base_rgb),           row=1, col=2)
fig.add_trace(go.Image(z=overlay_pred_rgba),  row=1, col=2)
# Bottom-left: MRI only
fig.add_trace(go.Image(z=base_rgb),           row=2, col=1)
# Bottom-right: Combined Pred vs GT
fig.add_trace(go.Image(z=base_rgb),                row=2, col=2)
fig.add_trace(go.Image(z=overlay_pred_only_rgba), row=2, col=2)
fig.add_trace(go.Image(z=overlay_gt_only_rgba),   row=2, col=2)

# Enable pan & zoom\ n
fig.update_layout(dragmode='pan', margin=dict(l=0, r=0, t=30, b=0), showlegend=False)

# Render interactive chart
st.plotly_chart(fig, use_container_width=True)

# Compute and display metrics\ nmetrics = compute_segmentation_metrics(pred_vols[selected_case], gt_vols[selected_case])
st.subheader('Segmentation Metrics Summary (Whole Volume)')
st.table(metrics)

# GPT analysis\ nif st.session_state.get('analysis_md') is None:
    st.session_state.analysis_md = random.randint(1, 1000)
if st.session_state.analysis_md:
    st.markdown(st.session_state.analysis_md, unsafe_allow_html=True)
