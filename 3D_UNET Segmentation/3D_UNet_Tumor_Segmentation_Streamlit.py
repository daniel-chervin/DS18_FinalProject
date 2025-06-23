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

# Set Streamlit to use full-width layout with minimal margins
#st.set_page_config(layout="wide")



# Constants for repo and data
REPO_URL = "https://github.com/daniel-chervin/DS18_FinalProject.git"
CLONE_DIR = "repo_clone"
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
PRED_DIR = os.path.join(data_path, 'pred_seg')
GT_DIR = data_path

# Load all volumes into cache
@st.cache_data
def load_all_volumes(input_dir, pred_dir, gt_dir):
    """Load MRI, prediction. GT optional."""
    files = os.listdir(input_dir)
    # Include files ending with '_T1.nii.gz' or '-T1.nii.gz'
    cases = sorted([
        f.replace('_T1.nii.gz', '').replace('-T1.nii.gz', '')
        for f in files
        if f.endswith('_T1.nii.gz') or f.endswith('-T1.nii.gz')
    ])
    mri_dict, pred_dict, gt_dict = {}, {}, {}
    for case in cases:
        # MRI file: detect suffix
        for suffix in ("_T1.nii.gz", "-T1.nii.gz"):
            mri_path = os.path.join(input_dir, f"{case}{suffix}")
            if os.path.exists(mri_path):
                mri_dict[case] = sitk.GetArrayFromImage(sitk.ReadImage(mri_path))
                break
        else:
            raise FileNotFoundError(f"No MRI volume found for case '{case}'")
        # Prediction file: accept both '_predict_seg' and '-predict_seg'
        pred_path = os.path.join(pred_dir, f"{case}_predict_seg.nii.gz")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"{case}-predict_seg.nii.gz")
        pred_dict[case] = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        # Ground truth
        gt_path = os.path.join(gt_dir, f"{case}.nii.gz")
        gt_dict[case] = (
            sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            if os.path.exists(gt_path) else None
        )
    return cases, mri_dict, pred_dict, gt_dict

cases, mri_vols, pred_vols, gt_vols = load_all_volumes(INPUT_DIR, PRED_DIR, GT_DIR)

# Sidebar: case and plane
st.sidebar.header('Controls')
selected_case = st.sidebar.selectbox('Case', cases)
plane = st.sidebar.selectbox('Plane', ['axial','sagittal','coronal'])

# Reset analysis when switching case
if st.session_state.get('selected_case') != selected_case:
    st.session_state.selected_case = selected_case
    st.session_state.analysis_md = None

# Determine slice index
shape = mri_vols[selected_case].shape
max_idx = {'axial':shape[0], 'sagittal':shape[2], 'coronal':shape[1]}[plane]
slice_idx = st.sidebar.slider('Slice', 0, max_idx-1, max_idx//2)

# Extract 2D slices
def get_slice(vol, plane, idx):
    return {'axial':vol[idx], 'sagittal':vol[:,:,idx], 'coronal':vol[:,idx]}[plane]

img_sl = get_slice(mri_vols[selected_case], plane, slice_idx)
pred_sl = get_slice(pred_vols[selected_case], plane, slice_idx)
gt_arr = gt_vols[selected_case]
gt_sl = get_slice(gt_arr, plane, slice_idx) if gt_arr is not None else None

# Prepare base RGB
img_norm = ((img_sl - img_sl.min()) / np.ptp(img_sl) * 255).astype(np.uint8)
base_rgb = np.stack([img_norm]*3, axis=-1)
black_frame = np.zeros_like(base_rgb)

# Blend function
def blend(base, overlay, alpha):
    return (base.astype(float)*(1-alpha[...,None]) + overlay.astype(float)*alpha[...,None]).astype(np.uint8)

# Build combined images
# GT overlay or black
if gt_sl is None:
    combined_gt = black_frame
    title_gt = 'No Ground Truth'
else:
    overlay_gt = np.zeros_like(base_rgb)
    for lbl,color in zip([1,2,3], [[255,0,0],[0,255,0],[0,0,255]]):
        overlay_gt[gt_sl==lbl] = color
    alpha_gt = (gt_sl>0).astype(float)*0.5
    combined_gt = blend(base_rgb, overlay_gt, alpha_gt)
    title_gt = 'Ground Truth'
# Prediction overlay
overlay_pred = np.zeros_like(base_rgb)
for lbl,color in zip([1,2,3], [[255,0,0],[0,255,0],[0,0,255]]):
    overlay_pred[pred_sl==lbl] = color
alpha_pred = (pred_sl>0).astype(float)*0.5
combined_pred = blend(base_rgb, overlay_pred, alpha_pred)
# Combined pred vs GT
if gt_sl is None:
    combined_pred_gt = combined_pred
else:
    pred_first = combined_pred
    combined_pred_gt = blend(pred_first, overlay_gt, alpha_gt)

# Create figure
fig = make_subplots(
    rows=2,cols=2,
    subplot_titles=(title_gt,'Prediction','MRI','Pred vs GT'),
    horizontal_spacing=0.01,vertical_spacing=0.1)
fig.add_trace(go.Image(z=combined_gt),1,1)
fig.add_trace(go.Image(z=combined_pred),1,2)
fig.add_trace(go.Image(z=base_rgb),2,1)
fig.add_trace(go.Image(z=combined_pred_gt),2,2)
fig.update_layout(
    dragmode='pan',autosize=False,
    width=1600,height=1100,
    margin=dict(l=0,r=0,t=80,b=0),showlegend=False)
st.plotly_chart(fig,use_container_width=True,height=1200)

# Metrics & analysis if GT exists
if gt_sl is not None:
    metrics = compute_segmentation_metrics(pred_vols[selected_case], gt_arr)
    col1,_ = st.columns([4,1])
    with col1:
        st.subheader('Segmentation Metrics Summary (Whole Volume)')
        st.table(metrics)
    if st.session_state.analysis_md is None:
        st.session_state.analysis_md = analyze_metrics_with_gpt(metrics)
    st.markdown(st.session_state.analysis_md,unsafe_allow_html=True)
else:
    st.info('Ground truth not available; skipping metrics.')
