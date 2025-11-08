# GBM Automatic Segmentation Project

## Project Overview

This is a deep learning project focused on automatic segmentation of Glioblastoma Multiforme (GBM) brain tumors using medical imaging. The project implements and compares different approaches for tumor segmentation, with a primary supervised method using a 3D U-Net architecture and experimental unsupervised methods using Autoencoders (AE) and Variational Autoencoders (VAE).

**Live Demo:** https://ds18finalproject-danielc.streamlit.app/

## Project Purpose

- Identify and segment GBM tumor volumes from brain MRI scans
- Compare supervised (3D U-Net) vs unsupervised (AE/VAE) segmentation strategies
- Provide visual comparison and metrics through an interactive Streamlit application

## Technology Stack

### Core Deep Learning
- **PyTorch 2.2.2** - Deep learning framework
- **MONAI 1.3.0** - Medical imaging AI library (provides 3D U-Net)
- **torchvision 0.17.2** - Computer vision utilities

### Medical Imaging
- **nibabel 5.1.0** - NIfTI format handling
- **SimpleITK 2.3.1** - Medical image processing and transformations

### Data Science & Visualization
- **numpy 1.26.4** - Numerical computing
- **matplotlib 3.8.4** - Plotting
- **plotly** - Interactive visualizations
- **scipy** - Scientific computing
- **scikit-learn 1.4.2** - Metrics and utilities

### Application & Deployment
- **Streamlit** - Interactive web application
- **OpenAI 1.88.0** - GPT-based metrics analysis
- **GitPython** - Repository management for data loading

## Directory Structure

```
DS18_FinalProject/
│
├── 3D_UNET Segmentation/          # Primary supervised segmentation implementation
│   ├── inference_test/            # Test dataset for demo
│   │   ├── BRATS_*.nii.gz        # Tumor scan samples (BRATS dataset)
│   │   ├── IXI*.nii.gz           # Healthy brain samples (IXI dataset)
│   │   └── pred_seg/             # Model predictions output
│   ├── 3D_UNet_Brain_Tumor_Segmentation_multiclass_complete.ipynb
│   │                              # Main training & inference notebook
│   ├── 3D_UNET_loss-0.4394_ep-60.pth  # Trained model weights (~19MB)
│   ├── 3D_UNet_Tumor_Segmentation_Streamlit.py  # Main Streamlit app
│   ├── 3D_UNet_Tumor_Segmentation_Streamlit_no_pan-zoom.py  # Alternative UI
│   ├── HelperFunctions.py        # Utility functions
│   ├── Interactive_BrainSeg_UI.ipynb  # Interactive notebook interface
│   └── Multi-Class Training-Validation Loss.png  # Training visualization
│
├── AE_Healthy/                    # Autoencoder implementation (unsupervised)
│   ├── AE_Healthy_T1_1ch_ep.ipynb  # Training notebook
│   ├── AE_Healthy_T1_1ch_ep(46)_loss(0.0402).pt  # Model weights
│   ├── AE_043_T1_ep46_recon.nii.gz  # Reconstruction output
│   ├── AE_043_T1_ep46_tumor_anomaly_mask.nii.gz  # Anomaly detection
│   └── AE_043_T1_ep46_tumor_error_map.nii.gz    # Error map
│
├── VAE_Healthy/                   # Variational Autoencoder implementations
│   ├── VAE/                       # Standard VAE
│   │   ├── VAE_Healthy_T1_1ch_BEST.ipynb
│   │   ├── VAE_Healthy_T1_1ch_ep_epoch28_loss0.1363.pt
│   │   └── *_anomaly_mask.nii.gz, *_error_map.nii.gz
│   └── VAE_MBConv/               # VAE with MobileNet Conv blocks
│       ├── VAE_Healthy_T1_1ch_BEST_With_MBConv.ipynb
│       ├── VAE_MBConv_Healthy_T1_1ch_ep5_loss0.1088.pt
│       └── *_anomaly_mask.nii.gz, *_error_map.nii.gz
│
├── .devcontainer/                 # VS Code dev container config
│   └── devcontainer.json         # Auto-launches Streamlit app
│
├── .streamlit/                    # Streamlit configuration
│   └── config.toml               # Logger settings
│
├── requirements.txt              # Python dependencies (pip)
├── environment.yml               # Conda environment specification
├── README.md                     # Project documentation
├── ds18_GBM_Autosegmentation.pdf   # Project presentation
└── ds18_GBM_Autosegmentation.pptx  # Project slides
```

## Key Components

### 1. 3D U-Net Segmentation (Supervised - Primary Method)

**Main Files:**
- `3D_UNet_Brain_Tumor_Segmentation_multiclass_complete.ipynb` - Complete training pipeline
- `3D_UNet_Tumor_Segmentation_Streamlit.py` - Production demo app
- `HelperFunctions.py` - Utility functions for data processing and metrics

**Architecture:**
- Uses MONAI's 3D U-Net implementation
- Multi-class segmentation (4 classes: background + 3 tumor regions)
- Trained on BRATS dataset with ground truth labels
- Final model: epoch 60, loss 0.4394

**Key Functions (HelperFunctions.py):**
- `Split4DNIFTY()` - Split 4D NIFTI files into separate modalities
- `resample_image()` - Resample images to 128x128x128 voxels
- `compute_segmentation_metrics()` - Calculate Dice, Jaccard, Hausdorff, etc.
- `analyze_metrics_with_gpt()` - AI-powered metrics interpretation
- `reorient_nifti_directory()` - Standardize NIFTI orientations

### 2. Autoencoder Approaches (Unsupervised - Experimental)

**AE_Healthy:**
- Trains on healthy brain scans only
- Detects tumors as reconstruction anomalies
- Epoch 46, loss 0.0402

**VAE_Healthy:**
- Two variants: standard VAE and VAE with MBConv blocks
- Similar anomaly detection approach
- MBConv version shows improved performance (loss 0.1088 at epoch 5)

### 3. Streamlit Application

**Features:**
- Interactive 3D volume viewer with axial/sagittal/coronal planes
- Real-time slice navigation
- Overlay visualization (prediction vs ground truth)
- Comprehensive metrics dashboard
- GPT-powered analysis of segmentation quality
- Plotly-based interactive pan/zoom

**Data Loading:**
- Automatically clones repository on first run
- Caches volumes for fast navigation
- Supports both BRATS (tumor) and IXI (healthy) datasets

## Datasets

### BRATS Dataset
- Source: https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/
- Content: Brain scans with GBM tumors + segmentation labels
- Format: NIfTI (.nii.gz)
- Used for: Training and evaluation of supervised model

### IXI Dataset  
- Source: https://brain-development.org/ixi-dataset/
- Content: Healthy brain scans
- Format: NIfTI (.nii.gz)
- Used for: Training unsupervised models (AE/VAE) and testing false positives

## Architectural Patterns

### 1. Medical Image Processing Pipeline
```
Raw NIFTI → Resampling (128³) → Orientation Normalization → Model Inference → Segmentation Mask
```

### 2. Supervised Learning (3D U-Net)
- Encoder-decoder architecture with skip connections
- Multi-class segmentation with cross-entropy loss
- Data augmentation for robustness
- Post-processing for smooth boundaries

### 3. Unsupervised Learning (AE/VAE)
- Reconstruction-based anomaly detection
- Trained only on healthy brains
- Tumor regions show high reconstruction error
- Threshold-based mask generation

### 4. Metrics Calculation
- **Overlap Metrics:** Dice, Jaccard (IoU), Precision, Recall, F1
- **Volume Metrics:** RVD (Relative Volume Difference)
- **Distance Metrics:** Hausdorff, HD95, ASSD (Average Symmetric Surface Distance)
- Surface extraction using binary erosion and distance transforms

## Configuration Files

### requirements.txt
Standard pip-installable dependencies. Use with:
```bash
pip install -r requirements.txt
```

### environment.yml
Complete conda environment with CUDA support:
```bash
conda env create -f environment.yml
conda activate brain_tumor_segmentation
```

**Key features:**
- Python 3.10
- PyTorch with CUDA 11.8 support
- Conda channels: pytorch, conda-forge, defaults

### .devcontainer/devcontainer.json
VS Code dev container for consistent development environment:
- Python 3.11 base image
- Auto-installs dependencies on container creation
- Automatically launches Streamlit app on port 8501
- Opens README.md and main Streamlit file by default

### .streamlit/config.toml
Streamlit configuration:
- Sets logger level to "error" to reduce console noise

### .gitignore
Standard Python ignores plus:
- `secrets.toml` - Excluded for security (contains OpenAI API key)
- Virtual environments (venv/, ENV/, etc.)
- Jupyter checkpoints

## Build/Run/Test Commands

### Setup Environment

**Option 1: Using conda (recommended for GPU)**
```bash
conda env create -f environment.yml
conda activate brain_tumor_segmentation
```

**Option 2: Using pip**
```bash
python -m venv ds18_venv
source ds18_venv/bin/activate  # On Windows: ds18_venv\Scriptsctivate
pip install -r requirements.txt
```

### Run Streamlit Application

**Local deployment:**
```bash
streamlit run "3D_UNET Segmentation/3D_UNet_Tumor_Segmentation_Streamlit.py"
```

**Using devcontainer:**
The app launches automatically when the container starts (port 8501)

**Production deployment:**
Already deployed at https://ds18finalproject-danielc.streamlit.app/

### Train Models

**3D U-Net (Supervised):**
Open and run `3D_UNET Segmentation/3D_UNet_Brain_Tumor_Segmentation_multiclass_complete.ipynb`

**Autoencoder (Unsupervised):**
Open and run `AE_Healthy/AE_Healthy_T1_1ch_ep.ipynb`

**VAE (Unsupervised):**
- Standard: `VAE_Healthy/VAE/VAE_Healthy_T1_1ch_BEST.ipynb`
- MBConv: `VAE_Healthy/VAE_MBConv/VAE_Healthy_T1_1ch_BEST_With_MBConv.ipynb`

### Run Inference

**Interactive notebook:**
`3D_UNET Segmentation/Interactive_BrainSeg_UI.ipynb`

**Programmatic:**
Load model from `3D_UNET_loss-0.4394_ep-60.pth` and use helper functions

## API Integration

### OpenAI GPT Integration
The app uses GPT-4o-mini to analyze segmentation metrics:
- Requires `secrets.toml` file with OpenAI API key
- Location: `.streamlit/secrets.toml`
- Format:
  ```toml
  [openai]
  api_key = "your-api-key-here"
  ```

## Git Workflow

Current branch: `main`

**Recent commits:**
- `27d5024` - env files cleanen & updated using claude
- `9dc7a3e` - final touches
- `aa3ec96` - final touches
- `878fc98` - readme.md
- `7af375b` - reorient healthy brains

**Modified files (uncommitted):**
- `3D_UNET Segmentation/3D_UNet_Brain_Tumor_Segmentation_multiclass_complete.ipynb`
- `environment.yml` (untracked)

## Development Notes

### Image Format
All medical images use NIfTI format (.nii.gz):
- 3D volumes: shape (Z, Y, X) typically 128x128x128 after resampling
- Multi-modal: 4D volumes split into separate files per modality
- Orientations standardized using SimpleITK's DICOMOrientImageFilter

### Data Organization
- **Input scans:** `*_T1.nii.gz` or `*-T1.nii.gz` (T1-weighted MRI)
- **Ground truth:** `*.nii.gz` (same base name as input)
- **Predictions:** `*_predict_seg.nii.gz` or `*-predict_seg.nii.gz`

### Caching Strategy
Streamlit app uses two levels of caching:
1. `@st.cache_resource` - Repository cloning (once per session)
2. `@st.cache_data` - Volume loading (persists across reruns)

### Performance Considerations
- All volumes loaded into memory for fast navigation
- Resampling to 128³ reduces computation while maintaining quality
- GPU acceleration recommended for training (CUDA 11.8)

## Common Tasks for Claude

### Analyzing Results
- Metrics are computed using `compute_segmentation_metrics()` in HelperFunctions.py
- GPT analysis via `analyze_metrics_with_gpt()` provides clinical interpretation

### Adding New Test Cases
1. Place T1 scan in `3D_UNET Segmentation/inference_test/`
2. Place ground truth label (optional) in same directory
3. Run inference to generate prediction in `pred_seg/` subdirectory
4. Restart Streamlit app to see new case

### Modifying the UI
- Main app: `3D_UNet_Tumor_Segmentation_Streamlit.py`
- Alternative (no zoom): `3D_UNet_Tumor_Segmentation_Streamlit_no_pan-zoom.py`
- Helper utilities: `HelperFunctions.py`

### Training New Models
- Training notebooks contain complete pipelines
- Adjust hyperparameters in notebook cells
- Model checkpoints saved automatically during training

## Resources

- **Project Documentation:** See `README.md`
- **Presentation:** `ds18_GBM_Autosegmentation.pdf` / `.pptx`
- **MONAI Documentation:** https://docs.monai.io/
- **BRATS Dataset:** https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/
- **IXI Dataset:** https://brain-development.org/ixi-dataset/

## Known Limitations

1. **Unsupervised methods (AE/VAE):** Currently postponed, experimental status
2. **Data size:** Only small subset included in repo for demo purposes
3. **GPU requirement:** Training requires CUDA-capable GPU
4. **Secrets management:** OpenAI API key must be configured locally

## Future Improvements

- Complete evaluation of unsupervised methods
- Expand test dataset
- Add more modalities (T1Gd, T2, FLAIR)
- Ensemble predictions
- Real-time inference optimization
