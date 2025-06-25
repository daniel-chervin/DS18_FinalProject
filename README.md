# GBM Automatic Segmentation 

Repository contains a 3d UNET model trained to identify and segment GBM tumor volumes. Two segmentation strategies were tested, 
unsupervised strategy (postponed) using Autoencoder (AE) & Variable Autoencoder (VAE) and supervised strategy using a 3D UNET from Monai library.
A Streamlit application to demonstrate the predicted segmentations compared to the ground truth; Provides visual & comparison metrics.

## Streamlit app

https://ds18finalproject-danielc.streamlit.app/ 

## Datasets
Healthy Brain full scans in nifty format.
* IXI dataset https://brain-development.org/ixi-dataset/

Brain full scans with GBM + labels (segmentation) nifty format.
* BRATS dataset https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/

## Directories
Inferred dataset for the demo (tumor & healthy scans)
* 3D_UNET Segmentation/inference_test

#### AE_Healthy & VAE_Healthy contain the Unsupervised (postponed) implementation 
* Implementation notebooks
* *_anomaly_mask.nii.gz, the inferred segmentation. 
* *.pt, the model parameters file.

#### 3D_UNET Segmentation
* *inference_test*, inferred dataset for the Streamlit app.
* *3D_UNet_Brain_Tumor_Segmentation_multiclass_complete.ipynb*, notebook containing the data handling, training & inference functionality. 
* *3D_UNET_loss-0.4394_ep-60.pth*, 3D UNET trained model parameters file.
* *3D_UNet_Tumor_Segmentation_Streamlit.py*, the Streamlit app script.
* *HelperFunctions.py*, script with useful helper functions.





