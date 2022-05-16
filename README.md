# Segmentation of Lung Lobes and Lesions for Severity Classification of COVID-19 CT Scans.
App Demo: https://share.streamlit.io/hds-69/csc-app/main/app.py


## About
COVID-19 pandemic has spread all over the world, including Thailand. The disease becomes more serious when the infection has spread to the lungs. In the long term, it can harm the patient's respiratory system and cause death. Analysis of the COVID-19 severity infection can be validated by Lung Computed Tomography Scans. In this study, we represent a model for Semantic Segmentation of lung lobe area and lesions by using 3D-UNet deep learning modeling technology in combination with a pre-trained model, e.g. DenseNet. The models were trained with the lung computed tomography dataset of 24 patients and tested with the lung CT dataset of 8 patients. The results of the lung lobe  segmentation model have Dice Similarity Coefficient (DSC) 92.89% and Intersection over Union (IoU) 90.44%. This model is the combination of the 3D-UNet model with DenseNet169. And the results of the lesion segmentation model have DSC 84.22% and IoU 79.61%. The lesion segmentation model is combined from the 3D-UNet model with DenseNet169. The proposed segmentation models are able to accurately predict lung lobe area while the lesion model is still inaccurate, especially the lesions from mild and moderate patients. However, the proposed models can be applied to segment lung lobe area and lesion area. The lung computed tomography dataset of 62 COVID-19 patients was used to test the model prediction of the Total Severity Score (TSS). The correlation between the TSS predicted by the model and the TSS value diagnosed by the radiologist was 0.9125 and the R Square was 0.8327.

## Step for work
1. Input Lung CT-scan image `.jpg` (Max 256 images per patient)
<img src="https://github.com/hds-69/csc-app/blob/f5f5645ab9675d7b73a79cc297e26cf8fa5ec60f/Project%20info/upload.gif" style="max-width: 20%;" align="center" />

2. Click `Predict` button.
<img src="https://github.com/hds-69/csc-app/blob/57db1150b52154eb33c64af3959c8c43b697c35b/Project%20info/predict.gif" style="max-width: 20%;" align="center" />

3. Click `Save` button.
<img src="https://github.com/hds-69/csc-app/blob/57db1150b52154eb33c64af3959c8c43b697c35b/Project%20info/save.gif" style="max-width: 20%;" align="center" />

## Development process
1. Dataset: Train 24 cases, Test1 8 cases, Test2 62 cases
2. Image Annotation by labelme for using label mask for Model Training
3. Traning Model: 3D-Unet + Backbone (DenseNet, ResNet) using API: https://github.com/ZFTurbo/segmentation_models_3D
4. Model Evaluation with Test1 dataset
5. User Interface (UI) Development using Streamlit
6. Develop the Percentage of Infection (PI%) and TSS calculator
7. TSS Evaluation with Test2 dataset

## Reference
1. https://github.com/qubvel/segmentation_models
2. https://github.com/ZFTurbo/segmentation_models_3D
