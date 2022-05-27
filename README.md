# Segmentation of Lung Lobes and Lesions for Severity Classification of COVID-19 CT Scans.
App Demo: https://share.streamlit.io/hds-69/csc-app/main/app.py
please use in local computer
<img src="https://github.com/hds-69/csc-app/blob/2abecc4e249e88afafad68fa137d6251f2a77d0b/Project%20info/framework.gif" style="max-width: 20%;" align="center" />

## About
Lung computed tomography (CT) severity score can be used for predicting clinical outcomes of patient with COVID-19. In this study, we propose a deep learning sematic segmentation for lung severity scoring of COVID-19 infection using the combination of 3D-UNet and pre-trained models, DenseNet and ResNet.
The segmentation model was trained with axial CT scans of 32 COVID-19 patients (training: 24, validation: 8) and tested with CT dataset of 8 patients. Next, the segmented masks were used to calculate the percentage of infection (PI), Total Severity Score (TSS) and define severity type. Lastly, correlation between model-predicted vs radiologist TSS was analyzed using CT scans of 62 patient.

## Use steps
1. Input Lung CT-scan image `.jpg` (Max 256 images per patient)
<img src="https://github.com/hds-69/csc-app/blob/f5f5645ab9675d7b73a79cc297e26cf8fa5ec60f/Project%20info/upload.gif" style="max-width: 20%;" align="center" />

2. Click `Predict` button.
<img src="https://github.com/hds-69/csc-app/blob/57db1150b52154eb33c64af3959c8c43b697c35b/Project%20info/predict.gif" style="max-width: 20%;" align="center" />

3. Click `Save` button.
<img src="https://github.com/hds-69/csc-app/blob/57db1150b52154eb33c64af3959c8c43b697c35b/Project%20info/save.gif" style="max-width: 20%;" align="center" />

## Development process
1. Dataset: Train 24 cases, Test1 8 cases, Test2 62 cases.
2. Image Annotation by labelme for using label mask for Model Training.
3. Traning Model: 3D-Unet + Backbone (DenseNet, ResNet) using API: https://github.com/ZFTurbo/segmentation_models_3D
4. Model Evaluation with Test1 dataset.
5. User Interface (UI) Development using Streamlit.
6. Develop the Percentage of Infection (PI%) and TSS calculation.
7. TSS Evaluation with Test2 dataset.

## Results
The model was evaluated on a dataset of 8 infected patients, and the results demonstrated that 3D-UNet + DenseNet169 achieved the best performance, yielding Dice Similarity Coefficient (DSC) of 92.89% and 84.22% for lung lobe and lesion segmentation, respectively. The proposed model can reliably segment lesions on CT scans of severe cases, but the model performed less accurately in segmenting lung lesions of mild and moderate cases. However, the TSS calculated by the proposed model were comparable to those assigned by radiologists. Using CT scans of 62 COVID-19 patients for evaluation, the correlation coefficient (r) was 0.9125, indicating a very strong correlation.

Model Testing Result Table(with Test set 1)

| Objective             | Model   | Backbone.   |  DSC | IoU  |
| --------------------- | ------- | ----------- |------|------|
| Lung lobe segmentation| 3D-Unet |DenseNet 169 |92.89%|90.44%|
| Lesion segmentation   | 3D-Unet |DenseNet 169 |84.22%|79.61%|


TSS calculation Testing Result Table(with Test set 2)
| Regression Statistics  | Value   |
| ---------------------- | ------- |
| Correlation Coefficient| 0.9125  |
| R Square               | 0.8327  |
| Adjusted R Square      | 0.8299  |
| Standard Error         | 2.5629  | 
| Observations           |   62    |
| p-value                | <0.01   |


## Reference
1. https://github.com/qubvel/segmentation_models
2. https://github.com/ZFTurbo/segmentation_models_3D
