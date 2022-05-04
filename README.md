# Segmentation of Lung Lobes and Lesions for Severity Classification of COVID-19 CT Scans.
App Demo: https://share.streamlit.io/hds-69/csc-app/main/app.py
<img src="https://github.com/hds-no-69/COVID-19_Severity_Calculator/blob/aa4af859751397c6d4553c718eef34c08a446835/Project%20info/workflow.png" style="max-width: 60%;" align="center" />

## About
COVID-19 pandemic has spread all over the world, including Thailand. The disease becomes more serious when the infection has spread to the lungs. In the long term, it can harm the patient's respiratory system and cause death. Analysis of the COVID-19 severity infection can be validated by Lung Computed Tomography Scans. In this study, we represent a model for Semantic Segmentation of lung lobe area and lesions by using 3D-UNet deep learning modeling technology in combination with a pre-trained model, e.g. DenseNet. The results of the lung lobe  segmentation model have Dice Similarity Coefficient (DSC) 92.89% and Intersection over Union (IoU) 90.44%. This model is the combination of the 3D-UNet model with DenseNet169. And the results of the lesion segmentation model have DSC 86.35% and IoU 82.57%. The lesion segmentation model is combined from the 3D-UNet model with DenseNet201. The proposed segmentation model is able to accurately predict lung lobe area while the lesion model is still inaccurate, especially the lesions from mild and moderate patients. However, the proposed model can be applied to segment lobe area and lesion in order to predict the Total Severity Score (TSS) for the classification of COVID-19 severity.

## Step for work
1. Input Lung CT-scan image `.jpg` (Max 256 images per patient)
2. Click `Predict` button.

## Reference
