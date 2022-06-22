# Segmentation of lung lobes and lesions in CT scans for severity classification of covid-19
Video: https://youtu.be/bAVM-OChI_k

<img src="https://github.com/hds-69/csc-app/blob/2abecc4e249e88afafad68fa137d6251f2a77d0b/Project%20info/framework.gif" style="max-width: 20%;" align="center" />

## About
Lung computed tomography (CT) severity score can be used for predicting clinical outcomes of patient with COVID-19. In this study, we propose a deep learning sematic segmentation for lung severity scoring of COVID-19 infection using the combination of 3D-UNet and pre-trained models, DenseNet and ResNet.
The segmentation model was trained with axial CT scans of 32 COVID-19 patients (training: 24, validation: 8) and tested with CT dataset of 8 patients. Next, the segmented masks were used to calculate the percentage of infection (PI), Total Severity Score (TSS) and define severity type. Lastly, correlation between model-predicted vs radiologist TSS was analyzed using CT scans of 62 patient.

App demo: please use in local computer.

## Requirements
1. You need install `Python >= 3.9.13`
2. Library requirement

        streamlit
        numpy
        pandas
        patchify==0.2.3
        segmentation-models-3D==1.0.3
        keras==2.8.0
        Keras-Applications==1.0.8
        tensorflow==2.8.0
        regex
        scikit-image==0.18.3
        pandas==1.3.5
        matplotlib
        numba
        zipp
        opencv-python==4.5.5.64
        Pillow==8.3.2
        fpdf

## Installation
1. clone this project followed by Github CLI command: 

       gh repo clone hds-69/csc-app
   or download project `zip` file 
2. Open a command prompt or terminal for `CD` command to change the directory to project location.
3. Installation with pip allows the usage of the install command:

        pip install -r requirements.txt
        
4. Run the App

        streamlit run app.py

## Use steps
Installation handbook: [https://docs.google.com/presentation/d/1DoHhD1M588GQGURy2NagAL8hC8nkRoxNFfkGxB3L2vs/edit?usp=sharing](https://github.com/hds-69/csc-app/blob/70d2c9619397a46d69f06923292e4f6a94cb819d/Document/Manual.pdf)

1. Input Lung CT-scan image `.jpg` (Min/Max range  80-256 images per patient)
<img src="https://github.com/hds-69/csc-app/blob/f5f5645ab9675d7b73a79cc297e26cf8fa5ec60f/Project%20info/upload.gif" style="max-width: 20%;" align="center" />

2. Click `Predict` button.
<img src="https://github.com/hds-69/csc-app/blob/57db1150b52154eb33c64af3959c8c43b697c35b/Project%20info/predict.gif" style="max-width: 20%;" align="center" />

3. Click `Save` button.
<img src="https://github.com/hds-69/csc-app/blob/57db1150b52154eb33c64af3959c8c43b697c35b/Project%20info/save.gif" style="max-width: 20%;" align="center" />

## Development process
1. Dataset: Train 32 cases (3,752 images), Test1 8 cases (1,314 images), Test2 62 cases (7,686 images)

   CT-scans image range: `92-208` images
   
   Case Type: `No lesion, Mild, Moderate, Severe`
   
2. Image Annotation by labelme for using label mask for Model Training. (for Train set and Test1 set)
3. Traning Model: 3D-Unet + Backbone (DenseNet, ResNet) using API: https://github.com/ZFTurbo/segmentation_models_3D
4. Model Evaluation with Test1 dataset.
5. User Interface (UI) Development using Streamlit.
6. Develop the Percentage of Infection (PI%) and TSS calculation.
7. TSS Evaluation with Test2 dataset.

## Results
The model was evaluated on a dataset of 8 infected patients, and the results demonstrated that 3D-UNet + DenseNet169 achieved the best performance, yielding Dice Similarity Coefficient (DSC) of `92.89%` and `84.22%` for lung lobe and lesion segmentation, respectively. The proposed model can reliably segment lesions on CT scans of severe cases, but the model performed less accurately in segmenting lung lesions of mild and moderate cases. However, the TSS calculated by the proposed model were comparable to those assigned by radiologists. Using CT scans of 62 COVID-19 patients for evaluation, the correlation coefficient (r) was `0.9125`, indicating a very strong correlation.

Model Testing Result Table `Test with Test set 1`

| Objective             | Model   | Backbone   | IoU  |  DSC | Accuracy | Precision | Sensitivity | Specificity |
| --------------------- | ------- | ----------- |------|------|----------|-----------|-------------|-------------|
| Lung lobe segmentation| 3D-Unet |DenseNet 169 |90.44%|92.89%| 98.49%   | 94.18%    | 95.52%      | 98.49%      |
| Lesion segmentation   | 3D-Unet |DenseNet 169 |79.61%|84.22%| 98.86%   | 86.7%     | 89.07%      | 98.91%      |


TSS calculation Testing Result Table `Test with Test set 2`
| Regression Statistics  | Value   |
| ---------------------- | ------- |
| Correlation Coefficient| 0.9125  |
| R Square               | 0.8327  |
| Adjusted R Square      | 0.8299  |
| Standard Error         | 2.5629  | 
| Observations           |   62    |
| p-value                | <0.001   |


## Citation

For more details,

If you find this code useful, please cite our project
```

```

## Reference
1. Roman Solovyev, Alexandr A Kalinin, and Tatiana Gabruseva, 2022, ``3D convolutional neural networks for stalled brain capillary detection,'' Comput. Biol. Med., vol. 141, no. 105089, pp. 105089, 2022. https://doi.org/10.1016/j.compbiomed.2021.105089
2. Pan F, Ye T, Sun P, et al. Time course of lung changes at chest CT during recovery from coronavirus disease 2019 (COVID-19). Radiology. 2020;295(3):715-721. doi:10.1148/radiol.2020200370
3. Francone M, Iafrate F, Masci GM, Coco S, Cilia F, Manganaro L, et al. Chest CT score in COVID-19 patients: correlation with disease severity and short-term prognosis. Eur Radiol. 2020;30(12):6808-6817. https://doi.org/10.1007/s00330-020-07033-y
4.  Guan X, Yao L, Tan Y et al. Quantitative and semi-quantitative CT assessments of lung lesion burden in COVID-19 pneumonia. Sci Rep. 2021;11(1). doi:10.1038/s41598-021-84561-7
5. Xiao Z, Liu B, Geng L, Zhang F, Liu Y. Segmentation of lung nodules using improved 3D-UNet neural network. Symmetry (Basel). 2020;12(11):1787. doi:10.3390/sym12111787
6. Qiblawey Y, Tahir A, Chowdhury M, et al. Detection and severity classification of COVID-19 in CT images using deep learning. Diagnostics. 2021;11(5):893. doi:10.3390/diagnostics11050893
7. Chen, M., Gu, Y., Qin, Y., Zheng, H., & Yang, J. (2020). LOBENET: A GLOBAL POSITION RESERVATION AND FISSURE-AWARE CONVOLUTIONAL NEURAL NETWORK FOR PULMONARY LOBE SEGMENTATION.
8. Tang H, Zhang C, Xie X. Automatic pulmonary lobe segmentation using deep learning. 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). 2019. doi:10.1109/isbi.2019.8759468
9. Visvanathan M, Balasubramanian V, Sathish R, Balasubramaniam S, Sheet D. Assessing Lobe-wise Burden of COVID-19 Infection in Computed Tomography of Lungs using Knowledge Fusion from Multiple Datasets. Annu Int Conf IEEE Eng Med Biol Soc. 2021 Nov;2021:3961-3964. doi: 10.1109/EMBC46164.2021.9629591. PMID: 34892098.
10. OpenCV: Histograms - 2: Histogram Equalization. Docs.opencv.org. https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html. Published 2022. Accessed May 8, 2022.
