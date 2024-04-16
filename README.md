# Classification of Human Breast Cancer Lesions on 3D Automated Breast Ultrasound Images

This Master's thesis focuses on the classification of human breast cancer lesions using 3D Automated Breast Ultrasound (ABUS) images. The project is conducted under the supervision of the Iran Image Processing Lab (IIPL), where cutting-edge research in image processing and medical imaging is pursued.

### Project Overview:
The aim of this research is to develop a robust and accurate classification system for detecting and categorizing breast cancer lesions from 3D ABUS images. The classification process involves analyzing various features extracted from the images and employing machine learning algorithms to differentiate between different lesion types, such as benign and malignant tumors.

### Methodology:
The project utilizes advanced image processing techniques to extract relevant features from 3D ABUS images, including texture analysis, shape descriptors, intensity histogram analysis, spatial relationships, and vascular architecture analysis. These features serve as inputs to machine learning models, which are trained on annotated datasets to learn the characteristics of different breast lesions.

### Dataset and Code Availability:
The source code developed for this project is openly accessible under the MIT license, allowing for transparency and reproducibility of results. However, access to the datasets used for training and evaluation is restricted to lab conditions due to privacy and confidentiality concerns. For more information on dataset availability and access, please contact me@alinaderiparizi.com.

### Expected Outcomes:
By the conclusion of this research, we anticipate developing a highly accurate and efficient classification system for breast cancer lesions on 3D ABUS images. The findings of this study have the potential to contribute significantly to the early detection and diagnosis of breast cancer, ultimately improving patient outcomes and healthcare practices.

## Build/Run

Before you get started please make sure that all the requirements are installed properly:

- Anaconda

> Note: For gpu support follow the instructions here: [Installing tensorflow with gpu support](https://www.tensorflow.org/install/pip)

Then create and activate a virtual environment for the project:

```bash
$ conda create --name abus-classification --python=3.9
$ conda activate abus-classification
```

And install project requirments:
```bash
(venv) $ python -m pip install -t requirements.txt 
```

## Datasets
For this work we used two datasets, one is collected by our team in Iran Image Processing Lab which
consists of 70 Volumes of 3D ABUS and segmented by our team under supervision of
2 expert radiologists and the other one is the data from TDSC challenge which we don't have the permission
to share and publish until the event in finished and event holder is not published the data publicly.

| Dataset Name | Number of Volumes | Malignant | Begin |
|:------------:|:-----------------:|:---------:|:-----:|
| IIPL-3D-ABUS |        70         |    55     |  15   |
|     TDSC     |        200        |    100    |  100  |  
# Recent works and papers to be implemented
- [ ] [BTC: 2D Convolutional Neural Networks for 3D Digital Breast Tomosynthesis Classification.](https://arxiv.org/pdf/2002.12314.pdf)
- [ ] [DLF: A deep learning framework for supporting the classification of breast lesions in ultrasound images.](https://pubmed.ncbi.nlm.nih.gov/28753132/)
- [ ] [A method for the automated classification of benign and malignant masses on digital breast tomosynthesis images using machine learning and radiomic features.](https://pubmed.ncbi.nlm.nih.gov/31686300/)
- [ ] [Breast Cancer Classification in Automated Breast Ultrasound Using Multiview Convolutional Neural Network with Transfer Learning.](https://pubmed.ncbi.nlm.nih.gov/32059918/)
- [ ] [Fully automatic classification of automated breast ultrasound (ABUS) imaging according to BI-RADS using a deep convolutional neural network](https://pubmed.ncbi.nlm.nih.gov/35147776/)
- [ ] [Multi-modal artifcial intelligence for the combination of automated 3D breast ultrasound and mammograms in a population of women with predominantly dense breasts.](https://google.com)
