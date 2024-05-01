![example workflow](https://github.com/mralinp/abus-classification/actions/workflows/pages/pages-build-deployment/badge.svg)
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

o utilize the project source code, you have two options: either using **Google Colab** or setting up the environment on your local machine. For local installation, it's recommended to have a GPU and properly install Anaconda with GPU support.

The core requirements of this project, such as `transformers`, base `models`, `datasets`, and `utilities`, are developed as a Python package named `abus_classification`, which you can import and utilize in your own code. `abus_classification` is written on the PyTorch framework.

### Google Colab
To use the project in colab, open a colab notebook, clone the project repository and add the path to the module in `sys.path`. To do so, insert the code below into the first code cell of your note book and run.

```python
!git clone https://github.com/mralinp/abus-classification.git
!pip install pynrrd

import sys
sys.path.append("abus-classification")
```
After that, you will be able to import and use `abus_classification` in your notebook.

Example:
```python
from abus_classification import datasets

tdsc = datasets.TDSC() # This will automatically download the data

x, y, m = tdsc[0] # x: volume, m: mask_volume, y: label (0:m, 1:b)
```

### Anaconda (Local machine)

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
For this work, we utilized two datasets. The first dataset was collected by our team at the Iran Image Processing Lab, comprising 70 volumes of 3D ABUS. This dataset was meticulously segmented by our team under the supervision of two expert radiologists.

The second dataset is from the TDSC challenge. However, we do not have permission to share or publish this data until the event is concluded and the event organizer has publicly released the data.

It's crucial to adhere to data sharing and publication policies, especially when working with sensitive or proprietary datasets. Once the event concludes and the data becomes publicly available, we may be able to share our findings and insights from the TDSC challenge dataset.

| Dataset Name | Number of Volumes | Malignant | Benign |
|:------------:|:-----------------:|:---------:|:------:|
| IIPL-3D-ABUS |        70         |    55     |   15   |
|     TDSC     |        100        |    58     |   42   |  

# Recent works and papers to be implemented
- [x] [Multi-Task Learning for Segmentation and Classification of Tumors in 3D Automated Breast Ultrasound Images](https://drive.google.com/file/d/1ONcpYI0-VXYNkmmtPxh-CVP1bsN-UgM3/view)
- [ ] [BTC: 2D Convolutional Neural Networks for 3D Digital Breast Tomosynthesis Classification.](https://arxiv.org/pdf/2002.12314.pdf)
- [ ] [DLF: A deep learning framework for supporting the classification of breast lesions in ultrasound images.](https://pubmed.ncbi.nlm.nih.gov/28753132/)
- [ ] [A method for the automated classification of benign and malignant masses on digital breast tomosynthesis images using machine learning and radiomic features.](https://pubmed.ncbi.nlm.nih.gov/31686300/)
- [ ] [Breast Cancer Classification in Automated Breast Ultrasound Using Multiview Convolutional Neural Network with Transfer Learning.](https://pubmed.ncbi.nlm.nih.gov/32059918/)
- [ ] [Fully automatic classification of automated breast ultrasound (ABUS) imaging according to BI-RADS using a deep convolutional neural network](https://pubmed.ncbi.nlm.nih.gov/35147776/)
