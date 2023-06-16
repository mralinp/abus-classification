# ABUS Classification

This project is a Master thesis with the subject "classification of human breast cancer lesions on 3D Automated Breast Ultra Sound images". The project is under supervision of Iran Image Processing Lab ([IIPL](https://iipl.ir)). Using the source codes has no restrictions under MIT licence but using datasets are private and only accessible under lab conditions. For more information please contact support@iipl.ir.

# Build/Run

Before you get started please make sure that all the requirements are installed properly:

- Anaconda

> Note: For gpu support follow the instructions here: [Installing tensorflow with gpu support](https://www.tensorflow.org/install/pip)

Then create and activate a virtual environment for the project:

```bash
$ conda create --name abus-classification --python=3.9
$ conda activate abus-classification
```

And installed project requirments:
```bash
(venv) $ python -m pip install -t requirements.txt 
```

# Datasets
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
