"""Multi-task learning for segmentation and classification of tumors in 3D ABUS images (Med. Image Anal. 2021)."""
from abus_classification.papers.zhou2021.model import MultiTaskVNet
from abus_classification.papers.zhou2021.train import ablations, main, train_eval
