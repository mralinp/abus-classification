import cv2
import numpy as np
import torch
import torchvision


model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)


