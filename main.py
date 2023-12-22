import torch
import torchvision
import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally

train_transforms = A.Sequential(
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Normalize(),
    ToTensorV2(),
)

validation_transforms = A.Sequential(
    A.Resize(224,224),
    A.Normalize(),
    ToTensorV2(),
)

train_dataset = datasets.TDSC_2D(transforms=train_transforms)
validation_dataset = datasets.TDSC_2D(transforms=validation_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
validation_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

vgg16 = torchvision.models.vgg16(weights="DEFAULT")


