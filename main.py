import numpy as np
import torch
import torch.optim
import torchvision
from datasets.carvana import Carvana
from models.unet import UNet
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

print ('Using GPU' if torch.cuda.is_available() else "Using CPU")

train_transform = A.Compose(
        [
            A.Resize(height=160, width=240),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

training_dataset = Carvana("./data/carvana/train")
validation_dataset = Carvana("./data/carvana/validation")

    
plt.figure(figsize=(20,20))
print("here")
for idx, sample in enumerate(np.random.randint(0, len(training_dataset), 10)):
    img, mask = training_dataset[sample]
    plt.subplot(10,2, idx*2 + 1)
    plt.imshow(img)
    plt.subplot(10,2, idx*2 + 2)
    plt.imshow(mask)
plt.show()
    
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimize_fn = torch.optim
scaler = torch.cuda.amp.GradScaler()