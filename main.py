import torch
import torchvision
import utils
import models
import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

validation_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2(),
])

train_dataset = datasets.TDSC2D(path="./data/tdsc/slices", train=True, transforms=train_transforms)
validation_dataset = datasets.TDSC2D(path="./data/tdsc/slices", train=False, transforms=validation_transforms)

print(len(train_dataset))
print(len(validation_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

sefnet = models.SEFNet(in_channels=3)
sefnet = sefnet.to(DEVICE)
optimizer = torch.optim.Adam(sefnet.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):
    data_loop = tqdm(train_dataloader)
    for data in data_loop:
        x, m, l = data
        x = x.to(DEVICE)
        m = m.unsqueeze(1).to(DEVICE)
        l = l.unsqueeze(1).to(DEVICE)
        x = x - x * m * 0.5

        #Forward
        with torch.cuda.amp.autocast():
            predictions = sefnet(x)
            loss = criterion(predictions, l)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        data_loop.set_postfix(loss=loss.item())

    utils.calculate_accuracy(validation_dataloader, sefnet, DEVICE)





