import torch
import torchvision
from torch.utils.data import DataLoader
from datasets.carvana import Carvana
import numpy as np
import cv2


def save_checkpoint(model, path="./checkpoint/checkpoint.pth.tar"):
    torch.save(model, path)

def load_checkpoint(model, path="./checkpoint/checkpoint.pth.tar"):
    model.load_state_dict(path)
    
def calculate_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    # torch no grad?
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
    
def get_loaders(dataset_path, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):

    train_ds = Carvana(path_to_dataset=f'{dataset_path}/train', transform=train_transform)
    val_ds = Carvana(path_to_dataset=f'{dataset_path}/validation', transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def ResizeData(sample, target_size):
    # Get the original image dimensions
    x, m = sample
    
    original_height, original_width, _ = x.shape
    
    # Set the target size
    primary_target_size = (490, 490)

    # Calculate the padding needed
    padding_height = max(0, primary_target_size[0] - original_height)
    padding_width = max(0, primary_target_size[1] - original_width)

    # Calculate the padding amounts for top, bottom, left, and right
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding
    left_padding = padding_width // 2
    right_padding = padding_width - left_padding

    # Create a border around the image with zero-padding
    image_with_padding = cv2.copyMakeBorder(
        x,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Set the padding color to black
    )
    
    mask_with_padding = cv2.copyMakeBorder(
        m,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Set the padding color to black
    )

    # Resize the image to the target size
    resized_image = cv2.resize(image_with_padding, target_size)
    resized_mask = cv2.resize(mask_with_padding, target_size)
    
    return resized_image, resized_mask
