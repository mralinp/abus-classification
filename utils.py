import torch
import torchvision


def calculate_accuracy(dataset, model, device="cuda"):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data in dataset:
            x, m, y = data
            x = x.to(device)
            m = m.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)
            x = x - x * m * 0.5

            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == y).sum()
            total += torch.numel(predictions)
    model.train()

    print(f"Got {num_correct}/{total} with acc {num_correct/total*100:.2f}")


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    # torch no grad?
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.nn.functional.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
    
def zero_pad_resize(img, size=(224,224)):
    res = np.zeros(size, dtype=np.float32)
    cx, cy = size[0]//2, size[1]//2
    w,h = img.shape
    if w > size[0] or h > size[1]:
        raise f"Cant resize with zero padding from origin with shape {img.shape} to size {size}"
    res[cx-w//2:cx+w//2, cy-h//2:cy+h//2] = img
    return res
