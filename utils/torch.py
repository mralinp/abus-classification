import torch
import torchvision


def calculate_accuracy(dataset, model, device="cuda"):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data in dataset:
            x, _, y = data
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            
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