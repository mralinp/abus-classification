class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice_coefficient = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # The Dice Loss is the complement of the Dice Coefficient
        dice_loss = 1 - dice_coefficient

        return dice_loss