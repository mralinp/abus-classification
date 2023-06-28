import torch


class VGG3D(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(VGG3D, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 8 * 8 * 8, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x