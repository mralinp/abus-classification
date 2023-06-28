import torch

class VGGLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=1):
        super(VGGLSTM, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lstm = torch.nn.LSTM(256 * 8 * 8, hidden_size, num_layers, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_classes),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = x.view(batch_size, seq_length, -1)
        _, (x, _) = self.lstm(x)
        x = x[-1]
        x = self.classifier(x)
        return x