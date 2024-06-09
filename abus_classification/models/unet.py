import torch
import torchvision


class DoubleConv(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):        
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        
    def forward(self, X):
        return self.conv(X)
    
class UNet(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, features=[16, 32, 128, 256]) -> None:
        super(UNet, self).__init__()
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottle neck
        self.bottle_neck = DoubleConv(features[-1], features[-1]*2)
        
        for feature in reversed(features):
            self.decoder.append(torch.nn.ConvTranspose2d(feature*2, feature, kernel_size=(3,3), stride=2))
            self.decoder.append(DoubleConv(feature*2, feature))
        
        # last layer
        self.last_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=(1,1))
        
        self.bottle_neck_output = None
        
        
    def forward(self, x):
        
        skip_connections = []
        
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.max_pool(x)
        
        x = self.bottle_neck(x)
        
        # save bottleneck output
        self.bottle_neck_output = x
        
        # reverse skip_connections
        skip_connections.reverse()
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        
        return self.last_conv(x)
    
    @property
    def bottleneck_output(self):
        return self.bottle_neck_output