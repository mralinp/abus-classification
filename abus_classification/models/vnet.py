import torch
import torchvision


class ConvBatchNormReLu(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):        
        super(ConvBatchNormReLu, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        
    def forward(self, X):
        return self.conv(X)
    

class ConvBatchNormReLuSigmoid(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ConvBatchNormReLuSigmoid, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.conv(x)


class VNet(torch.nn.Module):
    
    
    def __init__(self, in_channels, out_channels, features=[(16,1),(32,1),(64,2)], bottle_neck_layers=3) -> None:
        super(VNet, self).__init__()
        
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.output_conv = ConvBatchNormReLuSigmoid(features[0][0], out_channels)
        self.down_sampling = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottle_neck = torch.nn.Sequential()
        
        for num_features, num_layers in features:
            module = torch.nn.Sequential(ConvBatchNormReLu(in_channels, num_features))
            for _ in range(num_layers - 1):
                module.append(ConvBatchNormReLu(num_features, num_features))
            self.encoder.append(module)
            in_channels = num_features
            
        num_bootle_neck_features = in_channels*2
        self.bottle_neck.append(ConvBatchNormReLu(in_channels, num_bootle_neck_features))
        for _ in range(bottle_neck_layers - 1):
            self.bottle_neck.append(ConvBatchNormReLu(num_bootle_neck_features, num_bootle_neck_features))
        
        for num_features, num_layers in reversed(features):
            self.decoder.append(torch.nn.ConvTranspose3d(num_features*2, num_features, kernel_size=2, stride=2))
            # First decoder input has two inputs which are concated:
            #   1. Output of up-sampling layer
            #   2. Skip connection from the encode module at the same depth
            module = torch.nn.Sequential(ConvBatchNormReLu(num_features*2, num_features))
            for _ in range(num_layers-1):
                module.append(ConvBatchNormReLu(num_features, num_features))
            self.decoder.append(module)
                
        self.__bottle_neck_output = None
        self.__decoder_outputs = None
        self.__encoder_outputs = None


    @property
    def bottle_neck_output(self):
        return self.__bottle_neck_output
    
    
    @property 
    def encoder_outputs(self):
        return self.__encoder_outputs
    
    
    @property
    def decoder_output(self):
        return self.__decoder_outputs
        
        
    def forward(self, x):
        
        self.__decoder_outputs = []
        self.__encoder_outputs = []
        
        for encode in self.encoder:
            x = encode(x)
            self.__encoder_outputs.append(x)
            x = self.down_sampling(x)

        x = self.bottle_neck(x)        
        # save bottleneck output
        self.__bottle_neck_output = x
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = self.__encoder_outputs[-(idx//2+1)]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
            self.__decoder_outputs.append(x)
        
        return self.output_conv(x)    
  