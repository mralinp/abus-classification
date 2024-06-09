import torch
from .vnet import VNet


class Classifier(torch.nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()
        self.pooling = torch.nn.AdaptiveAvgPool3d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128+256+128, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, *features):
        feature_vector = self.pooling(features[0])
        for feature in features[1:]:
            feature = self.pooling(feature)
            feature_vector = torch.cat((feature_vector, feature), dim=1)        
        return self.classifier(feature_vector)
    

class MultiTaskSegmentationClassificationABUS3D(torch.nn.Module):
    
    def __init__(self):
        super(MultiTaskSegmentationClassificationABUS3D, self).__init__()
        self.vnet = VNet(1,1, features=[(16,1), (32,1), (64,2), (128,3)], bottle_neck_layers=3)
        self.classifier = Classifier()
    
    def forward(self, x):
        segmentation_outputs = self.vnet(x)
        classification_outputs = self.classifier(self.vnet.encoder_outputs[-1], self.vnet.bottle_neck_output, self.vnet.decoder_outputs[0])
        
        return segmentation_outputs, classification_outputs