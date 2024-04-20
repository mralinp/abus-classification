from .metric import Metric
import torch


class AccuracyMetric(Metric):
    
    def calculate(self, prediction:torch.Tensor, labels: torch.Tensor):
        
        _, predictions = predictions.max(1)
        num_corrects += (predictions == labels).float().sum()
        
        return num_corrects/labels.shape[0]
        