import torch
from abus_classification.models import VNet

def test_vnet():
    
    model = VNet(in_channels=1, out_channels=1)
    # Assuming input shape is [batch_size, channels, depth, height, width]
    input_tensor = torch.randn(1, 1, 128, 128, 64)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Print the shape of the output tensor