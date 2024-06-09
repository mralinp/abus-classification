import torch
from abus_classification.models import UNet

def test_unet():
    
     # Example usage
    model = UNet(in_channels=64, out_channels=64)
    # Assuming input shape is [batch_size, channels, depth, height, width]
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Print the shape of the output tensor