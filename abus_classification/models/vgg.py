import torch
import torch.nn as nn
from segment_anything.modeling import ImageEncoderViT

class VGG(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VGG, self).__init__()
        
        # Initialize the Medical SAM image encoder
        self.feature_extractor = ImageEncoderViT(
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=[2, 5, 8, 11],
        )
        
        # Load pretrained weights if specified
        if pretrained:
            try:
                checkpoint = torch.load('path_to_medical_sam_weights.pth')
                self.feature_extractor.load_state_dict(checkpoint, strict=True)
                print("Successfully loaded Medical SAM pretrained weights")
            except Exception as e:
                print(f"Failed to load pretrained weights: {e}")
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features using Medical SAM encoder
        features = self.feature_extractor(x)
        
        # Get the [CLS] token output (first token)
        x = features[0][:, 0]
        
        # Classification
        x = self.classifier(x)
        return x

    def unfreeze_layers(self, num_layers=0):
        """
        Unfreeze the last n layers of the feature extractor
        Args:
            num_layers: number of layers to unfreeze (0 means keep all frozen)
        """
        if num_layers > 0:
            for i, param in enumerate(reversed(list(self.feature_extractor.parameters()))):
                if i < num_layers:
                    param.requires_grad = True