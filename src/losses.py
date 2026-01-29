"""
Loss Functions for Anatomy-Guided Diffusion.
Implements Blueprint Section 7.1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Warning: torchvision not found. Perceptual loss will be disabled/dummy.")

class PerceptualLoss(nn.Module):
    """
    Simple VGG-based Perceptual Loss (Feature Matching).
    Extracts features from VGG16 (frozen) and computes MSE.
    """
    def __init__(self):
        super().__init__()
        if not HAS_TORCHVISION:
            raise RuntimeError("PerceptualLoss requires 'torchvision' library. Please install it or set lambda_percep=0.")
            
        vgg = models.vgg16(pretrained=True)
        # Use first few layers for texture/structure
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        """
        Input x, y are [B, 1, D, H, W] (3D)
        We allow x, y to be 2D [B, 1, H, W] or 3D.
        For 3D, we reshape Depth into Batch dimension.
        """
        # Handle 3D Input
        if x.ndim == 5:
            b, c, d, h, w = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            y_2d = y.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        else:
            x_2d = x
            y_2d = y
            
        # Convert 1 channel to 3 channels (repeat)
        if x_2d.shape[1] == 1:
            x_2d = x_2d.repeat(1, 3, 1, 1)
            y_2d = y_2d.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet mean/std (approx)
        # Assuming input is [-1, 1], map to [0, 1] then normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        x_2d = (x_2d + 1) * 0.5
        y_2d = (y_2d + 1) * 0.5
        
        x_2d = (x_2d - mean) / std
        y_2d = (y_2d - mean) / std
        
        # Extract features
        x_feat = self.feature_extractor(x_2d)
        y_feat = self.feature_extractor(y_2d)
        
        return F.mse_loss(x_feat, y_feat)

class TopologyLoss(nn.Module):
    """
    Topology-Aware Loss (Segmentation Consistency).
    Currently implements Cross-Entropy/Dice on segmentation logits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, seg_pred, seg_target):
        """
        seg_pred: [B, NumClasses, D, H, W] Logits
        seg_target: [B, D, H, W] Class Indices (Long)
        """
        # Cross Entropy
        loss_ce = F.cross_entropy(seg_pred, seg_target)
        
        # Optional: Add Dice Loss here if needed for clearer boundaries
        return loss_ce
