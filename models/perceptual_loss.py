"""
Perceptual loss for 3D medical images using pretrained features.

Implements:
- 3D adaptation of VGG-based perceptual loss
- Medical imaging-specific feature extraction
- Multi-scale perceptual matching

References:
- Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (ECCV 2016)
- Chen et al., "Brain MRI Super Resolution Using 3D Deep Densely Connected Neural Networks" (ISBI 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class PerceptualLoss3D(nn.Module):
    """
    3D Perceptual loss for medical images.
    
    Since 3D pretrained models are rare, we:
    1. Extract 2D slices from 3D volumes
    2. Use 2D VGG features
    3. Aggregate across slices
    
    Args:
        feature_layers: Which VGG layers to use for features
        slice_axis: Which axis to slice (0, 1, or 2)
        num_slices: How many slices to sample per volume
        use_gram: Whether to use Gram matrices (style loss)
    """
    def __init__(
        self,
        feature_layers: List[int] = [3, 8, 15, 22],  # relu1_2, relu2_2, relu3_3, relu4_3
        slice_axis: int = 2,
        num_slices: int = 5,
        use_gram: bool = False,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.slice_axis = slice_axis
        self.num_slices = num_slices
        self.use_gram = use_gram
        self.normalize_features = normalize_features
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
        
        # Store which layers to extract
        self.feature_layers = feature_layers
        
        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize grayscale image to RGB ImageNet stats.
        
        Args:
            x: Grayscale image (B, 1, H, W) in range [-1, 1] or [0, 1]
            
        Returns:
            RGB image (B, 3, H, W) normalized for VGG
        """
        # Convert to [0, 1] if needed
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Replicate to 3 channels
        x = x.repeat(1, 3, 1, 1)
        
        # Normalize using ImageNet stats
        x = (x - self.mean) / self.std
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-layer features from VGG.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            List of feature maps from specified layers
        """
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features
    
    def compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style loss.
        
        Args:
            x: Feature map (B, C, H, W)
            
        Returns:
            Gram matrix (B, C, C)
        """
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target 3D volumes.
        
        Args:
            pred: Predicted volumes (B, 1, D, H, W)
            target: Target volumes (B, 1, D, H, W)
            
        Returns:
            Perceptual loss (scalar)
        """
        batch_size = pred.shape[0]
        depth = pred.shape[2]
        
        # Sample slice indices
        slice_indices = torch.linspace(0, depth - 1, self.num_slices, dtype=torch.long)
        
        total_loss = 0.0
        
        for slice_idx in slice_indices:
            # Extract 2D slices
            if self.slice_axis == 0:
                pred_slice = pred[:, :, slice_idx, :, :]
                target_slice = target[:, :, slice_idx, :, :]
            elif self.slice_axis == 1:
                pred_slice = pred[:, :, :, slice_idx, :]
                target_slice = target[:, :, :, slice_idx, :]
            else:  # axis == 2
                pred_slice = pred[:, :, :, :, slice_idx]
                target_slice = target[:, :, :, :, slice_idx]
            
            # Normalize for VGG
            pred_norm = self.normalize_image(pred_slice)
            target_norm = self.normalize_image(target_slice)
            
            # Extract features
            pred_features = self.extract_features(pred_norm)
            target_features = self.extract_features(target_norm)
            
            # Compute loss for each layer
            for pred_feat, target_feat in zip(pred_features, target_features):
                if self.use_gram:
                    # Gram matrix (style) loss
                    pred_gram = self.compute_gram_matrix(pred_feat)
                    target_gram = self.compute_gram_matrix(target_gram)
                    total_loss += F.mse_loss(pred_gram, target_gram)
                else:
                    # Direct feature (content) loss
                    if self.normalize_features:
                        # Normalize features to unit norm
                        pred_feat = F.normalize(pred_feat, p=2, dim=1)
                        target_feat = F.normalize(target_feat, p=2, dim=1)
                    
                    total_loss += F.l1_loss(pred_feat, target_feat)
        
        # Average across slices and layers
        total_loss = total_loss / (self.num_slices * len(self.feature_layers))
        
        return total_loss


class MedicalPerceptualLoss(nn.Module):
    """
    Medical imaging-specific perceptual loss.
    
    Combines:
    1. VGG perceptual loss (texture/structure)
    2. Gradient loss (edge preservation)
    3. Frequency loss (spectral consistency)
    """
    def __init__(
        self,
        vgg_weight: float = 1.0,
        gradient_weight: float = 1.0,
        frequency_weight: float = 0.5,
        num_slices: int = 5,
    ):
        super().__init__()
        self.vgg_weight = vgg_weight
        self.gradient_weight = gradient_weight
        self.frequency_weight = frequency_weight
        
        self.vgg_loss = PerceptualLoss3D(num_slices=num_slices)
    
    def compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient difference loss for edge preservation.
        """
        # Compute gradients in all 3 dimensions
        pred_grad_x = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        pred_grad_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        pred_grad_z = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        target_grad_x = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        target_grad_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        target_grad_z = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        # L1 loss on gradients
        loss = (
            F.l1_loss(pred_grad_x, target_grad_x) +
            F.l1_loss(pred_grad_y, target_grad_y) +
            F.l1_loss(pred_grad_z, target_grad_z)
        ) / 3.0
        
        return loss
    
    def compute_frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss in frequency domain to preserve spectral properties.
        """
        # FFT on spatial dimensions
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # L1 loss on magnitude spectrum
        loss = F.l1_loss(pred_mag, target_mag)
        
        return loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined perceptual loss.
        
        Returns:
            Total loss and dictionary of components
        """
        losses = {}
        
        # VGG perceptual loss
        vgg_loss = self.vgg_loss(pred, target)
        losses['perceptual_vgg'] = vgg_loss.item()
        
        # Gradient loss
        grad_loss = self.compute_gradient_loss(pred, target)
        losses['perceptual_gradient'] = grad_loss.item()
        
        # Frequency loss
        freq_loss = self.compute_frequency_loss(pred, target)
        losses['perceptual_frequency'] = freq_loss.item()
        
        # Combine
        total_loss = (
            self.vgg_weight * vgg_loss +
            self.gradient_weight * grad_loss +
            self.frequency_weight * freq_loss
        )
        
        return total_loss, losses


if __name__ == "__main__":
    # Test perceptual losses
    print("Testing perceptual loss modules...")
    
    # Create dummy 3D volumes
    batch_size = 2
    pred = torch.randn(batch_size, 1, 32, 128, 128)
    target = torch.randn(batch_size, 1, 32, 128, 128)
    
    print("\n1. Testing PerceptualLoss3D...")
    perc_loss = PerceptualLoss3D(num_slices=3)
    loss = perc_loss(pred, target)
    print(f"   Perceptual loss: {loss.item():.4f}")
    
    print("\n2. Testing MedicalPerceptualLoss...")
    med_loss = MedicalPerceptualLoss(num_slices=3)
    loss, loss_dict = med_loss(pred, target)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Components: {loss_dict}")
    
    print("\n✓ Perceptual loss tests complete")
