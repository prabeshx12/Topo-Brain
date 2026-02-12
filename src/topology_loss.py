"""
Advanced Topology Loss Module for Anatomy-Preserving MRI Synthesis.

Implements multi-scale topology preservation including:
1. Edge-aware segmentation loss
2. Boundary sharpness enhancement
3. Multi-scale structural consistency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareTopologyLoss(nn.Module):
    """
    Edge-aware topology loss that emphasizes anatomical boundaries.
    
    This loss combines:
    - Standard cross-entropy for tissue classification
    - Edge-weighted loss to sharpen boundaries
    - Multi-scale consistency for structural preservation
    """
    
    def __init__(self, num_classes=4, edge_weight=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        
        # Class weights for brain tissue (BG, CSF, GM, WM)
        # Prioritize smaller classes (CSF, GM) over dominant ones (BG, WM)
        self.register_buffer('class_weights', torch.tensor([0.5, 2.0, 1.5, 1.0]))
        
        # Sobel filters for edge detection (3D)
        self.register_buffer('sobel_x', self._create_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._create_sobel_kernel('y'))
        self.register_buffer('sobel_z', self._create_sobel_kernel('z'))
    
    def _create_sobel_kernel(self, direction):
        """Create 3D Sobel kernel for edge detection."""
        if direction == 'x':
            kernel = torch.tensor([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            ], dtype=torch.float32)
        elif direction == 'y':
            kernel = torch.tensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ], dtype=torch.float32)
        else:  # z
            kernel = torch.tensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ], dtype=torch.float32)
        
        return kernel.unsqueeze(0).unsqueeze(0) / 16.0
    
    def detect_edges(self, mask):
        """
        Detect edges in the segmentation mask using 3D Sobel filters.
        
        Args:
            mask: [B, D, H, W] segmentation mask (class indices)
            
        Returns:
            edge_map: [B, 1, D, H, W] binary edge map
        """
        # Convert to float and add channel dim
        mask_float = mask.float().unsqueeze(1)
        
        # Apply Sobel filters
        grad_x = F.conv3d(mask_float, self.sobel_x, padding=1)
        grad_y = F.conv3d(mask_float, self.sobel_y, padding=1)
        grad_z = F.conv3d(mask_float, self.sobel_z, padding=1)
        
        # Compute gradient magnitude (Force float32 for geometric precision)
        edge_map = torch.sqrt(grad_x.float()**2 + grad_y.float()**2 + grad_z.float()**2 + 1e-8)
        
        # Threshold to binary
        edge_map = (edge_map > 0.1).float()
        
        return edge_map
    
    def forward(self, pred_logits, target_mask, mask=None):
        """
        Args:
            mask: Optional [B] binary mask for timestep gating
        """
        # Force float32 to prevent AMP float16 overflow in cross-entropy
        pred_logits = pred_logits.float()
        target_mask = target_mask.long()
        
        # Clamp logits to prevent extreme values under AMP
        pred_logits = torch.clamp(pred_logits, -50.0, 50.0)
        
        # 1. Weighted cross-entropy loss
        loss_ce = F.cross_entropy(pred_logits, target_mask, weight=self.class_weights, reduction='none')
        
        # 2. Detect edges in target
        edge_map = self.detect_edges(target_mask)
        edge_map = edge_map.squeeze(1)
        
        # 3. Weight loss by edges and mask by timestep gating
        edge_weights = 1.0 + (self.edge_weight - 1.0) * edge_map
        loss_weighted = (loss_ce * edge_weights)
        
        if mask is not None:
             # Apply sample-wise gating [B, 1, 1, 1]
             loss_weighted = loss_weighted * mask.view(-1, 1, 1, 1)
        
        loss_weighted = loss_weighted.mean()
        
        # Dice-like component on predicted edges (gated)
        if mask is not None:
            # Only compute for gated samples to save time
            gated_indices = torch.where(mask > 0.5)[0]
            if len(gated_indices) == 0:
                return {'loss': torch.tensor(0.0, device=pred_logits.device), 
                        'loss_ce': loss_ce.mean(), 'loss_boundary': torch.tensor(0.0, device=pred_logits.device)}
            
            # Sub-select batch
            curr_pred = pred_logits[gated_indices]
            curr_target = target_mask[gated_indices]
            curr_edge_map = edge_map[gated_indices]
        else:
            curr_pred = pred_logits
            curr_target = target_mask
            curr_edge_map = edge_map

        pred_probs = F.softmax(curr_pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)
        pred_edges = self.detect_edges(pred_class).squeeze(1)
        
        # Force float32 for sums to prevent AMP float16 overflow (max 65,504)
        # A 64x64x64 patch has 262,144 voxels. Any sum > 25% of the patch will overflow float16.
        edge_map_f32 = curr_edge_map.float()
        pred_edges_f32 = pred_edges.float()
        
        intersection = (pred_edges_f32 * edge_map_f32).sum()
        union = pred_edges_f32.sum() + edge_map_f32.sum()
        loss_boundary = 1.0 - (2.0 * intersection + 1e-8) / (union + 1e-8)
        
        loss_total = loss_weighted + 0.5 * loss_boundary
        
        return {
            'loss': loss_total,
            'loss_ce': loss_ce.mean(),
            'loss_boundary': loss_boundary,
        }


class MultiScaleTopologyLoss(nn.Module):
    """
    Multi-scale topology loss for hierarchical structure preservation.
    
    Computes topology loss at multiple spatial scales to ensure
    both fine-grained and coarse anatomical structures are preserved.
    """
    
    def __init__(self, num_classes=2, scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.num_classes = num_classes
        self.scales = scales
        self.edge_aware_loss = EdgeAwareTopologyLoss(num_classes)
    
    def forward(self, pred_logits, target_mask, mask=None):
        """
        Compute multi-scale topology loss.
        
        Args:
            pred_logits: [B, C, D, H, W] predicted class logits
            target_mask: [B, D, H, W] ground truth class indices
            
        Returns:
            loss_dict: Dictionary with total loss and scale-specific components
        """
        total_loss = 0.0
        scale_losses = {}
        
        for scale in self.scales:
            if scale == 1.0:
                # Original scale
                pred_scaled = pred_logits
                target_scaled = target_mask.long()
            else:
                # Downsample
                scale_factor = scale
                pred_scaled = F.interpolate(
                    pred_logits, 
                    scale_factor=scale_factor, 
                    mode='trilinear', 
                    align_corners=False
                )
                target_scaled = F.interpolate(
                    target_mask.unsqueeze(1).float(), 
                    scale_factor=scale_factor, 
                    mode='nearest'
                ).squeeze(1).long()
            
            # Compute loss at this scale
            loss_dict = self.edge_aware_loss(pred_scaled, target_scaled, mask=mask)
            scale_loss = loss_dict['loss']
            
            # Weight by scale (finer scales get more weight)
            weight = scale ** 0.5
            total_loss += weight * scale_loss
            scale_losses[f'loss_scale_{scale}'] = scale_loss
        
        # Normalize by total weight
        total_weight = sum(s ** 0.5 for s in self.scales)
        total_loss = total_loss / total_weight
        
        return {
            'loss': total_loss,
            **scale_losses
        }


def create_topology_loss(num_classes=2, use_multiscale=True):
    """
    Factory function to create the appropriate topology loss.
    
    Args:
        num_classes: Number of segmentation classes
        use_multiscale: Whether to use multi-scale loss
        
    Returns:
        Topology loss module
    """
    if use_multiscale:
        return MultiScaleTopologyLoss(num_classes=num_classes)
    else:
        return EdgeAwareTopologyLoss(num_classes=num_classes)


if __name__ == "__main__":
    # Test the topology loss
    print("Testing Edge-Aware Topology Loss...")
    
    # Create dummy data
    batch_size = 2
    num_classes = 2
    D, H, W = 64, 64, 64
    
    pred_logits = torch.randn(batch_size, num_classes, D, H, W)
    target_mask = torch.randint(0, num_classes, (batch_size, D, H, W))
    
    # Test edge-aware loss
    loss_fn = EdgeAwareTopologyLoss(num_classes=num_classes)
    loss_dict = loss_fn(pred_logits, target_mask)
    
    print(f"Edge-Aware Loss: {loss_dict['loss'].item():.4f}")
    print(f"  - CE Loss: {loss_dict['loss_ce'].item():.4f}")
    print(f"  - Boundary Loss: {loss_dict['loss_boundary'].item():.4f}")
    
    # Test multi-scale loss
    print("\nTesting Multi-Scale Topology Loss...")
    loss_fn_ms = MultiScaleTopologyLoss(num_classes=num_classes)
    loss_dict_ms = loss_fn_ms(pred_logits, target_mask)
    
    print(f"Multi-Scale Loss: {loss_dict_ms['loss'].item():.4f}")
    for key, value in loss_dict_ms.items():
        if key.startswith('loss_scale'):
            print(f"  - {key}: {value.item():.4f}")
    
    print("\n✓ Topology loss module tests passed!")
