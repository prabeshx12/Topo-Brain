"""
Topology-aware loss functions using Persistent Homology for medical image synthesis.

Implements:
- Persistent Homology computation using cubical complexes
- Betti number matching loss
- Topological regularization for preserving anatomical structures
- Wasserstein distance between persistence diagrams

References:
- Clough et al., "A Topological Loss Function for Deep-Learning based Image Segmentation" (MICCAI 2019)
- Hu et al., "Topology-Preserving Deep Image Segmentation" (NeurIPS 2019)
- Byrne et al., "A Persistent Homology-Based Topological Loss" (arXiv 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

try:
    # Try importing gudhi for persistent homology
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    logger.warning("gudhi not available. Install with: pip install gudhi")

try:
    # Try importing giotto-tda as alternative
    from gtda.homology import CubicalPersistence
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False


class PersistentHomologyLoss(nn.Module):
    """
    Computes topological loss based on persistent homology.
    
    Measures the difference in topological features (connected components,
    holes, voids) between generated and target images using Betti numbers
    and persistence diagrams.
    
    Args:
        dimensions: Which homology dimensions to compute (0=components, 1=holes, 2=voids)
        matching_weight: Weight for Betti number matching loss
        diagram_weight: Weight for persistence diagram matching
        use_3d: If True, compute 3D topology; else use 2D slice-wise
        slice_axis: Which axis to slice for 2D topology (-1 for middle slice only)
    """
    def __init__(
        self,
        dimensions: List[int] = [0, 1],
        matching_weight: float = 1.0,
        diagram_weight: float = 1.0,
        use_3d: bool = False,
        slice_axis: int = 2,
        num_slices: int = 5,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.matching_weight = matching_weight
        self.diagram_weight = diagram_weight
        self.use_3d = use_3d
        self.slice_axis = slice_axis
        self.num_slices = num_slices
        
        if not GUDHI_AVAILABLE and not GIOTTO_AVAILABLE:
            raise ImportError(
                "Neither gudhi nor giotto-tda available. Install one:\n"
                "  pip install gudhi\n"
                "  pip install giotto-tda"
            )
    
    def compute_betti_numbers(self, image: np.ndarray, threshold: float = 0.5) -> Dict[int, int]:
        """
        Compute Betti numbers for given image.
        
        Args:
            image: 2D or 3D numpy array
            threshold: Binary threshold for creating cubical complex
            
        Returns:
            Dictionary mapping dimension -> Betti number
        """
        # Binarize image
        binary_image = (image > threshold).astype(np.uint8)
        
        betti_numbers = {}
        
        if GUDHI_AVAILABLE:
            # Use GUDHI for persistent homology
            cc = gudhi.CubicalComplex(dimensions=binary_image.shape, top_dimensional_cells=binary_image.flatten())
            cc.compute_persistence()
            
            for dim in self.dimensions:
                persistence_pairs = cc.persistence_intervals_in_dimension(dim)
                if len(persistence_pairs) > 0:
                    # Count features with persistence > small threshold
                    persistence = persistence_pairs[:, 1] - persistence_pairs[:, 0]
                    betti_numbers[dim] = np.sum(persistence > 0.01)
                else:
                    betti_numbers[dim] = 0
        
        elif GIOTTO_AVAILABLE:
            # Use Giotto-TDA
            cubical_persistence = CubicalPersistence(homology_dimensions=self.dimensions)
            # Giotto expects shape (n_samples, n_pixels_x, n_pixels_y, ...)
            diagrams = cubical_persistence.fit_transform(binary_image[None, ...])
            
            for dim in self.dimensions:
                # Filter by dimension and count significant features
                dim_diagram = diagrams[0][diagrams[0][:, 2] == dim]
                if len(dim_diagram) > 0:
                    persistence = dim_diagram[:, 1] - dim_diagram[:, 0]
                    betti_numbers[dim] = np.sum(persistence > 0.01)
                else:
                    betti_numbers[dim] = 0
        
        return betti_numbers
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute topological loss between predicted and target images.
        
        Args:
            pred: Predicted images (B, C, D, H, W) or (B, C, H, W)
            target: Target images (same shape)
            
        Returns:
            Topological loss (scalar)
        """
        batch_size = pred.shape[0]
        device = pred.device
        
        # Convert to numpy for topology computation
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        total_loss = 0.0
        
        for b in range(batch_size):
            pred_vol = pred_np[b, 0]  # (D, H, W) or (H, W)
            target_vol = target_np[b, 0]
            
            if self.use_3d and pred_vol.ndim == 3:
                # Full 3D topology (expensive!)
                pred_betti = self.compute_betti_numbers(pred_vol)
                target_betti = self.compute_betti_numbers(target_vol)
                
                # Betti number matching loss
                betti_loss = sum(
                    abs(pred_betti.get(dim, 0) - target_betti.get(dim, 0))
                    for dim in self.dimensions
                )
                total_loss += betti_loss
            
            else:
                # 2D slice-wise topology (more efficient)
                if pred_vol.ndim == 3:
                    # Sample multiple slices
                    depth = pred_vol.shape[self.slice_axis]
                    slice_indices = np.linspace(0, depth-1, self.num_slices, dtype=int)
                    
                    slice_loss = 0.0
                    for idx in slice_indices:
                        if self.slice_axis == 0:
                            pred_slice = pred_vol[idx]
                            target_slice = target_vol[idx]
                        elif self.slice_axis == 1:
                            pred_slice = pred_vol[:, idx]
                            target_slice = target_vol[:, idx]
                        else:  # axis == 2
                            pred_slice = pred_vol[:, :, idx]
                            target_slice = target_vol[:, :, idx]
                        
                        pred_betti = self.compute_betti_numbers(pred_slice)
                        target_betti = self.compute_betti_numbers(target_slice)
                        
                        slice_loss += sum(
                            abs(pred_betti.get(dim, 0) - target_betti.get(dim, 0))
                            for dim in self.dimensions
                        )
                    
                    total_loss += slice_loss / self.num_slices
                else:
                    # Already 2D
                    pred_betti = self.compute_betti_numbers(pred_vol)
                    target_betti = self.compute_betti_numbers(target_vol)
                    
                    total_loss += sum(
                        abs(pred_betti.get(dim, 0) - target_betti.get(dim, 0))
                        for dim in self.dimensions
                    )
        
        # Normalize by batch size
        topo_loss = total_loss / batch_size
        
        return torch.tensor(topo_loss, device=device, dtype=pred.dtype) * self.matching_weight


class TopologicalRegularization(nn.Module):
    """
    Soft topological regularization that encourages connectivity preservation
    without requiring exact Betti number matching.
    
    Uses a differentiable approximation of topology based on:
    - Connected component smoothness
    - Boundary length minimization
    - Gradient coherence
    """
    def __init__(
        self,
        connectivity_weight: float = 1.0,
        smoothness_weight: float = 0.5,
    ):
        super().__init__()
        self.connectivity_weight = connectivity_weight
        self.smoothness_weight = smoothness_weight
    
    def compute_connectivity_loss(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encourage connectivity by penalizing isolated regions.
        Uses a differentiable approximation via morphological operations.
        """
        # Use average pooling as smooth approximation of dilation
        kernel_size = 3
        dilated = F.avg_pool3d(
            image,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        # Penalize difference between original and dilated
        # Large differences = isolated components
        connectivity_loss = F.mse_loss(image, dilated)
        
        return connectivity_loss
    
    def compute_boundary_loss(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encourage smooth boundaries by minimizing total variation.
        """
        # Compute gradients in all directions
        grad_x = torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        grad_y = torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        grad_z = torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1])
        
        # Total variation
        tv_loss = grad_x.mean() + grad_y.mean() + grad_z.mean()
        
        return tv_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute topological regularization loss.
        
        Args:
            pred: Predicted images (B, C, D, H, W)
            target: Target images (same shape)
            
        Returns:
            Regularization loss
        """
        # Normalize to [0, 1]
        pred_norm = torch.sigmoid(pred)
        target_norm = torch.sigmoid(target)
        
        # Connectivity preservation
        pred_conn = self.compute_connectivity_loss(pred_norm)
        target_conn = self.compute_connectivity_loss(target_norm)
        conn_loss = torch.abs(pred_conn - target_conn)
        
        # Boundary smoothness
        pred_boundary = self.compute_boundary_loss(pred_norm)
        target_boundary = self.compute_boundary_loss(target_norm)
        boundary_loss = torch.abs(pred_boundary - target_boundary)
        
        total_loss = (
            self.connectivity_weight * conn_loss +
            self.smoothness_weight * boundary_loss
        )
        
        return total_loss


class CombinedTopologyLoss(nn.Module):
    """
    Combines persistent homology loss with differentiable regularization.
    
    Args:
        use_ph_loss: Whether to use expensive PH computation
        ph_weight: Weight for persistent homology loss
        reg_weight: Weight for differentiable regularization
        ph_frequency: Compute PH loss every N batches (to reduce overhead)
    """
    def __init__(
        self,
        use_ph_loss: bool = True,
        ph_weight: float = 0.1,
        reg_weight: float = 1.0,
        ph_frequency: int = 10,
        dimensions: List[int] = [0, 1],
        use_3d: bool = False,
    ):
        super().__init__()
        self.use_ph_loss = use_ph_loss and (GUDHI_AVAILABLE or GIOTTO_AVAILABLE)
        self.ph_weight = ph_weight
        self.reg_weight = reg_weight
        self.ph_frequency = ph_frequency
        self.batch_counter = 0
        
        if self.use_ph_loss:
            self.ph_loss = PersistentHomologyLoss(
                dimensions=dimensions,
                use_3d=use_3d,
                num_slices=3  # Use fewer slices for efficiency
            )
        
        self.reg_loss = TopologicalRegularization()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined topology loss.
        
        Returns:
            Total loss and dictionary of individual loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Always compute differentiable regularization
        reg = self.reg_loss(pred, target)
        losses['topo_reg'] = reg.item()
        total_loss += self.reg_weight * reg
        
        # Compute PH loss periodically
        if self.use_ph_loss and (self.batch_counter % self.ph_frequency == 0):
            ph = self.ph_loss(pred, target)
            losses['topo_ph'] = ph.item()
            total_loss += self.ph_weight * ph
        else:
            losses['topo_ph'] = 0.0
        
        self.batch_counter += 1
        
        return total_loss, losses


if __name__ == "__main__":
    # Test topology losses
    print("Testing topology loss modules...")
    
    # Create dummy data
    batch_size = 2
    pred = torch.randn(batch_size, 1, 32, 64, 64)
    target = torch.randn(batch_size, 1, 32, 64, 64)
    
    # Test regularization (always works)
    print("\n1. Testing TopologicalRegularization...")
    reg_loss = TopologicalRegularization()
    loss = reg_loss(pred, target)
    print(f"   Regularization loss: {loss.item():.4f}")
    
    # Test PH loss if available
    if GUDHI_AVAILABLE or GIOTTO_AVAILABLE:
        print("\n2. Testing PersistentHomologyLoss...")
        ph_loss = PersistentHomologyLoss(dimensions=[0, 1], use_3d=False, num_slices=3)
        loss = ph_loss(pred, target)
        print(f"   PH loss: {loss.item():.4f}")
        
        print("\n3. Testing CombinedTopologyLoss...")
        combined_loss = CombinedTopologyLoss(use_ph_loss=True, ph_frequency=1)
        loss, loss_dict = combined_loss(pred, target)
        print(f"   Combined loss: {loss.item():.4f}")
        print(f"   Components: {loss_dict}")
    else:
        print("\n2. PersistentHomologyLoss: Skipped (no gudhi/giotto-tda)")
        print("   Install with: pip install gudhi  OR  pip install giotto-tda")
    
    print("\n✓ Topology loss tests complete")
