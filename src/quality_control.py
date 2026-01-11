"""
Quality Control (QC) pipeline for preprocessing validation.
Automatically detects issues and generates QC reports.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

logger = logging.getLogger(__name__)


class AlignmentQC:
    """
    Generate alignment quality control visualizations for 3T-7T paired images.
    Implements checkerboard overlays and edge comparison per Blueprint Section 5.10.
    """

    @staticmethod
    def create_checkerboard(
        image1: np.ndarray,
        image2: np.ndarray,
        block_size: int = 20,
    ) -> np.ndarray:
        """
        Create a checkerboard overlay of two images for alignment verification.

        Args:
            image1: First image (e.g., 3T)
            image2: Second image (e.g., 7T)
            block_size: Size of checkerboard blocks in voxels

        Returns:
            Checkerboard composite image
        """
        if image1.shape != image2.shape:
            raise ValueError(
                f"Image shapes must match: {image1.shape} vs {image2.shape}"
            )

        result = np.zeros_like(image1)
        for dim in range(len(image1.shape)):
            indices = np.arange(image1.shape[dim])
            block_indices = (indices // block_size) % 2
            slices = [slice(None)] * len(image1.shape)
            for i, val in enumerate(block_indices):
                slices[dim] = i
                if dim == 0:
                    result[tuple(slices)] = val
                elif dim == 1:
                    result[tuple(slices)] += val * 2
                else:
                    result[tuple(slices)] += val * 4

        # Create checkerboard pattern
        mask = (result % 2) == 0
        composite = np.where(mask, image1, image2)
        return composite

    @staticmethod
    def compute_edge_map(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Compute edge map using Sobel gradient magnitude."""
        from scipy.ndimage import sobel, gaussian_filter

        # Smooth slightly first
        smoothed = gaussian_filter(image.astype(np.float32), sigma=sigma)

        # Compute gradient magnitude
        edges = np.zeros_like(smoothed)
        for axis in range(len(image.shape)):
            edges += sobel(smoothed, axis=axis) ** 2
        return np.sqrt(edges)

    @staticmethod
    def create_edge_overlay(
        image1: np.ndarray,
        image2: np.ndarray,
        sigma: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create edge maps for both images and compute their difference.

        Returns:
            Tuple of (edges1, edges2, edge_difference)
        """
        edges1 = AlignmentQC.compute_edge_map(image1, sigma)
        edges2 = AlignmentQC.compute_edge_map(image2, sigma)
        edge_diff = np.abs(edges1 - edges2)
        return edges1, edges2, edge_diff

    @staticmethod
    def compute_alignment_score(
        image1: np.ndarray,
        image2: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute alignment quality metrics between two images.

        Returns:
            Dictionary with alignment metrics (NCC, edge_mse, etc.)
        """
        if mask is not None:
            img1_masked = image1[mask > 0]
            img2_masked = image2[mask > 0]
        else:
            img1_masked = image1[image1 > 0]
            img2_masked = image2[image2 > 0]

        # Normalize for comparison
        img1_norm = (img1_masked - np.mean(img1_masked)) / (np.std(img1_masked) + 1e-8)
        img2_norm = (img2_masked - np.mean(img2_masked)) / (np.std(img2_masked) + 1e-8)

        # Length might differ if masking differently
        min_len = min(len(img1_norm), len(img2_norm))
        img1_norm = img1_norm[:min_len]
        img2_norm = img2_norm[:min_len]

        # Normalized Cross-Correlation
        ncc = float(np.mean(img1_norm * img2_norm))

        # Edge-based alignment score
        edges1, edges2, edge_diff = AlignmentQC.create_edge_overlay(image1, image2)
        if mask is not None:
            edge_mse = float(np.mean(edge_diff[mask > 0] ** 2))
        else:
            edge_mse = float(np.mean(edge_diff[image1 > 0] ** 2))

        return {
            "ncc": ncc,
            "edge_mse": edge_mse,
            "alignment_quality": "good" if ncc > 0.7 and edge_mse < 0.1 else "check",
        }

    @staticmethod
    def save_alignment_visualization(
        image_3t: np.ndarray,
        image_7t: np.ndarray,
        output_path: Path,
        mask: Optional[np.ndarray] = None,
        slice_axis: int = 2,
        title: str = "3T-7T Alignment QC",
    ) -> Path:
        """
        Generate and save alignment visualization with checkerboard and edge overlays.

        Args:
            image_3t: 3T MRI volume
            image_7t: 7T MRI volume
            output_path: Path to save the visualization
            mask: Optional brain mask
            slice_axis: Axis for slice selection (0=sagittal, 1=coronal, 2=axial)
            title: Title for the visualization
        """
        # Get middle slice
        slice_idx = image_3t.shape[slice_axis] // 2
        slices = [slice(None)] * 3
        slices[slice_axis] = slice_idx

        slice_3t = image_3t[tuple(slices)].T
        slice_7t = image_7t[tuple(slices)].T

        # Normalize for display
        slice_3t = (slice_3t - slice_3t.min()) / (slice_3t.max() - slice_3t.min() + 1e-8)
        slice_7t = (slice_7t - slice_7t.min()) / (slice_7t.max() - slice_7t.min() + 1e-8)

        # Create 2D checkerboard
        checkerboard = AlignmentQC.create_checkerboard(
            slice_3t, slice_7t, block_size=15
        )

        # Create edge overlay
        edges_3t = AlignmentQC.compute_edge_map(slice_3t, sigma=0.5)
        edges_7t = AlignmentQC.compute_edge_map(slice_7t, sigma=0.5)

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Row 1: Original images and checkerboard
        axes[0, 0].imshow(slice_3t, cmap="gray")
        axes[0, 0].set_title("3T Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(slice_7t, cmap="gray")
        axes[0, 1].set_title("7T Image")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(checkerboard, cmap="gray")
        axes[0, 2].set_title("Checkerboard Overlay")
        axes[0, 2].axis("off")

        # Row 2: Edge maps and difference
        axes[1, 0].imshow(edges_3t, cmap="hot")
        axes[1, 0].set_title("3T Edges")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(edges_7t, cmap="hot")
        axes[1, 1].set_title("7T Edges")
        axes[1, 1].axis("off")

        # Edge overlay with color coding
        edge_overlay = np.zeros((*edges_3t.shape, 3))
        edge_overlay[..., 0] = edges_3t / (edges_3t.max() + 1e-8)  # Red = 3T
        edge_overlay[..., 1] = edges_7t / (edges_7t.max() + 1e-8)  # Green = 7T
        axes[1, 2].imshow(edge_overlay)
        axes[1, 2].set_title("Edge Overlay (R=3T, G=7T)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path


class MaskQualityValidator:
    """
    Validate brain mask quality to detect skull-stripping issues.
    Implements Blueprint Section 5.11 fail-safes.
    """

    # Expected brain volume ranges in mm³ (normal adult brain)
    MIN_BRAIN_VOLUME_MM3 = 900_000  # ~900 cm³
    MAX_BRAIN_VOLUME_MM3 = 1_800_000  # ~1800 cm³

    @staticmethod
    def compute_mask_volume(
        mask: np.ndarray, voxel_spacing: Tuple[float, float, float]
    ) -> float:
        """Compute mask volume in mm³."""
        voxel_volume = float(np.prod(voxel_spacing))
        num_voxels = np.sum(mask > 0)
        return num_voxels * voxel_volume

    @staticmethod
    def check_boundary_touching(
        mask: np.ndarray, margin: int = 2
    ) -> Dict[str, bool]:
        """
        Check if mask touches image boundaries (may indicate cut-off brain).

        Returns:
            Dictionary with flags for each boundary
        """
        touches = {}
        dims = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]

        # Check each boundary
        touches["x_min"] = np.any(mask[:margin, :, :] > 0)
        touches["x_max"] = np.any(mask[-margin:, :, :] > 0)
        touches["y_min"] = np.any(mask[:, :margin, :] > 0)
        touches["y_max"] = np.any(mask[:, -margin:, :] > 0)
        touches["z_min"] = np.any(mask[:, :, :margin] > 0)
        touches["z_max"] = np.any(mask[:, :, -margin:] > 0)

        return touches

    @staticmethod
    def count_connected_components(mask: np.ndarray) -> int:
        """Count number of connected components in the mask."""
        labeled, num_features = ndimage.label(mask > 0)
        return num_features

    @staticmethod
    def compute_sphericity(mask: np.ndarray) -> float:
        """
        Compute mask sphericity (1.0 = perfect sphere).
        Low sphericity may indicate fragmented or irregular mask.
        """
        volume = np.sum(mask > 0)
        if volume == 0:
            return 0.0

        # Compute surface area (approximate via erosion)
        eroded = ndimage.binary_erosion(mask > 0)
        surface = np.sum((mask > 0) & ~eroded)

        if surface == 0:
            return 0.0

        # Sphericity = (π^(1/3) * (6V)^(2/3)) / A
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface
        return float(min(sphericity, 1.0))  # Cap at 1.0

    @staticmethod
    def detect_holes(mask: np.ndarray) -> int:
        """Detect internal holes in the mask (may indicate eroded brain regions)."""
        # Fill holes and compare
        filled = ndimage.binary_fill_holes(mask > 0)
        holes = filled & (mask == 0)
        return int(np.sum(holes))

    @classmethod
    def validate_mask(
        cls,
        mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Comprehensive mask quality validation.

        Returns:
            Dictionary with validation results and any warnings
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # Volume check
        volume = cls.compute_mask_volume(mask, voxel_spacing)
        results["volume_mm3"] = volume

        if volume < cls.MIN_BRAIN_VOLUME_MM3:
            results["warnings"].append(
                f"Brain volume too small: {volume:.0f} mm³ (expected > {cls.MIN_BRAIN_VOLUME_MM3})"
            )
        elif volume > cls.MAX_BRAIN_VOLUME_MM3:
            results["warnings"].append(
                f"Brain volume too large: {volume:.0f} mm³ (expected < {cls.MAX_BRAIN_VOLUME_MM3})"
            )

        # Boundary check
        boundary_touches = cls.check_boundary_touching(mask)
        results["boundary_touches"] = boundary_touches
        touching_boundaries = [k for k, v in boundary_touches.items() if v]
        if touching_boundaries:
            results["warnings"].append(
                f"Mask touches boundaries: {touching_boundaries} (potential brain cut-off)"
            )

        # Connected components check
        num_components = cls.count_connected_components(mask)
        results["num_components"] = num_components
        if num_components > 1:
            results["warnings"].append(
                f"Mask has {num_components} disconnected components (expected 1)"
            )
        elif num_components == 0:
            results["errors"].append("Mask is empty!")
            results["valid"] = False

        # Sphericity check
        sphericity = cls.compute_sphericity(mask)
        results["sphericity"] = sphericity
        if sphericity < 0.3:
            results["warnings"].append(
                f"Low sphericity: {sphericity:.2f} (may indicate fragmented mask)"
            )

        # Hole detection
        num_holes = cls.detect_holes(mask)
        results["num_internal_holes"] = num_holes
        if num_holes > 1000:  # Threshold for significant holes
            results["warnings"].append(
                f"Mask has {num_holes} internal hole voxels"
            )

        # Check coverage if image provided
        if image is not None:
            # Check if high-intensity regions are outside mask
            threshold = np.percentile(image[image > 0], 90)
            high_intensity = image > threshold
            missed = high_intensity & (mask == 0)
            missed_ratio = np.sum(missed) / (np.sum(high_intensity) + 1e-8)
            results["missed_high_intensity_ratio"] = float(missed_ratio)

            if missed_ratio > 0.1:
                results["warnings"].append(
                    f"High-intensity regions outside mask: {missed_ratio*100:.1f}%"
                )

        # Set overall validity
        if results["errors"]:
            results["valid"] = False

        return results


class MRIQCParser:
    """
    Parse and integrate MRIQC JSON metrics.
    Implements Blueprint Section 5.10/5.12 reviewer concerns.
    """

    # MRIQC IQM (Image Quality Metric) keys of interest
    KEY_METRICS = [
        "cjv",       # Coefficient of Joint Variation
        "cnr",       # Contrast-to-Noise Ratio
        "efc",       # Entropy Focus Criterion
        "fber",      # Foreground-Background Energy Ratio
        "fwhm_avg",  # Full Width Half Maximum (average)
        "snr_total", # Signal-to-Noise Ratio
        "snrd_total",# SNR Dietrich
        "qi_1",      # Artifact detection (Mortamet)
        "qi_2",      # Artifact detection
        "wm2max",    # White matter to max ratio
    ]

    @classmethod
    def find_mriqc_json(
        cls,
        image_path: Path,
        derivatives_root: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Find MRIQC JSON for an image following BIDS conventions.

        Searches in:
        1. Same directory as image
        2. derivatives/mriqc/sub-XXX/ses-XXX/anat/
        """
        image_path = Path(image_path)

        # Try same directory (sidecar)
        sidecar = image_path.with_suffix("").with_suffix(".json")
        if sidecar.suffix != ".json":
            sidecar = image_path.parent / (image_path.stem.replace(".nii", "") + ".json")
        if sidecar.exists():
            # Check if it's an MRIQC JSON (has IQMs)
            try:
                with open(sidecar) as f:
                    data = json.load(f)
                    if any(k in data for k in cls.KEY_METRICS):
                        return sidecar
            except:
                pass

        # Try derivatives/mriqc path
        if derivatives_root is not None:
            # Extract subject/session from path
            parts = image_path.parts
            subject = None
            session = None
            for p in parts:
                if p.startswith("sub-"):
                    subject = p
                elif p.startswith("ses-"):
                    session = p

            if subject:
                mriqc_patterns = [
                    derivatives_root / "mriqc" / subject / session / "anat" / f"*{image_path.stem}*.json",
                    derivatives_root / "mriqc" / subject / "anat" / f"*{image_path.stem}*.json",
                    derivatives_root / "mriqc" / f"*{subject}*{image_path.stem}*.json",
                ]
                for pattern in mriqc_patterns:
                    matches = list(pattern.parent.glob(pattern.name)) if pattern.parent.exists() else []
                    for match in matches:
                        if match.exists():
                            return match

        return None

    @classmethod
    def parse_mriqc(cls, json_path: Path) -> Dict[str, object]:
        """
        Parse MRIQC JSON and extract key metrics.

        Returns:
            Dictionary with extracted IQMs
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        metrics = {"mriqc_path": str(json_path)}

        for key in cls.KEY_METRICS:
            if key in data:
                metrics[f"mriqc_{key}"] = data[key]

        # Add quality flags based on known thresholds
        quality_flags = []

        # CJV: Coefficient of Joint Variation (lower is better, >0.5 may indicate issues)
        if "cjv" in data and data["cjv"] > 0.5:
            quality_flags.append(f"High CJV: {data['cjv']:.3f}")

        # EFC: Entropy Focus Criterion (lower is better, closer to 0 indicates good focus)
        if "efc" in data and data["efc"] > 0.6:
            quality_flags.append(f"High EFC: {data['efc']:.3f}")

        # QI_1: Artifact indicator (should be close to 0)
        if "qi_1" in data and data["qi_1"] > 0.05:
            quality_flags.append(f"Artifacts detected (qi_1): {data['qi_1']:.3f}")

        metrics["mriqc_quality_flags"] = quality_flags
        metrics["mriqc_overall_quality"] = "good" if not quality_flags else "check"

        return metrics

    @classmethod
    def get_metrics_for_image(
        cls,
        image_path: Path,
        derivatives_root: Optional[Path] = None,
    ) -> Optional[Dict[str, object]]:
        """
        Get MRIQC metrics for an image if available.
        """
        json_path = cls.find_mriqc_json(image_path, derivatives_root)
        if json_path:
            try:
                return cls.parse_mriqc(json_path)
            except Exception as e:
                logger.warning(f"Failed to parse MRIQC JSON {json_path}: {e}")
        return None


class QCMetrics:
    """Compute quality control metrics for MRI volumes."""
    
    @staticmethod
    def compute_snr(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Signal-to-Noise Ratio.
        
        SNR = mean(signal) / std(background)
        """
        if mask is not None:
            signal = image[mask > 0]
            background = image[mask == 0]
        else:
            # Use Otsu to separate signal from background
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(image[image > 0])
                signal = image[image > threshold]
                background = image[(image > 0) & (image <= threshold)]
            except:
                # Fallback if Otsu fails
                signal = image[image > np.percentile(image, 50)]
                background = image[(image > 0) & (image <= np.percentile(image, 50))]
        
        if len(background) == 0 or len(signal) == 0:
            return 0.0
        
        signal_mean = np.mean(signal)
        background_std = np.std(background)
        
        if background_std == 0:
            return 0.0
        
        return float(signal_mean / background_std)
    
    @staticmethod
    def compute_cnr(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Contrast-to-Noise Ratio.
        
        CNR = |mean(GM) - mean(WM)| / std(background)
        """
        if mask is None:
            return 0.0
        
        # Simple approximation: use intensity quantiles as GM/WM
        signal = image[mask > 0]
        if len(signal) == 0:
            return 0.0
        
        # Approximate GM and WM intensities
        sorted_signal = np.sort(signal)
        n = len(sorted_signal)
        gm_approx = sorted_signal[int(0.3*n):int(0.5*n)]
        wm_approx = sorted_signal[int(0.7*n):int(0.9*n)]
        
        background = image[mask == 0]
        
        if len(background) == 0 or len(gm_approx) == 0 or len(wm_approx) == 0:
            return 0.0
        
        gm_mean = np.mean(gm_approx)
        wm_mean = np.mean(wm_approx)
        background_std = np.std(background)
        
        if background_std == 0:
            return 0.0
        
        return float(abs(gm_mean - wm_mean) / background_std)
    
    @staticmethod
    def compute_entropy(image: np.ndarray) -> float:
        """Compute image entropy (measure of information content)."""
        # Compute histogram
        hist, _ = np.histogram(image[image > 0], bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)
    
    @staticmethod
    def detect_artifacts(image: np.ndarray) -> Dict[str, bool]:
        """Detect common MRI artifacts."""
        artifacts = {
            'has_nan': bool(np.isnan(image).any()),
            'has_inf': bool(np.isinf(image).any()),
            'has_negative': bool((image < 0).any()),
            'is_empty': bool(np.all(image == 0)),
        }
        
        # Check for extreme values
        if not artifacts['is_empty']:
            nonzero = image[image > 0]
            mean = np.mean(nonzero)
            std = np.std(nonzero)
            
            # Flag if values are more than 10 std from mean
            artifacts['has_outliers'] = bool(np.any(np.abs(nonzero - mean) > 10 * std))
        else:
            artifacts['has_outliers'] = False
        
        return artifacts
    
    @staticmethod
    def compute_foreground_fraction(image: np.ndarray) -> float:
        """Compute fraction of non-zero voxels."""
        total_voxels = image.size
        nonzero_voxels = np.count_nonzero(image)
        return float(nonzero_voxels / total_voxels)


class PreprocessingQC:
    """
    Quality control for preprocessing pipeline.
    Detects outliers and generates reports.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_list = []
    
    def compute_volume_qc(
        self,
        image_path: Path,
        mask_path: Optional[Path] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Compute QC metrics for a single volume."""
        # Load image
        nib_img = nib.load(str(image_path))
        image = nib_img.get_fdata().astype(np.float32)
        
        # Load mask if available
        mask = None
        if mask_path is not None and mask_path.exists():
            mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
        
        # Compute metrics
        metrics = {
            'file_path': str(image_path),
            'file_name': image_path.name,
            'shape': image.shape,
            'spacing': tuple(nib_img.header.get_zooms()[:3]),
            'orientation': ''.join(nib.aff2axcodes(nib_img.affine)),
            'snr': QCMetrics.compute_snr(image, mask),
            'cnr': QCMetrics.compute_cnr(image, mask) if mask is not None else 0.0,
            'entropy': QCMetrics.compute_entropy(image),
            'foreground_fraction': QCMetrics.compute_foreground_fraction(image),
            'mean_intensity': float(np.mean(image[image > 0])) if np.any(image > 0) else 0.0,
            'std_intensity': float(np.std(image[image > 0])) if np.any(image > 0) else 0.0,
            'min_intensity': float(np.min(image)),
            'max_intensity': float(np.max(image)),
            'has_mask': mask is not None,
        }
        
        # Detect artifacts
        artifacts = QCMetrics.detect_artifacts(image)
        metrics.update(artifacts)
        
        # Add metadata if provided
        if metadata is not None:
            metrics.update(metadata)
        
        self.metrics_list.append(metrics)
        
        return metrics
    
    def detect_outliers(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
    ) -> List[str]:
        """
        Detect outlier volumes based on metrics.
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
            
        Returns:
            List of file paths flagged as outliers
        """
        if len(self.metrics_list) == 0:
            logger.warning("No metrics available for outlier detection")
            return []
        
        df = pd.DataFrame(self.metrics_list)
        
        outliers = set()
        
        # Metrics to check for outliers
        numeric_metrics = ['snr', 'cnr', 'entropy', 'foreground_fraction', 
                          'mean_intensity', 'std_intensity']
        
        for metric in numeric_metrics:
            if metric not in df.columns:
                continue
            
            values = df[metric].values
            
            if method == 'iqr':
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outlier_mask = (values < lower_bound) | (values > upper_bound)
                
            elif method == 'zscore':
                mean = np.mean(values)
                std = np.std(values)
                
                if std == 0:
                    continue
                
                z_scores = np.abs((values - mean) / std)
                outlier_mask = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add outliers
            outlier_files = df[outlier_mask]['file_path'].tolist()
            outliers.update(outlier_files)
        
        # Also flag files with artifacts
        artifact_cols = ['has_nan', 'has_inf', 'is_empty', 'has_outliers']
        for col in artifact_cols:
            if col in df.columns:
                artifact_files = df[df[col] == True]['file_path'].tolist()
                outliers.update(artifact_files)
        
        return sorted(list(outliers))
    
    def generate_qc_report(
        self,
        report_path: Optional[Path] = None,
    ) -> Path:
        """Generate HTML QC report with visualizations."""
        if report_path is None:
            report_path = self.output_dir / 'qc_report.html'
        
        df = pd.DataFrame(self.metrics_list)
        
        # Detect outliers
        outliers = self.detect_outliers()
        df['is_outlier'] = df['file_path'].isin(outliers)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Quality Control Report', fontsize=16, fontweight='bold')
        
        # Plot distributions
        metrics_to_plot = [
            ('snr', 'Signal-to-Noise Ratio'),
            ('cnr', 'Contrast-to-Noise Ratio'),
            ('entropy', 'Image Entropy'),
            ('foreground_fraction', 'Foreground Fraction'),
            ('mean_intensity', 'Mean Intensity'),
            ('std_intensity', 'Std Intensity'),
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            if metric in df.columns:
                # Plot histogram
                ax.hist(df[metric], bins=20, alpha=0.7, color='blue', edgecolor='black')
                
                # Mark outliers
                if df['is_outlier'].any():
                    outlier_values = df[df['is_outlier']][metric]
                    ax.hist(outlier_values, bins=20, alpha=0.7, color='red', 
                           edgecolor='black', label='Outliers')
                    ax.legend()
                
                ax.set_title(title)
                ax.set_xlabel(metric)
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'qc_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MRI Preprocessing QC Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .outlier {{ background-color: #ffcccc !important; }}
                .warning {{ color: red; font-weight: bold; }}
                .success {{ color: green; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>MRI Preprocessing Quality Control Report</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <h2>Summary</h2>
            <ul>
                <li>Total volumes: {len(df)}</li>
                <li>Outliers detected: <span class="{'warning' if len(outliers) > 0 else 'success'}">
                    {len(outliers)}</span></li>
                <li>Volumes with artifacts: <span class="{'warning' if df[['has_nan', 'has_inf', 'is_empty']].any().any() else 'success'}">
                    {df[['has_nan', 'has_inf', 'is_empty']].any(axis=1).sum()}</span></li>
            </ul>
            
            <h2>Metric Distributions</h2>
            <img src="qc_distributions.png" alt="QC Distributions">
            
            <h2>Statistical Summary</h2>
            {df.describe().to_html()}
            
            <h2>Outlier Volumes</h2>
            {'<p class="warning">The following volumes were flagged as outliers:</p>' if len(outliers) > 0 else '<p class="success">No outliers detected!</p>'}
            {'<ul>' + ''.join(f'<li>{Path(o).name}</li>' for o in outliers) + '</ul>' if len(outliers) > 0 else ''}
            
            <h2>All Volumes</h2>
            {df.to_html(classes='table', index=False, escape=False)}
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Save metrics to JSON
        json_path = self.output_dir / 'qc_metrics.json'
        with open(json_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_volumes': len(df),
                    'outliers': len(outliers),
                    'outlier_files': outliers,
                },
                'metrics': self.metrics_list,
            }, f, indent=2)
        
        logger.info(f"QC report saved to {report_path}")
        logger.info(f"QC metrics saved to {json_path}")
        
        return report_path
    
    def save_metrics(self, output_path: Path) -> None:
        """Save metrics to CSV file."""
        df = pd.DataFrame(self.metrics_list)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved QC metrics to {output_path}")


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_default_config()
    
    # Initialize QC
    qc = PreprocessingQC(config.logging.log_dir / 'qc')
    
    # Process all volumes in dataset
    for nifti_file in config.data.data_root.glob("**/*_defaced.nii.gz"):
        metrics = qc.compute_volume_qc(
            nifti_file,
            metadata={
                'subject': nifti_file.parts[-4],
                'session': nifti_file.parts[-3],
            }
        )
        print(f"Processed {nifti_file.name}: SNR={metrics['snr']:.2f}")
    
    # Generate report
    report_path = qc.generate_qc_report()
    print(f"\nQC Report: {report_path}")
