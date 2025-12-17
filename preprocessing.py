"""
Deterministic preprocessing pipeline for 3D brain MRI using MONAI.
Implements reorientation, bias correction, skull stripping, and normalization.
"""
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import json

import numpy as np
import torch
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    EnsureTyped,
    SaveImaged,
    CropForegroundd,
    SpatialPadd,
)
from monai.data import MetaTensor
import SimpleITK as sitk
from tqdm import tqdm

from config import MRIConfig, PreprocessingConfig


logger = logging.getLogger(__name__)


class N4BiasFieldCorrection:
    """
    N4ITK bias field correction using SimpleITK.
    More reliable than MONAI's implementation for medical imaging.
    """
    def __init__(
        self,
        num_iterations: int = 50,
        convergence_threshold: float = 0.001,
        shrink_factor: int = 4,
        num_histogram_bins: int = 200,
    ):
        self.num_iterations = num_iterations
        self.convergence_threshold = convergence_threshold
        self.shrink_factor = shrink_factor
        self.num_histogram_bins = num_histogram_bins
    
    def __call__(self, image_path: Union[str, Path], mask_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Apply N4 bias field correction to input image.
        
        Args:
            image_path: Path to input NIfTI image
            mask_path: Optional path to brain mask
            
        Returns:
            Corrected image as numpy array
        """
        # Load image with SimpleITK
        image = sitk.ReadImage(str(image_path))
        image = sitk.Cast(image, sitk.sitkFloat32)
        
        # Load mask if provided
        if mask_path is not None and Path(mask_path).exists():
            mask = sitk.ReadImage(str(mask_path))
            mask = sitk.Cast(mask, sitk.sitkUInt8)
        else:
            # Create mask from Otsu thresholding
            mask = sitk.OtsuThreshold(image, 0, 1)
        
        # Shrink image for faster computation
        if self.shrink_factor > 1:
            image_shrunk = sitk.Shrink(image, [self.shrink_factor] * image.GetDimension())
            mask_shrunk = sitk.Shrink(mask, [self.shrink_factor] * image.GetDimension())
        else:
            image_shrunk = image
            mask_shrunk = mask
        
        # Configure N4 bias field corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([self.num_iterations] * 4)
        corrector.SetConvergenceThreshold(self.convergence_threshold)
        corrector.SetNumberOfHistogramBins(self.num_histogram_bins)
        
        # Execute correction
        try:
            corrected = corrector.Execute(image_shrunk, mask_shrunk)
            
            # Get bias field
            log_bias_field = corrector.GetLogBiasFieldAsImage(image)
            
            # Apply bias field to original resolution image
            corrected_full = image / sitk.Exp(log_bias_field)
            
            # Convert to numpy
            corrected_array = sitk.GetArrayFromImage(corrected_full)
            
            logger.debug(f"N4 correction completed in {corrector.GetElapsedIterations()} iterations")
            
            return corrected_array.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"N4 correction failed: {e}. Returning original image.")
            return sitk.GetArrayFromImage(image).astype(np.float32)


class SkullStripping:
    """
    Apply brain mask for skull stripping.
    Can use pre-computed masks or generate them using simple thresholding.
    
    NOTE: For production, use HD-BET, SynthStrip, or similar tools to generate masks.
    """
    def __init__(self, mask_pattern: str = "*brain_mask.nii.gz"):
        self.mask_pattern = mask_pattern
    
    def find_mask(self, image_path: Path) -> Optional[Path]:
        """Find corresponding brain mask for an image."""
        parent_dir = image_path.parent
        
        # Try to find mask with similar naming
        mask_candidates = list(parent_dir.glob(self.mask_pattern))
        
        if mask_candidates:
            # Match based on subject/session/modality
            image_stem = image_path.stem.replace("_defaced", "").replace(".nii", "")
            for mask in mask_candidates:
                if image_stem in mask.stem:
                    return mask
            return mask_candidates[0]
        
        return None
    
    def create_simple_mask(self, image_array: np.ndarray, threshold_percentile: float = 10) -> np.ndarray:
        """
        Create a simple brain mask using intensity thresholding.
        
        WARNING: This is a fallback method. For production, use proper brain extraction tools.
        """
        # Calculate threshold
        threshold = np.percentile(image_array[image_array > 0], threshold_percentile)
        
        # Create binary mask
        mask = (image_array > threshold).astype(np.uint8)
        
        # Simple morphological operations to clean up mask
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_erosion(mask, iterations=1)
        mask = ndimage.binary_dilation(mask, iterations=2)
        
        # Keep largest connected component
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_component = np.argmax(sizes) + 1
            mask = (labeled == largest_component).astype(np.uint8)
        
        return mask
    
    def __call__(
        self, 
        image_array: np.ndarray, 
        image_path: Optional[Path] = None,
        mask_array: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply skull stripping."""
        if mask_array is None and image_path is not None:
            # Try to find pre-computed mask
            mask_path = self.find_mask(image_path)
            if mask_path is not None and mask_path.exists():
                mask_nib = nib.load(str(mask_path))
                mask_array = mask_nib.get_fdata()
            else:
                # Generate simple mask
                logger.warning(f"No mask found for {image_path}. Using simple thresholding.")
                mask_array = self.create_simple_mask(image_array)
        elif mask_array is None:
            # Generate simple mask
            mask_array = self.create_simple_mask(image_array)
        
        # Apply mask
        return image_array * mask_array


class IntensityNormalization:
    """
    Intensity normalization methods for MRI.
    """
    def __init__(
        self,
        method: str = "zscore",
        percentile_lower: float = 1.0,
        percentile_upper: float = 99.0,
        clip_lower: float = 0.5,
        clip_upper: float = 99.5,
    ):
        self.method = method
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
    
    def __call__(self, image_array: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize intensity values.
        
        Args:
            image_array: Input image
            mask: Optional mask to compute statistics only within brain
        """
        # Work with a copy
        normalized = image_array.copy()
        
        # Define region of interest for statistics
        if mask is not None:
            roi = image_array[mask > 0]
        else:
            roi = image_array[image_array > 0]
        
        if len(roi) == 0:
            logger.warning("Empty ROI for normalization")
            return normalized
        
        # Clip outliers before normalization
        if self.clip_lower is not None and self.clip_upper is not None:
            lower_val = np.percentile(roi, self.clip_lower)
            upper_val = np.percentile(roi, self.clip_upper)
            normalized = np.clip(normalized, lower_val, upper_val)
            roi = normalized[mask > 0] if mask is not None else normalized[normalized > 0]
        
        # Apply normalization
        if self.method == "zscore":
            mean = np.mean(roi)
            std = np.std(roi)
            if std > 0:
                normalized = (normalized - mean) / std
            else:
                logger.warning("Zero standard deviation, skipping normalization")
        
        elif self.method == "minmax":
            min_val = np.min(roi)
            max_val = np.max(roi)
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
            else:
                logger.warning("Zero range, skipping normalization")
        
        elif self.method == "percentile":
            lower_val = np.percentile(roi, self.percentile_lower)
            upper_val = np.percentile(roi, self.percentile_upper)
            if upper_val > lower_val:
                normalized = (normalized - lower_val) / (upper_val - lower_val)
                normalized = np.clip(normalized, 0, 1)
            else:
                logger.warning("Zero percentile range, skipping normalization")
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return normalized.astype(np.float32)


class MRIPreprocessor:
    """
    Complete preprocessing pipeline for brain MRI.
    Ensures deterministic and reproducible preprocessing.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Initialize components
        if config.use_bias_correction:
            self.bias_corrector = N4BiasFieldCorrection(
                num_iterations=config.n4_iterations,
                convergence_threshold=config.n4_convergence_threshold,
            )
        else:
            self.bias_corrector = None
        
        if config.use_skull_stripping:
            self.skull_stripper = SkullStripping(mask_pattern=config.brain_mask_pattern)
        else:
            self.skull_stripper = None
        
        self.normalizer = IntensityNormalization(
            method=config.normalization_method,
            percentile_lower=config.percentile_lower,
            percentile_upper=config.percentile_upper,
            clip_lower=config.clip_lower_percentile,
            clip_upper=config.clip_upper_percentile,
        )
        
        # Build MONAI transform pipeline
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> Compose:
        """Build MONAI preprocessing transform pipeline."""
        transforms_list = [
            LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Orientationd(keys=["image"], axcodes=self.config.target_orientation),
        ]
        
        # Add resampling if target spacing is specified
        if self.config.target_spacing is not None:
            transforms_list.append(
                Spacingd(
                    keys=["image"],
                    pixdim=self.config.target_spacing,
                    mode=self.config.interpolation_mode,
                )
            )
        
        # Add padding/cropping if target size is specified
        if self.config.target_size is not None:
            transforms_list.append(
                ResizeWithPadOrCropd(
                    keys=["image"],
                    spatial_size=self.config.target_size,
                )
            )
        
        transforms_list.append(EnsureTyped(keys=["image"], dtype=torch.float32))
        
        return Compose(transforms_list)
    
    def preprocess_single(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Preprocess a single MRI volume.
        
        Args:
            image_path: Path to input NIfTI file
            output_path: Path to save preprocessed output
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        image_path = Path(image_path)
        logger.info(f"Preprocessing: {image_path.name}")
        
        # Load image
        nib_image = nib.load(str(image_path))
        image_array = nib_image.get_fdata().astype(np.float32)
        original_shape = image_array.shape
        original_affine = nib_image.affine
        
        # Step 1: Bias field correction
        if self.bias_corrector is not None:
            logger.debug("Applying N4 bias field correction...")
            image_array = self.bias_corrector(image_path)
            
            if save_intermediate and output_path:
                intermediate_path = Path(output_path).parent / f"{image_path.stem}_bias_corrected.nii.gz"
                nib.save(nib.Nifti1Image(image_array, original_affine), str(intermediate_path))
        
        # Step 2: Skull stripping
        mask = None
        if self.skull_stripper is not None:
            logger.debug("Applying skull stripping...")
            image_array = self.skull_stripper(image_array, image_path)
            
            if save_intermediate and output_path:
                intermediate_path = Path(output_path).parent / f"{image_path.stem}_skull_stripped.nii.gz"
                nib.save(nib.Nifti1Image(image_array, original_affine), str(intermediate_path))
        
        # Step 3: Intensity normalization
        logger.debug(f"Applying {self.config.normalization_method} normalization...")
        image_array = self.normalizer(image_array, mask)
        
        # Step 4: Reorientation and resampling (MONAI)
        logger.debug("Applying spatial transforms...")
        data_dict = {"image": image_path}
        
        # Apply MONAI transforms
        transformed = self.transforms(data_dict)
        processed_image = transformed["image"]
        
        # Replace intensity values with normalized values
        # Need to reshape normalized array to match transformed shape
        if processed_image.shape[1:] == original_shape:
            # If no resampling was done, directly use normalized array
            processed_image[0] = torch.from_numpy(image_array)
        else:
            # If resampling was done, apply normalization after
            processed_array = processed_image[0].numpy()
            processed_array = self.normalizer(processed_array)
            processed_image[0] = torch.from_numpy(processed_array)
        
        # Compute statistics
        stats = self._compute_statistics(processed_image[0].numpy())
        
        # Save if output path specified
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy and save
            output_array = processed_image[0].numpy()
            
            # Get affine from metadata if available
            if hasattr(processed_image, 'affine'):
                output_affine = processed_image.affine.numpy()
            else:
                output_affine = original_affine
            
            nib.save(
                nib.Nifti1Image(output_array, output_affine),
                str(output_path)
            )
            
            # Save metadata
            metadata_path = output_path.parent / f"{output_path.stem.replace('.nii', '')}_metadata.json"
            metadata = {
                "original_path": str(image_path),
                "original_shape": original_shape,
                "processed_shape": tuple(output_array.shape),
                "statistics": stats,
                "preprocessing_config": {
                    "orientation": self.config.target_orientation,
                    "spacing": self.config.target_spacing,
                    "normalization": self.config.normalization_method,
                    "bias_correction": self.config.use_bias_correction,
                    "skull_stripping": self.config.use_skull_stripping,
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved to: {output_path}")
        
        return {
            "image": processed_image,
            "metadata": transformed.get("image_meta_dict", {}),
            "statistics": stats,
            "original_shape": original_shape,
            "processed_shape": tuple(processed_image.shape),
        }
    
    def _compute_statistics(self, image_array: np.ndarray) -> Dict[str, float]:
        """Compute image statistics."""
        nonzero = image_array[image_array != 0]
        
        if len(nonzero) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }
        
        return {
            "mean": float(np.mean(nonzero)),
            "std": float(np.std(nonzero)),
            "min": float(np.min(nonzero)),
            "max": float(np.max(nonzero)),
            "median": float(np.median(nonzero)),
            "p01": float(np.percentile(nonzero, 1)),
            "p99": float(np.percentile(nonzero, 99)),
        }
    
    def preprocess_dataset(
        self,
        input_paths: list,
        output_dir: Path,
        save_intermediate: bool = False,
    ) -> list:
        """
        Preprocess multiple volumes.
        
        Args:
            input_paths: List of input NIfTI file paths
            output_dir: Directory to save preprocessed outputs
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of preprocessed file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for input_path in tqdm(input_paths, desc="Preprocessing volumes"):
            input_path = Path(input_path)
            
            # Create output filename
            output_filename = input_path.name.replace("_defaced", "_preprocessed")
            output_path = output_dir / input_filename
            
            # Preprocess
            try:
                self.preprocess_single(
                    input_path,
                    output_path,
                    save_intermediate=save_intermediate
                )
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to preprocess {input_path}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(output_paths)}/{len(input_paths)} volumes")
        
        return output_paths


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_default_config()
    preprocessor = MRIPreprocessor(config.preprocessing)
    
    # Test on a single file
    test_file = Path(r"d:\11PrabeshX\Projects\major_\Nifti\sub-01\ses-1\anat\sub-01_ses-1_T1w_defaced.nii.gz")
    
    if test_file.exists():
        output_path = config.data.output_root / "test_output.nii.gz"
        result = preprocessor.preprocess_single(test_file, output_path, save_intermediate=True)
        
        print(f"Original shape: {result['original_shape']}")
        print(f"Processed shape: {result['processed_shape']}")
        print(f"Statistics: {result['statistics']}")
    else:
        print(f"Test file not found: {test_file}")
