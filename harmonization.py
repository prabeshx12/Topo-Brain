"""
Intensity harmonization for multi-field strength MRI data.
Handles differences between 3T and 7T scans.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Literal
import pickle

import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import nibabel as nib

logger = logging.getLogger(__name__)


class HistogramMatcher:
    """
    Match image histograms to a reference distribution.
    Useful for harmonizing intensity distributions across scanners.
    """
    
    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins
        self.reference_cdf = None
        self.reference_bins = None
        
    def fit(self, reference_images: List[np.ndarray]) -> 'HistogramMatcher':
        """
        Learn reference histogram from a set of images.
        
        Args:
            reference_images: List of reference image arrays
        """
        # Collect all non-zero values
        all_values = []
        for img in reference_images:
            values = img[img > 0].flatten()
            all_values.extend(values)
        
        all_values = np.array(all_values)
        
        # Compute reference histogram and CDF
        hist, bins = np.histogram(all_values, bins=self.n_bins, density=True)
        
        # Compute cumulative distribution
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]  # Normalize
        
        # Store reference
        self.reference_bins = bins[:-1]  # Remove last edge
        self.reference_cdf = cdf
        
        logger.info(f"Fitted histogram matcher on {len(reference_images)} images")
        
        return self
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Match image histogram to reference.
        
        Args:
            image: Input image array
            
        Returns:
            Histogram-matched image
        """
        if self.reference_cdf is None:
            raise ValueError("Must call fit() before transform()")
        
        # Get non-zero mask
        nonzero_mask = image > 0
        
        if not np.any(nonzero_mask):
            return image.copy()
        
        # Compute source histogram and CDF
        values = image[nonzero_mask]
        hist, bins = np.histogram(values, bins=self.n_bins, density=True)
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        
        # Interpolate to match reference
        # Map source CDF to reference bins
        interp_func = interp1d(
            cdf,
            bins[:-1],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Find corresponding reference values
        matched_values = interp_func(self.reference_cdf)
        
        # Create mapping from source to matched values
        mapping_func = interp1d(
            bins[:-1],
            matched_values,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Apply mapping
        result = image.copy()
        result[nonzero_mask] = mapping_func(values)
        
        return result.astype(image.dtype)
    
    def save(self, path: Path) -> None:
        """Save fitted matcher to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'reference_cdf': self.reference_cdf,
                'reference_bins': self.reference_bins,
                'n_bins': self.n_bins,
            }, f)
        logger.info(f"Saved histogram matcher to {path}")
    
    def load(self, path: Path) -> 'HistogramMatcher':
        """Load fitted matcher from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.reference_cdf = data['reference_cdf']
        self.reference_bins = data['reference_bins']
        self.n_bins = data['n_bins']
        
        logger.info(f"Loaded histogram matcher from {path}")
        return self


class IntensityHarmonizer:
    """
    Harmonize intensity distributions across different field strengths.
    
    This is important when training models on mixed 3T/7T data or
    when applying models trained on one field strength to another.
    """
    
    def __init__(
        self,
        method: Literal['histogram', 'zscore', 'quantile'] = 'histogram',
        reference_field_strength: str = '3T',
    ):
        """
        Initialize harmonizer.
        
        Args:
            method: Harmonization method
                - 'histogram': Histogram matching
                - 'zscore': Z-score normalization per field strength
                - 'quantile': Quantile normalization
            reference_field_strength: Reference field strength ('3T' or '7T')
        """
        self.method = method
        self.reference_field_strength = reference_field_strength
        
        if method == 'histogram':
            self.matcher = HistogramMatcher()
        elif method == 'zscore':
            self.scalers = {}
        elif method == 'quantile':
            self.quantiles = {}
        
        self.is_fitted = False
    
    def fit(
        self,
        images_by_field_strength: Dict[str, List[np.ndarray]]
    ) -> 'IntensityHarmonizer':
        """
        Fit harmonizer on training data.
        
        Args:
            images_by_field_strength: Dict mapping field strength to list of images
                e.g., {'3T': [img1, img2, ...], '7T': [img3, img4, ...]}
        """
        if self.method == 'histogram':
            # Fit on reference field strength
            if self.reference_field_strength in images_by_field_strength:
                reference_images = images_by_field_strength[self.reference_field_strength]
                self.matcher.fit(reference_images)
            else:
                raise ValueError(f"Reference field strength {self.reference_field_strength} not found")
        
        elif self.method == 'zscore':
            # Compute mean/std for each field strength
            for field_strength, images in images_by_field_strength.items():
                all_values = []
                for img in images:
                    all_values.extend(img[img > 0].flatten())
                
                all_values = np.array(all_values).reshape(-1, 1)
                
                scaler = StandardScaler()
                scaler.fit(all_values)
                
                self.scalers[field_strength] = scaler
                
                logger.info(f"{field_strength}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
        
        elif self.method == 'quantile':
            # Compute quantiles for each field strength
            quantile_points = np.linspace(0, 1, 100)
            
            for field_strength, images in images_by_field_strength.items():
                all_values = []
                for img in images:
                    all_values.extend(img[img > 0].flatten())
                
                all_values = np.array(all_values)
                quantiles = np.percentile(all_values, quantile_points * 100)
                
                self.quantiles[field_strength] = {
                    'points': quantile_points,
                    'values': quantiles,
                }
        
        self.is_fitted = True
        logger.info(f"Fitted {self.method} harmonizer")
        
        return self
    
    def transform(
        self,
        image: np.ndarray,
        field_strength: str,
    ) -> np.ndarray:
        """
        Harmonize image from specified field strength.
        
        Args:
            image: Input image
            field_strength: Field strength of input image ('3T' or '7T')
            
        Returns:
            Harmonized image
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        if self.method == 'histogram':
            # Match to reference histogram
            if field_strength != self.reference_field_strength:
                return self.matcher.transform(image)
            else:
                return image.copy()
        
        elif self.method == 'zscore':
            # Apply z-score normalization
            if field_strength not in self.scalers:
                logger.warning(f"Unknown field strength: {field_strength}, returning original")
                return image.copy()
            
            nonzero_mask = image > 0
            result = image.copy()
            
            if np.any(nonzero_mask):
                scaler = self.scalers[field_strength]
                values = image[nonzero_mask].reshape(-1, 1)
                normalized = scaler.transform(values).flatten()
                result[nonzero_mask] = normalized
            
            return result
        
        elif self.method == 'quantile':
            # Quantile normalization
            if field_strength not in self.quantiles:
                logger.warning(f"Unknown field strength: {field_strength}, returning original")
                return image.copy()
            
            source_quantiles = self.quantiles[field_strength]
            target_quantiles = self.quantiles[self.reference_field_strength]
            
            nonzero_mask = image > 0
            result = image.copy()
            
            if np.any(nonzero_mask):
                values = image[nonzero_mask]
                
                # Map to quantiles
                source_interp = interp1d(
                    source_quantiles['values'],
                    source_quantiles['points'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(0, 1),
                )
                
                quantile_positions = source_interp(values)
                
                # Map to target values
                target_interp = interp1d(
                    target_quantiles['points'],
                    target_quantiles['values'],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate',
                )
                
                harmonized_values = target_interp(quantile_positions)
                result[nonzero_mask] = harmonized_values
            
            return result
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def save(self, path: Path) -> None:
        """Save fitted harmonizer."""
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'reference_field_strength': self.reference_field_strength,
                'is_fitted': self.is_fitted,
                'matcher': self.matcher if self.method == 'histogram' else None,
                'scalers': self.scalers if self.method == 'zscore' else None,
                'quantiles': self.quantiles if self.method == 'quantile' else None,
            }, f)
        logger.info(f"Saved harmonizer to {path}")
    
    def load(self, path: Path) -> 'IntensityHarmonizer':
        """Load fitted harmonizer."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.method = data['method']
        self.reference_field_strength = data['reference_field_strength']
        self.is_fitted = data['is_fitted']
        
        if self.method == 'histogram':
            self.matcher = data['matcher']
        elif self.method == 'zscore':
            self.scalers = data['scalers']
        elif self.method == 'quantile':
            self.quantiles = data['quantiles']
        
        logger.info(f"Loaded harmonizer from {path}")
        return self


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_default_config()
    
    # Collect images by field strength
    images_3t = []
    images_7t = []
    
    for nifti_file in list(config.data.data_root.glob("**/ses-1/anat/*T1w*.nii.gz"))[:3]:
        img = nib.load(str(nifti_file)).get_fdata().astype(np.float32)
        images_3t.append(img)
    
    for nifti_file in list(config.data.data_root.glob("**/ses-2/anat/*T1w*.nii.gz"))[:3]:
        img = nib.load(str(nifti_file)).get_fdata().astype(np.float32)
        images_7t.append(img)
    
    # Fit harmonizer
    harmonizer = IntensityHarmonizer(method='histogram', reference_field_strength='3T')
    harmonizer.fit({
        '3T': images_3t,
        '7T': images_7t,
    })
    
    # Transform a 7T image
    if len(images_7t) > 0:
        harmonized = harmonizer.transform(images_7t[0], field_strength='7T')
        
        print(f"Original 7T - Mean: {np.mean(images_7t[0][images_7t[0] > 0]):.2f}")
        print(f"Harmonized - Mean: {np.mean(harmonized[harmonized > 0]):.2f}")
        print(f"Reference 3T - Mean: {np.mean(images_3t[0][images_3t[0] > 0]):.2f}")
