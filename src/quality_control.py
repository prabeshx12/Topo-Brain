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
