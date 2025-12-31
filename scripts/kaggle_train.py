"""
Kaggle/Colab Training Script for 3T → 7T MRI Super-Resolution GAN.

This script auto-detects preprocessed data in Kaggle/Colab environments
and handles the automatic tar extraction that these platforms perform.

Usage on Kaggle:
    !python Topo-Brain/scripts/kaggle_train.py --epochs 100
    
Usage on Colab:
    !python /content/Topo-Brain/scripts/kaggle_train.py --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path
import time
import os
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Setup basic logging first
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def detect_environment():
    """Detect running environment (Kaggle, Colab, or Local)."""
    IS_KAGGLE = Path('/kaggle/input').exists()
    IS_COLAB = Path('/content').exists() and not IS_KAGGLE
    
    if IS_KAGGLE:
        env = 'kaggle'
        base_dir = Path('/kaggle/working')
    elif IS_COLAB:
        env = 'colab'
        base_dir = Path('/content')
    else:
        env = 'local'
        base_dir = Path.cwd()
    
    return env, base_dir, IS_KAGGLE, IS_COLAB


def find_preprocessed_data(verbose=True):
    """
    Auto-detect preprocessed NIfTI files.
    Kaggle auto-extracts tar files, so we scan for actual data.
    
    Returns:
        tuple: (data_path, list_of_nifti_files) or (None, []) if not found
    """
    env, base_dir, IS_KAGGLE, IS_COLAB = detect_environment()
    
    if verbose:
        print("=" * 70)
        print("AUTO-DETECTING PREPROCESSED DATA")
        print("=" * 70)
        print(f"Environment: {env.upper()}")
    
    # Try multiple file patterns (in order of preference)
    patterns = ['*_preprocessed.nii.gz', '*.nii.gz', '*.nii']
    
    def search_directory(search_dir, verbose_name=""):
        """Search a directory for NIfTI files with various patterns."""
        for pattern in patterns:
            nifti_files = list(search_dir.rglob(pattern))
            if nifti_files:
                if verbose:
                    print(f"   ✓ Found {len(nifti_files)} files matching '{pattern}' in {verbose_name}")
                    for f in nifti_files[:3]:
                        print(f"      - {f.name}")
                    if len(nifti_files) > 3:
                        print(f"      ... and {len(nifti_files) - 3} more")
                
                # Find the root containing subject folders (sub-XX)
                for nifti in nifti_files:
                    for parent in [nifti.parent, nifti.parent.parent, 
                                   nifti.parent.parent.parent, search_dir]:
                        if parent.exists():
                            sub_dirs = [d for d in parent.iterdir() 
                                       if d.is_dir() and d.name.startswith('sub-')]
                            if sub_dirs:
                                if verbose:
                                    print(f"   ✓ Found subject folders in: {parent}")
                                return parent, nifti_files
                
                # No sub- folders found, return the search directory
                return search_dir, nifti_files
        return None, []
    
    if IS_KAGGLE:
        kaggle_input = Path('/kaggle/input')
        if verbose:
            print(f"🔍 Scanning Kaggle input directories...")
        
        if kaggle_input.exists():
            datasets = [d for d in kaggle_input.iterdir() if d.is_dir()]
            if verbose:
                print(f"   Found {len(datasets)} datasets: {[d.name for d in datasets]}")
            
            # Search each dataset
            for dataset_dir in datasets:
                if verbose:
                    print(f"\n   Searching '{dataset_dir.name}'...")
                result_path, nifti_files = search_directory(dataset_dir, dataset_dir.name)
                if result_path:
                    return result_path, nifti_files
        
        # Check working directory
        for working_path in [Path('/kaggle/working/preprocessed'), 
                            Path('/kaggle/working/preprocessed_no_n4')]:
            if working_path.exists():
                result_path, nifti_files = search_directory(working_path, str(working_path))
                if result_path:
                    return result_path, nifti_files
    
    elif IS_COLAB:
        for colab_path in [Path('/content/preprocessed'),
                          Path('/content/drive/MyDrive/preprocessed')]:
            if colab_path.exists():
                result_path, nifti_files = search_directory(colab_path, str(colab_path))
                if result_path:
                    return result_path, nifti_files
    
    else:
        # Local paths
        for local_path in [Path('./preprocessed'), Path('./preprocessed_registered'),
                          _project_root / 'preprocessed']:
            if local_path.exists():
                result_path, nifti_files = search_directory(local_path, str(local_path))
                if result_path:
                    return result_path, nifti_files
    
    return None, []


def show_available_data():
    """Show what data is available in the environment (for debugging)."""
    env, base_dir, IS_KAGGLE, IS_COLAB = detect_environment()
    
    print("\n🔍 Available data locations:")
    
    if IS_KAGGLE:
        kaggle_input = Path('/kaggle/input')
        if kaggle_input.exists():
            print(f"\n📁 /kaggle/input/ contents:")
            for d in kaggle_input.iterdir():
                if d.is_dir():
                    all_files = list(d.rglob('*'))
                    nifti_files = [f for f in all_files if f.suffix in ['.gz', '.nii']]
                    print(f"   {d.name}/")
                    print(f"      Total files: {len(all_files)}, NIfTI-like: {len(nifti_files)}")
                    # Show first few files
                    for f in list(d.iterdir())[:5]:
                        if f.is_dir():
                            print(f"      📁 {f.name}/")
                        else:
                            print(f"      📄 {f.name}")


# =============================================================================
# DATASET
# =============================================================================

class Paired3T7TDataset(Dataset):
    """
    PyTorch Dataset for paired 3T-7T MRI volumes.
    Extracts random 3D patches for training.
    """
    def __init__(
        self,
        data_pairs,
        patch_size=(64, 64, 64),
        num_patches_per_volume=10,
        transform=None,
        cache_data=False,
    ):
        self.data_pairs = data_pairs
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        
        self.total_patches = len(data_pairs) * num_patches_per_volume
        
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        pair_idx = idx // self.num_patches_per_volume
        pair = self.data_pairs[pair_idx]
        
        if self.cache_data and pair_idx in self.cache:
            vol_3t, vol_7t = self.cache[pair_idx]
        else:
            vol_3t = nib.load(pair['input_3t']).get_fdata().astype(np.float32)
            vol_7t = nib.load(pair['target_7t']).get_fdata().astype(np.float32)
            
            if self.cache_data:
                self.cache[pair_idx] = (vol_3t, vol_7t)
        
        patch_3t, patch_7t = self._extract_random_patch(vol_3t, vol_7t)
        
        patch_3t = torch.from_numpy(patch_3t[None, ...])
        patch_7t = torch.from_numpy(patch_7t[None, ...])
        
        if self.transform:
            patch_3t = self.transform(patch_3t)
            patch_7t = self.transform(patch_7t)
        
        return {
            'input_3t': patch_3t,
            'target_7t': patch_7t,
            'subject': pair.get('subject', 'unknown'),
        }
    
    def _extract_random_patch(self, vol_3t, vol_7t):
        """Extract patch, biased towards brain tissue (non-empty regions)."""
        d, h, w = vol_3t.shape
        pd, ph, pw = self.patch_size
        
        # Try up to 10 times to find a good patch
        best_patch_3t = None
        best_patch_7t = None
        best_mean = 0.0
        
        for _ in range(10):
            # Bias towards center (brain is typically in center)
            # Use truncated normal distribution centered on volume center
            center_d, center_h, center_w = d // 2, h // 2, w // 2
            
            # Random offset from center with some spread
            spread = 0.3  # 30% of volume size
            start_d = int(center_d + random.gauss(0, d * spread) - pd // 2)
            start_h = int(center_h + random.gauss(0, h * spread) - ph // 2)
            start_w = int(center_w + random.gauss(0, w * spread) - pw // 2)
            
            # Clamp to valid range
            start_d = max(0, min(start_d, d - pd))
            start_h = max(0, min(start_h, h - ph))
            start_w = max(0, min(start_w, w - pw))
            
            patch_3t = vol_3t[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
            patch_7t = vol_7t[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
            
            # Check if this patch has good brain content
            patch_mean = np.mean(patch_3t)
            if patch_mean > best_mean:
                best_mean = patch_mean
                best_patch_3t = patch_3t
                best_patch_7t = patch_7t
            
            # If patch has good content (mean > 0.1 of max), use it
            if patch_mean > 0.1 * np.max(vol_3t):
                break
        
        # Use best found patch
        patch_3t = best_patch_3t if best_patch_3t is not None else patch_3t
        patch_7t = best_patch_7t if best_patch_7t is not None else patch_7t
        
        if patch_3t.shape != self.patch_size:
            patch_3t = self._pad_to_size(patch_3t, self.patch_size)
            patch_7t = self._pad_to_size(patch_7t, self.patch_size)
        
        return patch_3t, patch_7t
    
    def _pad_to_size(self, volume, target_size):
        pad_width = []
        for i in range(3):
            diff = target_size[i] - volume.shape[i]
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        
        return np.pad(volume, pad_width, mode='constant', constant_values=0)


def create_paired_data_list(preprocessed_dir, modalities):
    """
    Create list of paired 3T-7T volumes.
    Assumes: ses-1 = 3T, ses-2 = 7T
    Handles files with or without '_preprocessed' suffix.
    """
    preprocessed_dir = Path(preprocessed_dir)
    
    if isinstance(modalities, str):
        modalities = [modalities]
    
    all_pairs = []
    
    # Try multiple patterns for each modality (both compressed and uncompressed)
    file_patterns = [
        '*{modality}_preprocessed.nii.gz',
        '*{modality}_preprocessed.nii',  # Uncompressed
        '*{modality}.nii.gz',
        '*{modality}.nii',  # Uncompressed
        '*{modality}_*.nii.gz',
        '*{modality}_*.nii',  # Uncompressed
    ]
    
    for modality in modalities:
        print(f"🔍 Searching for {modality} pairs...")
        
        files_by_subject = {}
        
        # Try each pattern
        for pattern_template in file_patterns:
            pattern = pattern_template.format(modality=modality)
            for file in preprocessed_dir.rglob(pattern):
                # Parse filename to extract subject and session
                # Expected formats:
                #   sub-01_ses-1_T1w_preprocessed.nii.gz
                #   sub-01_ses-1_T1w.nii.gz
                stem = file.stem.replace('.nii', '')  # Handle .nii.gz
                
                # Remove common suffixes
                for suffix in ['_preprocessed', '_registered', '_brain']:
                    stem = stem.replace(suffix, '')
                
                parts = stem.split('_')
                
                # Find subject and session
                subject = None
                session = None
                for part in parts:
                    if part.startswith('sub-'):
                        subject = part
                    elif part.startswith('ses-'):
                        session = part
                
                if subject and session:
                    if subject not in files_by_subject:
                        files_by_subject[subject] = {}
                    # Only store if not already found (prefer _preprocessed)
                    if session not in files_by_subject[subject]:
                        files_by_subject[subject][session] = file
        
        # Create pairs from found files
        for subject, sessions in files_by_subject.items():
            if 'ses-1' in sessions and 'ses-2' in sessions:
                all_pairs.append({
                    'subject': subject,
                    'input_3t': str(sessions['ses-1']),
                    'target_7t': str(sessions['ses-2']),
                    'modality': modality,
                })
        
        n_pairs = sum(1 for p in all_pairs if p['modality'] == modality)
        print(f"   ✓ Found {n_pairs} {modality} pairs")
        
        # Show sample files found
        if files_by_subject:
            sample_subj = list(files_by_subject.keys())[0]
            sample_files = files_by_subject[sample_subj]
            print(f"   Sample: {sample_subj} has sessions: {list(sample_files.keys())}")
    
    return all_pairs


# =============================================================================
# MODELS (Simplified fallbacks - use actual models if available)
# =============================================================================

try:
    from models.generator_unet3d import UNet3DGenerator
    from models.discriminator_patchgan3d import PatchGANDiscriminator3D
    print("✓ Loaded models from repository")
except ImportError:
    print("⚠️ Using simplified fallback models")
    
    class UNet3DGenerator(nn.Module):
        """Simplified 3D U-Net Generator."""
        def __init__(self, in_channels=1, out_channels=1, base_features=32, **kwargs):
            super().__init__()
            bf = base_features
            
            # Encoder
            self.enc1 = self._conv_block(in_channels, bf)
            self.enc2 = self._conv_block(bf, bf * 2)
            self.enc3 = self._conv_block(bf * 2, bf * 4)
            
            # Bottleneck
            self.bottleneck = self._conv_block(bf * 4, bf * 8)
            
            # Decoder
            self.up3 = nn.ConvTranspose3d(bf * 8, bf * 4, 2, stride=2)
            self.dec3 = self._conv_block(bf * 8, bf * 4)
            self.up2 = nn.ConvTranspose3d(bf * 4, bf * 2, 2, stride=2)
            self.dec2 = self._conv_block(bf * 4, bf * 2)
            self.up1 = nn.ConvTranspose3d(bf * 2, bf, 2, stride=2)
            self.dec1 = self._conv_block(bf * 2, bf)
            
            self.out = nn.Conv3d(bf, out_channels, 1)
            self.pool = nn.MaxPool3d(2)
        
        def _conv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            
            b = self.bottleneck(self.pool(e3))
            
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            
            return self.out(d1)
        
        def get_num_parameters(self):
            return sum(p.numel() for p in self.parameters())
    
    class PatchGANDiscriminator3D(nn.Module):
        """Simplified 3D PatchGAN Discriminator."""
        def __init__(self, in_channels=1, base_features=64, **kwargs):
            super().__init__()
            bf = base_features
            
            self.model = nn.Sequential(
                nn.Conv3d(in_channels, bf, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(bf, bf * 2, 4, stride=2, padding=1),
                nn.InstanceNorm3d(bf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(bf * 2, bf * 4, 4, stride=2, padding=1),
                nn.InstanceNorm3d(bf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv3d(bf * 4, 1, 4, stride=1, padding=1),
            )
        
        def forward(self, x):
            return self.model(x)
        
        def get_num_parameters(self):
            return sum(p.numel() for p in self.parameters())


# =============================================================================
# TRAINING
# =============================================================================

def train_gan(args):
    """Main training function."""
    
    # Detect environment
    env, base_dir, IS_KAGGLE, IS_COLAB = detect_environment()
    print(f"\n{'='*70}")
    print(f"TOPO-BRAIN GAN TRAINING")
    print(f"{'='*70}")
    print(f"Environment: {env.upper()}")
    print(f"Base directory: {base_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Find data
    data_path, nifti_files = find_preprocessed_data(verbose=True)
    
    if data_path is None:
        print("\n❌ ERROR: No preprocessed data found!")
        show_available_data()
        raise FileNotFoundError("No preprocessed NIfTI files found!")
    
    print(f"\n✓ Found {len(nifti_files)} preprocessed files")
    print(f"  Data path: {data_path}")
    
    # Create output directories
    output_dir = base_dir / 'gan_output'
    checkpoint_dir = output_dir / 'checkpoints'
    vis_dir = output_dir / 'visualizations'
    log_dir = output_dir / 'logs'
    
    for d in [checkpoint_dir, vis_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output: {output_dir}")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create data pairs
    print(f"\n{'='*70}")
    print("CREATING DATA PAIRS")
    print(f"{'='*70}")
    
    all_pairs = create_paired_data_list(data_path, args.modalities)
    
    if len(all_pairs) == 0:
        raise ValueError("No paired 3T-7T data found! Check your data structure.")
    
    print(f"✓ Total pairs: {len(all_pairs)}")
    
    # Split data (patient-level)
    subjects = list(set([p['subject'] for p in all_pairs]))
    random.shuffle(subjects)
    
    n_train = int(len(subjects) * 0.6)
    n_val = int(len(subjects) * 0.2)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    train_pairs = [p for p in all_pairs if p['subject'] in train_subjects]
    val_pairs = [p for p in all_pairs if p['subject'] in val_subjects]
    
    print(f"✓ Train: {len(train_pairs)} pairs from {len(train_subjects)} subjects")
    print(f"✓ Val: {len(val_pairs)} pairs from {len(val_subjects)} subjects")
    
    # Save split info
    split_info = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'created': datetime.now().isoformat(),
    }
    with open(output_dir / 'data_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create datasets
    print(f"\n{'='*70}")
    print("CREATING DATALOADERS")
    print(f"{'='*70}")
    
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    train_dataset = Paired3T7TDataset(
        data_pairs=train_pairs,
        patch_size=patch_size,
        num_patches_per_volume=args.num_patches,
        cache_data=False,
    )
    
    val_dataset = Paired3T7TDataset(
        data_pairs=val_pairs,
        patch_size=patch_size,
        num_patches_per_volume=5,
        cache_data=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create models
    print(f"\n{'='*70}")
    print("INITIALIZING MODELS")
    print(f"{'='*70}")
    
    generator = UNet3DGenerator(
        in_channels=1,
        out_channels=1,
        base_features=32,
    ).to(device)
    
    discriminator = PatchGANDiscriminator3D(
        in_channels=1,
        base_features=64,
    ).to(device)
    
    print(f"✓ Generator params: {generator.get_num_parameters():,}")
    print(f"✓ Discriminator params: {discriminator.get_num_parameters():,}")
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_l1 = nn.L1Loss()
    criterion_adv = nn.MSELoss()  # LSGAN
    
    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler_g = GradScaler(enabled=use_amp)
    scaler_d = GradScaler(enabled=use_amp)
    
    print(f"✓ Mixed precision: {use_amp}")
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Patch size: {patch_size}")
    
    # Training dynamics configuration
    WARMUP_EPOCHS = 20  # L1 only for first 20 epochs
    LABEL_SMOOTHING = 0.9  # Real labels: 1.0 -> 0.9
    G_UPDATES_PER_D = 2  # Train G twice per D update
    GRADIENT_CLIP = 1.0  # Gradient clipping value
    LAMBDA_L1 = 100.0  # L1 loss weight
    LAMBDA_ADV = 1.0  # Adversarial loss weight
    
    print(f"Warmup epochs: {WARMUP_EPOCHS} (L1 only)")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"G:D update ratio: {G_UPDATES_PER_D}:1")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Check if in warmup phase
        in_warmup = epoch <= WARMUP_EPOCHS
        
        # Train
        generator.train()
        discriminator.train()
        
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_loss_l1 = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for batch in pbar:
            input_3t = batch['input_3t'].to(device)
            target_7t = batch['target_7t'].to(device)
            
            # ===== Train Discriminator (skip during warmup) =====
            if not in_warmup:
                optimizer_d.zero_grad()
                
                with autocast(enabled=use_amp):
                    with torch.no_grad():
                        fake_7t = generator(input_3t)
                    
                    pred_real = discriminator(target_7t)
                    pred_fake = discriminator(fake_7t.detach())
                    
                    # Labels with smoothing
                    real_label = torch.ones_like(pred_real) * LABEL_SMOOTHING
                    fake_label = torch.zeros_like(pred_fake)
                    
                    loss_d = 0.5 * (criterion_adv(pred_real, real_label) + 
                                   criterion_adv(pred_fake, fake_label))
                
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRADIENT_CLIP)
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d = torch.tensor(0.0)
            
            # ===== Train Generator (multiple times per D update) =====
            for g_step in range(G_UPDATES_PER_D):
                optimizer_g.zero_grad()
                
                with autocast(enabled=use_amp):
                    fake_7t = generator(input_3t)
                    loss_l1 = criterion_l1(fake_7t, target_7t)
                    
                    if not in_warmup:
                        pred_fake = discriminator(fake_7t)
                        real_label = torch.ones_like(pred_fake)
                        loss_adv = criterion_adv(pred_fake, real_label)
                    else:
                        loss_adv = torch.tensor(0.0, device=device)
                    
                    loss_g = LAMBDA_L1 * loss_l1 + LAMBDA_ADV * loss_adv
                
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRADIENT_CLIP)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item() if isinstance(loss_d, torch.Tensor) else loss_d
            epoch_loss_l1 += loss_l1.item()
            
            if in_warmup:
                pbar.set_postfix({'L1': f'{loss_l1.item():.4f}', 'Phase': 'Warmup'})
            else:
                pbar.set_postfix({'G': f'{loss_g.item():.3f}', 'D': f'{loss_d.item():.3f}'})
        
        avg_loss_g = epoch_loss_g / len(train_loader)
        avg_loss_d = epoch_loss_d / len(train_loader)
        train_losses.append(avg_loss_g)
        
        # Validate
        generator.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_3t = batch['input_3t'].to(device)
                target_7t = batch['target_7t'].to(device)
                fake_7t = generator(input_3t)
                val_loss += criterion_l1(fake_7t, target_7t).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        avg_loss_l1 = epoch_loss_l1 / len(train_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Logging with warmup indicator
        phase_str = "[WARMUP] " if in_warmup else ""
        print(f"{phase_str}Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
              f"G: {avg_loss_g:.4f}, D: {avg_loss_d:.4f}, L1: {avg_loss_l1:.4f}, Val L1: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pth')
            print(f"   ✓ New best model saved!")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth')
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best validation L1: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Kaggle/Colab GAN Training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_patches', type=int, default=10)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--modalities', type=str, nargs='+', default=['T1w', 'T2w'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=10)
    
    args = parser.parse_args()
    
    train_gan(args)


if __name__ == '__main__':
    main()
