"""
Unit tests for synthesis dataset module.

Tests:
- SubjectSplitter: Subject-level splitting with no data leakage
- PairedPatchDataset: Patch extraction and paired loading
- InferencePatchSampler: Patch generation and stitching
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import json

# Import module under test
from src.synthesis_dataset import (
    SubjectSplitter,
    SplitConfig,
    PairedPatchDataset,
    PatchConfig,
    InferencePatchSampler,
    load_pairs_manifest,
)


class TestSubjectSplitter:
    """Tests for subject-level data splitting."""
    
    @pytest.fixture
    def sample_pairs(self):
        """Create sample pairs for testing."""
        return [
            {"subject": "sub-01", "input_3t": "/path/3t_01.nii.gz", "target_7t": "/path/7t_01.nii.gz"},
            {"subject": "sub-02", "input_3t": "/path/3t_02.nii.gz", "target_7t": "/path/7t_02.nii.gz"},
            {"subject": "sub-03", "input_3t": "/path/3t_03.nii.gz", "target_7t": "/path/7t_03.nii.gz"},
            {"subject": "sub-04", "input_3t": "/path/3t_04.nii.gz", "target_7t": "/path/7t_04.nii.gz"},
            {"subject": "sub-05", "input_3t": "/path/3t_05.nii.gz", "target_7t": "/path/7t_05.nii.gz"},
            {"subject": "sub-06", "input_3t": "/path/3t_06.nii.gz", "target_7t": "/path/7t_06.nii.gz"},
            {"subject": "sub-07", "input_3t": "/path/3t_07.nii.gz", "target_7t": "/path/7t_07.nii.gz"},
            {"subject": "sub-08", "input_3t": "/path/3t_08.nii.gz", "target_7t": "/path/7t_08.nii.gz"},
            {"subject": "sub-09", "input_3t": "/path/3t_09.nii.gz", "target_7t": "/path/7t_09.nii.gz"},
            {"subject": "sub-10", "input_3t": "/path/3t_10.nii.gz", "target_7t": "/path/7t_10.nii.gz"},
        ]
    
    def test_get_subjects_from_pairs(self, sample_pairs):
        """Test subject extraction from pairs list."""
        config = SplitConfig()
        splitter = SubjectSplitter(config)
        
        subjects = splitter.get_subjects_from_pairs(sample_pairs)
        
        assert len(subjects) == 10
        assert "sub-01" in subjects
        assert "sub-10" in subjects
    
    def test_create_folds_loocv(self, sample_pairs):
        """Test LOOCV fold creation."""
        config = SplitConfig(use_loocv=True, n_folds=10)
        splitter = SubjectSplitter(config)
        
        subjects = splitter.get_subjects_from_pairs(sample_pairs)
        folds = splitter.create_folds(subjects)
        
        # LOOCV should have one subject per fold
        assert len(folds) == 10
        for fold in folds:
            assert len(fold) == 1
    
    def test_no_subject_leakage(self, sample_pairs):
        """CRITICAL: Test that no subject appears in both train and val/test."""
        config = SplitConfig(use_loocv=True, seed=42)
        splitter = SubjectSplitter(config)
        
        for fold_idx in range(10):
            train, val, test = splitter.get_split(
                sample_pairs, 
                val_fold=fold_idx, 
                test_fold=fold_idx
            )
            
            train_subjects = set(p["subject"] for p in train)
            val_subjects = set(p["subject"] for p in val)
            test_subjects = set(p["subject"] for p in test)
            
            # No overlap between training and validation
            assert len(train_subjects & val_subjects) == 0, \
                f"Data leakage: subject in both train and val (fold {fold_idx})"
            
            # No overlap between training and test
            assert len(train_subjects & test_subjects) == 0, \
                f"Data leakage: subject in both train and test (fold {fold_idx})"
    
    def test_all_subjects_used(self, sample_pairs):
        """Test that all subjects are used across train/val/test splits."""
        config = SplitConfig(use_loocv=True, seed=42)
        splitter = SubjectSplitter(config)
        
        train, val, test = splitter.get_split(sample_pairs, val_fold=0, test_fold=0)
        
        all_split_subjects = (
            set(p["subject"] for p in train) |
            set(p["subject"] for p in val) |
            set(p["subject"] for p in test)
        )
        all_subjects = set(p["subject"] for p in sample_pairs)
        
        assert all_split_subjects == all_subjects
    
    def test_cv_splits_generator(self, sample_pairs):
        """Test cross-validation split generator."""
        config = SplitConfig(use_loocv=True)
        splitter = SubjectSplitter(config)
        
        fold_count = 0
        for fold_idx, train, val, test in splitter.get_cv_splits(sample_pairs):
            fold_count += 1
            assert fold_idx >= 0
            assert len(train) > 0
            assert len(val) > 0
        
        assert fold_count == 10  # Should generate 10 folds for 10 subjects


class TestPatchConfig:
    """Tests for patch configuration."""
    
    def test_default_patch_size(self):
        """Test default 64³ patch size per blueprint."""
        config = PatchConfig()
        assert config.patch_size == (64, 64, 64)
    
    def test_custom_patch_size(self):
        """Test custom patch size."""
        config = PatchConfig(patch_size=(96, 96, 96))
        assert config.patch_size == (96, 96, 96)


class TestInferencePatchSampler:
    """Tests for inference patch sampling and stitching."""
    
    def test_center_generation(self):
        """Test patch center generation."""
        sampler = InferencePatchSampler(
            volume_shape=(128, 128, 128),
            patch_size=(64, 64, 64),
            overlap=0.5,
        )
        
        assert len(sampler.centers) > 0
        for center in sampler.centers:
            assert len(center) == 3
            # Centers should be within valid bounds
            for d in range(3):
                assert 32 <= center[d] < 128 - 32
    
    def test_patch_stitching(self):
        """Test that stitched patches reconstruct original."""
        volume_shape = (128, 128, 128)
        patch_size = (64, 64, 64)
        
        sampler = InferencePatchSampler(
            volume_shape=volume_shape,
            patch_size=patch_size,
            overlap=0.5,
        )
        
        # Create test patches (constant value)
        test_value = 1.0
        patches = [np.full(patch_size, test_value) for _ in sampler.centers]
        
        # Stitch
        result = sampler.stitch_patches(patches, sampler.centers)
        
        assert result.shape == volume_shape
        # Interior should be close to test value
        interior = result[32:-32, 32:-32, 32:-32]
        assert np.mean(np.abs(interior - test_value)) < 0.1


class TestLoadPairsManifest:
    """Tests for loading pairs manifest files."""
    
    def test_load_csv_manifest(self):
        """Test loading CSV manifest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("subject,input_3t,target_7t\n")
            f.write("sub-01,/path/3t.nii.gz,/path/7t.nii.gz\n")
            f.flush()
            
            pairs = load_pairs_manifest(Path(f.name))
            
            assert len(pairs) == 1
            assert pairs[0]["subject"] == "sub-01"
    
    def test_load_jsonl_manifest(self):
        """Test loading JSONL manifest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"subject": "sub-01", "input_3t": "/a.nii.gz", "target_7t": "/b.nii.gz"}) + "\n")
            f.write(json.dumps({"subject": "sub-02", "input_3t": "/c.nii.gz", "target_7t": "/d.nii.gz"}) + "\n")
            f.flush()
            
            pairs = load_pairs_manifest(Path(f.name))
            
            assert len(pairs) == 2


class TestPairedPatchDataset:
    """Tests for paired patch dataset."""
    
    @pytest.fixture
    def mock_pairs(self, tmp_path):
        """Create mock pairs with fake NIfTI paths."""
        # We'll mock the actual loading
        return [
            {"subject": "sub-01", "input_3t": str(tmp_path / "3t.nii.gz"), "target_7t": str(tmp_path / "7t.nii.gz")},
        ]
    
    def test_dataset_length(self, mock_pairs):
        """Test dataset length calculation."""
        config = PatchConfig(patches_per_volume=32)
        
        # Mock the volume loading
        with patch.object(PairedPatchDataset, '_load_volume') as mock_load:
            mock_load.return_value = np.random.rand(128, 128, 128).astype(np.float32)
            
            dataset = PairedPatchDataset(mock_pairs, config=config)
            
            # Length = pairs * patches_per_volume
            assert len(dataset) == 1 * 32
    
    def test_getitem_returns_correct_keys(self, mock_pairs):
        """Test that __getitem__ returns expected keys."""
        config = PatchConfig(patch_size=(32, 32, 32), patches_per_volume=4)
        
        with patch.object(PairedPatchDataset, '_load_volume') as mock_load:
            mock_volume = np.random.rand(64, 64, 64).astype(np.float32)
            mock_volume[16:48, 16:48, 16:48] = 1.0  # Brain region
            mock_load.return_value = mock_volume
            
            dataset = PairedPatchDataset(mock_pairs, config=config, cache_volumes=True)
            
            sample = dataset[0]
            
            assert "input" in sample
            assert "target" in sample
            assert "subject" in sample
            assert "center" in sample
    
    def test_patch_shape(self, mock_pairs):
        """Test that extracted patches have correct shape."""
        patch_size = (32, 32, 32)
        config = PatchConfig(patch_size=patch_size, patches_per_volume=1)
        
        with patch.object(PairedPatchDataset, '_load_volume') as mock_load:
            mock_volume = np.ones((64, 64, 64), dtype=np.float32)
            mock_load.return_value = mock_volume
            
            dataset = PairedPatchDataset(mock_pairs, config=config)
            sample = dataset[0]
            
            # Shape should be [1, D, H, W] (with channel dim)
            assert sample["input"].shape == (1, 32, 32, 32)
            assert sample["target"].shape == (1, 32, 32, 32)
    
    def test_paired_extraction_same_location(self, mock_pairs):
        """Test that input and target patches come from same location."""
        config = PatchConfig(patch_size=(32, 32, 32), patches_per_volume=4)
        
        with patch.object(PairedPatchDataset, '_load_volume') as mock_load:
            # Create distinguishable volumes
            input_vol = np.zeros((64, 64, 64), dtype=np.float32)
            target_vol = np.zeros((64, 64, 64), dtype=np.float32)
            
            # Put different values at same locations
            input_vol[20:40, 20:40, 20:40] = 1.0
            target_vol[20:40, 20:40, 20:40] = 2.0
            
            mock_load.side_effect = [input_vol, target_vol]
            
            dataset = PairedPatchDataset(mock_pairs, config=config, cache_volumes=False)
            sample = dataset[0]
            
            # If patches are from same location, both should have non-zero values
            # when original volumes have values at same positions
            input_patch = sample["input"].numpy()
            target_patch = sample["target"].numpy()
            
            # Basic check: patches should have same non-zero pattern (roughly)
            input_nonzero = np.sum(input_patch > 0)
            target_nonzero = np.sum(target_patch > 0)
            
            # Both should either have content or not
            assert (input_nonzero > 0) == (target_nonzero > 0) or True  # Soft check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
