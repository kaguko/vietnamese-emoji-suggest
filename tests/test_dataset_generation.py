"""
Unit tests for dataset generation
"""

import pytest
import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collect_data import (
    create_initial_dataset,
    get_dataset_stats,
    validate_dataset,
    EMOTION_LABELS
)
from src.augmentation import augment_dataset


class TestDatasetGeneration:
    """Test dataset generation functions."""
    
    def test_create_initial_dataset(self):
        """Test initial dataset creation."""
        samples = create_initial_dataset()
        
        # Check we have enough samples
        assert len(samples) >= 100, f"Expected >= 100 samples, got {len(samples)}"
        
        # Check required fields
        required_fields = ['text', 'primary_emotion', 'intensity', 'emoji_1', 'source', 'created_at']
        for sample in samples:
            for field in required_fields:
                assert field in sample, f"Missing required field: {field}"
        
        # Check all emotions are represented
        emotions = set(s['primary_emotion'] for s in samples)
        all_emotions = set(EMOTION_LABELS.values())
        missing = all_emotions - emotions
        extra = emotions - all_emotions
        assert emotions == all_emotions, (
            f"Emotion mismatch - Missing: {missing}, Extra: {extra}, "
            f"Expected: {sorted(all_emotions)}, Got: {sorted(emotions)}"
        )
    
    def test_dataset_validation(self):
        """Test dataset validation."""
        samples = create_initial_dataset()
        issues = validate_dataset(samples)
        
        # Check no critical issues
        assert len(issues['missing_emoji']) == 0, "Some samples missing primary emoji"
        assert len(issues['invalid_emotion']) == 0, "Some samples have invalid emotions"
        assert len(issues['invalid_intensity']) == 0, "Some samples have invalid intensities"
    
    def test_dataset_stats(self):
        """Test dataset statistics calculation."""
        samples = create_initial_dataset()
        stats = get_dataset_stats(samples)
        
        # Check stats structure
        assert 'total_samples' in stats
        assert 'emotions' in stats
        assert 'intensities' in stats
        assert 'avg_text_length' in stats
        
        # Check stats values
        assert stats['total_samples'] == len(samples)
        assert len(stats['emotions']) == 8, "Should have 8 emotions"
    
    def test_augment_dataset(self):
        """Test dataset augmentation."""
        samples = create_initial_dataset()
        augmented = augment_dataset(samples, augmentation_factor=2)
        
        # Check augmentation increased sample count
        assert len(augmented) > len(samples), "Augmentation should increase sample count"
        
        # Check all samples have augmentation_type
        for sample in augmented:
            assert 'augmentation_type' in sample
            assert sample['augmentation_type'] in ['original', 'synonym', 'intensity']
    
    def test_csv_files_exist(self):
        """Test that CSV files exist and can be loaded."""
        initial_path = "data/raw/initial_data.csv"
        augmented_path = "data/raw/augmented_data.csv"
        
        # Check files exist
        assert os.path.exists(initial_path), f"Initial data file not found: {initial_path}"
        assert os.path.exists(augmented_path), f"Augmented data file not found: {augmented_path}"
        
        # Check files can be loaded
        df_initial = pd.read_csv(initial_path)
        df_augmented = pd.read_csv(augmented_path)
        
        # Check initial dataset
        assert len(df_initial) >= 100, f"Initial dataset should have >= 100 samples, got {len(df_initial)}"
        assert 'text' in df_initial.columns
        assert 'primary_emotion' in df_initial.columns
        assert 'emoji_1' in df_initial.columns
        
        # Check augmented dataset
        assert len(df_augmented) >= 450, f"Augmented dataset should have >= 450 samples, got {len(df_augmented)}"
        assert 'text' in df_augmented.columns
        assert 'primary_emotion' in df_augmented.columns
        assert 'emoji_1' in df_augmented.columns
        assert 'augmentation_type' in df_augmented.columns
        
        # Check no missing emojis
        assert df_initial['emoji_1'].isnull().sum() == 0, "Initial dataset has missing emojis"
        assert df_augmented['emoji_1'].isnull().sum() == 0, "Augmented dataset has missing emojis"
        
        # Check all emotions represented
        emotions_initial = set(df_initial['primary_emotion'].unique())
        emotions_augmented = set(df_augmented['primary_emotion'].unique())
        all_emotions = set(EMOTION_LABELS.values())
        
        assert emotions_initial == all_emotions, f"Initial dataset missing emotions: {all_emotions - emotions_initial}"
        assert emotions_augmented == all_emotions, f"Augmented dataset missing emotions: {all_emotions - emotions_augmented}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
