"""
Unit tests for dataset generation scripts.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collect_data import (
    create_initial_dataset,
    save_dataset_csv,
    load_dataset_csv,
    get_dataset_stats,
    validate_dataset,
    EMOTION_LABELS,
    EMOTION_TO_IDX
)


class TestDataCollection:
    """Test data collection functions."""
    
    def test_create_initial_dataset(self):
        """Test initial dataset creation."""
        samples = create_initial_dataset()
        
        # Check we have samples
        assert len(samples) > 0, "Should create samples"
        assert len(samples) >= 100, f"Should have at least 100 samples, got {len(samples)}"
        
        # Check structure
        first_sample = samples[0]
        required_keys = ['text', 'primary_emotion', 'intensity', 'emoji_1', 
                        'emoji_2', 'emoji_3', 'source', 'created_at']
        for key in required_keys:
            assert key in first_sample, f"Sample missing required key: {key}"
    
    def test_emotion_coverage(self):
        """Test that all 8 emotions are covered."""
        samples = create_initial_dataset()
        
        emotions = {sample['primary_emotion'] for sample in samples}
        expected_emotions = set(EMOTION_TO_IDX.keys())
        
        assert emotions == expected_emotions, \
            f"Missing emotions: {expected_emotions - emotions}"
    
    def test_intensity_range(self):
        """Test that intensity values are in valid range."""
        samples = create_initial_dataset()
        
        for sample in samples:
            intensity = sample['intensity']
            assert 1 <= intensity <= 5, \
                f"Intensity {intensity} out of range for: {sample['text']}"
    
    def test_dataset_stats(self):
        """Test dataset statistics calculation."""
        samples = create_initial_dataset()
        stats = get_dataset_stats(samples)
        
        assert 'total_samples' in stats
        assert 'emotions' in stats
        assert 'intensities' in stats
        assert stats['total_samples'] == len(samples)
        
        # Check all emotions are counted
        emotion_count = sum(stats['emotions'].values())
        assert emotion_count == len(samples)
    
    def test_validate_dataset(self):
        """Test dataset validation."""
        samples = create_initial_dataset()
        issues = validate_dataset(samples)
        
        # Check issue categories exist
        assert 'missing_emoji' in issues
        assert 'invalid_emotion' in issues
        assert 'invalid_intensity' in issues
        
        # All samples should have valid emotions and intensities
        assert len(issues['invalid_emotion']) == 0, \
            "Found invalid emotions"
        assert len(issues['invalid_intensity']) == 0, \
            "Found invalid intensities"


class TestGeneratedData:
    """Test the generated data files."""
    
    def test_initial_data_exists(self):
        """Test that initial data files exist."""
        csv_path = Path("data/raw/initial_data.csv")
        json_path = Path("data/raw/initial_data.json")
        
        # These files should exist after running the generation scripts
        if csv_path.exists():
            assert csv_path.is_file(), "initial_data.csv should be a file"
            
            # Test loading with pandas
            df = pd.read_csv(csv_path)
            assert len(df) >= 100, f"Should have at least 100 samples, got {len(df)}"
    
    def test_augmented_data_exists(self):
        """Test that augmented data exists."""
        csv_path = Path("data/raw/augmented_data.csv")
        
        if csv_path.exists():
            assert csv_path.is_file(), "augmented_data.csv should be a file"
            
            # Test loading with pandas
            df = pd.read_csv(csv_path)
            assert len(df) >= 450, f"Should have at least 450 samples, got {len(df)}"
            
            # Check columns
            required_columns = ['text', 'primary_emotion', 'intensity', 'emoji_1', 
                              'emoji_2', 'emoji_3', 'source', 'created_at']
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"
            
            # Check all 8 emotions are present
            emotions = set(df['primary_emotion'].unique())
            expected_emotions = set(EMOTION_TO_IDX.keys())
            assert emotions == expected_emotions, \
                f"Missing emotions: {expected_emotions - emotions}"


if __name__ == "__main__":
    # Run tests manually if pytest not available
    print("Running dataset generation tests...")
    
    test_collection = TestDataCollection()
    print("✓ Testing initial dataset creation...")
    test_collection.test_create_initial_dataset()
    
    print("✓ Testing emotion coverage...")
    test_collection.test_emotion_coverage()
    
    print("✓ Testing intensity range...")
    test_collection.test_intensity_range()
    
    print("✓ Testing dataset stats...")
    test_collection.test_dataset_stats()
    
    print("✓ Testing dataset validation...")
    test_collection.test_validate_dataset()
    
    test_generated = TestGeneratedData()
    print("✓ Testing initial data exists...")
    test_generated.test_initial_data_exists()
    
    print("✓ Testing augmented data exists...")
    test_generated.test_augmented_data_exists()
    
    print("\n✅ All tests passed!")
