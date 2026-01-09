#!/usr/bin/env python3
"""
Convenience script to generate complete dataset for Vietnamese Emoji Suggestion System.

This script:
1. Runs data collection to create initial dataset
2. Applies augmentation to expand the dataset
3. Validates the output
4. Prints statistics

Usage:
    python scripts/generate_dataset.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.collect_data import (
    create_initial_dataset,
    save_dataset_csv,
    save_dataset_json,
    get_dataset_stats,
    validate_dataset,
    load_dataset_csv
)
from src.augmentation import augment_dataset, validate_dataset as validate_augmented
from src.models import EMOTION_EMOJI_MAP


def generate_initial_dataset():
    """Generate initial manually-curated dataset."""
    print("=" * 70)
    print("STEP 1: Generating Initial Dataset")
    print("=" * 70)
    
    samples = create_initial_dataset()
    print(f"\nCreated {len(samples)} initial samples")
    
    # Save to files
    save_dataset_csv(samples, "data/raw/initial_data.csv")
    save_dataset_json(samples, "data/raw/initial_data.json")
    
    # Get and print stats
    stats = get_dataset_stats(samples)
    print(f"\nInitial Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Emotions: {stats['emotions']}")
    print(f"  Avg text length: {stats['avg_text_length']:.1f} words")
    
    # Validate
    issues = validate_dataset(samples)
    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        print("\n✓ Dataset validation passed!")
    else:
        print(f"\n⚠ Found {total_issues} issues:")
        for issue_type, indices in issues.items():
            if indices:
                print(f"  {issue_type}: {len(indices)} samples")
    
    return samples


def generate_augmented_dataset(initial_samples, target_samples=450):
    """Generate augmented dataset with target number of samples."""
    print("\n" + "=" * 70)
    print("STEP 2: Generating Augmented Dataset")
    print("=" * 70)
    
    # Calculate needed augmentation factor to reach target
    # We need to account for the fact that not all augmentations produce new samples
    initial_count = len(initial_samples)
    
    # Start with a high augmentation factor
    augmentation_factor = 4
    
    print(f"\nApplying augmentation with factor {augmentation_factor}...")
    augmented = augment_dataset(
        initial_samples,
        augmentation_factor=augmentation_factor,
        include_weak_labeled=False
    )
    
    print(f"Generated {len(augmented)} augmented samples")
    
    # Update source field for augmented samples
    for sample in augmented:
        aug_type = sample.get('augmentation_type', 'original')
        if aug_type != 'original':
            sample['source'] = 'augmented'
        sample['created_at'] = sample.get('created_at', '')
    
    # Save augmented dataset
    save_dataset_csv(augmented, "data/raw/augmented_data.csv")
    print(f"\nSaved augmented dataset to data/raw/augmented_data.csv")
    
    return augmented


def print_final_statistics(augmented_samples):
    """Print final statistics and validation."""
    print("\n" + "=" * 70)
    print("STEP 3: Final Validation and Statistics")
    print("=" * 70)
    
    # Basic stats
    stats = get_dataset_stats(augmented_samples)
    print(f"\nFinal Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Average text length: {stats['avg_text_length']:.1f} words")
    
    print(f"\n  Emotion Distribution:")
    for emotion, count in sorted(stats['emotions'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"    {emotion:15s}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\n  Intensity Distribution:")
    for intensity, count in sorted(stats['intensities'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"    Level {intensity}: {count:3d} ({percentage:5.1f}%)")
    
    # Augmentation type breakdown
    aug_types = {}
    for sample in augmented_samples:
        aug_type = sample.get('augmentation_type', 'unknown')
        aug_types[aug_type] = aug_types.get(aug_type, 0) + 1
    
    print(f"\n  Augmentation Types:")
    for aug_type, count in sorted(aug_types.items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"    {aug_type:15s}: {count:3d} ({percentage:5.1f}%)")
    
    # Validation
    issues = validate_dataset(augmented_samples)
    total_issues = sum(len(v) for v in issues.values())
    
    print(f"\n  Validation:")
    if total_issues == 0:
        print("    ✓ All checks passed!")
    else:
        print(f"    ⚠ Found {total_issues} issues:")
        for issue_type, indices in issues.items():
            if indices:
                print(f"      {issue_type}: {len(indices)} samples")
    
    # Check target
    print(f"\n  Target Achievement:")
    if stats['total_samples'] >= 450:
        print(f"    ✓ Target met: {stats['total_samples']} >= 450 samples")
    else:
        print(f"    ⚠ Target not met: {stats['total_samples']} < 450 samples")
    
    # Check all emotions present
    expected_emotions = set(EMOTION_EMOJI_MAP.keys())
    present_emotions = set(stats['emotions'].keys())
    if present_emotions == expected_emotions:
        print(f"    ✓ All 8 emotions represented")
    else:
        missing = expected_emotions - present_emotions
        print(f"    ⚠ Missing emotions: {missing}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Vietnamese Emoji Suggestion System - Dataset Generation")
    print("=" * 70)
    
    try:
        # Step 1: Generate initial dataset
        initial_samples = generate_initial_dataset()
        
        # Step 2: Generate augmented dataset
        augmented_samples = generate_augmented_dataset(initial_samples)
        
        # Step 3: Print statistics and validate
        print_final_statistics(augmented_samples)
        
        print("\n" + "=" * 70)
        print("✓ Dataset generation completed successfully!")
        print("=" * 70)
        
        print("\nGenerated files:")
        print("  - data/raw/initial_data.csv (initial samples)")
        print("  - data/raw/initial_data.json (initial samples in JSON)")
        print("  - data/raw/augmented_data.csv (augmented dataset)")
        
    except Exception as e:
        print(f"\n✗ Error during dataset generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
