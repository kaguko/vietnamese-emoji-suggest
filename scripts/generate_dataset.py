#!/usr/bin/env python3
"""
Script to generate initial and augmented datasets for Vietnamese Emoji Suggestion System.

This script:
1. Creates initial manually-curated dataset (100+ samples)
2. Applies augmentation to expand the dataset (to 450+ samples)
3. Validates the output
4. Prints statistics

Usage:
    python scripts/generate_dataset.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collect_data import (
    create_initial_dataset,
    save_dataset_csv,
    get_dataset_stats,
    validate_dataset,
    EMOTION_LABELS
)
from src.augmentation import augment_dataset
from src.models import EMOTION_EMOJI_MAP


def print_stats(samples, title="Dataset Statistics"):
    """Print statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    stats = get_dataset_stats(samples)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} words")
    
    print(f"\nEmotion distribution:")
    for emotion, count in sorted(stats['emotions'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  {emotion:15} : {count:4} ({percentage:5.1f}%)")
    
    print(f"\nIntensity distribution:")
    for intensity, count in sorted(stats['intensities'].items()):
        percentage = (count / stats['total_samples']) * 100
        print(f"  Level {intensity}: {count:4} ({percentage:5.1f}%)")


def validate_and_report(samples, dataset_name="Dataset"):
    """Validate dataset and report issues."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Validation")
    print(f"{'='*60}")
    
    issues = validate_dataset(samples)
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        print("✓ All validation checks passed!")
        return True
    else:
        print(f"⚠ Found {total_issues} issues:")
        for issue_type, indices in issues.items():
            if indices:
                print(f"  {issue_type}: {len(indices)} samples (indices: {indices[:5]}{'...' if len(indices) > 5 else ''})")
        return False


def main():
    """Main function to generate datasets."""
    print("="*60)
    print("VIETNAMESE EMOJI SUGGESTION - DATASET GENERATION")
    print("="*60)
    
    # Step 1: Create initial dataset
    print("\n[1/5] Creating initial manually-curated dataset...")
    initial_samples = create_initial_dataset()
    print(f"✓ Created {len(initial_samples)} initial samples")
    
    # Validate initial dataset
    validate_and_report(initial_samples, "Initial Dataset")
    print_stats(initial_samples, "Initial Dataset Statistics")
    
    # Check emotion coverage
    emotions_present = set(s['primary_emotion'] for s in initial_samples)
    all_emotions = set(EMOTION_LABELS.values())
    missing_emotions = all_emotions - emotions_present
    
    if missing_emotions:
        print(f"\n⚠ Warning: Missing emotions: {missing_emotions}")
    else:
        print(f"\n✓ All 8 emotions are represented!")
    
    # Step 2: Save initial dataset
    print("\n[2/5] Saving initial dataset...")
    initial_csv_path = "data/raw/initial_data.csv"
    save_dataset_csv(initial_samples, initial_csv_path)
    
    # Verify file exists and can be loaded
    import pandas as pd
    try:
        df_initial = pd.read_csv(initial_csv_path)
        print(f"✓ Initial dataset saved and verified: {initial_csv_path}")
        print(f"  Shape: {df_initial.shape}")
    except Exception as e:
        print(f"✗ Error loading initial CSV: {e}")
        return 1
    
    # Step 3: Augment dataset
    print("\n[3/5] Augmenting dataset...")
    # We need to generate enough augmented samples to reach 450+ total
    # Initial: ~107 samples
    # We need to reach 450+, so we need ~343+ more samples
    # Let's use a very high augmentation factor to ensure we reach the goal
    augmented_samples = augment_dataset(initial_samples, augmentation_factor=10)
    
    # Add weak-labeled samples to reach target
    print(f"  Augmented samples so far: {len(augmented_samples)}")
    if len(augmented_samples) < 450:
        print(f"  Adding weak-labeled samples to reach 450...")
        from src.augmentation import generate_weak_labeled_samples, SAMPLE_UNLABELED_TEXTS
        weak_samples = generate_weak_labeled_samples(SAMPLE_UNLABELED_TEXTS, EMOTION_EMOJI_MAP)
        augmented_samples.extend(weak_samples)
        print(f"  Added {len(weak_samples)} weak-labeled samples")
    print(f"✓ Generated {len(augmented_samples)} total samples (including originals)")
    
    # Show augmentation breakdown
    aug_types = {}
    for s in augmented_samples:
        aug_type = s.get('augmentation_type', 'unknown')
        aug_types[aug_type] = aug_types.get(aug_type, 0) + 1
    
    print("\nAugmentation breakdown:")
    for aug_type, count in sorted(aug_types.items()):
        print(f"  {aug_type:15} : {count:4}")
    
    # Step 4: Validate augmented dataset
    validate_and_report(augmented_samples, "Augmented Dataset")
    print_stats(augmented_samples, "Augmented Dataset Statistics")
    
    # Check emotion coverage in augmented
    emotions_present_aug = set(s['primary_emotion'] for s in augmented_samples)
    missing_emotions_aug = all_emotions - emotions_present_aug
    
    if missing_emotions_aug:
        print(f"\n⚠ Warning: Missing emotions in augmented: {missing_emotions_aug}")
    else:
        print(f"\n✓ All 8 emotions are represented in augmented dataset!")
    
    # Step 5: Save augmented dataset
    print("\n[4/5] Saving augmented dataset...")
    augmented_csv_path = "data/raw/augmented_data.csv"
    save_dataset_csv(augmented_samples, augmented_csv_path)
    
    # Verify file exists and can be loaded
    try:
        df_augmented = pd.read_csv(augmented_csv_path)
        print(f"✓ Augmented dataset saved and verified: {augmented_csv_path}")
        print(f"  Shape: {df_augmented.shape}")
    except Exception as e:
        print(f"✗ Error loading augmented CSV: {e}")
        return 1
    
    # Step 6: Final summary
    print("\n[5/5] Final Summary")
    print("="*60)
    print(f"✓ Initial dataset: {len(initial_samples)} samples")
    print(f"✓ Augmented dataset: {len(augmented_samples)} samples")
    
    # Check success criteria
    success_criteria = [
        (len(initial_samples) >= 100, f"Initial dataset has {len(initial_samples)} samples (>= 100)"),
        (len(augmented_samples) >= 450, f"Augmented dataset has {len(augmented_samples)} samples (>= 450)"),
        (len(missing_emotions) == 0, "All 8 emotions represented in initial"),
        (len(missing_emotions_aug) == 0, "All 8 emotions represented in augmented"),
        (os.path.exists(initial_csv_path), f"Initial CSV exists at {initial_csv_path}"),
        (os.path.exists(augmented_csv_path), f"Augmented CSV exists at {augmented_csv_path}"),
    ]
    
    all_passed = True
    print("\nSuccess Criteria:")
    for passed, message in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {message}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("SUCCESS! All criteria met.")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("FAILURE: Some criteria not met.")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
