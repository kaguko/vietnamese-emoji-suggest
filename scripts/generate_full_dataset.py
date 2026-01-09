"""
Generate full augmented dataset from initial dataset.

This script:
1. Loads initial data from data/raw/initial_data.csv
2. Applies augmentation using src/augmentation.py
3. Saves augmented dataset to data/raw/augmented_data.csv
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.collect_data import load_dataset_csv, save_dataset_csv, get_dataset_stats
from src.augmentation import augment_dataset, generate_weak_labeled_samples, SAMPLE_UNLABELED_TEXTS
from src.models import EMOTION_EMOJI_MAP


def main():
    print("=" * 70)
    print("GENERATING FULL AUGMENTED DATASET")
    print("=" * 70)
    
    # 1. Load initial dataset
    initial_csv_path = "data/raw/initial_data.csv"
    print(f"\n1. Loading initial dataset from {initial_csv_path}...")
    
    if not os.path.exists(initial_csv_path):
        print(f"❌ Error: {initial_csv_path} not found!")
        print("   Please run 'python data/collect_data.py' first.")
        return 1
    
    initial_samples = load_dataset_csv(initial_csv_path)
    print(f"   ✓ Loaded {len(initial_samples)} initial samples")
    
    # 2. Augment dataset
    print("\n2. Applying augmentation...")
    print("   - Synonym replacement")
    print("   - Intensity variations")
    
    # First round of augmentation
    augmented_samples = augment_dataset(
        initial_samples,
        augmentation_factor=3,  # Creates 3 augmented versions per sample
        include_weak_labeled=True
    )
    print(f"   ✓ Generated {len(augmented_samples)} samples after first round")
    
    # Second round: augment the augmented samples
    print("   - Applying second round of augmentation...")
    second_round = []
    for sample in augmented_samples:
        # Only augment high-confidence samples
        if sample.get('confidence', 1.0) >= 0.9:
            second_round.append(sample)
    
    additional_augmented = augment_dataset(
        second_round[:104],  # Use all original samples for second round
        augmentation_factor=2,
        include_weak_labeled=False
    )
    
    # Combine, removing original duplicates from second round
    for aug in additional_augmented:
        if aug.get('augmentation_type') != 'original':
            augmented_samples.append(aug)
    
    print(f"   ✓ Total after second round: {len(augmented_samples)} samples")
    
    # 3. Generate weak-labeled samples
    print("\n3. Generating weak-labeled samples...")
    weak_labeled_samples = generate_weak_labeled_samples(
        SAMPLE_UNLABELED_TEXTS,
        EMOTION_EMOJI_MAP
    )
    print(f"   ✓ Generated {len(weak_labeled_samples)} weak-labeled samples")
    
    # 4. Combine all samples
    print("\n4. Combining datasets...")
    all_samples = augmented_samples + weak_labeled_samples
    print(f"   ✓ Total samples: {len(all_samples)}")
    
    # 5. Add metadata
    print("\n5. Adding metadata...")
    for sample in all_samples:
        # Ensure all required fields are present
        if 'source' not in sample:
            if sample.get('augmentation_type') == 'original':
                sample['source'] = 'manual'
            elif sample.get('augmentation_type') in ['synonym', 'intensity']:
                sample['source'] = 'augmented'
            else:
                sample['source'] = 'augmented'
        
        # Ensure created_at is present
        if 'created_at' not in sample or not sample['created_at']:
            sample['created_at'] = datetime.now().isoformat()
    
    # 6. Get statistics
    print("\n6. Dataset statistics:")
    stats = get_dataset_stats(all_samples)
    print(f"   - Total samples: {stats['total_samples']}")
    print(f"   - Emotions: {stats['emotions']}")
    print(f"   - Average text length: {stats['avg_text_length']:.1f} words")
    
    # 7. Save augmented dataset
    output_path = "data/raw/augmented_data.csv"
    print(f"\n7. Saving to {output_path}...")
    save_dataset_csv(all_samples, output_path)
    print(f"   ✓ Saved successfully!")
    
    # 8. Final validation
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Initial samples: {len(initial_samples)}")
    print(f"✓ After augmentation: {len(augmented_samples)}")
    print(f"✓ Weak-labeled samples: {len(weak_labeled_samples)}")
    print(f"✓ Total samples: {len(all_samples)}")
    
    if len(all_samples) >= 450:
        print(f"\n✅ SUCCESS: Generated {len(all_samples)} samples (target: 450+)")
    else:
        print(f"\n⚠️  WARNING: Only {len(all_samples)} samples (target: 450+)")
        print("   You may need to increase augmentation_factor or add more unlabeled texts")
    
    return 0


if __name__ == "__main__":
    exit(main())
