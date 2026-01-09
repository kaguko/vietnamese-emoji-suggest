"""
Data verification script for Vietnamese Emoji Suggestion System.

This script:
- Checks data folder structure
- Validates CSV can be loaded
- Prints emotion distribution
- Confirms 450+ samples exist
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_folder_structure():
    """Check if required folders exist."""
    print("=" * 70)
    print("1. CHECKING FOLDER STRUCTURE")
    print("=" * 70)
    
    required_folders = [
        "data",
        "data/raw",
        "data/processed",
    ]
    
    all_exist = True
    for folder in required_folders:
        exists = os.path.exists(folder)
        status = "✓" if exists else "✗"
        print(f"   {status} {folder}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_initial_data():
    """Check initial_data.csv."""
    print("\n" + "=" * 70)
    print("2. CHECKING INITIAL DATASET")
    print("=" * 70)
    
    csv_path = "data/raw/initial_data.csv"
    json_path = "data/raw/initial_data.json"
    
    if not os.path.exists(csv_path):
        print(f"   ✗ {csv_path} not found!")
        return False
    
    if not os.path.exists(json_path):
        print(f"   ✗ {json_path} not found!")
        return False
    
    print(f"   ✓ {csv_path} exists")
    print(f"   ✓ {json_path} exists")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✓ CSV loaded successfully")
        print(f"   ✓ Sample count: {len(df)}")
        
        if len(df) < 100:
            print(f"   ⚠️  Warning: Only {len(df)} samples (expected 100+)")
            return False
        else:
            print(f"   ✓ Meets requirement: {len(df)} >= 100")
        
        # Check columns
        required_columns = ['text', 'primary_emotion', 'intensity', 'emoji_1', 
                          'emoji_2', 'emoji_3', 'source', 'created_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   ✗ Missing columns: {missing_columns}")
            return False
        else:
            print(f"   ✓ All required columns present")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error loading CSV: {e}")
        return False


def check_augmented_data():
    """Check augmented_data.csv."""
    print("\n" + "=" * 70)
    print("3. CHECKING AUGMENTED DATASET")
    print("=" * 70)
    
    csv_path = "data/raw/augmented_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"   ✗ {csv_path} not found!")
        return False, None
    
    print(f"   ✓ {csv_path} exists")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✓ CSV loaded successfully")
        print(f"   ✓ Sample count: {len(df)}")
        
        if len(df) < 450:
            print(f"   ⚠️  Warning: Only {len(df)} samples (expected 450+)")
            return False, None
        else:
            print(f"   ✓ Meets requirement: {len(df)} >= 450")
        
        # Check columns
        required_columns = ['text', 'primary_emotion', 'intensity', 'emoji_1', 
                          'emoji_2', 'emoji_3', 'source', 'created_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   ✗ Missing columns: {missing_columns}")
            return False, None
        else:
            print(f"   ✓ All required columns present")
        
        return True, df
        
    except Exception as e:
        print(f"   ✗ Error loading CSV: {e}")
        return False, None


def check_emotion_distribution(df):
    """Check emotion distribution."""
    print("\n" + "=" * 70)
    print("4. EMOTION DISTRIBUTION")
    print("=" * 70)
    
    if df is None:
        print("   ✗ No data available")
        return False
    
    emotion_counts = df['primary_emotion'].value_counts().sort_index()
    
    print(f"\n   {'Emotion':<15} {'Count':<10} {'Percentage':<10}")
    print("   " + "-" * 40)
    
    required_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 
                        'disgust', 'trust', 'anticipation']
    
    all_present = True
    for emotion in required_emotions:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / len(df) * 100) if len(df) > 0 else 0
        status = "✓" if count > 0 else "✗"
        print(f"   {status} {emotion:<14} {count:<10} {percentage:.1f}%")
        if count == 0:
            all_present = False
    
    if all_present:
        print(f"\n   ✓ All 8 emotions are represented")
    else:
        print(f"\n   ✗ Some emotions are missing")
    
    return all_present


def check_data_quality(df):
    """Check data quality."""
    print("\n" + "=" * 70)
    print("5. DATA QUALITY CHECKS")
    print("=" * 70)
    
    if df is None:
        print("   ✗ No data available")
        return False
    
    all_good = True
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts['text'] > 0:
        print(f"   ✗ {null_counts['text']} null values in 'text' column")
        all_good = False
    else:
        print(f"   ✓ No null values in 'text' column")
    
    if null_counts['primary_emotion'] > 0:
        print(f"   ✗ {null_counts['primary_emotion']} null values in 'primary_emotion' column")
        all_good = False
    else:
        print(f"   ✓ No null values in 'primary_emotion' column")
    
    if null_counts['emoji_1'] > 0:
        print(f"   ✗ {null_counts['emoji_1']} null values in 'emoji_1' column")
        all_good = False
    else:
        print(f"   ✓ No null values in 'emoji_1' column")
    
    # Check intensity range
    if df['intensity'].min() < 1 or df['intensity'].max() > 5:
        print(f"   ✗ Intensity values out of range (1-5)")
        all_good = False
    else:
        print(f"   ✓ Intensity values in valid range (1-5)")
    
    # Check source values
    source_values = df['source'].unique()
    expected_sources = ['manual', 'augmented', 'auto']
    invalid_sources = [s for s in source_values if s not in expected_sources]
    if invalid_sources:
        print(f"   ⚠️  Unexpected source values: {invalid_sources}")
    else:
        print(f"   ✓ All source values are valid")
    
    return all_good


def main():
    """Run all verification checks."""
    os.chdir('/home/runner/work/vietnamese-emoji-suggest/vietnamese-emoji-suggest')
    
    print("\n" + "=" * 70)
    print(" DATA VERIFICATION SCRIPT")
    print("=" * 70)
    
    results = []
    
    # Check folder structure
    results.append(("Folder Structure", check_folder_structure()))
    
    # Check initial data
    results.append(("Initial Dataset", check_initial_data()))
    
    # Check augmented data
    augmented_ok, df = check_augmented_data()
    results.append(("Augmented Dataset", augmented_ok))
    
    # Check emotion distribution
    if df is not None:
        results.append(("Emotion Distribution", check_emotion_distribution(df)))
        results.append(("Data Quality", check_data_quality(df)))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
