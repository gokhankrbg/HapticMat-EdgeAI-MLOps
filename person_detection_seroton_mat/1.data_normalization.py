"""
Simple CSV Normalization Script
Input: CSV files with BLE data
Output: Normalized CSV file (ready for training)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def normalize_csv(empty_csv_path, full_csv_path, output_csv_path):
    """
    Normalize CSV data and save to new CSV file

    Args:
        empty_csv_path: Path to empty state CSV
        full_csv_path: Path to occupied state CSV
        output_csv_path: Path to save normalized CSV
    """

    # Load CSV files
    print("📥 Loading CSV files...")
    empty_df = pd.read_csv(empty_csv_path)
    full_df = pd.read_csv(full_csv_path)

    print(f"  Empty data: {len(empty_df)} rows")
    print(f"  Full data: {len(full_df)} rows")

    # Get feature columns (Float1, Float2, Float3, Float4)
    feature_cols = [col for col in empty_df.columns if col.startswith('Float')]
    print(f"  Features: {feature_cols}")

    # Extract data
    empty_features = empty_df[feature_cols].values
    full_features = full_df[feature_cols].values

    # Combine data
    all_data = np.vstack([empty_features, full_features])
    print(f"\n📊 Total samples: {len(all_data)}")

    # Normalize using StandardScaler
    print("\n🔄 Normalizing data...")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(all_data)

    print(f"  Original range: [{all_data.min():.2f}, {all_data.max():.2f}]")
    print(f"  Normalized range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")

    # Create output dataframe
    output_df = pd.DataFrame(normalized_data, columns=feature_cols)

    # Add labels (0 = empty, 1 = occupied)
    output_df['Label'] = [0] * len(empty_df) + [1] * len(full_df)
    output_df['State'] = ['Empty'] * len(empty_df) + ['Occupied'] * len(full_df)

    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved to: {output_csv_path}")

    # Print sample
    print("\n📋 Sample (first 5 rows):")
    print(output_df.head())

    print("\n📊 Statistics:")
    print(output_df[feature_cols].describe())

    return output_df


if __name__ == "__main__":
    # Input paths
    empty_csv = '0.empty_or_cat_on_mat.csv'
    full_csv = '0.human_on_mat.csv'
    output_csv = 'normalized_data.csv'

    # Run normalization
    result_df = normalize_csv(empty_csv, full_csv, output_csv)

    print("\n" + "=" * 60)
    print("✨ Done! Your normalized CSV is ready:")
    print(f"   {output_csv}")
    print("=" * 60)