"""
Data Augmentation Script
Expands dataset using multiple augmentation techniques
Input: normalized_data.csv
Output: augmented_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('normalized_data.csv')

# Separate by class
empty_data = df[df['Label'] == 0].copy()
occupied_data = df[df['Label'] == 1].copy()

feature_cols = ['Float1', 'Float2', 'Float3', 'Float4']

# ============================================================================
# AUGMENTATION TECHNIQUE 1: GAUSSIAN NOISE
# ============================================================================
print("\n2️⃣  Augmentation 1: Adding Gaussian Noise...")

def add_gaussian_noise(data, noise_level=0.05):
    """Add random Gaussian noise to data (5% of std deviation)"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

augmented_data_list = [df.copy()]  # Keep original

# Add noise to both classes
noise_levels = [0.02, 0.05, 0.08]  # Different noise intensities
for noise in noise_levels:
    noisy_empty = empty_data[feature_cols].copy()
    noisy_empty[feature_cols] = add_gaussian_noise(noisy_empty[feature_cols].values, noise)
    noisy_empty['Label'] = 0
    augmented_data_list.append(noisy_empty)
    
    noisy_occupied = occupied_data[feature_cols].copy()
    noisy_occupied[feature_cols] = add_gaussian_noise(noisy_occupied[feature_cols].values, noise)
    noisy_occupied['Label'] = 1
    augmented_data_list.append(noisy_occupied)

print(f"✓ Added {len(noise_levels) * 2} noise augmented batches")

# ============================================================================
# AUGMENTATION TECHNIQUE 2: MIXUP
# ============================================================================
print("\n3️⃣  Augmentation 2: Mixup (Blending similar samples)...")

def mixup_samples(data, num_samples=50, alpha=0.3):
    """Create new samples by mixing existing ones"""
    new_samples = []
    for _ in range(num_samples):
        idx1, idx2 = np.random.choice(len(data), 2, replace=False)
        sample1 = data[feature_cols].iloc[idx1].values
        sample2 = data[feature_cols].iloc[idx2].values
        
        # Mix with random weight
        weight = np.random.beta(alpha, alpha)
        mixed_sample = weight * sample1 + (1 - weight) * sample2
        new_samples.append(mixed_sample)
    
    new_df = pd.DataFrame(new_samples, columns=feature_cols)
    new_df['Label'] = data['Label'].iloc[0]
    return new_df

mixup_empty = mixup_samples(empty_data, num_samples=100)
mixup_occupied = mixup_samples(occupied_data, num_samples=100)
augmented_data_list.extend([mixup_empty, mixup_occupied])
print(f"✓ Created {len(mixup_empty) + len(mixup_occupied)} mixup samples")

# ============================================================================
# AUGMENTATION TECHNIQUE 3: JITTERING (Small random shifts)
# ============================================================================
print("\n4️⃣  Augmentation 3: Jittering (Small random shifts)...")

def jitter_data(data, jitter_std=0.03):
    """Add small random shifts to each feature"""
    jittered = data.copy()
    for col in feature_cols:
        jitter = np.random.normal(0, jitter_std, len(jittered))
        jittered[col] = data[col] + jitter
    return jittered

for jitter_level in [0.02, 0.04]:
    jittered_empty = jitter_data(empty_data[feature_cols + ['Label']].copy(), jitter_level)
    jittered_occupied = jitter_data(occupied_data[feature_cols + ['Label']].copy(), jitter_level)
    augmented_data_list.extend([jittered_empty, jittered_occupied])

print(f"✓ Created jittered variations")

# ============================================================================
# AUGMENTATION TECHNIQUE 4: SCALING
# ============================================================================
print("\n5️⃣  Augmentation 4: Scaling (Magnitude variations)...")

def scale_features(data, scale_factor):
    """Scale all features by a constant factor"""
    scaled = data[feature_cols].copy() * scale_factor
    scaled['Label'] = data['Label'].values
    return scaled

for scale in [0.95, 1.05, 0.90, 1.10]:  # ±5-10% variations
    scaled_empty = scale_features(empty_data, scale)
    scaled_occupied = scale_features(occupied_data, scale)
    augmented_data_list.extend([scaled_empty, scaled_occupied])

print(f"✓ Created {len([0.95, 1.05, 0.90, 1.10]) * 2} scaled variations")

# ============================================================================
# COMBINE ALL DATA
# ============================================================================
print("\n6️⃣  Combining augmented data...")

augmented_df = pd.concat(augmented_data_list, ignore_index=True)
print(f"✓ Total augmented dataset: {len(augmented_df)} samples")
print(f"   - Original: {len(df)}")
print(f"   - New samples added: {len(augmented_df) - len(df)}")
print(f"   - Expansion ratio: {len(augmented_df) / len(df):.2f}x")

# ============================================================================
# CHECK CLASS BALANCE
# ============================================================================
print("\n7️⃣  Class Balance Check:")
class_dist = augmented_df['Label'].value_counts().sort_index()
empty_count = class_dist.get(0, 0)
occupied_count = class_dist.get(1, 0)

print(f"✓ Empty samples: {empty_count}")
print(f"✓ Occupied samples: {occupied_count}")

balance_ratio = min(empty_count, occupied_count) / max(empty_count, occupied_count) * 100
print(f"✓ Balance ratio: {balance_ratio:.2f}% (100% = perfect balance)")

if abs(empty_count - occupied_count) > 50:
    print(f"⚠️  Slight imbalance detected. Balancing...")
    
    # Balance by oversampling minority class
    if empty_count < occupied_count:
        minority = empty_data
        majority = occupied_data
        minority_label = 0
    else:
        minority = occupied_data
        majority = empty_data
        minority_label = 1
    
    num_to_add = abs(empty_count - occupied_count)
    samples_to_add = minority[feature_cols + ['Label']].sample(num_to_add, replace=True, random_state=42)
    augmented_df = pd.concat([augmented_df, samples_to_add], ignore_index=True)
    
    print(f"✓ Added {num_to_add} samples to balance dataset")
    class_dist_after = augmented_df['Label'].value_counts().sort_index()
    print(f"✓ After balancing:")
    print(f"   - Empty: {class_dist_after.get(0, 0)}")
    print(f"   - Occupied: {class_dist_after.get(1, 0)}")

# ============================================================================
# SAVE AUGMENTED DATA
# ============================================================================
print("\n8️⃣  Saving augmented data...")

augmented_df.to_csv('augmented_data.csv', index=False)
print(f"✓ Saved: augmented_data.csv ({len(augmented_df)} samples)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n9️⃣  Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original vs Augmented distribution
ax1 = axes[0, 0]
sizes_orig = [np.sum(df['Label'] == 0), np.sum(df['Label'] == 1)]
sizes_aug = [np.sum(augmented_df['Label'] == 0), np.sum(augmented_df['Label'] == 1)]
ax1.bar(['Original\nEmpty', 'Original\nOccupied'], sizes_orig, color=['#FF6B6B', '#4ECDC4'], alpha=0.7, label='Original')
ax1.set_title('Original Dataset Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
ax1.grid(True, alpha=0.3, axis='y')

# Augmented distribution
ax2 = axes[0, 1]
ax2.bar(['Augmented\nEmpty', 'Augmented\nOccupied'], sizes_aug, color=['#FF6B6B', '#4ECDC4'], alpha=0.7, label='Augmented')
ax2.set_title('Augmented Dataset Distribution', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count')
ax2.grid(True, alpha=0.3, axis='y')

# Feature comparison - Float1
ax3 = axes[1, 0]
ax3.hist(df[df['Label'] == 0]['Float1'], bins=30, alpha=0.5, label='Original Empty', color='red')
ax3.hist(augmented_df[augmented_df['Label'] == 0]['Float1'], bins=30, alpha=0.5, label='Augmented Empty', color='darkred')
ax3.set_title('Float1 Distribution - Empty Class', fontsize=12, fontweight='bold')
ax3.set_xlabel('Float1 Value')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Feature comparison - Float2
ax4 = axes[1, 1]
ax4.hist(df[df['Label'] == 1]['Float2'], bins=30, alpha=0.5, label='Original Occupied', color='blue')
ax4.hist(augmented_df[augmented_df['Label'] == 1]['Float2'], bins=30, alpha=0.5, label='Augmented Occupied', color='darkblue')
ax4.set_title('Float2 Distribution - Occupied Class', fontsize=12, fontweight='bold')
ax4.set_xlabel('Float2 Value')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('augmentation_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: augmentation_analysis.png")
