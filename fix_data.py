"""
Fix: Add realistic noise so behavioral scores overlap between classes.
Real-world neurological data has high inter-subject variability.
"""
import numpy as np

# Reload original and add realistic noise
labels     = np.load('outputs/data/labels.npy')
behavioral = np.load('outputs/data/behavioral.npy')
rng        = np.random.default_rng(99)

# Add heavy noise to make classes realistically overlapping
# ETVS: add ±0.6 noise (large variance across subjects)
# GAI:  add ±0.08 noise
# SSCS: add ±0.15 noise
noise = rng.normal(0, 1, behavioral.shape) * np.array([0.55, 0.07, 0.14])
behavioral_noisy = behavioral + noise

# Clip to valid ranges
behavioral_noisy[:, 0] = np.clip(behavioral_noisy[:, 0], 0.5, 4.0)
behavioral_noisy[:, 1] = np.clip(behavioral_noisy[:, 1], 0.01, 0.40)
behavioral_noisy[:, 2] = np.clip(behavioral_noisy[:, 2], 0.05, 0.95)
behavioral_noisy = behavioral_noisy.astype(np.float32)

# Check separation
for cid, cname in enumerate(['Normal', 'Early', 'High Risk']):
    m = labels == cid
    print(f"Class {cid} ({cname}): ETVS={behavioral_noisy[m,0].mean():.2f}±{behavioral_noisy[m,0].std():.2f}  "
          f"GAI={behavioral_noisy[m,1].mean():.3f}±{behavioral_noisy[m,1].std():.3f}  "
          f"SSCS={behavioral_noisy[m,2].mean():.2f}±{behavioral_noisy[m,2].std():.2f}")

np.save('outputs/data/behavioral.npy', behavioral_noisy)
print("\nBehavioral scores updated with realistic noise.")
