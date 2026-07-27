import numpy as np


def add_jitter_noise(features, noise_level=0.01):
    noisy_features = features.copy()
    noise = np.random.normal(0, noise_level, noisy_features.shape)
    return noisy_features + noise


def time_warp_sequence(features, warp_factor=0.8):
    """Speeds up or slows down the gesture sequence."""
    orig_length = features.shape[0]
    new_length = int(orig_length * warp_factor)

    # Resample indices across time
    orig_indices = np.linspace(0, orig_length - 1, num=orig_length)
    new_indices = np.linspace(0, orig_length - 1, num=new_length)

    warped_features = np.zeros((new_length, features.shape[1]))
    for col in range(features.shape[1]):
        warped_features[:, col] = np.interp(new_indices, orig_indices, features[:, col])

    return warped_features



def random_scaling(features, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return features * scale
