"""Model and hash-grid hyper-parameters shared by the interpolation scripts."""

omega = 30  # Frequency of each sine activation
layers = 2
features = 128  # Number of neurons per layer
n_levels = 8
finest_resolution = 512
hash_config = dict(
    n_levels=n_levels,
    n_features_per_level=2,
    log2_hashmap_size=19,
    base_resolution=16,
    finest_resolution=finest_resolution,
)
