# %%
"""
Multimodal CCA Analysis: Relating Neural and Behavioral Data
-----------------------------------------------------------
This script demonstrates how to use Canonical Correlation Analysis (CCA) to find shared latent variables between neural recordings and behavioral measurements.

- Loads neural data and behavioral data using neo_helpers and pandas
- Extracts neural features (spike band power, PCA)
- Extracts behavioral features (velocity, target, etc)
- Runs CCA to find joint structure
- Visualizes canonical correlations and projections
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neo.io.blackrockio import BlackrockIO
from neo_helpers import load_by_time

# --- PARAMETERS ---
NEURAL_FILE = "data/neural.ns6"
BEHAVIORAL_FILE = "data/actions.csv"
DURATION_SEC = 10  # seconds of data to load
FS = 30000  # Hz, will be overwritten by file

# --- LOAD DATA ---
print("Loading neural data...")
reader = BlackrockIO(NEURAL_FILE, verbose=True)
fs = float(reader.get_signal_sampling_rate())
print(f"Sampling rate: {fs} Hz")

# Get neural file start time (from neo proxy)
block = reader.read_block(lazy=True)
proxy = block.segments[0].analogsignals[0]
neural_file_t_start = float(proxy.t_start)  # in seconds

# Load behavioral data
behavior = pd.read_csv(BEHAVIORAL_FILE)
print(behavior.head())
# %%
# Behavioral timestamps are already in UNIX seconds!
neural_time = behavior['timestamp'] - neural_file_t_start

# Use only behavioral events that are within the neural file’s time window
valid_mask = (neural_time >= 0) & (neural_time < (proxy.t_stop - proxy.t_start))
behavior = behavior[valid_mask]

# Set start_time for neural chunk
first_behavior_event = neural_time.min()
start_time = first_behavior_event
print(f"Neural file t_start: {neural_file_t_start} s")
print(f"First behavioral event (relative to neural): {first_behavior_event} s")
print(f"Loading neural data from {start_time:.3f} seconds after neural file start...")

chunk = load_by_time(reader, duration_sec=DURATION_SEC, start_time=start_time)
print(f"Neural data shape: {chunk.shape}")

# %%
# --- FEATURE EXTRACTION ---
def extract_spike_band_power(data, fs=30000, freq_band=(200, 400)):
    """Extract high-gamma power (spike band power)"""
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    b, a = signal.butter(4, freq_band, btype='band', fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=0)
    power = np.abs(signal.hilbert(filtered, axis=0))**2
    return gaussian_filter1d(power, sigma=fs//100, axis=0)  # 10ms smoothing

# Neural features: spike band power, then PCA
print("Extracting neural features (spike band power)...")
sbp = extract_spike_band_power(chunk, fs)

# Downsample for speed (to 100 Hz)
ds_factor = int(fs // 100)
sbp_ds = sbp[::ds_factor]
print(f"Downsampled neural features shape: {sbp_ds.shape}")

# PCA to reduce neural dimensionality
n_pca = 10
pca = PCA(n_components=n_pca)
X_neural = pca.fit_transform(StandardScaler().fit_transform(sbp_ds))
print(f"Neural PCA shape: {X_neural.shape}")

# --- BEHAVIORAL FEATURES ---
print("Extracting behavioral features...")
# Interpolate behavioral features to neural timebase
n_samples = sbp_ds.shape[0]
time_neural = np.arange(n_samples) * ds_factor / fs  # seconds from neural chunk start

# Interpolate velocity to neural timebase
vel_x = np.interp(time_neural, behavior['neural_time'], behavior['velocity_x'])
vel_y = np.interp(time_neural, behavior['neural_time'], behavior['velocity_y'])

# Use target as categorical (take most recent target for each time)
targets = behavior['target_index'].fillna(method='ffill').values if 'target_index' in behavior.columns else behavior['target'].fillna(method='ffill').values
# Interpolate target index (nearest)
target_interp = np.interp(time_neural, behavior['neural_time'], targets, left=targets[0], right=targets[-1])

# One-hot encode target
n_targets = int(np.nanmax(targets)) + 1
Y_behavior = np.column_stack([
    vel_x,
    vel_y,
    np.eye(n_targets)[target_interp.astype(int)]
])
print(f"Behavioral feature matrix shape: {Y_behavior.shape}")
print(f"vel_x (first 10): {vel_x[:10]}")
print(f"vel_y (first 10): {vel_y[:10]}")
print(f"target_interp (first 10): {target_interp[:10]}")

# --- RUN CCA ---
n_cca = min(X_neural.shape[1], Y_behavior.shape[1], 5)
cca = CCA(n_components=n_cca, max_iter=1000)
X_c, Y_c = cca.fit_transform(X_neural, Y_behavior)

# --- VISUALIZATION ---
plt.figure(figsize=(10, 4))
plt.plot(cca.score(X_neural, Y_behavior), 'o-', label='Canonical correlation')
plt.title('Canonical Correlation (overall score)')
plt.xlabel('Component')
plt.ylabel('Correlation')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(n_cca):
    plt.subplot(n_cca, 1, i+1)
    plt.plot(X_c[:, i], label=f'Neural CCA {i+1}')
    plt.plot(Y_c[:, i], label=f'Behavioral CCA {i+1}', alpha=0.7)
    plt.legend()
    plt.ylabel('Projection')
plt.xlabel('Sample')
plt.suptitle('Canonical Variates (Neural vs Behavioral)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Done! This is a basic CCA demo. For DCCA, TDA, or graph-based analysis, see comments in this script.")

# --- EXTENSIONS ---
# - To use Deep CCA (DCCA), see: https://github.com/VahidooX/DCCA or similar PyTorch implementations
# - For TDA, try the 'giotto-tda' or 'scikit-tda' packages
# - For graph-based analysis, see 'networkx' or 'graspologic' for graph construction and analysis 
# %%
