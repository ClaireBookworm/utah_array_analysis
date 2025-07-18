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
import datetime
from analysis import align_timestamps, make_lagged_matrix
from neo_helpers import load_by_time

# --- CONSTANTS ---
NEURAL_FILE = "data/neural.ns6"
BEHAVIORAL_FILE = "data/actions.csv"
DURATION_SEC = 15
FS = 30000
N_PCA = 10

# --- LOAD DATA ---
print("Loading neural data...")
reader = BlackrockIO(NEURAL_FILE, verbose=True)
fs = float(reader.get_signal_sampling_rate())
print(f"Sampling rate: {fs} Hz")

block = reader.read_block(lazy=True)
proxy = block.segments[0].analogsignals[0]
neural_file_t_start = float(proxy.t_start)

behavior = pd.read_csv(BEHAVIORAL_FILE)
print(behavior.head())

neural_start_unix = datetime.datetime(2025, 3, 25, 21, 22, 53, 360000).timestamp()
offset_sec = 0
neural_load_start_unix = behavior['timestamp'].min() + offset_sec
neural_load_start_time = (neural_load_start_unix - neural_start_unix)
if neural_load_start_time < 0:
    neural_load_start_time = 0.0
print(f"Loading neural data starting at: {neural_load_start_time:.2f} seconds")

chunk = load_by_time(reader, duration_sec=DURATION_SEC, start_time=neural_load_start_time)
print(f"Neural data shape: {chunk.shape}")

behavior['neural_time'] = behavior['timestamp'] - neural_start_unix
print(f"Added 'neural_time' column to behavior DataFrame. First 5 values:\n{behavior['neural_time'].head()}")

trials_samples = behavior['timestamp'].apply(lambda x: align_timestamps(x, neural_start_unix, fs))
behavioral_first_event = trials_samples.min() + int(offset_sec * fs)
behavioral_end = behavioral_first_event + (DURATION_SEC * fs)
valid_mask = (trials_samples >= behavioral_first_event) & (trials_samples < behavioral_end)
trials_subset = behavior[valid_mask].copy()
trials_subset['timestamp'] = trials_samples[valid_mask] - behavioral_first_event
print(f"First behavioral event (relative to neural): {behavioral_first_event} s")

# --- FEATURE EXTRACTION ---
def extract_spike_band_power(data, fs=30000, freq_band=(200, 400)):
    """Extract high-gamma power (spike band power)"""
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    b, a = signal.butter(4, freq_band, btype='band', fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=0)
    power = np.abs(signal.hilbert(filtered, axis=0))**2
    return gaussian_filter1d(power, sigma=fs//100, axis=0)  # 10ms smoothing

print("Extracting neural features (spike band power)...")
sbp = extract_spike_band_power(chunk, fs)
ds_factor = int(fs // 100)
sbp_ds = sbp[::ds_factor]
print(f"Downsampled neural features shape: {sbp_ds.shape}")

pca = PCA(n_components=N_PCA)
X_neural = pca.fit_transform(StandardScaler().fit_transform(sbp_ds))
print(f"Neural PCA shape: {X_neural.shape}")

print("Extracting behavioral features...")
n_samples = sbp_ds.shape[0]
time_neural = np.arange(n_samples) * ds_factor / fs
vel_x = np.interp(time_neural, behavior['neural_time'], behavior['velocity_x'])
vel_y = np.interp(time_neural, behavior['neural_time'], behavior['velocity_y'])
targets = behavior['target_index'].fillna(method='ffill').values if 'target_index' in behavior.columns else behavior['target'].fillna(method='ffill').values
target_ordinal = np.interp(time_neural, behavior['neural_time'], targets)
Y_behavior = np.column_stack([vel_x, vel_y, target_ordinal])

velocity_threshold = 0.1

# ^^ the below graph shows 0.1 prob gets rid of the low velocity samples
# speed = np.sqrt(vel_x**2 + vel_y**2)
# plt.hist(speed, bins=50, alpha=0.7)
# plt.xlabel('speed')
# plt.ylabel('count')
# plt.yscale('log')  # helps see the tail
# plt.show()

# print(f"speed percentiles: 50%={np.percentile(speed, 50):.3f}, 75%={np.percentile(speed, 75):.3f}, 90%={np.percentile(speed, 90):.3f}")

movement_mask = np.sqrt(vel_x**2 + vel_y**2) > velocity_threshold
X_neural_move = X_neural[movement_mask]
Y_behavior_move = Y_behavior[movement_mask]

print(f"Behavioral feature matrix shape: {Y_behavior.shape}")
print(f"vel_x (first 10): {vel_x[:10]}")
print(f"vel_y (first 10): {vel_y[:10]}")

n_cca = min(X_neural.shape[1], Y_behavior.shape[1], 10)
print(f"Running CCA with {n_cca} components...")
cca = CCA(n_components=n_cca, max_iter=1000)
X_c, Y_c = cca.fit_transform(X_neural, Y_behavior)
from numpy import corrcoef
correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_cca)]

# --- VISUALIZATION ---
plt.figure(figsize=(10, 4))
plt.plot(range(n_cca), correlations, 'o-', label='Canonical correlation')
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
    plt.ylabel('Proj')
plt.xlabel('Sample')
plt.suptitle('Canonical Variates (Neural vs Behavioral)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# --- EXTENSIONS ---
# - To use Deep CCA (DCCA), see: https://github.com/VahidooX/DCCA or similar PyTorch implementations
# - For TDA, try the 'giotto-tda' or 'scikit-tda' packages
# - For graph-based analysis, see 'networkx' or 'graspologic' for graph construction and analysis 
# %%
import seaborn as sns

# Neural CCA weights
plt.figure(figsize=(10, 6))
sns.heatmap(cca.x_weights_, annot=True, cmap='vlag',
            xticklabels=[f'CCA{i+1}' for i in range(n_cca)],
            yticklabels=[f'PC{i+1}' for i in range(X_neural.shape[1])])
plt.title("Neural Feature Loadings (CCA weights)")
plt.xlabel("Canonical Component")
plt.ylabel("Neural PCA Feature")
plt.tight_layout()
plt.show()

# Behavioral CCA weights
plt.figure(figsize=(10, 4))
sns.heatmap(cca.y_weights_, annot=True, cmap='vlag',
            xticklabels=[f'CCA{i+1}' for i in range(n_cca)],
            yticklabels=['vel_x', 'vel_y'] + [f'target_{i}' for i in range(Y_behavior.shape[1] - 2)])
plt.title("Behavioral Feature Loadings (CCA weights)")
plt.xlabel("Canonical Component")
plt.ylabel("Behavioral Feature")
plt.tight_layout()
plt.show()

# %%
C = np.corrcoef(np.hstack([X_c, Y_c]).T)
plt.figure(figsize=(8, 8))
sns.heatmap(C, cmap='coolwarm', center=0,
            xticklabels=[f'N{i+1}' for i in range(n_cca)] + [f'B{i+1}' for i in range(n_cca)],
            yticklabels=[f'N{i+1}' for i in range(n_cca)] + [f'B{i+1}' for i in range(n_cca)])
plt.title("Correlation Matrix of Canonical Variates")
plt.show()

# %%
# Multiply PCA components by CCA weights to get channel-level attribution
neural_channels = chunk.shape[1]
pca_components = pca.components_  # shape (n_pca, n_channels)
cca_contrib = np.dot(cca.x_weights_, pca_components)  # shape (n_cca, n_channels)

plt.figure(figsize=(10, 5))
for i in range(n_cca):
    plt.plot(cca_contrib[i], label=f'CCA{i+1}')
plt.legend()
plt.xlabel('Electrode #')
plt.ylabel('Relative Contribution')
plt.title('Electrode Contributions to Each Canonical Component')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(6, 5))
sc = plt.scatter(vel_x, vel_y, c=X_c[:, 0], cmap='coolwarm', s=10)
plt.colorbar(sc, label='Neural CCA1 projection')
plt.xlabel('Velocity X')
plt.ylabel('Velocity Y')
plt.title('Behavior Colored by Neural CCA1')
plt.grid(True)
plt.show()

# %%
