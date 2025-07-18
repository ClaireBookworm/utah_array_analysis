# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from analysis import extract_spike_band_power, align_behavioral_trials, load_neural_chunk
import matplotlib.pyplot as plt
import os
import tqdm
import json
import datetime
import spikeinterface as se
from neo_helpers import load_by_time
from neo.io.blackrockio import BlackrockIO
output_dir = "kalman_wiener_outputs"
os.makedirs(output_dir, exist_ok=True)
# %%

# --- PARAMETERS ---
NEURAL_FILE = "data/neural.ns6"
BEHAVIORAL_FILE = "data/actions.csv"
FS = 30000
DURATION_SEC = 10

# --- LOAD & ALIGN DATA ---
print("Loading neural and behavioral data...")
trials = pd.read_csv('data/actions.csv')
trials = trials.rename(columns={
    'timestamp': 'movement_onset',
    'trial_start': 'trial_idx',
    'num_targets': 'target_count',
    'target_index': 'target'	
})
# Load neural data chunk
# offset_sec = 25.2  # based on behavioral start vs. neural time_origin
neural_start_unix = datetime.datetime(2025, 3, 25, 21, 22, 53, 360000).timestamp()
def align_timestamps(behavioral_unix, neural_start_unix=neural_start_unix, fs=FS):
    """convert behavioral unix time to neural sample index"""
    time_offset = behavioral_unix - neural_start_unix  # seconds since neural start
    sample_index = int(time_offset * fs)
    return sample_index

trials_samples = trials['movement_onset'].apply(lambda x: align_timestamps(x, neural_start_unix, FS))
time_offset = trials['movement_onset'].min() - neural_start_unix
neural_data = load_neural_chunk(NEURAL_FILE, n_samples=int(DURATION_SEC * FS), start_sample=align_timestamps(time_offset))

# neural_data = load_neural_chunk(NEURAL_FILE, n_samples=int(DURATION_SEC*FS))
# Load and align behavioral data
behavioral_df = pd.read_csv(BEHAVIORAL_FILE)
trials = align_behavioral_trials(behavioral_df)
# %%
# --- FEATURE EXTRACTION ---
print("Extracting neural features (spike band power)...")
sbp = extract_spike_band_power(neural_data, fs=FS)
# Average across channels for simplicity (or use PCA for dimensionality reduction)
sbp_mean = np.mean(sbp, axis=1)
from sklearn.decomposition import PCA
n_components = 10
pca = PCA(n_components=n_components)
X = pca.fit_transform(sbp)
# %%
# --- SAVE PCA FEATURES ---
# np.save(os.path.join(output_dir, "pca_features.npy"), X)
X = np.load(os.path.join(output_dir, "pca_features.npy"))
# %%
# --- Extract behavioral velocity aligned to neural chunk ---
print("Extracting behavioral velocity targets...")

def align_behavioral_to_neural(trials_df, neural_start_unix, reader, fs, 
                              duration_sec=5, offset_sec=0):
    """
    Extract neural chunk and align behavioral trials to neural timeline.
    
    Args:
        trials_df: behavioral data with 'movement_onset' unix timestamps
        neural_start_unix: when neural recording began (unix timestamp)  
        reader: BlackrockIO reader object
        fs: sampling rate (Hz)
        duration_sec: how much neural data to load
        offset_sec: start loading this many seconds after first movement
        
    Returns:
        neural_chunk: (samples, channels) array
        aligned_trials: trials df with movement_onset as sample indices relative to chunk
    """
    # convert behavioral timestamps to neural sample indices
    def unix_to_samples(unix_time):
        return int((unix_time - neural_start_unix) * fs)
    
    trials_samples = trials_df['movement_onset'].apply(unix_to_samples)
    
    # determine neural data loading window
    neural_start_time = (trials_df['movement_onset'].min() - neural_start_unix) + offset_sec
    neural_start_time = max(0, neural_start_time)  # can't load before recording start
    
    # load neural chunk
    neural_chunk = load_by_time(reader, duration_sec, neural_start_time)
    
    # filter trials that fall within this neural chunk
    chunk_start_sample = unix_to_samples(neural_start_unix + neural_start_time)
    chunk_end_sample = chunk_start_sample + int(duration_sec * fs)
    
    valid_mask = (trials_samples >= chunk_start_sample) & (trials_samples < chunk_end_sample)
    aligned_trials = trials_df[valid_mask].copy()
    
    # convert movement times to sample indices relative to chunk start
    aligned_trials['movement_onset'] = trials_samples[valid_mask] - chunk_start_sample
    
    print(f"loaded {duration_sec}s neural chunk, aligned {len(aligned_trials)} trials")
    
    return neural_chunk, aligned_trials

# Load neural chunk and align behavioral trials
reader = BlackrockIO(NEURAL_FILE)
neural_chunk, trials = align_behavioral_to_neural(trials, neural_start_unix, reader, FS, 
												  duration_sec=DURATION_SEC, offset_sec=0)
# Make sure behavioral timestamps are in seconds
behavioral_df = behavioral_df.copy()
if behavioral_df['timestamp'].max() > 1e6:
    behavioral_df['timestamp'] = behavioral_df['timestamp'] / 1e6

# Create time vector for neural chunk samples
n_neural_samples = neural_chunk.shape[0] 
neural_timebase = np.arange(n_neural_samples) / FS

# Find behavioral data time range that overlaps with neural chunk
behavioral_start = behavioral_df['timestamp'].min()
neural_start_in_behavioral_time = behavioral_start  # assume they start together

# Create absolute time vector for neural chunk
neural_times_absolute = neural_start_in_behavioral_time + neural_timebase

# Interpolate velocities
vel_x = np.interp(neural_times_absolute, behavioral_df['timestamp'], behavioral_df['velocity_x'])

from scipy.interpolate import interp1d
f = interp1d(behavioral_df['timestamp'], behavioral_df['velocity_y'], 
             kind='nearest', bounds_error=False, fill_value=0)
vel_y = f(neural_times_absolute)

Y = np.column_stack([vel_x, vel_y])

print(f"Y shape: {Y.shape}")
print(f"vel_x range: {vel_x.min():.4f} to {vel_x.max():.4f}")
print(f"vel_y range: {vel_y.min():.4f} to {vel_y.max():.4f}")
print(f"nonzero vel_x: {np.count_nonzero(vel_x)}/{len(vel_x)}")
print(f"nonzero vel_y: {np.count_nonzero(vel_y)}/{len(vel_y)}")

plt.plot(Y[:1000, 0], label='vel_x')
plt.plot(Y[:1000, 1], label='vel_y')
plt.legend()
plt.title("Behavioral Velocity Aligned to Neural Chunk")
plt.show()

# %%
# --- SAVE & LOAD VELOCITY TARGETS ---
np.save(os.path.join(output_dir, "velocity_targets.npy"), Y)
# Y = np.load(os.path.join(output_dir, "velocity_targets.npy"))
# %%
# --- TRAIN/TEST SPLIT ---
n_train = int(0.8 * len(sbp_mean))

lag = 3  # e.g., predict Y[t] using X[t-3], X[t-2], ..., X[t]

X_lagged = np.hstack([X[lag - i : -i if i != 0 else None] for i in range(lag)])
# X_lagged = X_lagged[lag:]  # remove initial lag samples
# Y_lagged = np.hstack([Y[lag + i : len(Y) - (lag - i) if (lag - i) != 0 else None] for i in range(lag)])
Y_lagged = Y[lag:]

# Sanity check
print("X_lagged:", X_lagged.shape)
print("Y_lagged:", Y_lagged.shape)

n_train = int(0.8 * len(X_lagged))
X_train, X_test = X_lagged[:n_train], X_lagged[n_train:]
Y_train, Y_test = Y_lagged[:n_train], Y_lagged[n_train:]


# --- 1. LINEAR REGRESSION ---
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
print(f"Linear Regression R^2: {r2_score(Y_test, Y_pred_lr, multioutput='raw_values')}")
print(f"Linear Regression MSE: {mean_squared_error(Y_test, Y_pred_lr, multioutput='raw_values')}")
# %%
# --- 2. KALMAN FILTER ---
try:
    from pykalman import KalmanFilter
    print("Training Kalman Filter...")
    kf = KalmanFilter(transition_matrices=np.eye(2), observation_matrices=np.array([[lr.coef_[0,0], 0],[0, lr.coef_[1,0]]]),
                     transition_covariance=1e-4*np.eye(2), observation_covariance=1e-1*np.eye(2))
    kf = kf.em(Y_train, n_iter=5)
    (Y_filt, Y_filt_cov) = kf.filter(Y_test)
    print(f"Kalman Filter output shape: {Y_filt.shape}")
    print(f"Kalman Filter R^2: {r2_score(Y_test, Y_filt, multioutput='raw_values')}")
    print(f"Kalman Filter MSE: {mean_squared_error(Y_test, Y_filt, multioutput='raw_values')}")
except ImportError:
    print("pykalman not installed, skipping Kalman Filter.")
# %%
# Y_filt, Y_filt_cov = kf.filter(Y_test)
kalman_r2 = r2_score(Y_test, Y_filt, multioutput='raw_values')
kalman_mse = mean_squared_error(Y_test, Y_filt, multioutput='raw_values')
print(f"Kalman Filter output shape: {Y_filt.shape}")
print(f"Kalman Filter R²: {kalman_r2}")
print(f"Kalman Filter MSE: {kalman_mse}")

np.save(os.path.join(output_dir, "Y_filt.npy"), Y_filt)
#%%
# # --- 3. WIENER FILTER ---
# def wiener_filter(X, Y, order=5):
#     # X: [n_samples, n_features], Y: [n_samples, n_targets]
#     # Stack lagged versions of X
#     X_lagged = np.hstack([np.roll(X, i, axis=0) for i in range(order)])
#     X_lagged = X_lagged[order:]
#     Y = Y[order:]
#     # Fit linear regression
#     wf = LinearRegression()
#     wf.fit(X_lagged, Y)
#     return wf, X_lagged, Y

# print("Training Wiener Filter...")
# wf, Xw_test, Yw_test = wiener_filter(X_test, Y_test, order=5)
# Y_pred_wf = wf.predict(Xw_test)
# print(f"Wiener Filter R^2: {r2_score(Yw_test, Y_pred_wf, multioutput='raw_values')}")
# print(f"Wiener Filter MSE: {mean_squared_error(Yw_test, Y_pred_wf, multioutput='raw_values')}")

# --- 3. WIENER FILTER ---
def wiener_filter(X, Y, order=5):
    # X: [n_samples, n_features], Y: [n_samples, n_targets]
    X_lagged = np.hstack([
        np.roll(X, i, axis=0) for i in tqdm.tqdm(range(order), desc="Lagging features")
    ])
    X_lagged = X_lagged[order:]
    Y = Y[order:]

    wf = LinearRegression()
    wf.fit(X_lagged, Y)
    return wf, X_lagged, Y

print("Training Wiener Filter...")
wf_model, X_lagged, Y_lagged = wiener_filter(X_train, Y_train, order=5)
Y_pred_wiener = wf_model.predict(np.hstack([np.roll(X_test, i, axis=0) for i in range(5)])[5:])
Y_test_lagged = Y_test[5:]

wiener_r2 = r2_score(Y_test_lagged, Y_pred_wiener, multioutput='raw_values')
wiener_mse = mean_squared_error(Y_test_lagged, Y_pred_wiener, multioutput='raw_values')

print(f"Wiener Filter R²: {wiener_r2}")
print(f"Wiener Filter MSE: {wiener_mse}")

# Save Wiener results
np.save(os.path.join(output_dir, "Y_pred_wiener.npy"), Y_pred_wiener)

# Save metrics
metrics = {
    "kalman_r2": kalman_r2.tolist() if kalman_r2 is not None else None,
    "kalman_mse": kalman_mse.tolist() if kalman_mse is not None else None,
    "wiener_r2": wiener_r2.tolist(),
    "wiener_mse": wiener_mse.tolist(),
}
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# %%
# --- OPTIONAL: PLOT RESULTS ---
plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
plt.plot(Y_test[:,0], label='True vel_x')
plt.plot(Y_pred_lr[:,0], label='LR pred vel_x', alpha=0.7)
plt.title('Linear Regression: True vs Predicted Velocity X')
plt.legend()
plt.subplot(2,1,2)
plt.plot(Y_test[:,1], label='True vel_y')
plt.plot(Y_pred_lr[:,1], label='LR pred vel_y', alpha=0.7)
plt.title('Linear Regression: True vs Predicted Velocity Y')
plt.legend()
plt.tight_layout()
plt.show() 
# %%
