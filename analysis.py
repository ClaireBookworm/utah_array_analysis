import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import struct

# Feature extraction
def extract_spike_band_power(data, fs=30000, freq_band=(200, 400)):
    """Extract high-gamma power (spike band power)"""
    b, a = signal.butter(4, freq_band, btype='band', fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=0)
    power = np.abs(signal.hilbert(filtered, axis=0))**2
    return gaussian_filter1d(power, sigma=fs//100, axis=0)  # 10ms smoothing

def extract_lfp_bands(data, fs=30000):
    """Extract multiple LFP frequency bands"""
    bands = {'beta': (13, 30), 'gamma': (30, 100), 'high_gamma': (100, 200)}
    features = {}
    
    for name, (low, high) in bands.items():
        b, a = signal.butter(4, [low, high], btype='band', fs=fs)
        filtered = signal.filtfilt(b, a, data, axis=0)
        features[name] = np.abs(signal.hilbert(filtered, axis=0))**2
    
    return features

def get_spike_threshold_crossings(data, threshold_std=4):
    """Simple spike detection via threshold crossing"""
    thresholds = -threshold_std * np.std(data, axis=0)
    spikes = data < thresholds[None, :]
    spike_counts = np.sum(spikes, axis=0)
    return spikes, spike_counts

# Data loading utilities
def load_neural_chunk(filepath, start_sample=0, n_samples=300000):
    """Load chunk of neural data"""
    with open(filepath, 'rb') as f:
        # Skip to data section (assume starts at byte 316 for now)
        f.seek(316)
        
        # Find first data packet
        while True:
            byte = f.read(1)
            if byte and byte[0] == 0x01:
                break
        
        # Read timestamp and data info
        timestamp = struct.unpack('<Q', f.read(8))[0]
        num_points = struct.unpack('<I', f.read(4))[0]
        
        # Skip to desired start
        if start_sample > 0:
            f.seek(start_sample * 96 * 2, 1)
        
        # Read data
        data_bytes = f.read(n_samples * 96 * 2)
        data = struct.unpack(f'<{len(data_bytes)//2}h', data_bytes)
        return np.array(data).reshape(-1, 96) * 0.25  # Convert to µV

def align_behavioral_trials(behavioral_df):
    """Extract trial structure from behavioral data"""
    print(f"Behavioral data shape: {behavioral_df.shape}")
    print(f"Columns: {list(behavioral_df.columns)}")
    print(f"Timestamp range: {behavioral_df['timestamp'].min()} to {behavioral_df['timestamp'].max()}")
    
    starts = behavioral_df[behavioral_df['trial_start'] == True].copy()
    wins = behavioral_df[behavioral_df['trial_win'] == True].copy()
    
    print(f"Found {len(starts)} starts, {len(wins)} wins")
    
    trials = []
    for i, (_, start) in enumerate(starts.iterrows()):
        start_ts = start['timestamp']
        target = start['target_index']
        
        # Find movement onset (first non-zero velocity after start)
        # Look for velocity changes in a reasonable window
        window_start = start_ts
        window_end = start_ts + 180000  # 6 seconds at 30khz
        
        move_data = behavioral_df[
            (behavioral_df['timestamp'] >= window_start) & 
            (behavioral_df['timestamp'] <= window_end)
        ]
        
        # Look for velocity above threshold
        moving = move_data[
            (move_data['velocity_x'].abs() > 0.5) | 
            (move_data['velocity_y'].abs() > 0.5)
        ]
        
        if len(moving) > 0:
            move_ts = moving.iloc[0]['timestamp']
            trials.append({
                'trial_start': start_ts,
                'movement_onset': move_ts,
                'target': target,
                'prep_time': move_ts - start_ts,
                'trial_idx': i
            })
        else:
            # Use a fixed delay if no movement detected
            trials.append({
                'trial_start': start_ts,
                'movement_onset': start_ts + 30000,  # 1 second delay
                'target': target,
                'prep_time': 30000,
                'trial_idx': i
            })
    
    trials_df = pd.DataFrame(trials)
    print(f"Created {len(trials_df)} trials")
    if len(trials_df) > 0:
        print(f"Target distribution: {trials_df['target'].value_counts().sort_index()}")
    
    return trials_df

# Visualization functions
def plot_neural_overview(data, fs=30000, duration=10):
    """Plot overview of neural data"""
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Raw signals (subset of channels)
    axes[0].plot(time, data[:n_samples, ::10], alpha=0.7, lw=0.5)
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title('Raw Neural Signals (every 10th channel)')
    
    # Spike band power
    sbp = extract_spike_band_power(data[:n_samples])
    axes[1].plot(time, np.mean(sbp, axis=1), 'r', lw=2)
    axes[1].set_ylabel('SBP (µV²)')
    axes[1].set_title('Average Spike Band Power')
    
    # LFP bands
    lfp = extract_lfp_bands(data[:n_samples])
    for name, power in lfp.items():
        axes[2].plot(time, np.mean(power, axis=1), label=name, lw=1.5)
    axes[2].set_ylabel('LFP Power (µV²)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].set_title('LFP Frequency Bands')
    
    plt.tight_layout()
    return fig

# convert neural file start time to unix timestamp
neural_start_str = "2025-03-25T9:22:53Z"
neural_start_unix = pd.to_datetime(neural_start_str).timestamp()

def make_lagged_matrix(X, order):
    """
    Construct a lagged feature matrix for time series decoding.
    X: [n_samples, n_features]
    order: number of lags (int)
    Returns: [n_samples - order, n_features * order] lagged matrix
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    return np.hstack([np.roll(X, i, axis=0) for i in range(order)])[order:]

# Standard timestamp alignment function (for all scripts)
def align_timestamps(behavioral_unix, neural_start_unix, fs):
    """
    Convert behavioral unix time to neural sample index.
    behavioral_unix: float or array, unix timestamp(s) of behavioral events
    neural_start_unix: float, unix timestamp of neural file start
    fs: float, sampling rate
    Returns: int or array, sample index/indices
    """
    time_offset = behavioral_unix - neural_start_unix
    sample_index = (time_offset * fs).astype(int) if hasattr(time_offset, '__iter__') else int(time_offset * fs)
    return sample_index

def plot_trial_aligned_activity(data, trials, fs=30000, feature='sbp'):
    """Plot neural activity aligned to behavioral events"""
    if len(trials) == 0:
        print("No trials to align")
        return None
    
    print(f"Aligning {len(trials)} trials to neural data of shape {data.shape}")
    
    window = [-1, 3]  # -1 to +3 seconds around movement
    window_samples = [int(w * fs) for w in window]
    time = np.linspace(window[0], window[1], window_samples[1] - window_samples[0])
    
    # Extract features
    print("Extracting neural features...")
    if feature == 'sbp':
        neural_features = extract_spike_band_power(data)
    else:
        neural_features = data
    
    print(f"Neural features shape: {neural_features.shape}")
    
    # Convert behavioral timestamps to sample indices
    # Assume behavioral timestamps are in microseconds (common for NSx)
    # and neural data starts at sample 0
    aligned_data = []
    valid_trials = []
    
    for _, trial in trials.iterrows():
        move_sample = align_timestamps(trial['movement_onset'], neural_start_unix, fs)
        start_idx = move_sample + window_samples[0]
        end_idx = move_sample + window_samples[1]
                
        if start_idx >= 0 and end_idx < len(neural_features):
            trial_data = neural_features[start_idx:end_idx]
            aligned_data.append(trial_data)
            valid_trials.append(trial)
        else:
            print(f"Trial {trial['trial_idx']} out of bounds: samples {start_idx}-{end_idx}, data length {len(neural_features)}")
    
    if len(aligned_data) == 0:
        print("No valid trials found for alignment")
        return None
    
    print(f"Successfully aligned {len(aligned_data)} trials")
    
    aligned_array = np.stack(aligned_data)  # [trials, time, channels]
    valid_trials_df = pd.DataFrame(valid_trials)
    
    # Plot by target direction (assume targets are 0-7)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    targets_found = valid_trials_df['target'].unique()
    print(f"Targets found: {sorted(targets_found)}")
    
    for target in range(0, 8):
        ax_idx = target
        target_mask = valid_trials_df['target'] == target
        n_trials = np.sum(target_mask)
        
        if n_trials == 0:
            axes[ax_idx].text(0.5, 0.5, f'No trials\nfor target {target}', 
                            ha='center', va='center', transform=axes[ax_idx].transAxes)
            axes[ax_idx].set_title(f'Target {target} (n=0)')
            continue
            
        target_data = aligned_array[target_mask]
        mean_activity = np.mean(target_data, axis=(0, 2))  # Average across trials and channels
        sem_activity = np.std(target_data, axis=(0, 2)) / np.sqrt(n_trials)
        
        axes[ax_idx].plot(time, mean_activity, lw=2)
        axes[ax_idx].fill_between(time, mean_activity - sem_activity, 
                                 mean_activity + sem_activity, alpha=0.3)
        axes[ax_idx].axvline(0, color='r', linestyle='--', alpha=0.7)
        axes[ax_idx].set_title(f'Target {target} (n={n_trials})')
        axes[ax_idx].set_xlabel('Time from movement (s)')
        
        if target == 0:
            axes[ax_idx].set_ylabel(f'{feature.upper()} (µV²)')
    
    plt.suptitle(f'Trial-aligned {feature.upper()} by target direction')
    plt.tight_layout()
    return fig, aligned_array, valid_trials_df

def plot_population_dynamics(aligned_data, valid_trials_df):
    """Plot neural population dynamics in PC space"""
    # Reshape for PCA: [time*trials, channels]
    n_trials, n_time, n_channels = aligned_data.shape
    reshaped_data = aligned_data.reshape(-1, n_channels)
    
    # PCA
    pca = PCA(n_components=3)
    pc_data = pca.fit_transform(reshaped_data)
    pc_data = pc_data.reshape(n_trials, n_time, 3)
    
    # Plot trajectories by target
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for target in range(1, 9):
        target_mask = valid_trials_df['target'] == target
        if np.sum(target_mask) == 0:
            continue
        
        target_traj = pc_data[target_mask]
        mean_traj = np.mean(target_traj, axis=0)
        
        ax1.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2], 
                color=colors[target-1], linewidth=2, label=f'Target {target}')
        ax1.scatter(mean_traj[0, 0], mean_traj[0, 1], mean_traj[0, 2], 
                   color=colors[target-1], s=50, marker='o')  # Start
        ax1.scatter(mean_traj[-1, 0], mean_traj[-1, 1], mean_traj[-1, 2], 
                   color=colors[target-1], s=50, marker='s')  # End
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax1.set_title('Neural Trajectories (3D)')
    
    # 2D projections
    ax2 = fig.add_subplot(132)
    for target in range(1, 9):
        target_mask = valid_trials_df['target'] == target
        if np.sum(target_mask) == 0:
            continue
        target_traj = pc_data[target_mask]
        mean_traj = np.mean(target_traj, axis=0)
        ax2.plot(mean_traj[:, 0], mean_traj[:, 1], color=colors[target-1], 
                linewidth=2, label=f'Target {target}')
        ax2.scatter(mean_traj[0, 0], mean_traj[0, 1], color=colors[target-1], s=30, marker='o')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('Neural Trajectories (PC1 vs PC2)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Variance explained
    ax3 = fig.add_subplot(133)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax3.plot(range(1, len(cumvar)+1), cumvar[:20], 'bo-')
    ax3.set_xlabel('PC Number')
    ax3.set_ylabel('Cumulative Variance Explained')
    ax3.set_title('PCA Variance Explained')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, pca

def decode_movement_direction(aligned_data, targets, test_size=0.3):
    """Decode movement direction from neural population"""
    n_trials, n_time, n_channels = aligned_data.shape
    
    # Use activity around movement onset (0 to +500ms)
    move_window = slice(int(n_time*0.4), int(n_time*0.6))  # Rough movement period
    
    # Features: average activity in movement window
    X = np.mean(aligned_data[:, move_window, :], axis=1)  # [trials, channels]
    y = targets
    
    # Cross-validated decoding
    clf = LinearDiscriminantAnalysis()
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    
    print(f"Movement direction decoding: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Chance level: {1/len(np.unique(y)):.3f}")
    
    # Fit full model for visualization
    clf.fit(X, y)
    
    return scores, clf

# Utah array visualization
def plot_utah_array_map(channel_data, array_size=(10, 10)):
    """Plot data on Utah array spatial layout"""
    # Standard 96-channel Utah array is 10x10 with corner electrodes missing
    utah_map = np.full(array_size, np.nan)
    
    # Fill in the 96 channels (excluding corners)
    ch_idx = 0
    for i in range(10):
        for j in range(10):
            if (i == 0 and j == 0) or (i == 0 and j == 9) or \
               (i == 9 and j == 0) or (i == 9 and j == 9):
                continue  # Skip corners
            if ch_idx < len(channel_data):
                utah_map[i, j] = channel_data[ch_idx]
                ch_idx += 1
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(utah_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, shrink=0.8)
    plt.title('Utah Array Spatial Map')
    plt.xlabel('Array Column')
    plt.ylabel('Array Row')
    
    # Add channel numbers
    ch_idx = 0
    for i in range(10):
        for j in range(10):
            if not np.isnan(utah_map[i, j]):
                plt.text(j, i, str(ch_idx+1), ha='center', va='center', 
                        color='white', fontsize=8)
                ch_idx += 1
    
    return plt.gcf()

# Main analysis workflow
def run_analysis(neural_file, behavioral_file):
    """Complete analysis pipeline"""
    print("Loading data...")
    neural_data = load_neural_chunk(neural_file, n_samples=300000)  # 10 seconds
    behavioral_df = pd.read_csv(behavioral_file)
    trials = align_behavioral_trials(behavioral_df)
    
    print(f"Loaded {neural_data.shape} neural data, {len(trials)} trials")
    
    # Basic neural overview
    print("Plotting neural overview...")
    fig1 = plot_neural_overview(neural_data)
    
    # Trial-aligned activity
    print("Analyzing trial-aligned activity...")
    result = plot_trial_aligned_activity(neural_data, trials, feature='sbp')
    
    if result is not None:
        fig2, aligned_data, valid_trials = result
        
        # Population dynamics
        print("Computing population dynamics...")
        fig3, pca = plot_population_dynamics(aligned_data, valid_trials)
        
        # Decoding
        print("Testing movement direction decoding...")
        scores, decoder = decode_movement_direction(
            aligned_data, valid_trials['target'].values
        )
        
        # Utah array visualization
        print("Creating Utah array visualization...")
        channel_activity = np.mean(np.mean(aligned_data, axis=0), axis=0)
        fig4 = plot_utah_array_map(channel_activity)
        
        return {
            'neural_data': neural_data,
            'trials': trials,
            'aligned_data': aligned_data,
            'decoder_scores': scores,
            'figures': [fig1, fig2, fig3, fig4]
        }
    else:
        # Just show basic plots if alignment fails
        print("Trial alignment failed, showing basic neural overview only")
        
        # Show some trial timing info
        if len(trials) > 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.hist(trials['target'], bins=8, alpha=0.7)
            plt.xlabel('Target Index')
            plt.ylabel('Count')
            plt.title('Trial Distribution by Target')
            
            plt.subplot(1, 2, 2)
            plt.plot(trials['trial_start'], 'o-', alpha=0.7)
            plt.xlabel('Trial Number')
            plt.ylabel('Trial Start Timestamp')
            plt.title('Trial Timing')
            plt.tight_layout()
            fig_trials = plt.gcf()
            
            return {
                'neural_data': neural_data,
                'trials': trials,
                'figures': [fig1, fig_trials]
            }
        
        return {'neural_data': neural_data, 'trials': trials, 'figures': [fig1]}

# Usage
# results = run_analysis('data/neural.ns6', 'data/behavioral.csv')
# plt.show()