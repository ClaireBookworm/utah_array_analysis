# %%
from neo.io.blackrockio import BlackrockIO
import datetime
import quantities as pq
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
from neo_helpers import load_by_time, load_by_samples
# %%
# load reader
reader = BlackrockIO("data/neural.ns6", verbose=True)
fs = float(reader.get_signal_sampling_rate())
print(f"sampling rate: {fs} Hz")
# chunk1 = load_by_samples(reader, 30000)
chunk2 = load_by_time(reader, 10)
# print(f"sample method: {chunk1.shape}, time method: {chunk2.shape}")
# %%

# create behavioral dataframe 
trials = pd.read_csv('data/actions.csv')
# timestamp,velocity_x,velocity_y,trial_start,trial_win,trial_lose,num_targets,target_index
trials = trials.rename(columns={
    'timestamp': 'movement_onset',
    'trial_start': 'trial_idx',
    'num_targets': 'target_count',
    'target_index': 'target'	
})
print(trials.head())
# %%
# process single channel
# ch1 = chunk2[:, 1].flatten()

def bandpass(data, low, high, fs=30000, graph=False): 
    """bandpass filter for neural data"""
    b, a = butter(4, [low, high], btype="band", fs=fs)
    filtered = filtfilt(b, a, data)
    if graph:
        plt.figure(figsize=(12, 4))
        plt.plot(filtered)
        plt.title(f'bandpass filtered ({low} - {high} Hz)')
        plt.xlabel('sample')
        plt.ylabel('amplitude (V)')
        plt.show()
    return filtered
# %%
# %%
## FILTERS
data = chunk2[:, 0].flatten()  # raw neural signal
b, a = butter(4, 300, btype='low', fs=fs)
lfp = filtfilt(b, a, data) # this gives you the LFG (through low pass filter)
print(lfp.shape)
plt.figure(figsize=(12, 4))
plt.plot(lfp)
plt.title('Low Frequency Potential (LFP)')
plt.xlabel('Sample')
plt.ylabel('Amplitude (V)')
plt.show()
# %%
# SPIKE DETECTION
# notch for 60hz
from scipy.signal import iirnotch
b, a = iirnotch(60, 30, fs)
clean = filtfilt(b, a, data)

## SPIKE DETECTION 
# threshold crossing
data = chunk2[:, 0].flatten()  # first channel
spikes = bandpass(data, 300, 6000, fs, graph=True)  # bandpass filter for spikes
threshold = - 4.5 * np.std(spikes)
crossings = np.where(spikes < threshold)[0]

# fix refractory period
isi_samples = np.diff(crossings)  # samples between crossings
refractory_samples = int(0.001 * fs)  # 1ms in samples
valid = np.concatenate([[True], isi_samples > refractory_samples])
spike_times = crossings[valid]

print(f"Raw crossings: {len(crossings)}")
print(f"After refractory: {len(spike_times)}")
print(f"Threshold: {threshold:.4f} V")

# histogram
plt.figure(figsize=(5, 2))
plt.hist(spikes, bins=100, alpha=0.7, color='blue')
plt.axvline(threshold, color='r', label=f'threshold: {threshold:.4f}V')
plt.xlabel('amplitude (V)')
plt.ylabel('count')
plt.title('signal amplitude distribution')
plt.legend()
plt.show()

# spike detection plot
plt.figure(figsize=(12, 4))
plt.plot(spikes, label='bandpass filtered', alpha=0.7)
plt.plot(spike_times, spikes[spike_times], 'ro', markersize=4, label=f'{len(spike_times)} detected spikes')
plt.axhline(threshold, color='r', linestyle='--', alpha=0.5, label='threshold')
plt.xlabel('sample')
plt.ylabel('amplitude (V)')
plt.title('spike detection on channel 0')
plt.legend()
plt.show()

# example: streaming spike detection
# all_spikes = []
# for i in range(0, total_samples, chunk_size):
#     chunk = load_chunk(i, chunk_size)
#     spikes = detect_spikes(chunk)
#     all_spikes.extend(spikes + i)  # adjust for global time
# %%

## POWER SPECTRUM 
from scipy.signal import welch

# Randomly select 8 different channels (from available channels)
num_channels = chunk2.shape[1]
np.random.seed(42)
channels = np.random.choice(num_channels, size=8, replace=False)
print(f"Randomly selected channels: {channels}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, ch in enumerate(channels):
    data = chunk2[:, ch].flatten()
    freqs, psd = welch(data, fs, nperseg=1024)
    axes[idx].semilogy(freqs, psd)
    axes[idx].set_title(f'Channel {ch}')
    axes[idx].set_xlabel('Frequency (Hz)')
    axes[idx].set_ylabel('PSD (V²/Hz)')
    axes[idx].grid(True)

plt.suptitle('Power Spectrum Density (PSD) for 8 Random Channels')
plt.tight_layout()
plt.show()

# The power spectrum shows how the power of a signal is distributed across different frequencies.
# It helps identify dominant oscillations or rhythms in neural data, such as LFP bands or noise.
# Peaks in the power spectrum indicate frequencies where the signal has higher energy.

# %%
# convert neural file start time to unix timestamp
# neural_start_str = "2025-03-25T9:22:53Z"
neural_start_unix = datetime.datetime(2025, 3, 25, 21, 22, 53, 360000).timestamp() # pd.to_datetime(neural_start_str).timestamp()

print("Difference behavioral start - neural start:", str(trials['movement_onset'].min() - neural_start_unix))

# align behavioral events to neural timeline
def align_timestamps(behavioral_unix, neural_start_unix, fs):
    """convert behavioral unix time to neural sample index"""
    time_offset = behavioral_unix - neural_start_unix  # seconds since neural start
    # earliest behavioral event occurs 25.19 seconds before neural recordings time_origin
    sample_index = int(time_offset * fs)
    return sample_index

# %%
def plot_aligned_trials(data, trials, fs=30000, feature='sbp'):
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
        neural_features = analysis.extract_spike_band_power(data)
    else:
        neural_features = data
    
    print(f"Neural features shape: {neural_features.shape}")
    
    # Convert behavioral timestamps to sample indices
    # Assume behavioral timestamps are in microseconds (common for NSx)
    # and neural data starts at sample 0
    aligned_data = []
    valid_trials = []
    
    for _, trial in trials.iterrows():
        # move_sample = align_timestamps(trial['movement_onset'], neural_start_unix, fs)
        move_sample = int(trial['movement_onset'])
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
# %%
import analysis
def get_trials_chunk(trials, neural_start_unix, reader, fs, duration=5, offset_sec=15):
    """
    Extract a chunk of neural data and corresponding trials within a time window.
    Args:
        trials (pd.DataFrame): Behavioral trials dataframe.
        neural_start_unix (float): Neural file start time (unix timestamp).
        reader (BlackrockIO): Neo reader object.
        fs (float): Sampling rate.
        duration (float): Duration of chunk in seconds.
        offset_sec (float): Offset from first movement onset in seconds.
    Returns:
        chunk_from_trials (np.ndarray): Neural data chunk.
        trials_subset (pd.DataFrame): Trials within chunk, movement_onset as sample index relative to chunk.
    """
    trials_samples = trials['movement_onset'].apply(lambda x: align_timestamps(x, neural_start_unix, fs))
    # units are all in seconds! 
    neural_load_start_unix = trials['movement_onset'].min() + offset_sec
    neural_load_start_time = (neural_load_start_unix - neural_start_unix) 
    print(f"Loading neural data starting at: {neural_load_start_time:.2f} seconds")
    
    if neural_load_start_time < 0:
        print(f"Warning: Calculated neural_load_start_time is negative ({neural_load_start_time:.2f}s). "
              "Adjusting to 0s (start of neural recording) as data prior to time_origin is typically unavailable in NS6 files.")
        neural_load_start_time = 0.0 # Cap the start time at the beginning of the recording

    chunk_from_trials = load_by_time(reader, duration_sec=duration, start_time=neural_load_start_time)

    sample_offset = int(offset_sec * fs)
    trial_filter_start = trials_samples.min() + sample_offset # adding the sample index offset
    trial_filter_end = trial_filter_start + (duration * fs)
    valid_mask = (trials_samples >= trial_filter_start) & (trials_samples < trial_filter_end)
    trials_subset = trials[valid_mask].copy()
    trials_subset['movement_onset'] = trials_samples[valid_mask] - trial_filter_start
    print(f"trials in {duration} sec chunk: {len(trials_subset)}")
    print(f"trial spacing: {np.diff(trials_subset['movement_onset'].head(5))} samples")

    return chunk_from_trials, trials_subset
# %%
offsets = [0, 45.76, 85.4, 118.46, 134.79]
aligned_concat = []
for offset in offsets:
    print(f"Loading chunk with offset {offset} seconds")
    chunk_from_trials, trials_subset = get_trials_chunk(trials=trials, neural_start_unix=neural_start_unix, reader=reader, fs=fs, duration=5, offset_sec=offset)
    
    fig, aligned_data, valid_trials = plot_aligned_trials(
        data=chunk_from_trials,
        trials=trials_subset,
        fs=30000,
        feature='sbp'
    )
    plt.show()
    if aligned_data is not None:
        aligned_concat.append(aligned_data)

if aligned_concat:
    # Concatenate along first axis (trials), ensure shape [x, y, 96]
    aligned_concat = np.concatenate(aligned_concat, axis=0)
    print(f"Concatenated shape: {aligned_concat.shape}")
else:
    aligned_concat = np.empty((0, 0, 96))
    print("No aligned data found.")
    chunk_from_trials, trials_subset = get_trials_chunk(trials=trials, neural_start_unix=neural_start_unix, reader=reader, fs=fs, duration=5, offset_sec=offset)
    
    fig, aligned_data, valid_trials = plot_aligned_trials(
        data=chunk_from_trials,
        trials=trials_subset,
        fs=30000,
        feature='sbp'
    )
    plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem # For Standard Error of the Mean

# --- Assuming 'aligned_concat' is already populated from your loop ---
# Its shape is (192, 120000, 96) as confirmed

if aligned_concat.shape[0] > 0: # Proceed only if there's data after concatenation

    # --- NEW: Average SBP across the 96 channels ---
    # This assumes that 'aligned_concat' already contains the SBP values per channel.
    # The result will be a 2D array: (total_trials, num_timepoints)
    aligned_concat_sbp_averaged_across_channels = np.mean(aligned_concat, axis=2)
    
    print(f"Original aligned_concat shape: {aligned_concat.shape}")
    print(f"Shape after averaging SBP across channels: {aligned_concat_sbp_averaged_across_channels.shape}")
    # --- END NEW ---

    # Now calculate mean and standard error across trials
    # The axis=0 here means averaging across the trials dimension
    mean_sbp = np.mean(aligned_concat_sbp_averaged_across_channels, axis=0)
    sem_sbp = sem(aligned_concat_sbp_averaged_across_channels, axis=0)

    # Determine the time vector for the x-axis
    # Based on your previous plot example, the time axis goes from -1s to 3s.
    # The number of time points is the second dimension of the channel-averaged data.
    num_timepoints = aligned_concat_sbp_averaged_across_channels.shape[1]
    time_vector = np.linspace(-1, 3, num_timepoints) # Adjust these bounds if your actual aligned window differs

    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, mean_sbp, label='Mean SBP')
    plt.fill_between(time_vector, mean_sbp - sem_sbp, mean_sbp + sem_sbp, alpha=0.2, label='SEM')

    # Add a vertical line at the alignment point (movement onset, typically 0s)
    plt.axvline(x=0, color='red', linestyle='--', label='Movement Onset')

    plt.xlabel('Time from Movement Onset (s)')
    plt.ylabel('Mean SBP (across channels)') # Updated label to reflect averaging
    plt.title(f'Trial-Aligned Mean SBP (All Trials, N={aligned_concat_sbp_averaged_across_channels.shape[0]})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("No data in aligned_concat to plot.")
# %%
def get_target_transition_times(
    trials_df: pd.DataFrame,
    time_col: str = 'movement_onset',
    target_col: str = 'target',
    verbose: bool = True # Changed default to True for your immediate need
) -> np.ndarray:
    """
    Identifies Unix timestamps where the target index changes and optionally prints details.

    Args:
        trials_df (pd.DataFrame): Behavioral DataFrame.
        time_col (str): Name of the timestamp column (e.g., 'movement_onset').
        target_col (str): Name of the target index column (e.g., 'target').
        verbose (bool): If True, prints details of each target transition.

    Returns:
        np.ndarray: Sorted Unix timestamps where the 'target_index' changes.
                    Includes the timestamp of the very first trial.
    """
    if time_col not in trials_df.columns:
        raise ValueError(f"Column '{time_col}' not found in behavioral DataFrame.")
    if target_col not in trials_df.columns:
        raise ValueError(f"Column '{target_col}' not found in behavioral DataFrame.")

    trials_sorted = trials_df.sort_values(by=time_col).reset_index(drop=True)

    # Get the target of the previous trial for comparison
    previous_target_series = trials_sorted[target_col].shift(1)
    
    # Create a mask for rows where the target *changes* from the previous one
    target_changes_mask = (trials_sorted[target_col] != previous_target_series)

    # Extract the timestamps at these change points
    transition_timestamps = trials_sorted[time_col][target_changes_mask].values

    if verbose:
        print("\n--- Target Transition Details ---")
        # Filter the DataFrame to show only the rows where a change occurred
        change_rows = trials_sorted[target_changes_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Add the 'old_target' column for clear "from X to Y" printing
        change_rows['old_target'] = previous_target_series[target_changes_mask]
        
        for idx, row in change_rows.iterrows():
            old_t = row['old_target']
            new_t = row[target_col]
            timestamp = row[time_col]
            
            # For the very first trial, 'old_target' will be NaN (no previous trial)
            if pd.isna(old_t):
                print(f"At Unix {timestamp - neural_start_unix}: First trial, target set to {int(new_t)}")
            else:
                print(f"At Unix {timestamp - neural_start_unix}: Target changed from {int(old_t)} to {int(new_t)}")
        print("-------------------------------\n")

    return transition_timestamps
# --- How to use it with your renamed DataFrame ---
target_change_unix_times = get_target_transition_times(trials)
print(f"Identified {len(target_change_unix_times)} target transition timestamps.")
# Convert transition timestamps to seconds (assuming they are in sample indices or unix time)
# If they are unix timestamps, subtract neural_start_unix to get seconds relative to neural recording
transition_times_sec = (target_change_unix_times - neural_start_unix)
print("Transition times (seconds relative to neural start):", transition_times_sec[:5])
print(f"First 5 transition times: {target_change_unix_times[:5]}")
# %%
from analysis import load_neural_chunk, align_behavioral_trials, plot_neural_overview, plot_trial_aligned_activity, plot_population_dynamics, decode_movement_direction, plot_utah_array_map
def run_analysis(neural_file = "data/neural.ns6", behavioral_file="data/actions.csv"):
    """Complete analysis pipeline"""
    print("Loading data...")
    neural_chunk = load_neural_chunk(neural_file, n_samples=300000)  # 10 seconds
    behavioral_df = pd.read_csv(behavioral_file)
    trials = align_behavioral_trials(behavioral_df)
    
    print(f"Loaded {neural_chunk.shape} neural data, {len(trials)} trials")
    
    # Basic neural overview
    print("Plotting neural overview...")
    fig1 = plot_neural_overview(neural_chunk)

    # Trial-aligned activity
    print("Analyzing trial-aligned activity...")
    result = plot_trial_aligned_activity(neural_chunk, trials, feature='sbp')

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
            'neural_data': neural_chunk,
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
                'neural_data': neural_chunk,
                'trials': trials,
                'figures': [fig1, fig_trials]
            }
        
        return {'neural_data': neural_chunk, 'trials': trials, 'figures': [fig1]}
# %%
# Usage
results = run_analysis()
plt.show()
# %%
