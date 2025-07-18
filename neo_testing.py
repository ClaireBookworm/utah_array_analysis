# %%
from neo.io.blackrockio import BlackrockIO
import quantities as pq
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
# %%
def load_by_samples(reader, n_samples=30000):
    """direct chunk loading by sample count"""
    raw_chunk = reader.get_analogsignal_chunk(i_stop=n_samples)
    # rescaling it turns it into voltages
    return reader.rescale_signal_raw_to_float(raw_chunk, dtype="float64")

def load_by_time(reader, duration_sec=10, start_time=0):
    """time-based loading using neo blocks"""
    block = reader.read_block(lazy=True)
    # block : segments [ analog signals, spike trains ], channel_indexes
    proxy = block.segments[0].analogsignals[0]
    # signal starts at 1739738583.3854 s, ends at 1739740084.5254   s
    if start_time == 0 or start_time is None: 
        t_start = proxy.t_start
    else:
        # start_time is seconds from beginning of recording
        t_start = proxy.t_start + start_time * pq.s
    
    t_end = t_start + duration_sec * pq.s
    
    # check bounds
    if t_end > proxy.t_stop:
        print(f"warning: requested end time {t_end} exceeds file end {proxy.t_stop}")
        t_end = proxy.t_stop
    
    subset = proxy.time_slice(t_start, t_end)
    return subset.magnitude
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
freqs, psd = welch(data, fs, nperseg=1024)
plt.figure(figsize=(10, 4))
plt.semilogy(freqs, psd)
plt.title('Power Spectrum Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V²/Hz)')
plt.grid(True)
plt.show()

# The power spectrum shows how the power of a signal is distributed across different frequencies.
# It helps identify dominant oscillations or rhythms in neural data, such as LFP bands or noise.
# Peaks in the power spectrum indicate frequencies where the signal has higher energy.

# %%
# convert neural file start time to unix timestamp
neural_start_str = "2025-03-25T9:22:53Z"
neural_start_unix = pd.to_datetime(neural_start_str).timestamp()

# align behavioral events to neural timeline
def align_timestamps(behavioral_unix, neural_start_unix, fs):
    """convert behavioral unix time to neural sample index"""
    time_offset = behavioral_unix - neural_start_unix  # seconds since neural start
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
duration = 8
trials_samples = trials['movement_onset'].apply(lambda x: align_timestamps(x, neural_start_unix, fs))

trial_start_sample = trials['movement_onset'].min()
trial_start_time = (trial_start_sample - neural_start_unix) / fs
print(f"Trial start time: {trial_start_time:.2f} seconds")
chunk_from_trials = load_by_time(reader, duration_sec=duration, start_time=trial_start_time)

trial_start_sample = trials_samples.min()
chunk_end_sample = trial_start_sample + (duration * fs)
valid_mask = (trials_samples >= trial_start_sample) & (trials_samples < chunk_end_sample)
trials_subset = trials[valid_mask].copy()
trials_subset['movement_onset'] = trials_samples[valid_mask] - trial_start_sample # adjust to chunk-relative sample indices
print(f"trials in {duration} sec chunk: {len(trials_subset)}")
print(f"trial spacing: {np.diff(trials_subset['movement_onset'].head(5))} samples")
# %%
# then call it
fig, aligned_data, valid_trials = plot_aligned_trials(
    data=chunk_from_trials,
    trials=trials_subset,
    fs=30000,
    feature='sbp'
)
# %%
