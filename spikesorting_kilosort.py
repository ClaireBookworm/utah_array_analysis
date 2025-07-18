# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from neo.io.blackrockio import BlackrockIO
import spikeinterface as si
from kilosort import run_kilosort
from neo_helpers import load_by_time
from analysis import align_timestamps

# --- CONSTANTS ---
NEURAL_FILE = "data/neural.ns6"
BEHAVIORAL_FILE = "data/actions.csv"
FS = 30000
N_CHANNELS = 96

reader = BlackrockIO(NEURAL_FILE, verbose=True)
fs = float(reader.get_signal_sampling_rate())
print(f"sampling rate: {fs} Hz")

trials = pd.read_csv(BEHAVIORAL_FILE)
trials = trials.rename(columns={
    'timestamp': 'movement_onset',
    'trial_start': 'trial_idx',
    'num_targets': 'target_count',
    'target_index': 'target'
})

def extract_waveforms(recording, sorting, ms_before=1.0, ms_after=2.0):
    """extract spike waveforms for quality metrics"""
    we = si.extract_waveforms(
        recording,
        sorting,
        folder='./waveforms',
        ms_before=ms_before,
        ms_after=ms_after,
        max_spikes_per_unit=1000
    )
    return we


def prep_for_ks4(raw_data, fs, output_dir='./ks4_data'):
    """convert to kilosort4 format"""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # kilosort wants (n_samples, n_channels)
    if raw_data.shape[0] < raw_data.shape[1]:
        raw_data = raw_data.T

    # save as binary
    # raw_data.astype('int16').tofile(f'{output_dir}/data.bin')
    bin_file = output_dir / 'data.bin'
    raw_data.astype('int16').tofile(bin_file)


    # utah array geometry - 10x10 grid, 400um spacing
    n_channels = raw_data.shape[1]
    grid_size = int(np.sqrt(n_channels))  # assume square grid
    spacing = 400  # microns

    xc = []
    yc = []
    for i in range(grid_size):
        for j in range(grid_size):
            xc.append(j * spacing)
            yc.append(i * spacing)

    """
    ops dict DEFAULT SETTINGS options
    dict_keys(['n_chan_bin', 'fs', 'batch_size', 'nblocks', 'Th_universal', 'Th_learned', 'tmin', 'tmax', 'nt', 'shift', 'scale', 'artifact_threshold', 'nskip', 'whitening_range', 'highpass_cutoff', 'binning_depth', 'sig_interp', 'drift_smoothing', 'nt0min', 'dmin', 'dminx', 'min_template_size', 'template_sizes', 'nearest_chans', 'nearest_templates', 'max_channel_distance', 'max_peels', 'templates_from_data', 'n_templates', 'n_pcs', 'Th_single_ch', 'acg_threshold', 'ccg_threshold', 'cluster_neighbors', 'cluster_downsampling', 'max_cluster_subset', 'x_centers', 'duplicate_spike_ms', 'position_limit'])
    """
    ops = {
        'fs': fs,
        'data_dir': output_dir,
        'n_chan_bin': 96,
        'probe_path': 'utah96.prb',
        'Th_single_ch': 3,     # lower single channel threshold
        'Th_universal': 4,     # lower universal threshold
        'Th_learned': 6,       # learned template threshold
    }
    return ops

chunk2 = load_by_time(reader, 20)
print("Loaded chunk")
print(f"data range: {chunk2.min():.3f} to {chunk2.max():.3f}")
print(f"data std: {chunk2.std():.3f}")

plt.figure(figsize=(10, 4))
plt.plot(chunk2[:1000, 0])  # first 1000 samples, channel 0
plt.title('raw data - should see spikes')
plt.ylabel('amplitude')
plt.show()

ops = prep_for_ks4(chunk2, fs)
ops = run_kilosort(ops)
print(ops)

# dict_keys(['n_chan_bin', 'fs', 'batch_size', 'nblocks', 'Th_universal', 'Th_learned', 'tmin', 'tmax', 'nt', 'shift', 'scale', 'artifact_threshold', 'nskip', 'whitening_range', 'highpass_cutoff', 'binning_depth', 'sig_interp', 'drift_smoothing', 'nt0min', 'dmin', 'dminx', 'min_template_size', 'template_sizes', 'nearest_chans', 'nearest_templates', 'max_channel_distance', 'max_peels', 'templates_from_data', 'n_templates', 'n_pcs', 'Th_single_ch', 'acg_threshold', 'ccg_threshold', 'cluster_neighbors', 'cluster_downsampling', 'max_cluster_subset', 'x_centers', 'duplicate_spike_ms', 'position_limit', 'data_dir', 'probe_path', 'filename', 'settings', 'probe', 'data_dtype', 'do_CAR', 'invert_sign', 'NTbuff', 'Nchan', 'duplicate_spike_bins', 'torch_device', 'save_preprocessed_copy', 'chanMap', 'xc', 'yc', 'kcoords', 'n_chan', 'Nbatches', 'preprocessing', 'Wrot', 'fwav', 'runtime_preproc', 'usage_preproc', 'wPCA', 'wTEMP', 'yup', 'xup', 'ycup', 'xcup', 'iC', 'iC2', 'weigh', 'yblk', 'dshift', 'iKxx', 'runtime_drift', 'usage_drift', 'cuda_drift', 'runtime_st0', 'usage_st0', 'cuda_st0', 'runtime_clu0', 'usage_clu0', 'cuda_clu0', 'iCC', 'iCC_mask', 'iU', 'runtime_st', 'usage_st', 'cuda_st', 'runtime_clu', 'usage_clu', 'cuda_clu', 'runtime_merge', 'usage_merge', 'cuda_merge', 'n_units_total', 'n_units_good', 'n_spikes', 'mean_drift', 'runtime_postproc', 'usage_postproc', 'runtime', 'cuda_postproc'])
# %%
data_dir = Path('./ks4_data/kilosort4')
# try to find results in data_dir instead
results_files = ['spike_times.npy', 'spike_clusters.npy', 'templates.npy', 'amplitudes.npy']
results = {}

for fname in results_files:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        results[fname.split('.')[0]] = np.load(fpath)
        print(f"loaded {fname}: shape {results[fname.split('.')[0]].shape}")
    else:
        print(f"missing: {fname}")

# %%

# %%

def analyze_spike_sorting_results(spike_times, spike_clusters, templates, raw_data, fs):
    """comprehensive spike sorting analysis plots"""

    unit_ids = np.unique(spike_clusters)
    n_units = len(unit_ids)
    recording_duration = raw_data.shape[0] / fs

    print(f"=== spike sorting summary ===")
    print(f"units found: {n_units}")
    print(f"total spikes: {len(spike_times)}")
    print(f"recording duration: {recording_duration:.1f}s")
    print(f"overall firing rate: {len(spike_times)/recording_duration:.1f} spikes/s")

    # 1. RASTER PLOT
    plt.figure(figsize=(15, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, min(n_units, 50)))

    for i, unit in enumerate(unit_ids[:50]):  # limit to 50 units for visibility
        unit_spikes = spike_times[spike_clusters == unit] / fs
        plt.scatter(unit_spikes, [unit] * len(unit_spikes),
                   s=1, alpha=0.8, c=[colors[i]], label=f'unit {unit}' if i < 10 else "")

    plt.ylabel('unit id')
    plt.xlabel('time (s)')
    plt.title(f'raster plot: {n_units} units (showing first 50)')
    plt.grid(True, alpha=0.3)
    if n_units <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2. FIRING RATE DISTRIBUTION
    firing_rates = []
    spike_counts = []
    for unit in unit_ids:
        n_spikes = np.sum(spike_clusters == unit)
        rate = n_spikes / recording_duration
        firing_rates.append(rate)
        spike_counts.append(n_spikes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(firing_rates, bins=min(20, n_units//2), alpha=0.7, color='skyblue')
    ax1.axvline(np.median(firing_rates), color='red', linestyle='--',
                label=f'median: {np.median(firing_rates):.2f} Hz')
    ax1.set_xlabel('firing rate (Hz)')
    ax1.set_ylabel('number of units')
    ax1.set_title('firing rate distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(spike_counts, bins=min(20, n_units//2), alpha=0.7, color='lightcoral')
    ax2.set_xlabel('total spike count')
    ax2.set_ylabel('number of units')
    ax2.set_title('spike count distribution')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. TEMPLATE WAVEFORMS
    if templates is not None and templates.size > 0:
        n_show = min(16, n_units)
        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        axes = axes.flatten()

        for i, unit in enumerate(unit_ids[:n_show]):
            if unit < len(templates):
                # template shape is usually (n_units, n_timepoints, n_channels)
                template = templates[unit]
                if len(template.shape) == 2:  # (timepoints, channels)
                    # show template on channel with max amplitude
                    max_chan = np.argmax(np.abs(template).max(axis=0))
                    waveform = template[:, max_chan]
                else:  # might be flattened
                    waveform = template

                axes[i].plot(waveform, 'k-', linewidth=1)
                axes[i].set_title(f'unit {unit}', fontsize=10)
                axes[i].set_xlim(0, len(waveform))
                axes[i].grid(True, alpha=0.3)

        # hide unused subplots
        for i in range(n_show, 16):
            axes[i].set_visible(False)

        plt.suptitle('template waveforms (max amplitude channel)', fontsize=14)
        plt.tight_layout()
        plt.show()

    # 4. ISI (INTER-SPIKE INTERVAL) DISTRIBUTIONS
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, unit in enumerate(unit_ids[:6]):  # first 6 units
        unit_spikes = spike_times[spike_clusters == unit] / fs
        if len(unit_spikes) > 1:
            isi = np.diff(unit_spikes) * 1000  # convert to ms
            isi = isi[isi < 100]  # limit to 100ms for visibility

            if len(isi) > 0:
                axes[i].hist(isi, bins=50, alpha=0.7, density=True)
                axes[i].axvline(1, color='red', linestyle='--', alpha=0.7,
                               label='1ms refractory')
                axes[i].set_xlabel('ISI (ms)')
                axes[i].set_ylabel('density')
                axes[i].set_title(f'unit {unit} (n={len(unit_spikes)})')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

    plt.suptitle('inter-spike interval distributions', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 5. AMPLITUDE OVER TIME (drift check)
    amplitudes = results.get('amplitudes', None)
    if 'amplitudes' in globals() and len(amplitudes) > 0:
        plt.figure(figsize=(12, 6))

        for unit in unit_ids[:10]:  # first 10 units
            unit_mask = spike_clusters == unit
            unit_times = spike_times[unit_mask] / fs
            unit_amps = amplitudes[unit_mask]

            if len(unit_times) > 10:
                plt.scatter(unit_times, unit_amps, s=1, alpha=0.6, label=f'unit {unit}')

        plt.xlabel('time (s)')
        plt.ylabel('spike amplitude')
        plt.title('amplitude drift over time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 6. CROSS-CORRELOGRAM (for first few units)
    if n_units >= 2:
        fig, axes = plt.subplots(1, min(3, n_units-1), figsize=(12, 4))
        if n_units == 2:
            axes = [axes]

        for i in range(min(3, n_units-1)):
            unit1, unit2 = unit_ids[0], unit_ids[i+1]
            spikes1 = spike_times[spike_clusters == unit1] / fs
            spikes2 = spike_times[spike_clusters == unit2] / fs

            # compute cross-correlogram
            max_lag = 0.05  # 50ms
            lags = []
            for spike in spikes1[:1000]:  # limit for speed
                diffs = spikes2 - spike
                diffs = diffs[(np.abs(diffs) <= max_lag)]
                lags.extend(diffs)

            if len(lags) > 0:
                axes[i].hist(lags, bins=50, alpha=0.7)
                axes[i].axvline(0, color='red', linestyle='--', alpha=0.7)
                axes[i].set_xlabel('lag (s)')
                axes[i].set_ylabel('count')
                axes[i].set_title(f'unit {unit1} vs {unit2}')
                axes[i].grid(True, alpha=0.3)

        plt.suptitle('cross-correlograms', fontsize=14)
        plt.tight_layout()
        plt.show()

    # 7. QUALITY METRICS SUMMARY
    print(f"\n=== quality summary ===")
    print(f"units with >10 spikes: {np.sum(np.array(spike_counts) > 10)}")
    print(f"units with >1 Hz: {np.sum(np.array(firing_rates) > 1)}")
    print(f"median firing rate: {np.median(firing_rates):.2f} Hz")
    print(f"highest firing rate: {np.max(firing_rates):.2f} Hz")

    return {
        'firing_rates': firing_rates,
        'spike_counts': spike_counts,
        'unit_ids': unit_ids
    }
# %%
analyze_spike_sorting_results(
    results.get('spike_times', np.array([])),
    results.get('spike_clusters', np.array([])),
    results.get('templates', np.array([])),
    chunk2,
    fs
)
# %%

chunk3 = load_by_time(reader, 40, start_time=40)
print("Loaded chunk")
print(f"data range: {chunk3.min():.3f} to {chunk3.max():.3f}")
print(f"data std: {chunk3.std():.3f}")

plt.figure(figsize=(10, 4))
plt.plot(chunk3[:5000, 0])  # first 1000 samples, channel 0
plt.title('raw data - should see spikes')
plt.ylabel('amplitude')
plt.show()

ops = prep_for_ks4(chunk3, fs, output_dir='./ks4_data2')
ops = run_kilosort(ops)
print(ops)

# %%
data_dir = Path('./ks4_data2/kilosort4')
# try to find results in data_dir instead
results_files = ['spike_times.npy', 'spike_clusters.npy', 'templates.npy', 'amplitudes.npy']
results = {}

for fname in results_files:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        results[fname.split('.')[0]] = np.load(fpath)
        print(f"loaded {fname}: shape {results[fname.split('.')[0]].shape}")
    else:
        print(f"missing: {fname}")

analyze_spike_sorting_results(
    results.get('spike_times', np.array([])),
    results.get('spike_clusters', np.array([])),
    results.get('templates', np.array([])),
    chunk3,
    fs
)
# %%
chunk3 = load_by_time(reader, 60, start_time=10)
print("Loaded chunk")
print(f"data range: {chunk3.min():.3f} to {chunk3.max():.3f}")
print(f"data std: {chunk3.std():.3f}")

plt.figure(figsize=(10, 4))
plt.plot(chunk3[:5000, 0])  # first 1000 samples, channel 0
plt.title('raw data - should see spikes')
plt.ylabel('amplitude')
plt.show()

ops = prep_for_ks4(chunk3, fs, output_dir='./ks4_data3')
ops = run_kilosort(ops)
print(ops)

# %%
data_dir = Path('./ks4_data3/kilosort4')
# try to find results in data_dir instead
results_files = ['spike_times.npy', 'spike_clusters.npy', 'templates.npy', 'amplitudes.npy']
results = {}

for fname in results_files:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        results[fname.split('.')[0]] = np.load(fpath)
        print(f"loaded {fname}: shape {results[fname.split('.')[0]].shape}")
    else:
        print(f"missing: {fname}")

analyze_spike_sorting_results(
    results.get('spike_times', np.array([])),
    results.get('spike_clusters', np.array([])),
    results.get('templates', np.array([])),
    chunk3,
    fs
)
# %%

