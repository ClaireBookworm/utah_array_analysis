# neo helpers 
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from neo.io.blackrockio import BlackrockIO

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