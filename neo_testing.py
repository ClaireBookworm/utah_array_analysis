from neo.io.blackrockio import BlackrockIO
import quantities as pq
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def load_by_samples(reader, n_samples=30000):
    """direct chunk loading by sample count"""
    raw_chunk = reader.get_analogsignal_chunk(i_stop=n_samples)
    return reader.rescale_signal_raw_to_float(raw_chunk, dtype="float64")

def load_by_time(reader, duration_sec=10):
    """time-based loading using neo blocks"""
    block = reader.read_block(lazy=True)
    proxy = block.segments[0].analogsignals[0]
    subset = proxy.time_slice(proxy.t_start, proxy.t_start + duration_sec*pq.s)
    return subset.magnitude

# load reader
reader = BlackrockIO("data/neural.ns6", verbose=True)
fs = float(reader.get_signal_sampling_rate())
print(f"sampling rate: {fs} Hz")

# test both methods
chunk1 = load_by_samples(reader, 30000)
chunk2 = load_by_time(reader, 10)
print(f"sample method: {chunk1.shape}, time method: {chunk2.shape}")

# process single channel
ch1 = chunk2[:, 1].flatten()
b, a = butter(4, [300, 6000], btype='band', fs=fs)
filtered = filtfilt(b, a, ch1)

# plot
plt.figure(figsize=(12, 4))
plt.plot(filtered)
plt.title('bandpass filtered (300-6000 Hz)')
plt.xlabel('sample')
plt.ylabel('amplitude (V)')
plt.show()