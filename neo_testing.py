# %%
from neo.io.blackrockio import BlackrockIO
from neo import Block, Segment, AnalogSignal, SpikeTrain
import quantities as pq
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
# %%
reader = BlackrockIO(filename="data/neural.ns6", verbose=True)
# reader.parse_header()
print(reader)
# print(reader.header)
# %%
sampling_rate = reader.get_signal_sampling_rate()
t_start = reader.get_signal_t_start(block_index=0, seg_index=0)
units = reader.header["signal_channels"][0]["units"]
print(f"{sampling_rate=}, {t_start=}, {units=}")
# %%
# one way to load chunk
raw_chunk = reader.get_analogsignal_chunk(i_stop=30000)
chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype="float64") # makes it voltages, memmap file format
print(raw_chunk.shape, chunk.shape)
# %%
# second way to load chunk
block = reader.read_block(lazy=True)  # lazy=True for metadata only
# block : segments [ analog signals, spike trains ], channel_indexes
continuous_data = block.segments[0].analogsignals[0]  # first analog signal
fs = float(continuous_data.sampling_rate)
# signal starts at 1739738583.3854105 s, ends at 1739740084.5254107 s
t_end = continuous_data.t_start + 10 * pq.s  # add 5 seconds
subset = continuous_data.time_slice(continuous_data.t_start, t_end)
raw_data = subset.magnitude  # actual numpy array
print(raw_data.shape)
# %%
# single channel
data = raw_data[:, 1].flatten()
b, a = butter(4, [300, 6000], btype='band', fs=fs) # bandpass filter
filtered = filtfilt(b, a, data)
print(filtered.shape)

plt.figure(figsize=(12, 4))
plt.plot(filtered, label='Filtered Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude (V)')
plt.title('Bandpass Filtered Signal (300-6000 Hz)')
plt.legend()
plt.tight_layout()
plt.show()
# %%
# %%

