# %%
from nsx_parser import NSxParser, MotorCortexAnalyzer
from analysis import run_analysis
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# %%
# Usage for the challenge
def analyze_motor_cortex_data(nsx_path: str, behavioral_path: str):
    """Main analysis pipeline for the challenge"""
    analyzer = MotorCortexAnalyzer(nsx_path, behavioral_path)
    
    # Load and align data
    header, channels = analyzer.load_data()
    trials = analyzer.extract_trials()
    
    # Extract neural features
    neural_sample = analyzer.extract_spike_band_power()
    
    print(f"\nDataset Summary:")
    print(f"- {header['channel_count']} neural channels")
    print(f"- {len(trials)} complete trials") 
    print(f"- {len(trials['target_index'].unique())} unique targets")
    print(f"- Neural data sample: {neural_sample.shape}")
    
    return analyzer, trials, neural_sample
# %%
analyzer, trials, neural_sample = analyze_motor_cortex_data("data/neural.ns6", "data/actions.csv")
# %% 
print(trials.iloc[0])
# %%
analyzer = MotorCortexAnalyzer("data/neural.ns6", "data/actions.csv")
parser = NSxParser("data/neural.ns6")
# %%
parser.parse_header()
# %%
parser.try_parse_extended_headers()
# %%
analyzer.load_data()
analyzer.extract_trials()
analyzer.extract_spike_band_power()
# %%
results = run_analysis('data/neural.ns6', 'data/actions.csv')
plt.show()

# %%
import struct
from datetime import datetime, timezone, timedelta

def ns6_to_datetime(sample_timestamp):
    origin = datetime(2025, 3, 29, 9, 22, 53, tzinfo=timezone.utc)
    return origin + timedelta(seconds=sample_timestamp / 30000)
# %%
print(ns6_to_datetime(1742937748.171605))
print(ns6_to_datetime(1742939256.433396))
# %%
import numpy as np
from scipy import signal

def kaiser_filter_neural_data(raw_data, fs=30000, band_type='lfp'):
   """
   Simple Kaiser filter for 96-channel neural data
   
   raw_data: (time_samples, 96_channels) - raw voltage traces
   fs: sampling rate (30kHz typical for neural data)
   band_type: 'lfp', 'spike', 'h1', 'h2'
   """
   
   # Define frequency bands (like in the paper)
   bands = {
       'lfp': (0.3, 2),      # Low freq LFP
       'spike': (300, 3000), # Spike band  
       'h1': (100, 200),     # High freq band 1
       'h2': (200, 400)      # High freq band 2
   }
   
   low, high = bands[band_type]
   
   # Kaiser filter design
   nyquist = fs / 2
   low_norm = low / nyquist
   high_norm = high / nyquist
   
   if band_type == 'lfp':
       # Low-pass filter
       b = signal.firwin(101, low_norm, window=('kaiser', 8))
   else:
       # Band-pass filter
       b = signal.firwin(101, [low_norm, high_norm], 
                        pass_zero=False, window=('kaiser', 8))
   
   # Apply to all 96 channels
   filtered_data = np.zeros_like(raw_data)
   for ch in range(96):
       # Zero-phase filtering (like filtfilt in paper)
       filtered_data[:, ch] = signal.filtfilt(b, 1, raw_data[:, ch])
   
   # Downsample LFP to 1kHz (like paper)
   if band_type == 'lfp':
       filtered_data = filtered_data[::30]  # 30kHz -> 1kHz
   
   return filtered_data

# Usage:
# lfp_signals = kaiser_filter_neural_data(raw_voltage, band_type='lfp')
# spike_signals = kaiser_filter_neural_data(raw_voltage, band_type='spike')

# %%

parser.parse_header()
parser.try_parse_extended_headers()

# Read all data packets (this may use a lot of RAM for long recordings!)
timestamps, digital_data = parser.read_data_packets()

# Convert to analog (microvolts)
analog_data = parser.convert_to_analog(digital_data)

# analog_data is now a numpy array of shape [samples, 96]
# You can save it as a .npy file for fast future loading:
import numpy as np
np.save("data/neural_raw_voltage.npy", analog_data)
# %%
