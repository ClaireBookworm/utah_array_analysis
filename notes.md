# NOTES

## file

Loading data...
Behavioral data shape: (58298, 8)
Columns: ['timestamp', 'velocity_x', 'velocity_y', 'trial_start', 'trial_win', 'trial_lose', 'num_targets', 'target_index']
Timestamp range: 1742937748.171605 to 1742939256.433396
Found 141 starts, 127 wins
Created 141 trials
Target distribution: target
0    28
1    14
2    16
3    13
4    19
5    15
6    19
7    17
Name: count, dtype: int64
Loaded (300000, 96) neural data, 141 trials
Plotting neural overview...
Analyzing trial-aligned activity...
Aligning 141 trials to neural data of shape (300000, 96)
Extracting neural features...
Neural features shape: (300000, 96)
Successfully aligned 141 trials
Targets found: [np.float64(0.0), np.float64(1.0), np.float64(2.0), np.float64(3.0), np.float64(4.0), np.float64(5.0), np.float64(6.0), np.float64(7.0)]
Computing population dynamics...
Testing movement direction decoding...
Movement direction decoding: 0.191 ± 0.028
Chance level: 0.125
Creating Utah array visualization...

Header: `{'file_type': 'BRSMPGRP', 'spec_version': '3.0', 'header_bytes': 6650, 'label': 'raw', 'comment': '', 'sampling_rate': 30000.0, 'period': 1, 'timestamp_resolution': 1000000000, 'time_origin': datetime.datetime(2025, 3, 25, 21, 22, 53, 360000), 'channel_count': 96}`

Loading neural data...
No extended headers found, creating default channel info
Neural data: 96 channels at 30000.0 Hz
Neural time origin: 2025-03-25 21:22:53.360000+00:00
Loading behavioral data...
Time offset: 43225.36 seconds
Found 141 trial starts and 127 successful trials
Matched 140 complete trials
Extracting spike band power...
First packet: timestamp=1739738583385410481, points=1
Loaded neural data shape: (1, 96)

Dataset Summary:
- 96 neural channels
- 140 complete trials
- 8 unique targets
- Neural data sample: (1, 96)

———
## hand knob region
hand knob - omega shaped anatomical landmark in the primary motor cortex that specifically controls hand and finger movements in primates. 

in cynomolgus macaques (macaca fascicularis) the hand representation sits in the rostral bank of the central sulcus 

cytoarchitecture
- giant betz cells (layer 5 pyramidal neurons)
- dense corticospinal projections 
- high density of gaba interneurons 
- distinct laminar organization vs. adjacent areas 

functional organization:

somatotopic hand/finger map
thumb typically most lateral/ventral
digits 2-5 arranged medial to thumb
intrinsic hand muscles vs extrinsic forearm muscles separated
precision grip circuits heavily represented

contralateral control:

~90% of corticospinal fibers cross at pyramidal decussation
strong contralateral hand control
some ipsilateral connections for proximal muscles
bilateral coordination circuits present

electrophysiology:

high firing rates during skilled movements
directional tuning for reaching
grip force encoding
movement preparation activity
plasticity with motor learning

## nsx file format 

binary formarts for storing continuously sampled extracellular neural data from blackrock 
-> header with sampling info (rate, channels, timestamps)
-> extended headers describing each channel's config
-> data packets w/ timestamped blocks of multichannel samples 

compabnion to NEV files (store spike events)
diff nsx nubmers = diff sampling rates 
stores 16-bit signed integers per sample
channels map to physical conenctors/pins on recording hardware
includes filer specs and calibration factors 