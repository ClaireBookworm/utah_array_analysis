
# Neural & Behavioral Data Analysis Codebase

This repository provides a comprehensive pipeline for analyzing neural and behavioral data, including spike sorting, feature extraction, multimodal analysis, and offline decoding. The code is modular, with each script and module serving a specific role in the workflow.

---

## File Summaries

### 1. `ml_multimodal_cca.py`
**Purpose:**  
Demonstrates Canonical Correlation Analysis (CCA) to find shared latent variables between neural recordings and behavioral measurements.

**Main Details:**
- Loads neural data (Blackrock NSx) and behavioral data (CSV).
- Extracts neural features (spike band power, PCA).
- Extracts behavioral features (velocity, target index).
- Runs CCA to find joint structure between neural and behavioral data.
- Visualizes canonical correlations, projections, and feature loadings.
- Includes permutation testing for statistical significance.
- Provides code for visualizing electrode contributions and behavioral projections.

---

### 2. `spikesorting_kilosort.py`
**Purpose:**  
Handles spike sorting using Kilosort (via SpikeInterface) and provides extensive quality control and visualization.

**Main Details:**
- Loads neural data and behavioral trial info.
- Prepares data for Kilosort4, including binary conversion and probe geometry.
- Runs Kilosort and loads results (spike times, clusters, templates, amplitudes).
- Provides comprehensive analysis and visualization:
  - Raster plots, firing rate and spike count distributions.
  - Template waveform plots.
  - ISI (inter-spike interval) histograms.
  - Amplitude drift over time.
  - Cross-correlograms.
  - Quality metrics summary.
- Supports multiple data chunks and batch processing.

---

### 3. `offline_decode.py`
**Purpose:**  
Performs offline decoding of behavioral variables from neural features using multiple models.

**Main Details:**
- Loads and aligns neural and behavioral data, handling time offsets.
- Extracts neural features (spike band power, PCA).
- Interpolates behavioral velocity to neural timebase.
- Saves and loads features and targets for reproducibility.
- Implements and evaluates:
  - Linear Regression
  - Kalman Filter (via pykalman)
  - Wiener Filter (lagged regression)
- Reports R² and MSE for each model.
- Saves predictions and metrics to disk.
- Includes plotting for true vs. predicted velocities.

---

### 4. `neo_testing.py`
**Purpose:**  
Provides a playground for neural data exploration, filtering, spike detection, and trial alignment.

**Main Details:**
- Loads neural and behavioral data.
- Implements bandpass and notch filtering, LFP extraction.
- Performs spike detection via threshold crossing.
- Computes and plots power spectra for multiple channels.
- Aligns behavioral events to neural timeline.
- Provides functions for trial-aligned feature extraction and visualization.
- Includes code for chunked data loading and concatenation.
- Offers tools for analyzing target transitions and running a full analysis pipeline.

---

### 5. `analysis.py`
**Purpose:**  
Core analysis utilities and pipeline functions for neural and behavioral data.

**Main Details:**
- Feature extraction: spike band power, LFP bands, spike detection.
- Data loading: chunked neural data loading from binary files.
- Behavioral trial alignment and event extraction.
- Visualization: neural overview, trial-aligned activity, population dynamics (PCA), Utah array spatial maps.
- Decoding: movement direction classification (LDA), cross-validation.
- Modular functions for use in other scripts (e.g., `offline_decode.py`, `neo_testing.py`).

---

### 6. `neo_helpers.py`
**Purpose:**  
Helper functions for loading neural data by time or sample index, used throughout the codebase.

---

### 7. `convert.py`
**Purpose:**  
Utility for data conversion (details depend on code, likely for converting between formats or preprocessing).

---

### 8. `nsx_parser.py`
**Purpose:**  
Parser for Blackrock NSx files, likely for low-level data access or custom extraction.

---