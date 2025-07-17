import struct
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class NSxParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = {}
        self.channels = []
        self.data_start_byte = 0
        
    def parse_header(self) -> Dict[str, Any]:
        """Parse the basic header section"""
        with open(self.filepath, 'rb') as f:
            # Basic header
            file_type = f.read(8).decode('ascii').rstrip('\x00')
            
            spec_major, spec_minor = struct.unpack('<BB', f.read(2))
            header_bytes = struct.unpack('<I', f.read(4))[0]
            
            label = f.read(16).decode('ascii').rstrip('\x00')
            comment = f.read(256).decode('ascii').rstrip('\x00')
            
            period = struct.unpack('<I', f.read(4))[0]
            timestamp_res = struct.unpack('<I', f.read(4))[0]
            
            # UTC time origin - THIS IS KEY FOR ALIGNMENT
            time_vals = struct.unpack('<8H', f.read(16))
            time_origin = datetime(time_vals[0], time_vals[1], time_vals[3], 
                                 time_vals[4], time_vals[5], time_vals[6], 
                                 time_vals[7] * 1000, tzinfo=timezone.utc)
            
            channel_count = struct.unpack('<I', f.read(4))[0]
            
            self.header = {
                'file_type': file_type,
                'spec_version': f"{spec_major}.{spec_minor}",
                'header_bytes': header_bytes,
                'label': label,
                'comment': comment,
                'sampling_rate': 30000 / period if period > 0 else 30000,
                'period': period,
                'timestamp_resolution': timestamp_res,
                'time_origin': time_origin,
                'channel_count': channel_count
            }
            
            self.data_start_byte = header_bytes
            return self.header
    
    def try_parse_extended_headers(self) -> List[Dict[str, Any]]:
        """Try to parse extended headers, fall back if not present"""
        if not self.header:
            self.parse_header()
            
        with open(self.filepath, 'rb') as f:
            f.seek(316)  # Standard position after basic header
            
            # Check if extended headers are present
            raw = f.read(2)
            if raw == b'CC':
                print("Extended headers found, parsing...")
                f.seek(316)  # Reset position
                return self._parse_extended_headers_full(f)
            else:
                print("No extended headers found, creating default channel info")
                # Create default channel info for 96-channel Utah array
                channels = []
                for i in range(self.header['channel_count']):
                    channels.append({
                        'electrode_id': i + 1,
                        'label': f'elec{i+1:03d}',
                        'connector': (i // 32) + 1,  # Bank A,B,C
                        'pin': (i % 32) + 1,
                        'conversion_factor': 0.25,  # Default µV per bit
                        'units': 'µV'
                    })
                self.channels = channels
                # Data starts right after basic header
                self.data_start_byte = self.header['header_bytes']
                return channels
    
    def _parse_extended_headers_full(self, f) -> List[Dict[str, Any]]:
        """Parse full extended headers when present"""
        channels = []
        for _ in range(self.header['channel_count']):
            ch_type = f.read(2).decode('ascii').rstrip('\x00')
            electrode_id = struct.unpack('<H', f.read(2))[0]
            label = f.read(16).decode('ascii', errors='replace').rstrip('\x00')
            
            connector = struct.unpack('<B', f.read(1))[0]
            pin = struct.unpack('<B', f.read(1))[0]
            
            min_digital = struct.unpack('<h', f.read(2))[0]
            max_digital = struct.unpack('<h', f.read(2))[0]
            min_analog = struct.unpack('<h', f.read(2))[0]
            max_analog = struct.unpack('<h', f.read(2))[0]
            
            units = f.read(16).decode('ascii', errors='replace').rstrip('\x00')
            
            # Skip filter info for now
            f.read(16)
            
            # Calculate conversion factor
            digital_range = max_digital - min_digital
            analog_range = max_analog - min_analog
            conversion_factor = analog_range / digital_range if digital_range != 0 else 0.25
            
            channels.append({
                'electrode_id': electrode_id,
                'label': label,
                'connector': connector,
                'pin': pin,
                'conversion_factor': conversion_factor,
                'units': units
            })
        
        self.channels = channels
        return channels

    def parse_extended_headers(self) -> List[Dict[str, Any]]:
        """Parse channel configuration headers"""
        if not self.header:
            self.parse_header()
            
        with open(self.filepath, 'rb') as f:
            # Use the actual header size from the file, not hardcoded 316
            basic_header_end = 316  # This is where extended headers start
            f.seek(basic_header_end)
            
            print(f"Starting extended headers at byte {basic_header_end}")
            print(f"Expected {self.header['channel_count']} channels")
            
            channels = []
            for ch_idx in range(self.header['channel_count']):
                current_pos = f.tell()
                print(f"Channel {ch_idx} starting at byte {current_pos}")
                
                # Read and check the header type
                raw = f.read(2)
                print(f"Raw bytes for ch_type: {raw}")
                
                try:
                    ch_type = raw.decode('ascii').rstrip('\x00')
                except UnicodeDecodeError:
                    print(f"Failed to decode header type at position {current_pos}: {raw}")
                    # Try to find the next "CC" pattern
                    f.seek(current_pos)
                    data_chunk = f.read(100)  # Read ahead to see what's there
                    print(f"Next 100 bytes: {data_chunk}")
                    raise ValueError(f"Cannot decode extended header {ch_idx} at position {current_pos}")
                
                if ch_type != 'CC':
                    print(f"Expected 'CC', got '{ch_type}' at position {current_pos}")
                    # This might not be an extended header section
                    # Check if we've hit the data section instead
                    f.seek(current_pos)
                    next_bytes = f.read(10)
                    print(f"Next 10 bytes: {next_bytes}")
                    if next_bytes[0:1] == b'\x01':  # Data packet marker
                        print("Hit data section - no extended headers present")
                        self.data_start_byte = current_pos
                        break
                    raise ValueError(f"Expected CC header, got {ch_type} at position {current_pos}")
                
                electrode_id = struct.unpack('<H', f.read(2))[0]
                label = f.read(16).decode('ascii', errors='replace').rstrip('\x00')
                
                connector = struct.unpack('<B', f.read(1))[0]
                pin = struct.unpack('<B', f.read(1))[0]
                
                min_digital = struct.unpack('<h', f.read(2))[0]
                max_digital = struct.unpack('<h', f.read(2))[0]
                min_analog = struct.unpack('<h', f.read(2))[0]
                max_analog = struct.unpack('<h', f.read(2))[0]
                
                units = f.read(16).decode('ascii', errors='replace').rstrip('\x00')
                
                # Filter info
                high_freq = struct.unpack('<I', f.read(4))[0]
                high_order = struct.unpack('<I', f.read(4))[0]
                high_type = struct.unpack('<H', f.read(2))[0]
                
                low_freq = struct.unpack('<I', f.read(4))[0]
                low_order = struct.unpack('<I', f.read(4))[0]
                low_type = struct.unpack('<H', f.read(2))[0]
                
                # Calculate conversion factor
                digital_range = max_digital - min_digital
                analog_range = max_analog - min_analog
                conversion_factor = analog_range / digital_range if digital_range != 0 else 1
                
                channel = {
                    'electrode_id': electrode_id,
                    'label': label,
                    'connector': connector,
                    'pin': pin,
                    'min_digital': min_digital,
                    'max_digital': max_digital,
                    'min_analog': min_analog,
                    'max_analog': max_analog,
                    'units': units,
                    'conversion_factor': conversion_factor,
                    'high_freq_hz': high_freq / 1000,  # Convert mHz to Hz
                    'low_freq_hz': low_freq / 1000,
                    'high_filter_order': high_order,
                    'low_filter_order': low_order
                }
                channels.append(channel)
                print(f"Parsed channel {ch_idx}: {channel['label']} (ID: {channel['electrode_id']})")
            
            self.channels = channels
            return channels
    
    def read_data_packets(self, max_packets: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Read data packets and return timestamps and data arrays"""
        if not self.channels:
            self.parse_extended_headers()
        
        timestamps = []
        data_blocks = []
        
        with open(self.filepath, 'rb') as f:
            f.seek(self.data_start_byte)
            
            packet_count = 0
            while True:
                if max_packets and packet_count >= max_packets:
                    break
                
                # Check for packet header
                header_byte = f.read(1)
                if not header_byte or header_byte[0] != 0x01:
                    break
                
                # Read timestamp and data point count
                timestamp = struct.unpack('<Q', f.read(8))[0]
                num_points = struct.unpack('<I', f.read(4))[0]
                
                # Read data (2 bytes per channel per time point)
                data_bytes = f.read(num_points * self.header['channel_count'] * 2)
                if len(data_bytes) < num_points * self.header['channel_count'] * 2:
                    break
                
                # Unpack as signed 16-bit integers
                data = struct.unpack(f'<{num_points * self.header["channel_count"]}h', data_bytes)
                
                # Reshape to [time_points, channels]
                data_array = np.array(data).reshape(num_points, self.header['channel_count'])
                
                timestamps.append(timestamp)
                data_blocks.append(data_array)
                packet_count += 1
        
        if not data_blocks:
            return np.array([]), np.array([])
        
        # Concatenate all data blocks
        all_data = np.vstack(data_blocks)
        all_timestamps = np.array(timestamps)
        
        return all_timestamps, all_data
    
    def convert_to_analog(self, digital_data: np.ndarray) -> np.ndarray:
        """Convert digital values to analog using channel calibration"""
        analog_data = digital_data.copy().astype(np.float32)
        
        for i, channel in enumerate(self.channels):
            min_digital = channel.get('min_digital', 0)
            min_analog = channel.get('min_analog', 0)
            analog_data[:, i] = (digital_data[:, i] - min_digital) * channel['conversion_factor'] + min_analog
            # Apply offset and scaling
            # analog_data[:, i] = (digital_data[:, i] - channel['min_digital']) * channel['conversion_factor'] + channel['min_analog']
        
        return analog_data
    
    def get_channel_info(self) -> str:
        """Get human-readable channel information"""
        if not self.channels:
            self.parse_extended_headers()
        
        info = f"NSx File: {self.header['label']}\n"
        info += f"Sampling Rate: {self.header['sampling_rate']:.1f} Hz\n"
        info += f"Channels: {self.header['channel_count']}\n"
        info += f"Time Origin: {self.header['time_origin']}\n\n"
        
        for i, ch in enumerate(self.channels):
            info += f"Ch{i:2d}: {ch['label']:12s} (ID:{ch['electrode_id']:4d}) "
            info += f"Connector {ch['connector']}-{ch['pin']:2d} "
            info += f"Range: {ch['min_analog']:+5d} to {ch['max_analog']:+5d} {ch['units']}\n"
        
        return info


    def debug_file_structure(self):
        """Debug function to inspect file structure"""
        with open(self.filepath, 'rb') as f:
            # Check basic header
            f.seek(0)
            file_type = f.read(8)
            print(f"File type: {file_type}")
            
            f.seek(10)
            header_bytes = struct.unpack('<I', f.read(4))[0]
            print(f"Header bytes: {header_bytes}")
            
            f.seek(310)
            channel_count = struct.unpack('<I', f.read(4))[0]
            print(f"Channel count: {channel_count}")
            
            # Check what's at the expected extended header location
            print(f"\nInspecting bytes at position 316:")
            f.seek(316)
            next_100 = f.read(100)
            print(f"Next 100 bytes: {next_100}")
            
            # Look for data section markers
            print(f"\nLooking for data section (0x01 markers):")
            f.seek(0)
            data = f.read()
            data_markers = []
            for i in range(len(data) - 1):
                if data[i] == 0x01:
                    # Check if this looks like a data packet header
                    if i + 13 < len(data):
                        timestamp = struct.unpack('<Q', data[i+1:i+9])[0]
                        num_points = struct.unpack('<I', data[i+9:i+13])[0]
                        data_markers.append((i, timestamp, num_points))
            
            print(f"Found {len(data_markers)} potential data packets:")
            for pos, ts, npts in data_markers[:5]:  # Show first 5
                print(f"  Position {pos}: timestamp={ts}, num_points={npts}")


class MotorCortexAnalyzer:
    def __init__(self, nsx_path: str, behavioral_path: str):
        self.nsx_path = nsx_path
        self.behavioral_path = behavioral_path
        self.neural_parser = None
        self.behavioral_data = None
        
    def load_data(self):
        """Load and align neural and behavioral data"""
        print("Loading neural data...")
        self.neural_parser = NSxParser(self.nsx_path)
        header = self.neural_parser.parse_header()
        channels = self.neural_parser.try_parse_extended_headers()
        
        print(f"Neural data: {header['channel_count']} channels at {header['sampling_rate']} Hz")
        print(f"Neural time origin: {header['time_origin']}")
        
        print("Loading behavioral data...")
        self.behavioral_data = pd.read_csv(self.behavioral_path)
        
        # Convert behavioral timestamps to match neural timebase
        # Behavioral starts at 2025-03-25T9:22:28Z
        # Neural starts at 2025-03-25T9:22:53Z (25 seconds later)
        behavioral_start = datetime(2025, 3, 25, 9, 22, 28, tzinfo=timezone.utc)
        neural_start = header['time_origin']
        
        time_offset_seconds = (neural_start - behavioral_start).total_seconds()
        print(f"Time offset: {time_offset_seconds} seconds")
        
        # Convert behavioral timestamps to neural timestamp format
        self.behavioral_data['neural_timestamp'] = (
            self.behavioral_data['timestamp'] + time_offset_seconds * header['timestamp_resolution']
        )
        
        return header, channels
    
    def extract_trials(self):
        """Extract trial information from behavioral data"""
        # Find trial starts
        trial_starts = self.behavioral_data[self.behavioral_data['trial_start'] == True].copy()
        trial_wins = self.behavioral_data[self.behavioral_data['trial_win'] == True].copy()
        
        print(f"Found {len(trial_starts)} trial starts and {len(trial_wins)} successful trials")
        
        # Match wins to starts
        trials = []
        for _, start_row in trial_starts.iterrows():
            start_time = start_row['neural_timestamp']
            target = start_row['target_index']
            
            # Find corresponding win within reasonable time window
            win_candidates = trial_wins[
                (trial_wins['neural_timestamp'] > start_time) & 
                (trial_wins['neural_timestamp'] < start_time + 90000)  # 3 seconds max
            ]
            
            if len(win_candidates) > 0:
                win_time = win_candidates.iloc[0]['neural_timestamp']
                trials.append({
                    'start_time': start_time,
                    'win_time': win_time,
                    'target_index': target,
                    'duration': win_time - start_time
                })
        
        self.trials = pd.DataFrame(trials)
        print(f"Matched {len(self.trials)} complete trials")
        return self.trials
    
    def extract_spike_band_power(self, window_size=1000):
        """Extract spike band power (200-400 Hz) features"""
        print("Extracting spike band power...")
        
        # Read neural data around trials
        with open(self.nsx_path, 'rb') as f:
            f.seek(self.neural_parser.data_start_byte)
            
            # Read first data packet to understand structure
            header_byte = f.read(1)[0]
            if header_byte != 0x01:
                raise ValueError("Expected data packet header")
                
            timestamp = struct.unpack('<Q', f.read(8))[0]
            num_points = struct.unpack('<I', f.read(4))[0]
            
            print(f"First packet: timestamp={timestamp}, points={num_points}")
            
            # For now, read a chunk of data
            data_bytes = f.read(num_points * self.neural_parser.header['channel_count'] * 2)
            data = struct.unpack(f'<{len(data_bytes)//2}h', data_bytes)
            
            # Reshape to [time, channels]
            neural_data = np.array(data).reshape(num_points, self.neural_parser.header['channel_count'])
            
            print(f"Loaded neural data shape: {neural_data.shape}")
            
            # Convert to microvolts
            for i, ch in enumerate(self.neural_parser.channels):
                neural_data[:, i] = neural_data[:, i] * ch['conversion_factor']
            
            return neural_data[:10000, :]  # Return first 10k samples for now


# Usage example
def analyze_nsx_file(filepath: str):
    """Complete analysis of an NSx file"""
    parser = NSxParser(filepath)
    
    # Debug first
    print("=== DEBUG FILE STRUCTURE ===")
    parser.debug_file_structure()
    print("\n=== PARSING HEADERS ===")
    
    # Parse headers
    header = parser.parse_header()
    print(f"Header parsed successfully: {header}")
    
    try:
        channels = parser.parse_extended_headers()
        print(f"Extended headers parsed: {len(channels)} channels")
    except Exception as e:
        print(f"Extended header parsing failed: {e}")
        print("Trying to find data section directly...")
        
        # If extended headers fail, try to find data section
        with open(filepath, 'rb') as f:
            f.seek(316)  # Start after basic header
            while True:
                pos = f.tell()
                byte = f.read(1)
                if not byte:
                    break
                if byte[0] == 0x01:  # Found data packet
                    print(f"Found data section at byte {pos}")
                    parser.data_start_byte = pos
                    break
        return parser, None, None