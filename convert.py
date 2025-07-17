import numpy as np
from nsx_parser import NSxParser
import struct 

filepath = "data/neural.ns6"
parser = NSxParser(filepath)
parser.parse_header()
parser.try_parse_extended_headers()

# Estimate total samples (optional, for preallocating the memmap)
# You may need to parse all data packet headers to get the exact number
# For now, let's just process in chunks

chunk_size = 30000 * 10  # 10 seconds at 30kHz
output_path = "data/neural_raw_voltage.npy"

# Open file for reading
with open(filepath, 'rb') as f:
    f.seek(parser.data_start_byte)
    first = True
    total_samples = 0
    while True:
        header_byte = f.read(1)
        if not header_byte or header_byte[0] != 0x01:
            break
        timestamp = struct.unpack('<Q', f.read(8))[0]
        num_points = struct.unpack('<I', f.read(4))[0]
        data_bytes = f.read(num_points * parser.header['channel_count'] * 2)
        if len(data_bytes) < num_points * parser.header['channel_count'] * 2:
            break
        data = struct.unpack(f'<{num_points * parser.header["channel_count"]}h', data_bytes)
        data_array = np.array(data).reshape(num_points, parser.header['channel_count'])
        analog_chunk = parser.convert_to_analog(data_array)
        if first:
            # Create memmap file with unknown total length, so use a list and save at the end
            all_chunks = [analog_chunk]
            first = False
        else:
            all_chunks.append(analog_chunk)
        total_samples += num_points

# Concatenate and save
all_data = np.concatenate(all_chunks, axis=0)
np.save(output_path, all_data)
print(f"Saved {all_data.shape} samples to {output_path}")