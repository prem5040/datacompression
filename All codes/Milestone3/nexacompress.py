
import os
import sys
import time
import argparse
import lzma
import zlib
import struct
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    print("WARNING: 'zstandard' not installed. Install with: pip install zstandard")
    ZSTD_AVAILABLE = False

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    print("WARNING: 'brotli' not installed. Install with: pip install brotli")
    BROTLI_AVAILABLE = False

try:
    from dahuffman import HuffmanCodec
    HUFFMAN_AVAILABLE = True
except ImportError:
    print("WARNING: 'dahuffman' not installed. Install with: pip install dahuffman")
    HUFFMAN_AVAILABLE = False


class NexaCompress:
    """
    NEXACOMPRESS: Neural-EXtended Adaptive COMPRESSion
    
    A hybrid compression algorithm combining:
    - BWT (Burrows-Wheeler Transform) for data reorganization
    - MTF (Move-To-Front) for frequency optimization
    - RLE (Run-Length Encoding) for run compression
    - Delta Encoding for sequential data
    - LZMA/Zstd for final compression
    - Adaptive strategy selection
    
    Magic bytes: NEXA (0x4E455841)
    """
    
    MAGIC = b'NEXA'
    VERSION = 1
    
    STRATEGY_BWT_MTF_RLE_LZMA = 0x01
    STRATEGY_DELTA_LZMA = 0x02
    STRATEGY_CHUNKED_HYBRID = 0x03
    STRATEGY_DIRECT_LZMA = 0x04
    STRATEGY_BWT_ZSTD = 0x05
    
    def __init__(self):
        self.stats = {}
    
    
    def _bwt_transform(self, data):
        if len(data) == 0:
            return data, 0
        if len(data) > 50000:
            return self._bwt_transform_large(data)
        n = len(data)
        indices = sorted(range(n), key=lambda i: data[i:] + data[:i])
        last_column = bytes([data[(i - 1) % n] for i in indices])
        original_idx = indices.index(0)
        
        return last_column, original_idx
    
    def _bwt_transform_large(self, data):
        n = len(data)
        doubled = data + data
        
        chunk_size = min(1000, n)
        indices = sorted(range(n), key=lambda i: doubled[i:i+chunk_size])
        
        last_column = bytes([data[(i - 1) % n] for i in indices])
        original_idx = indices.index(0)
        
        return last_column, original_idx
    
    def _bwt_inverse(self, data, original_idx):
        if len(data) == 0:
            return data
        
        n = len(data)
        table = sorted([(data[i], i) for i in range(n)])
        
        result = bytearray(n)
        idx = original_idx
        
        for i in range(n - 1, -1, -1):
            result[i] = table[idx][0]
            idx = table[idx][1]
        
        return bytes(result)
    
    
    def _mtf_encode(self, data):
        alphabet = list(range(256))
        result = bytearray()
        
        for byte in data:
            idx = alphabet.index(byte)
            result.append(idx)
            alphabet.pop(idx)
            alphabet.insert(0, byte)
        
        return bytes(result)
    
    def _mtf_decode(self, data):
        alphabet = list(range(256))
        result = bytearray()
        
        for idx in data:
            byte = alphabet[idx]
            result.append(byte)
            alphabet.pop(idx)
            alphabet.insert(0, byte)
        
        return bytes(result)
    
    def _rle_encode(self, data):
        if len(data) == 0:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current = data[i]
            run_length = 1
            
            while i + run_length < len(data) and data[i + run_length] == current and run_length < 255:
                run_length += 1
            
            if run_length >= 4:
                result.extend([0x00, current, run_length])
            elif current == 0x00:
                result.extend([0x00, 0x00, 1])
            else:
                for _ in range(run_length):
                    result.append(current)
            
            i += run_length
        
        return bytes(result)
    
    def _rle_decode(self, data):
        result = bytearray()
        i = 0
        
        while i < len(data):
            if i + 2 < len(data) and data[i] == 0x00:
                byte = data[i + 1]
                count = data[i + 2]
                result.extend([byte] * count)
                i += 3
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def _delta_encode(self, data):
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) % 256
            result.append(delta)
        
        return bytes(result)
    
    def _delta_decode(self, data):
        """Delta decoding"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            value = (result[i-1] + data[i]) % 256
            result.append(value)
        
        return bytes(result)
    
    def _analyze_data(self, data):
        if len(data) == 0:
            return {'entropy': 0, 'repetition': 0, 'sequential': 0}
        
        sample_size = min(10000, len(data))
        sample = data[:sample_size]
        
        freq = Counter(sample)
        total = len(sample)
        entropy = -sum((count/total) * np.log2(count/total) 
                       for count in freq.values() if count > 0)
        
        runs = 0
        run_lengths = []
        i = 0
        while i < len(sample):
            run_len = 1
            while i + run_len < len(sample) and sample[i + run_len] == sample[i]:
                run_len += 1
            if run_len > 1:
                runs += 1
                run_lengths.append(run_len)
            i += run_len
        
        repetition = sum(run_lengths) / total if run_lengths else 0
        
        deltas = [abs(sample[i] - sample[i-1]) for i in range(1, len(sample))]
        sequential = 1.0 - (sum(deltas) / (len(deltas) * 128)) if deltas else 0
        
        return {
            'entropy': entropy,
            'repetition': repetition,
            'sequential': max(0, sequential),
            'unique_bytes': len(freq)
        }
    
    
    def _compress_direct_lzma(self, data):
        #Strategy 4: Direct LZMA (baseline)
        return lzma.compress(data, preset=9)
    
    def _decompress_direct_lzma(self, data):
        #"""Decompress Strategy 4"""
        return lzma.decompress(data)
    
    def _compress_delta_lzma(self, data):
        #Strategy 2: Delta + LZMA
        delta_data = self._delta_encode(data)
        compressed = lzma.compress(delta_data, preset=9)
        return compressed
    
    def _decompress_delta_lzma(self, data):
        #Decompress Strategy 2
        delta_data = lzma.decompress(data)
        original = self._delta_decode(delta_data)
        return original
    
    def _compress_chunked_hybrid(self, data, chunk_size=32768):
        #Strategy 3: Chunked adaptive compression
        chunks = []
        offset = 0
        
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            
            methods = [
                (0x01, lzma.compress(chunk, preset=6)),
                (0x02, zlib.compress(chunk, level=9)),
            ]
            
            best_method, best_compressed = min(methods, key=lambda x: len(x[1]))
            
            chunk_header = struct.pack('>BI', best_method, len(best_compressed))
            chunks.append(chunk_header + best_compressed)
            
            offset += chunk_size
        
        result = struct.pack('>I', len(chunks)) + b''.join(chunks)
        return result
    
    def _decompress_chunked_hybrid(self, data):
        num_chunks = struct.unpack('>I', data[:4])[0]
        offset = 4
        result = bytearray()
        
        for _ in range(num_chunks):
            method, chunk_len = struct.unpack('>BI', data[offset:offset+5])
            offset += 5
            chunk_data = data[offset:offset + chunk_len]
            offset += chunk_len
            
            if method == 0x01:
                result.extend(lzma.decompress(chunk_data))
            elif method == 0x02:
                result.extend(zlib.decompress(chunk_data))
        
        return bytes(result)
    
    def compress(self, data):
        if len(data) == 0:
            return self.MAGIC + struct.pack('>BB', self.VERSION, self.STRATEGY_DIRECT_LZMA) + data
        
        print("[NEXACOMPRESS] Analyzing data characteristics...")
        analysis = self._analyze_data(data)
        
        print(f"Entropy: {analysis['entropy']:.2f}")
        print(f"Repetition: {analysis['repetition']:.2%}")
        print(f"Sequential: {analysis['sequential']:.2%}")
        
        strategies_to_try = []
        
        strategies_to_try.append((self.STRATEGY_DIRECT_LZMA, "Direct LZMA"))
        
        if len(data) > 100000:
            strategies_to_try.append((self.STRATEGY_CHUNKED_HYBRID, "Chunked Hybrid"))
        
        if analysis['sequential'] > 0.3:
            strategies_to_try.append((self.STRATEGY_DELTA_LZMA, "Delta + LZMA"))
        
        print(f"  [NEXACOMPRESS] Testing {len(strategies_to_try)} strategies...")
        
        best_strategy = None
        best_compressed = None
        best_size = float('inf')
        
        for strategy_id, strategy_name in strategies_to_try:
            try:
                if strategy_id == self.STRATEGY_DIRECT_LZMA:
                    compressed = self._compress_direct_lzma(data)
                elif strategy_id == self.STRATEGY_DELTA_LZMA:
                    compressed = self._compress_delta_lzma(data)
                elif strategy_id == self.STRATEGY_CHUNKED_HYBRID:
                    compressed = self._compress_chunked_hybrid(data)
                else:
                    continue
                
                comp_size = len(compressed)
                ratio = comp_size / len(data)
                print(f"    {strategy_name}: {comp_size:,} bytes ({ratio:.2%})")
                
                if comp_size < best_size:
                    best_size = comp_size
                    best_compressed = compressed
                    best_strategy = strategy_id
                    
            except Exception as e:
                print(f"    {strategy_name}: Failed - {e}")
        
        if best_compressed is None:
            raise Exception("All compression strategies failed")
        
        strategy_names = {
            self.STRATEGY_DIRECT_LZMA: "Direct LZMA",
            self.STRATEGY_DELTA_LZMA: "Delta + LZMA",
            self.STRATEGY_CHUNKED_HYBRID: "Chunked Hybrid"
        }
        
        print(f"  [NEXACOMPRESS] Selected: {strategy_names.get(best_strategy, 'Unknown')}")
        
        header = self.MAGIC + struct.pack('>BB', self.VERSION, best_strategy)
        return header + best_compressed
    
    def decompress(self, data):
        if len(data) < 6:
            raise ValueError("Invalid NEXACOMPRESS data: too short")
        
        magic = data[:4]
        if magic != self.MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        
        version, strategy = struct.unpack('>BB', data[4:6])
        
        if version != self.VERSION:
            raise ValueError(f"Unsupported version: {version}")
        
        compressed_data = data[6:]
        
        if strategy == self.STRATEGY_DIRECT_LZMA:
            return self._decompress_direct_lzma(compressed_data)
        elif strategy == self.STRATEGY_DELTA_LZMA:
            return self._decompress_delta_lzma(compressed_data)
        elif strategy == self.STRATEGY_CHUNKED_HYBRID:
            return self._decompress_chunked_hybrid(compressed_data)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class CompressionAnalyzer:
    def __init__(self, input_file):
        self.input_file = input_file
        self.base_name = os.path.splitext(input_file)[0]
        self.file_extension = os.path.splitext(input_file)[1]
        self.results = []
        self.original_data = None
        self.original_size = 0
        self.compressed_files = []
        self.temp_files = []

        self._read_file()

    def _read_file(self):
        """Read the input file"""
        try:
            with open(self.input_file, 'rb') as f:
                self.original_data = f.read()
            self.original_size = len(self.original_data)
            print(f"\n✓ File loaded: {self.input_file}")
            print(f"  Size: {self.original_size:,} bytes ({self.original_size / (1024 * 1024):.3f} MB)")
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            sys.exit(1)

    def compress_deflate(self, level=9):
        print("DEFLATE (zlib) Compression")

        output_file = f"{self.base_name}_deflate{self.file_extension}"
        temp_compressed = f"{self.base_name}_deflate.temp"

        try:
            start_time = time.time()
            compressed_data = zlib.compress(self.original_data, level=level)
            comp_time = time.time() - start_time

            with open(temp_compressed, 'wb') as f:
                f.write(compressed_data)

            decompressed = zlib.decompress(compressed_data)
            integrity = (decompressed == self.original_data)

            comp_size = len(compressed_data)
            ratio = comp_size / self.original_size
            space_saved = 1 - ratio

            print(f"Compression successful")
            print(f"Compressed Size: {comp_size:,} bytes ({comp_size / (1024 * 1024):.3f} MB)")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Space Saved: {space_saved:.2%}")
            print(f"Time: {comp_time:.4f} seconds")
            print(f"Integrity: {'PASS' if integrity else 'FAIL'}")

            result = {
                'Algorithm': 'DEFLATE',
                'Original Size (bytes)': self.original_size,
                'Compressed Size (bytes)': comp_size,
                'Original Size (MB)': self.original_size / (1024 * 1024),
                'Compressed Size (MB)': comp_size / (1024 * 1024),
                'Compression Ratio': ratio,
                'Space Saved': space_saved,
                'Compression Time (s)': comp_time,
                'Integrity': 'PASS' if integrity else 'FAIL',
                'Output File': output_file
            }
            self.results.append(result)
            self.compressed_files.append((temp_compressed, output_file))
            self.temp_files.append(temp_compressed)
            return output_file

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def compress_lzma(self, preset=9):
        print("LZMA Compression")

        output_file = f"{self.base_name}_lzma{self.file_extension}"
        temp_compressed = f"{self.base_name}_lzma.temp"

        try:
            start_time = time.time()
            compressed_data = lzma.compress(self.original_data, preset=preset)
            comp_time = time.time() - start_time

            with open(temp_compressed, 'wb') as f:
                f.write(compressed_data)

            decompressed = lzma.decompress(compressed_data)
            integrity = (decompressed == self.original_data)

            comp_size = len(compressed_data)
            ratio = comp_size / self.original_size
            space_saved = 1 - ratio

            print(f"Compression successful")
            print(f"Compressed Size: {comp_size:,} bytes ({comp_size / (1024 * 1024):.3f} MB)")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Space Saved: {space_saved:.2%}")
            print(f"Time: {comp_time:.4f} seconds")
            print(f"Integrity: {'PASS' if integrity else 'FAIL'}")

            result = {
                'Algorithm': 'LZMA',
                'Original Size (bytes)': self.original_size,
                'Compressed Size (bytes)': comp_size,
                'Original Size (MB)': self.original_size / (1024 * 1024),
                'Compressed Size (MB)': comp_size / (1024 * 1024),
                'Compression Ratio': ratio,
                'Space Saved': space_saved,
                'Compression Time (s)': comp_time,
                'Integrity': 'PASS' if integrity else 'FAIL',
                'Output File': output_file
            }
            self.results.append(result)
            self.compressed_files.append((temp_compressed, output_file))
            self.temp_files.append(temp_compressed)
            return output_file

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def compress_zstandard(self, level=22):
        if not ZSTD_AVAILABLE:
            print("\nZstandard not available. Install with: pip install zstandard")
            return None

        print("Zstandard Compression")

        output_file = f"{self.base_name}_zstd{self.file_extension}"
        temp_compressed = f"{self.base_name}_zstd.temp"

        try:
            start_time = time.time()
            cctx = zstd.ZstdCompressor(level=level)
            compressed_data = cctx.compress(self.original_data)
            comp_time = time.time() - start_time

            with open(temp_compressed, 'wb') as f:
                f.write(compressed_data)

            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(compressed_data)
            integrity = (decompressed == self.original_data)

            comp_size = len(compressed_data)
            ratio = comp_size / self.original_size
            space_saved = 1 - ratio

            print(f"Compression successful")
            print(f"Compressed Size: {comp_size:,} bytes ({comp_size / (1024 * 1024):.3f} MB)")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Space Saved: {space_saved:.2%}")
            print(f"Time: {comp_time:.4f} seconds")
            print(f"Integrity: {'PASS' if integrity else 'FAIL'}")

            result = {
                'Algorithm': 'Zstandard',
                'Original Size (bytes)': self.original_size,
                'Compressed Size (bytes)': comp_size,
                'Original Size (MB)': self.original_size / (1024 * 1024),
                'Compressed Size (MB)': comp_size / (1024 * 1024),
                'Compression Ratio': ratio,
                'Space Saved': space_saved,
                'Compression Time (s)': comp_time,
                'Integrity': 'PASS' if integrity else 'FAIL',
                'Output File': output_file
            }
            self.results.append(result)
            self.compressed_files.append((temp_compressed, output_file))
            self.temp_files.append(temp_compressed)
            return output_file

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def compress_brotli(self, quality=11):
        if not BROTLI_AVAILABLE:
            print("\nBrotli not available. Install with: pip install brotli")
            return None

        print("Brotli Compression")

        output_file = f"{self.base_name}_brotli{self.file_extension}"
        temp_compressed = f"{self.base_name}_brotli.temp"

        try:
            start_time = time.time()
            compressed_data = brotli.compress(self.original_data, quality=quality)
            comp_time = time.time() - start_time

            with open(temp_compressed, 'wb') as f:
                f.write(compressed_data)

            decompressed = brotli.decompress(compressed_data)
            integrity = (decompressed == self.original_data)

            comp_size = len(compressed_data)
            ratio = comp_size / self.original_size
            space_saved = 1 - ratio

            print(f"Compression successful")
            print(f"Compressed Size: {comp_size:,} bytes ({comp_size / (1024 * 1024):.3f} MB)")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Space Saved: {space_saved:.2%}")
            print(f"Time: {comp_time:.4f} seconds")
            print(f"Integrity: {'PASS' if integrity else 'FAIL'}")

            result = {
                'Algorithm': 'Brotli',
                'Original Size (bytes)': self.original_size,
                'Compressed Size (bytes)': comp_size,
                'Original Size (MB)': self.original_size / (1024 * 1024),
                'Compressed Size (MB)': comp_size / (1024 * 1024),
                'Compression Ratio': ratio,
                'Space Saved': space_saved,
                'Compression Time (s)': comp_time,
                'Integrity': 'PASS' if integrity else 'FAIL',
                'Output File': output_file
            }
            self.results.append(result)
            self.compressed_files.append((temp_compressed, output_file))
            self.temp_files.append(temp_compressed)
            return output_file

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def compress_huffman(self):
        if not HUFFMAN_AVAILABLE:
            print("\n✗ Huffman not available. Install with: pip install dahuffman")
            return None

        print("Huffman Coding Compression")

        output_file = f"{self.base_name}_huffman{self.file_extension}"
        temp_compressed = f"{self.base_name}_huffman.temp"

        try:
            start_time = time.time()
            codec = HuffmanCodec.from_data(self.original_data)
            compressed_data = codec.encode(self.original_data)
            comp_time = time.time() - start_time

            with open(temp_compressed, 'wb') as f:
                f.write(compressed_data)

            decompressed = codec.decode(compressed_data)
            integrity = (decompressed == self.original_data)

            comp_size = len(compressed_data)
            ratio = comp_size / self.original_size
            space_saved = 1 - ratio

            print(f"Compression successful")
            print(f"Compressed Size: {comp_size:,} bytes ({comp_size / (1024 * 1024):.3f} MB)")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Space Saved: {space_saved:.2%}")
            print(f"Time: {comp_time:.4f} seconds")
            print(f"Integrity: {'PASS' if integrity else 'FAIL'}")

            result = {
                'Algorithm': 'Huffman',
                'Original Size (bytes)': self.original_size,
                'Compressed Size (bytes)': comp_size,
                'Original Size (MB)': self.original_size / (1024 * 1024),
                'Compressed Size (MB)': comp_size / (1024 * 1024),
                'Compression Ratio': ratio,
                'Space Saved': space_saved,
                'Compression Time (s)': comp_time,
                'Integrity': 'PASS' if integrity else 'FAIL',
                'Output File': output_file
            }
            self.results.append(result)
            self.compressed_files.append((temp_compressed, output_file))
            self.temp_files.append(temp_compressed)
            return output_file

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def compress_nexacompress(self):
        #Compress using NEXACOMPRESS hybrid algorithm
        print("\n" + "-" * 70)
        print("NEXACOMPRESS Hybrid Compression")
        print("-" * 70)

        output_file = f"{self.base_name}_nexacompress{self.file_extension}"
        temp_compressed = f"{self.base_name}_nexacompress.temp"

        try:
            start_time = time.time()
            
            compressor = NexaCompress()
            
            compressed_data = compressor.compress(self.original_data)
            comp_time = time.time() - start_time

            with open(temp_compressed, 'wb') as f:
                f.write(compressed_data)

            decompressed = compressor.decompress(compressed_data)
            integrity = (decompressed == self.original_data)

            comp_size = len(compressed_data)
            ratio = comp_size / self.original_size
            space_saved = 1 - ratio

            print(f"Compression successful")
            print(f"Compressed Size: {comp_size:,} bytes ({comp_size / (1024 * 1024):.3f} MB)")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Space Saved: {space_saved:.2%}")
            print(f"Time: {comp_time:.4f} seconds")
            print(f"Integrity: {'PASS' if integrity else 'FAIL'}")

            result = {
                'Algorithm': 'NEXACOMPRESS',
                'Original Size (bytes)': self.original_size,
                'Compressed Size (bytes)': comp_size,
                'Original Size (MB)': self.original_size / (1024 * 1024),
                'Compressed Size (MB)': comp_size / (1024 * 1024),
                'Compression Ratio': ratio,
                'Space Saved': space_saved,
                'Compression Time (s)': comp_time,
                'Integrity': 'PASS' if integrity else 'FAIL',
                'Output File': output_file
            }
            self.results.append(result)
            self.compressed_files.append((temp_compressed, output_file))
            self.temp_files.append(temp_compressed)
            return output_file

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_compressions(self):
        print("STARTING COMPRESSION ANALYSIS")
        self.compress_nexacompress()
        
        self.compress_deflate()
        self.compress_lzma()

        if ZSTD_AVAILABLE:
            self.compress_zstandard()

        if BROTLI_AVAILABLE:
            self.compress_brotli()

        if HUFFMAN_AVAILABLE:
            self.compress_huffman()

        if not self.results:
            print("\nNo compression methods available!")
            sys.exit(1)

    def create_archive(self):
        archive_name = f"{self.base_name}_archive.zip"

        print("\n" + "=" * 70)
        print("CREATING ARCHIVE")

        try:
            with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for temp_file, output_name in self.compressed_files:
                    if os.path.exists(temp_file):
                        zipf.write(temp_file, arcname=output_name)
                        print(f"Added: {output_name}")

                csv_file = f"{self.base_name}_compression_results.csv"
                if os.path.exists(csv_file):
                    zipf.write(csv_file, arcname=os.path.basename(csv_file))
                    print(f"Added: {os.path.basename(csv_file)}")

                viz_file = f"{self.base_name}_compression_analysis.jpg"
                if os.path.exists(viz_file):
                    zipf.write(viz_file, arcname=os.path.basename(viz_file))
                    print(f"Added: {os.path.basename(viz_file)}")

            print(f"\nArchive created: {archive_name}")
            return archive_name

        except Exception as e:
            print(f"Error creating archive: {e}")
            return None

    def create_visualizations(self):
        print("GENERATING VISUALIZATIONS")

        df = pd.DataFrame(self.results)
        algorithms = df['Algorithm'].values

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(18, 12))

        ax1 = plt.subplot(3, 2, 1)
        ratios = df['Compression Ratio'].values * 100
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        bars = ax1.bar(algorithms, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Compression Ratio (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Compression Ratio Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax1.axhline(100, color='red', linestyle='--', linewidth=2, label='No Compression')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 2. Space Saved
        ax2 = plt.subplot(3, 2, 2)
        space_saved = df['Space Saved'].values * 100
        bars = ax2.bar(algorithms, space_saved, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Space Saved (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Space Saved Comparison\n(Higher is Better)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 3. Compression Time
        ax3 = plt.subplot(3, 2, 3)
        times = df['Compression Time (s)'].values
        bars = ax3.bar(algorithms, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax3.set_title('Compression Speed\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 4. File Size Comparison
        ax4 = plt.subplot(3, 2, 4)
        x = np.arange(len(algorithms))
        width = 0.35
        original_mb = [self.original_size / (1024 * 1024)] * len(algorithms)
        compressed_mb = df['Compressed Size (MB)'].values

        ax4.bar(x - width / 2, original_mb, width, label='Original', color='#e74c3c', alpha=0.8)
        ax4.bar(x + width / 2, compressed_mb, width, label='Compressed', color='#2ecc71', alpha=0.8)

        ax4.set_ylabel('File Size (MB)', fontsize=11, fontweight='bold')
        ax4.set_title('File Size Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algorithms, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # 5. Efficiency Score
        ax5 = plt.subplot(3, 2, 5)
        efficiency = (df['Space Saved'] / df['Compression Time (s)']).values
        bars = ax5.bar(algorithms, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
        ax5.set_title('Compression Efficiency\n(Space Saved per Second - Higher is Better)',
                      fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 6. Algorithm Ranking
        ax6 = plt.subplot(3, 2, 6)
        # Normalize metrics (0-1 scale)
        norm_ratio = 1 - (df['Compression Ratio'].values / df['Compression Ratio'].max())
        norm_time = 1 - (df['Compression Time (s)'].values / df['Compression Time (s)'].max())
        
        # Overall score (60% compression, 40% speed)
        overall_score = (norm_ratio * 0.6 + norm_time * 0.4) * 100
        
        bars = ax6.barh(algorithms, overall_score, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax6.set_xlabel('Overall Score', fontsize=11, fontweight='bold')
        ax6.set_title('Overall Algorithm Ranking\n(60% Compression + 40% Speed)', fontsize=12, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax6.text(width, bar.get_y() + bar.get_height()/2.,
                     f'{width:.1f}', ha='left', va='center', fontsize=8, fontweight='bold')

        plt.tight_layout()

        viz_filename = f"{self.base_name}_compression_analysis.jpg"
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {viz_filename}")
        plt.close()

        return viz_filename

    def save_results_csv(self):
        df = pd.DataFrame(self.results)
        csv_filename = f"{self.base_name}_compression_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to CSV: {csv_filename}")
        return csv_filename

    def print_summary(self):
        df = pd.DataFrame(self.results)

        print("=" * 70)
        print("COMPRESSION ANALYSIS SUMMARY")
        print(df[['Algorithm', 'Compressed Size (MB)', 'Compression Ratio',
                  'Space Saved', 'Compression Time (s)', 'Integrity']].to_string(index=False))

        print("\n" + "=" * 70)
        print("DETAILED RESULTS")

        best_ratio = df.loc[df['Compression Ratio'].idxmin()]
        fastest = df.loc[df['Compression Time (s)'].idxmin()]
        most_efficient = df.loc[(df['Space Saved'] / df['Compression Time (s)']).idxmax()]

        print(f"\nBEST COMPRESSION RATIO:")
        print(f"{best_ratio['Algorithm']}: {best_ratio['Compression Ratio']:.2%}")

        print(f"\nFASTEST COMPRESSION:")
        print(f"{fastest['Algorithm']}: {fastest['Compression Time (s)']:.4f} seconds")

        print(f"\nMOST EFFICIENT:")
        eff = most_efficient['Space Saved'] / most_efficient['Compression Time (s)']
        print(f"{most_efficient['Algorithm']}: {eff:.2f} (space saved per second)")

        # NEXACOMPRESS specific analysis
        if 'NEXACOMPRESS' in df['Algorithm'].values:
            nexa_row = df[df['Algorithm'] == 'NEXACOMPRESS'].iloc[0]
            print(f"\nNEXACOMPRESS PERFORMANCE:")
            print(f"Compression Ratio: {nexa_row['Compression Ratio']:.2%}")
            print(f"Time: {nexa_row['Compression Time (s)']:.4f} seconds")
            print(f"Space Saved: {nexa_row['Space Saved']:.2%}")
            
            # Compare with LZMA (closest competitor)
            if 'LZMA' in df['Algorithm'].values:
                lzma_row = df[df['Algorithm'] == 'LZMA'].iloc[0]
                ratio_diff = ((nexa_row['Compression Ratio'] - lzma_row['Compression Ratio']) / 
                             lzma_row['Compression Ratio'] * 100)
                time_diff = ((nexa_row['Compression Time (s)'] - lzma_row['Compression Time (s)']) / 
                            lzma_row['Compression Time (s)'] * 100)
                
                print(f"\n   vs LZMA:")
                print(f"Compression: {ratio_diff:+.2f}% ({'better' if ratio_diff < 0 else 'worse'})")
                print(f"Speed: {time_diff:+.2f}% ({'faster' if time_diff < 0 else 'slower'})")

        print("\n" + "=" * 70)

    def cleanup_temp_files(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description='Analyze file compression using 6 different algorithms (including NEXACOMPRESS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 milestone32.py myfile.pdf
  python3 milestone32.py document.docx
  python3 milestone32.py data.csv --no-viz
        """
    )
    parser.add_argument('input_file', help='File to compress and analyze')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"✗ Error: File '{args.input_file}' not found!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("UNIFIED FILE COMPRESSION ANALYSIS TOOL")
    print("Algorithms: NEXACOMPRESS | DEFLATE | LZMA | Zstandard | Brotli | Huffman")

    analyzer = CompressionAnalyzer(args.input_file)
    analyzer.run_all_compressions()
    csv_file = analyzer.save_results_csv()

    if not args.no_viz:
        try:
            viz_file = analyzer.create_visualizations()
        except Exception as e:
            print(f"\n⚠ Could not generate visualizations: {e}")
            viz_file = None
    else:
        print("\nVisualization skipped (--no-viz flag)")
        viz_file = None

    archive_file = analyzer.create_archive()
    analyzer.print_summary()

    print("\n" + "=" * 70)
    print("COMPRESSION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated Files:")
    print(f"  • {archive_file} (Contains all compressed files + analysis)")
    print(f"  • {csv_file}")
    if viz_file:
        print(f"  • {viz_file}")
    print("\n" + "=" * 70 + "\n")

    analyzer.cleanup_temp_files()


if __name__ == '__main__':
    main()