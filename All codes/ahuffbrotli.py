import sys
import os
import zlib
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import struct


try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    print("WARNING: 'brotli' library not found. Brotli compression will be skipped.")
    print("Install with: pip install brotli")
    BROTLI_AVAILABLE = False

file_name_list = []

class AdaptiveHuffman:
    def __init__(self):
        self.NYT = 512 
    
    def compress(self, data):
        """Compress data using Adaptive Huffman coding"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        tree = self._build_huffman_tree(freq)
        codes = self._generate_codes(tree)
        
        encoded_bits = []
        for byte in data:
            encoded_bits.append(codes[byte])
        
        bit_string = ''.join(encoded_bits)
        compressed = self._pack_bits(bit_string)
        
        header = self._create_header(freq)
        
        return header + compressed
    
    def decompress(self, compressed_data):
        """Decompress Adaptive Huffman encoded data"""
        try:
            header_size = struct.unpack('I', compressed_data[:4])[0]
            header_data = compressed_data[4:4+header_size]
            compressed = compressed_data[4+header_size:]
            
            freq = self._parse_header(header_data)
            

            tree = self._build_huffman_tree(freq)
            codes = self._generate_codes(tree)
            reverse_codes = {v: k for k, v in codes.items()}
            

            bit_string = self._unpack_bits(compressed)
            

            decoded = []
            current = ""
            for bit in bit_string:
                current += bit
                if current in reverse_codes:
                    decoded.append(reverse_codes[current])
                    current = ""
            
            return bytes(decoded)
        except Exception as e:
            print(f"Decompression error: {e}")
            return compressed_data 
    
    def _build_huffman_tree(self, freq):
        """Build Huffman tree from character frequencies"""
        import heapq
        
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    
    def _generate_codes(self, tree):
        return {symbol: code for symbol, code in tree}
    
    def _pack_bits(self, bit_string):
        padding = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * padding
        
        packed = bytearray()
        for i in range(0, len(bit_string), 8):
            byte = bit_string[i:i+8]
            packed.append(int(byte, 2))
        
        return bytes([padding]) + bytes(packed)
    
    def _unpack_bits(self, packed_data):
        """Unpack bytes into bit string"""
        padding = packed_data[0]
        bit_string = ''.join(format(byte, '08b') for byte in packed_data[1:])

        if padding > 0:
            bit_string = bit_string[:-padding]
        
        return bit_string
    
    def _create_header(self, freq):
        header = bytearray()
        

        for symbol, count in freq.items():
            header.append(symbol)
            header.extend(struct.pack('I', count))
        
        header_data = bytes(header)
        return struct.pack('I', len(header_data)) + header_data
    
    def _parse_header(self, header_data):
        freq = {}
        i = 0
        
        while i < len(header_data):
            symbol = header_data[i]
            count = struct.unpack('I', header_data[i+1:i+5])[0]
            freq[symbol] = count
            i += 5
        
        return freq

def create_sample_files():
    global file_name_list
    
    print("\n" + "="*70)
    print("CREATING TEST FILES")
    print("="*70)

    large_text_file = "large_text.txt"
    file_name_list.append(large_text_file)
    print(f"Creating: {large_text_file}")
    with open(large_text_file, "w") as file:
        file.write("A fallstreak hole (also known as a cavum, hole punch cloud, punch hole cloud, skypunch, cloud canal or cloud hole) is a large gap, usually circular or elliptical, that can appear in cirrocumulus or altocumulus clouds. The holes are caused by supercooled water in the clouds suddenly evaporating or freezing, and may be triggered by passing aircraft. Such clouds are not unique to any one geographic area and have been photographed from many places.\nBecause of their rarity and unusual appearance, fallstreak holes have been mistaken for or attributed to unidentified flying objects.\nSuch holes are formed when the water temperature in the clouds is below freezing, but the water, in a supercooled state, has not frozen yet due to the lack of ice nucleation. When ice crystals do form, a domino effect is set off due to the Wegener-Bergeron-Findeisen process, causing the water droplets around the crystals to evaporate; this leaves a large, often circular, hole in the cloud. It is thought that the introduction of large numbers of tiny ice crystals into the cloud layer sets off this domino effect of fusion which creates the hole.\nThe ice crystals can be formed by passing aircraft, which often have a large reduction in pressure behind the wing-tip or propeller-tips. This cools the air very quickly, and can produce a ribbon of ice crystals trailing in the aircraft's wake. These ice crystals find themselves surrounded by droplets, and grow quickly by the Bergeron process, causing the droplets to evaporate and creating a hole with brush-like streaks of ice crystals below it. An early satellite documentation of elongated fallstreak holes over the Florida Panhandle that likely were induced by passing aircraft appeared in Corfidi and Brandli (1986). Fallstreak holes are more routinely seen by the higher resolution satellites of today (e.g., see fourth example image in this article).")
    print(f"  ✓ Size: {os.path.getsize(large_text_file)} bytes")
    

    phrase_text_file = "phrase.txt"
    file_name_list.append(phrase_text_file)
    print(f"Creating: {phrase_text_file}")
    with open(phrase_text_file, "w") as file:
        phrase = "Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy."
        file.write(phrase * 40)
    print(f"  ✓ Size: {os.path.getsize(phrase_text_file)} bytes")
    
    random_text_file = "random.txt"
    file_name_list.append(random_text_file)
    print(f"Creating: {random_text_file}")
    with open(random_text_file, "w") as file:
        file.write("!3{VQTZ9B3dV}[M,L|kb$LQ/~D0urI:w5?x.2ai_X:,xqDT?cfB@+7=8G#k2pjD!e6W6}*2@2Q1d[X!2.HLss,}}`fXhodOHDe=D|I-hcmP._X_MbPx0e88b3s9]1aI[K)l%cN,e{xo5P~tu9{]50lB%_v:VVRt<4n;'Kc7SE=1JFq_7C^^MhY4+]k^<='!@JokP%mziaR7Bk,6sVaA}@^7K4y@l<!I42OiIC#b9Z}W<LQV--waXNv5O1}~|:Yx%HiI_9bS=!qE{=H[&Rh9CqiS6;%*#sBKB|2$Pf?R5O@6y'sB{2;e;`qC_6y2gZWT0CyMJpxbXlqbwF8;oW=s7e+4dZVyG4AT0x>1'M*:AQcIO{[p*ogq=`J[5=k{[~H7i6zg4Fe@57vTYRD9nWPmSUg0j4iGtX_[o~+]ftT~0N;8/WS=NC|)yJ~f4Sqy8&//VA^T2NT$!D|9Gv_Ehr#G3u7>hC[kK0Y~=[j;]UgoSP6bH{G;'x53(.!4UrRm!4:F=;d,C4{~-f:i'g'-<~SMDfE9^$uh3O-hS&6k5z?99gE-i}&w}hIpNBm`k6%Nh@#(Ui9_*a4*x~13RbsC(2Z_5:oRV*3c@/DY(NjseU$#GiY.`FNjY+f$")
    print(f"  ✓ Size: {os.path.getsize(random_text_file)} bytes")
    
    print("✓ All test files created successfully!\n")
    return [large_text_file, phrase_text_file, random_text_file]

def compress_adaptive_huffman(test_file, compressed_file):
    huffman = AdaptiveHuffman()
    start_time = time.time()
    
    try:
        with open(test_file, "rb") as file:
            data = file.read()
        
        compressed_data = huffman.compress(data)
        
        with open(compressed_file, "wb") as file:
            file.write(compressed_data)
        
        compression_time = time.time() - start_time
        
        # Verify decompression
        decompressed = huffman.decompress(compressed_data)
        integrity = (data == decompressed)
        
        return compression_time, integrity
    except Exception as e:
        print(f"  ✗ Error during Adaptive Huffman compression: {e}")
        return None, False

def compress_deflate(test_file, compressed_file, level=9):
    start_time = time.time()
    
    try:
        with open(test_file, 'rb') as file:
            data = file.read()
        
        compressed_data = zlib.compress(data, level=level)
        
        with open(compressed_file, "wb") as file:
            file.write(compressed_data)
        
        compression_time = time.time() - start_time
        
        decompressed = zlib.decompress(compressed_data)
        integrity = (data == decompressed)
        
        return compression_time, integrity
    except Exception as e:
        print(f"  ✗ Error during DEFLATE compression: {e}")
        return None, False

def compress_brotli(test_file, compressed_file, quality=11):
    if not BROTLI_AVAILABLE:
        return None, False
    
    start_time = time.time()
    
    try:
        with open(test_file, 'rb') as file:
            data = file.read()
        
        compressed_data = brotli.compress(data, quality=quality)
        
        with open(compressed_file, "wb") as file:
            file.write(compressed_data)
        
        compression_time = time.time() - start_time
        
        decompressed = brotli.decompress(compressed_data)
        integrity = (data == decompressed)
        
        return compression_time, integrity
    except Exception as e:
        print(f"  ✗ Error during Brotli compression: {e}")
        return None, False

def run_benchmark(input_files):
    results = []
    
    print("="*70)
    print("RUNNING COMPRESSION BENCHMARKS")
    print("="*70)
    
    for test_file in input_files:
        print(f"\n{'─'*70}")
        print(f"│ Benchmarking: {test_file}")
        print(f"{'─'*70}")
        
        original_size = os.path.getsize(test_file)
        print(f"│ Original Size: {original_size:,} bytes\n│")
        
        # Adaptive Huffman Compression
        compressed_huff = os.path.splitext(test_file)[0] + "_adaptive_huffman.huff"
        file_name_list.append(compressed_huff)
        huff_time, huff_integrity = compress_adaptive_huffman(test_file, compressed_huff)
        
        if huff_time is not None:
            compressed_size_huff = os.path.getsize(compressed_huff)
            huff_ratio = compressed_size_huff / original_size
            huff_saved = 1 - huff_ratio
            
            print(f"│ Adaptive Huffman:")
            print(f"│   Compressed: {compressed_size_huff:,} bytes")
            print(f"│   Ratio: {huff_ratio*100:.2f}%")
            print(f"│   Space Saved: {huff_saved*100:.2f}%")
            print(f"│   Time: {huff_time:.4f}s")
            print(f"│   Integrity: {'✓ PASS' if huff_integrity else '✗ FAIL'}\n│")
            
            results.append({
                'file': test_file,
                'algorithm': 'Adaptive Huffman',
                'original_size': original_size,
                'compressed_size': compressed_size_huff,
                'ratio': huff_ratio,
                'space_saved': huff_saved,
                'time': huff_time,
                'integrity': huff_integrity
            })
        
        # DEFLATE Compression
        compressed_deflate = os.path.splitext(test_file)[0] + "_deflate.zlib"
        file_name_list.append(compressed_deflate)
        deflate_time, deflate_integrity = compress_deflate(test_file, compressed_deflate)
        
        if deflate_time is not None:
            compressed_size_deflate = os.path.getsize(compressed_deflate)
            deflate_ratio = compressed_size_deflate / original_size
            deflate_saved = 1 - deflate_ratio
            
            print(f"│ DEFLATE (zlib):")
            print(f"│   Compressed: {compressed_size_deflate:,} bytes")
            print(f"│   Ratio: {deflate_ratio*100:.2f}%")
            print(f"│   Space Saved: {deflate_saved*100:.2f}%")
            print(f"│   Time: {deflate_time:.4f}s")
            print(f"│   Integrity: {'✓ PASS' if deflate_integrity else '✗ FAIL'}\n│")
            
            results.append({
                'file': test_file,
                'algorithm': 'DEFLATE',
                'original_size': original_size,
                'compressed_size': compressed_size_deflate,
                'ratio': deflate_ratio,
                'space_saved': deflate_saved,
                'time': deflate_time,
                'integrity': deflate_integrity
            })
        
        # Brotli Compression
        if BROTLI_AVAILABLE:
            compressed_brotli = os.path.splitext(test_file)[0] + "_brotli.br"
            file_name_list.append(compressed_brotli)
            brotli_time, brotli_integrity = compress_brotli(test_file, compressed_brotli)
            
            if brotli_time is not None:
                compressed_size_brotli = os.path.getsize(compressed_brotli)
                brotli_ratio = compressed_size_brotli / original_size
                brotli_saved = 1 - brotli_ratio
                
                print(f"│ Brotli:")
                print(f"│   Compressed: {compressed_size_brotli:,} bytes")
                print(f"│   Ratio: {brotli_ratio*100:.2f}%")
                print(f"│   Space Saved: {brotli_saved*100:.2f}%")
                print(f"│   Time: {brotli_time:.4f}s")
                print(f"│   Integrity: {'✓ PASS' if brotli_integrity else '✗ FAIL'}")
                
                results.append({
                    'file': test_file,
                    'algorithm': 'Brotli',
                    'original_size': original_size,
                    'compressed_size': compressed_size_brotli,
                    'ratio': brotli_ratio,
                    'space_saved': brotli_saved,
                    'time': brotli_time,
                    'integrity': brotli_integrity
                })
        
        print(f"{'─'*70}")
    
    return results

def visualize_benchmarks(results):
    df = pd.DataFrame(results)
    
    files = sorted(df['file'].unique())
    algorithms = sorted(df['algorithm'].unique())
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(files))
    width = 0.25
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        ratios = [algo_data[algo_data['file'] == f]['ratio'].values[0] * 100 
                  if len(algo_data[algo_data['file'] == f]) > 0 else 0 
                  for f in files]
        ax1.bar(x + i*width, ratios, width, label=algo)
    
    ax1.set_ylabel('Compression Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Compression Ratio: Adaptive Huffman vs DEFLATE vs Brotli\n(Lower is Better)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f.replace('.txt', '') for f in files])
    ax1.legend()
    ax1.axhline(100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No Compression')
    ax1.grid(axis='y', alpha=0.3)
    fig1.tight_layout()
    plt.savefig('01_compression_ratio.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: 01_compression_ratio.jpg")
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        space_saved = [algo_data[algo_data['file'] == f]['space_saved'].values[0] * 100 
                       if len(algo_data[algo_data['file'] == f]) > 0 else 0 
                       for f in files]
        ax2.bar(x + i*width, space_saved, width, label=algo)
    
    ax2.set_ylabel('Space Saved (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Space Savings: All Algorithms\n(Higher is Better)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f.replace('.txt', '') for f in files])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    plt.savefig('02_space_saved.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: 02_space_saved.jpg")
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        times = [algo_data[algo_data['file'] == f]['time'].values[0] * 1000 
                 if len(algo_data[algo_data['file'] == f]) > 0 else 0 
                 for f in files]
        ax3.bar(x + i*width, times, width, label=algo)
    
    ax3.set_ylabel('Compression Time (milliseconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Compression Speed Comparison\n(Lower is Better)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f.replace('.txt', '') for f in files])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    plt.savefig('03_compression_time.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: 03_compression_time.jpg")
    

    fig4, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, file in enumerate(files):
        file_data = df[df['file'] == file]
        algo_list = file_data['algorithm'].values
        space_saved = file_data['space_saved'].values * 100
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        axes[idx].barh(algo_list, space_saved, color=colors[:len(algo_list)])
        axes[idx].set_xlabel('Space Saved (%)', fontweight='bold')
        axes[idx].set_title(file.replace('.txt', '').replace('_', ' ').title(), 
                           fontweight='bold', fontsize=12)
        axes[idx].grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(space_saved):
            axes[idx].text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
    
    fig4.suptitle('Space Savings by File Type', fontsize=14, fontweight='bold')
    fig4.tight_layout()
    plt.savefig('04_by_file_type.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("Saved: 04_by_file_type.jpg")
    

    fig5, ax5 = plt.subplots(figsize=(12, 7))
    
    df['efficiency'] = df['space_saved'] / df['time']
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        efficiency = [algo_data[algo_data['file'] == f]['efficiency'].values[0] 
                      if len(algo_data[algo_data['file'] == f]) > 0 else 0 
                      for f in files]
        ax5.bar(x + i*width, efficiency, width, label=algo)
    
    ax5.set_ylabel('Efficiency Score (Space Saved/Time)', fontsize=12, fontweight='bold')
    ax5.set_title('Compression Efficiency\n(Higher is Better)', 
                  fontsize=14, fontweight='bold')
    ax5.set_xticks(x + width)
    ax5.set_xticklabels([f.replace('.txt', '') for f in files])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    fig5.tight_layout()
    plt.savefig('05_efficiency_score.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("Saved: 05_efficiency_score.jpg")
    
 
    csv_file = 'compression_benchmark_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(df[['file', 'algorithm', 'compressed_size', 'ratio', 'space_saved', 'time', 'integrity']].to_string(index=False))


def cleanup_files():
    print("\n" + "="*70)
    print("CLEANING UP FILES")
    print("="*70)
    
    for file_name in file_name_list:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
                print(f"  ✓ Deleted: {file_name}")
        except OSError as e:
            print(f"  ✗ Error deleting {file_name}: {e}")



if __name__ == "__main__":
    print("\n" + "="*70)
    print(" DATA COMPRESSION BENCHMARKING TOOL")
    print(" Algorithms: Adaptive Huffman | DEFLATE | Brotli")
    print("="*70)
    
    try:
        input_files = create_sample_files()
        
        results = run_benchmark(input_files)
        
        if results:
            visualize_benchmarks(results)
            
            print("\n" + "="*70)
            print("✓ COMPRESSION BENCHMARKING COMPLETE!")
            print("="*70)
            print("\nGenerated Output Files:")
            print("  • 01_compression_ratio.jpg")
            print("  • 02_space_saved.jpg")
            print("  • 03_compression_time.jpg")
            print("  • 04_by_file_type.jpg")
            print("  • 05_efficiency_score.jpg")
            print("  • compression_benchmark_results.csv")
            print("\n" + "="*70)
        else:
            print("\n✗ No results generated. Please check for errors above.")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user.")
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_files()
        print("\n" + "="*70)
        print("Benchmark session ended.")
        print("="*70 + "\n")
