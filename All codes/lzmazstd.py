import sys
import os
import zlib
import lzma
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from dahuffman import HuffmanCodec
    HUFFMAN_AVAILABLE = True
except ImportError:
    print("WARNING: 'dahuffman' library not found. Huffman compression will be skipped.")
    print("Install with: pip install dahuffman")
    HUFFMAN_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    print("WARNING: 'zstandard' library not found. Zstandard compression will be skipped.")
    print("Install with: pip install zstandard")
    ZSTD_AVAILABLE = False

file_name_list = []
all_results = []
def create_sample_files():
    global file_name_list

    large_text_file = "large_text.txt"
    file_name_list.append(large_text_file)
    print(f"Creating sample file: {large_text_file}")
    with open(large_text_file, "w") as file:
        file.write("A fallstreak hole (also known as a cavum, hole punch cloud, punch hole cloud, skypunch, cloud canal or cloud hole) is a large gap, usually circular or elliptical, that can appear in cirrocumulus or altocumulus clouds. The holes are caused by supercooled water in the clouds suddenly evaporating or freezing, and may be triggered by passing aircraft. Such clouds are not unique to any one geographic area and have been photographed from many places.\nBecause of their rarity and unusual appearance, fallstreak holes have been mistaken for or attributed to unidentified flying objects.\nSuch holes are formed when the water temperature in the clouds is below freezing, but the water, in a supercooled state, has not frozen yet due to the lack of ice nucleation. When ice crystals do form, a domino effect is set off due to the Wegener-Bergeron-Findeisen process, causing the water droplets around the crystals to evaporate; this leaves a large, often circular, hole in the cloud. It is thought that the introduction of large numbers of tiny ice crystals into the cloud layer sets off this domino effect of fusion which creates the hole.\nThe ice crystals can be formed by passing aircraft, which often have a large reduction in pressure behind the wing-tip or propeller-tips. This cools the air very quickly, and can produce a ribbon of ice crystals trailing in the aircraft's wake. These ice crystals find themselves surrounded by droplets, and grow quickly by the Bergeron process, causing the droplets to evaporate and creating a hole with brush-like streaks of ice crystals below it. An early satellite documentation of elongated fallstreak holes over the Florida Panhandle that likely were induced by passing aircraft appeared in Corfidi and Brandli (1986). Fallstreak holes are more routinely seen by the higher resolution satellites of today (e.g., see fourth example image in this article).")

    phrase_text_file = "phrase.txt"
    file_name_list.append(phrase_text_file)
    print(f"Creating sample file: {phrase_text_file}")
    with open(phrase_text_file, "w") as file:
        phrase = "Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy."
        file.write(phrase * 40)

    random_text_file = "random.txt"
    file_name_list.append(random_text_file)
    print(f"Creating sample file: {random_text_file}")
    with open(random_text_file, "w") as file:
        file.write("!3{VQTZ9B3dV}[M,L|kb$LQ/~D0urI:w5?x.2ai_X:,xqDT?cfB@+7=8G#k2pjD!e6W6}*2@2Q1d[X!2.HLss,}}`fXhodOHDe=D|I-hcmP._X_MbPx0e88b3s9]1aI[K)l%cN,e{xo5P~tu9{]50lB%_v:VVRt<4n;'Kc7SE=1JFq_7C^^MhY4+]k^<='!@JokP%mziaR7Bk,6sVaA}@^7K4y@l<!I42OiIC#b9Z}W<LQV--waXNv5O1}~|:Yx%HiI_9bS=!qE{=H[&Rh9CqiS6;%*#sBKB|2$Pf?R5O@6y'sB{2;e;`qC_6y2gZWT0CyMJpxbXlqbwF8;oW=s7e+4dZVyG4AT0x>1'M*:AQcIO{[p*ogq=`J[5=k{[~H7i6zg4Fe@57vTYRD9nWPmSUg0j4iGtX_[o~+]ftT~0N;8/WS=NC|)yJ~f4Sqy8&//VA^T2NT$!D|9Gv_Ehr#G3u7>hC[kK0Y~=[j;]UgoSP6bH{G;'x53(.!4UrRm!4:F=;d,C4{~-f:i'g'-<~SMDfE9^$uh3O-hS&6k5z?99gE-i}&w}hIpNBm`k6%Nh@#(Ui9_*a4*x~13RbsC(2Z_5:oRV*3c@/DY(NjseU$#GiY.`FNjY+f$")
    
    return [large_text_file, phrase_text_file, random_text_file]


def compress_huffman(test_file, compressed_file):
    if not HUFFMAN_AVAILABLE:
        return None, None
    
    start_time = time.time()
    try:
        with open(test_file, "r") as file:
            data = file.read()
        
        codec = HuffmanCodec.from_data(data)
        encoded = codec.encode(data)
        
        with open(compressed_file, "wb") as file:
            file.write(encoded)
            
        compression_time = time.time() - start_time
        return compression_time, codec
    except Exception as e:
        print(f"Error during Huffman compression: {e}")
        return None, None

def compress_deflate(test_file, compressed_file, level=9):
    """Compresses a file using DEFLATE (zlib)."""
    start_time = time.time()
    try:
        with open(test_file, 'rb') as file:
            data = file.read()
        
        compressed_data = zlib.compress(data, level=level)
        
        with open(compressed_file, "wb") as file:
            file.write(compressed_data)
            
        compression_time = time.time() - start_time
        return compression_time
    except Exception as e:
        print(f"Error during DEFLATE compression: {e}")
        return None

def compress_lzma(test_file, compressed_file, preset=9):
    """Compresses a file using LZMA."""
    start_time = time.time()
    try:
        with open(test_file, 'rb') as file:
            data = file.read()
        
        compressed_data = lzma.compress(data, preset=preset)
        
        with open(compressed_file, "wb") as file:
            file.write(compressed_data)
            
        compression_time = time.time() - start_time
        return compression_time
    except Exception as e:
        print(f"Error during LZMA compression: {e}")
        return None

def compress_zstandard(test_file, compressed_file, level=22):
    if not ZSTD_AVAILABLE:
        return None
    
    start_time = time.time()
    try:
        with open(test_file, 'rb') as file:
            data = file.read()
        
        cctx = zstd.ZstdCompressor(level=level)
        compressed_data = cctx.compress(data)
        
        with open(compressed_file, "wb") as file:
            file.write(compressed_data)
            
        compression_time = time.time() - start_time
        return compression_time
    except Exception as e:
        print(f"Error during Zstandard compression: {e}")
        return None

def run_benchmark(input_files):
    """Runs all compression algorithms on the files."""
    results = []

    for test_file in input_files:
        print("\n" + "-" * 60)
        print("| Benchmarking File: " + test_file)
        print("-" * 60)
        
        original_size = os.path.getsize(test_file)
        print("Original Size: " + str(original_size) + " bytes\n")

        if HUFFMAN_AVAILABLE:
            compressed_huff = os.path.splitext(test_file)[0] + "_huff.huff"
            file_name_list.append(compressed_huff)
            huff_result = compress_huffman(test_file, compressed_huff)
            huff_time = huff_result[0] if huff_result[0] is not None else 0
            compressed_size_huff = os.path.getsize(compressed_huff)
            huff_ratio = compressed_size_huff / original_size
            huff_saved = 1 - huff_ratio
            print("| Huffman: " + str(compressed_size_huff) + " bytes | Ratio: " + str(round(huff_ratio*100, 2)) + "% | Time: " + str(round(huff_time, 4)) + "s")
            results.append({
                'file': test_file,
                'algorithm': 'Huffman',
                'original_size': original_size,
                'compressed_size': compressed_size_huff,
                'ratio': huff_ratio,
                'space_saved': huff_saved,
                'time': huff_time
            })

        compressed_deflate = os.path.splitext(test_file)[0] + "_deflate.zlib"
        file_name_list.append(compressed_deflate)
        deflate_time = compress_deflate(test_file, compressed_deflate)
        compressed_size_deflate = os.path.getsize(compressed_deflate)
        deflate_ratio = compressed_size_deflate / original_size
        deflate_saved = 1 - deflate_ratio
        print("| DEFLATE: " + str(compressed_size_deflate) + " bytes | Ratio: " + str(round(deflate_ratio*100, 2)) + "% | Time: " + str(round(deflate_time, 4)) + "s")
        results.append({
            'file': test_file,
            'algorithm': 'DEFLATE',
            'original_size': original_size,
            'compressed_size': compressed_size_deflate,
            'ratio': deflate_ratio,
            'space_saved': deflate_saved,
            'time': deflate_time
        })

        # --- LZMA Compression ---
        compressed_lzma = os.path.splitext(test_file)[0] + "_lzma.xz"
        file_name_list.append(compressed_lzma)
        lzma_time = compress_lzma(test_file, compressed_lzma)
        compressed_size_lzma = os.path.getsize(compressed_lzma)
        lzma_ratio = compressed_size_lzma / original_size
        lzma_saved = 1 - lzma_ratio
        print("| LZMA:    " + str(compressed_size_lzma) + " bytes | Ratio: " + str(round(lzma_ratio*100, 2)) + "% | Time: " + str(round(lzma_time, 4)) + "s")
        results.append({
            'file': test_file,
            'algorithm': 'LZMA',
            'original_size': original_size,
            'compressed_size': compressed_size_lzma,
            'ratio': lzma_ratio,
            'space_saved': lzma_saved,
            'time': lzma_time
        })

        # --- Zstandard Compression ---
        if ZSTD_AVAILABLE:
            compressed_zstd = os.path.splitext(test_file)[0] + "_zstd.zst"
            file_name_list.append(compressed_zstd)
            zstd_time = compress_zstandard(test_file, compressed_zstd)
            compressed_size_zstd = os.path.getsize(compressed_zstd)
            zstd_ratio = compressed_size_zstd / original_size
            zstd_saved = 1 - zstd_ratio
            print("| Zstandard: " + str(compressed_size_zstd) + " bytes | Ratio: " + str(round(zstd_ratio*100, 2)) + "% | Time: " + str(round(zstd_time, 4)) + "s")
            results.append({
                'file': test_file,
                'algorithm': 'Zstandard',
                'original_size': original_size,
                'compressed_size': compressed_size_zstd,
                'ratio': zstd_ratio,
                'space_saved': zstd_saved,
                'time': zstd_time
            })

        print()

    return results

# --- 4. Visualization Functions ---

def visualize_benchmarks(results):
    df = pd.DataFrame(results)
    
    # Get unique files
    files = sorted(df['file'].unique())
    algorithms = sorted(df['algorithm'].unique())
    
    print("\nGenerating and saving visualization graphs...")

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(files))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        ratios = []
        for f in files:
            file_data = algo_data[algo_data['file'] == f]
            if len(file_data) > 0:
                ratios.append(file_data['ratio'].values[0])
            else:
                ratios.append(0)
        ax1.bar(x + i*width, ratios, width, label=algo)
    
    ax1.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Compression Ratio Comparison Across All Algorithms', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(algorithms)-1) / 2)
    ax1.set_xticklabels([f.replace('.txt', '') for f in files])
    ax1.legend()
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, label='No Compression')
    ax1.grid(axis='y', alpha=0.3)
    
    fig1.tight_layout()
    plt.savefig('01_compression_ratio_all.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: 01_compression_ratio_all.jpg")

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        space_saved = []
        for f in files:
            file_data = algo_data[algo_data['file'] == f]
            if len(file_data) > 0:
                space_saved.append(file_data['space_saved'].values[0] * 100)
            else:
                space_saved.append(0)
        ax2.bar(x + i*width, space_saved, width, label=algo)
    
    ax2.set_ylabel('Space Saved (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Space Saved: All Algorithms Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(algorithms)-1) / 2)
    ax2.set_xticklabels([f.replace('.txt', '') for f in files])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    fig2.tight_layout()
    plt.savefig('02_space_saved_all.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: 02_space_saved_all.jpg")

    # --- Plot 3: Compression Time ---
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        times = []
        for f in files:
            file_data = algo_data[algo_data['file'] == f]
            if len(file_data) > 0:
                times.append(file_data['time'].values[0] * 1000)
            else:
                times.append(0)
        ax3.bar(x + i*width, times, width, label=algo)
    
    ax3.set_ylabel('Compression Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Compression Speed: All Algorithms (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width * (len(algorithms)-1) / 2)
    ax3.set_xticklabels([f.replace('.txt', '') for f in files])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    fig3.tight_layout()
    plt.savefig('03_compression_time_all.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: 03_compression_time_all.jpg")

    # --- Plot 4: Overall Algorithm Comparison (by file type) ---
    fig4, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, file in enumerate(files):
        file_data = df[df['file'] == file]
        algo_list = file_data['algorithm'].values
        ratios = file_data['ratio'].values * 100
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(algo_list)))
        axes[idx].barh(algo_list, ratios, color=colors)
        axes[idx].set_xlabel('Compression Ratio (%)', fontweight='bold')
        axes[idx].set_title(file.replace('.txt', ''), fontweight='bold', fontsize=12)
        axes[idx].axvline(100, color='red', linestyle='--', linewidth=2)
        axes[idx].grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(ratios):
            axes[idx].text(v + 1, i, str(round(v, 1)) + "%", va='center', fontweight='bold')
    
    fig4.suptitle('Compression Ratio by File Type and Algorithm', fontsize=14, fontweight='bold')
    fig4.tight_layout()
    plt.savefig('04_by_file_type.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("Saved: 04_by_file_type.jpg")

    # --- Plot 5: Efficiency Score (Space Saved per Second) ---
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    
    df['efficiency'] = df['space_saved'] / df['time']
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        efficiency = []
        for f in files:
            file_data = algo_data[algo_data['file'] == f]
            if len(file_data) > 0:
                efficiency.append(file_data['efficiency'].values[0])
            else:
                efficiency.append(0)
        ax5.bar(x + i*width, efficiency, width, label=algo)
    
    ax5.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
    ax5.set_title('Compression Efficiency (Space Saved per Second)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x + width * (len(algorithms)-1) / 2)
    ax5.set_xticklabels([f.replace('.txt', '') for f in files])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    fig5.tight_layout()
    plt.savefig('05_efficiency_score.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("Saved: 05_efficiency_score.jpg")

    # --- Save results to CSV ---
    csv_file = 'compression_benchmark_results.csv'
    df.to_csv(csv_file, index=False)
    print("Saved: " + csv_file)
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPRESSION BENCHMARK SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

def cleanup_files():
    print("\n" + "="*60)
    print("Cleaning up generated files...")
    for file_name in file_name_list:
        try:
            os.remove(file_name)
            print("Deleted: " + file_name)
        except OSError as e:
            if os.path.exists(file_name):
                print("Error deleting " + file_name + ": " + str(e))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNIFIED DATA COMPRESSION BENCHMARKING TOOL")
    print("Algorithms: Huffman, DEFLATE, LZMA, Zstandard")
    print("="*60 + "\n")
    
    input_files = create_sample_files()
    
    try:
        results = run_benchmark(input_files)
        
        if results:
            visualize_benchmarks(results)
            print("\n" + "="*60)
            print("COMPRESSION BENCHMARKING COMPLETE!")
            print("="*60)
            print("\nGenerated files:")
            print("  - 01_compression_ratio_all.jpg")
            print("  - 02_space_saved_all.jpg")
            print("  - 03_compression_time_all.jpg")
            print("  - 04_by_file_type.jpg")
            print("  - 05_efficiency_score.jpg")
            print("  - compression_benchmark_results.csv")

    except Exception as e:
        print("\nFATAL ERROR: " + str(e))
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup_files()
