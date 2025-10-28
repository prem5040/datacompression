#This file was for Milstone 1
#For Milestone2, lzmastd.py and ahuffbrotli.py is used
import sys
import os
import zlib
import time
import matplotlib.pyplot as plt
import numpy as np

try:
    from dahuffman import HuffmanCodec
except ImportError:
    print("The 'dahuffman' library is required.")
    print("Please install it using: pip install dahuffman")
    sys.exit(1)



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
        file.write("!3{VQTZ9B\u00a33dV}[M,L|kb$LQ/~D0urI:w5?x.2ai_X:,xqDT?cfB\u00a3@+7=8G#k2pjD!e6W6}*2@2Q1d[X\!2.HLss,}}`fXhodOHDe=D|I-hcmP._X_MbPx0\e88b3s9]1\u00a3aI[K)l%cN,e{xo5P~tu9{]\50lB%_v:VVRt<4n;'Kc7S`E=1JFq_7C`^`MhY4+]k^<='!@JokP%m\u00a3z\u00a3iaR7Bk\,6sVaA}@\u00a3^7K4y@l<!I42OiIC#b9Z}W<LQV--waXNv5O1}~|:Yx%HiI_9bS=!qE{=H[&Rh9CqiS6;%*#sBKB|2$Pf?R5O@6y'sB{2;e;`qC_6y2gZWT0CyMJpxbXlqbwF8;oW=s7e+4dZVyG4AT0x>1'M*\u00a3:AQcIO{[p*ogq=`J[5=k{[~H7i6zg4Fe@57vTYRD9nWPmSUg0j4iGtX_[o~+]ftT~0N;8/WS=NC|)yJ~f4Sqy8&//VA^T2NT$!D|9Gv_E\u00a3hr#G3u7>hC[kK0Y~=[j;]UgoSP6bH{G;'x53(.!4UrRm!4:F=;d,C4{~-f:i'g'-<~SMDfE9^$uh3O-hS&6k5z?99gE-i}&w}hIpNBm`k6%Nh@\#(Ui9_*a4*\u00a3x~13RbsC(2Z_5:oRV*3c@/DY\(NjseU$#GiY.`FNjY+f$")
    
    return [large_text_file, phrase_text_file, random_text_file]


def compress_huffman(test_file, compressed_file):
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
        print(f"An error occurred during Huffman compression of '{test_file}': {e}")
        return None, None

def compress_zlib(test_file, compressed_file, level=9):
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
        print(f"An error occurred during DEFLATE compression of '{test_file}': {e}")
        return None


def run_benchmark(input_files):
    results = []

    for test_file in input_files:
        print("-" * 50)
        print(f"| Benchmarking File: {test_file}")
        print("-" * 50)
        
        original_size = os.path.getsize(test_file)
        print(f"Original Size: {original_size} bytes")

        compressed_huff = f"{os.path.splitext(test_file)[0]}_huff.huff"
        file_name_list.append(compressed_huff)
        
        huff_time, _ = compress_huffman(test_file, compressed_huff)
        compressed_size_huff = os.path.getsize(compressed_huff)
        
        compressed_zlib = f"{os.path.splitext(test_file)[0]}_zlib.zlib"
        file_name_list.append(compressed_zlib)
        
        zlib_time = compress_zlib(test_file, compressed_zlib, level=9)
        compressed_size_zlib = os.path.getsize(compressed_zlib)

        huff_ratio = compressed_size_huff / original_size
        zlib_ratio = compressed_size_zlib / original_size
        huff_saved = 1 - huff_ratio
        zlib_saved = 1 - zlib_ratio

        print("\n| **Huffman Coding Results**")
        print(f"| Compressed Size: {compressed_size_huff} bytes")
        print(f"| Compression Ratio: {huff_ratio:.2%}")
        print(f"| Space Saved: {huff_saved:.2%}")
        print(f"| Time: {huff_time:.4f} seconds")

        print("\n| **DEFLATE (zlib) Results**")
        print(f"| Compressed Size: {compressed_size_zlib} bytes")
        print(f"| Compression Ratio: {zlib_ratio:.2%}")
        print(f"| Space Saved: {zlib_saved:.2%}")
        print(f"| Time: {zlib_time:.4f} seconds")
        print("\n" + "=" * 50)


        results.append({
            'file': test_file,
            'original_size': original_size,
            'huff_size': compressed_size_huff,
            'zlib_size': compressed_size_zlib,
            'huff_time': huff_time,
            'zlib_time': zlib_time
        })

    return results

def visualize_benchmarks(results):
    file_names = [r['file'].replace(".txt", "") for r in results]
    huff_ratios = np.array([r['huff_size'] / r['original_size'] for r in results])
    zlib_ratios = np.array([r['zlib_size'] / r['original_size'] for r in results])
    

    huff_times = np.array([r['huff_time'] for r in results])
    zlib_times = np.array([r['zlib_time'] for r in results])

    x = np.arange(len(file_names))  # the label locations
    width = 0.35  # the width of the bars

    print("\n[INFO] Generating and saving visualization graphs...")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width/2, huff_ratios, width, label='Huffman Coding', color='skyblue')
    ax1.bar(x + width/2, zlib_ratios, width, label='DEFLATE (zlib)', color='salmon')

    ax1.set_ylabel('Compression Ratio (Compressed Size / Original Size)')
    ax1.set_title('Compression Ratio: Huffman vs. DEFLATE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_names)
    ax1.legend()
    ax1.axhline(1.0, color='gray', linestyle='--', label='No Compression (100%)') # Line for 100% (no compression)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    
    ratio_filename = 'compression_ratio_comparison.jpg'
    fig1.tight_layout()
    plt.savefig(ratio_filename, dpi=300)
    plt.close(fig1) # Close the figure to free memory
    print(f"Saved: {ratio_filename}")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - width/2, huff_times * 1000, width, label='Huffman Coding', color='lightgreen')
    ax2.bar(x + width/2, zlib_times * 1000, width, label='DEFLATE (zlib)', color='gold')

    ax2.set_ylabel('Compression Time (ms)')
    ax2.set_title('Compression Speed: Huffman vs. DEFLATE (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(file_names)
    ax2.legend()
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    speed_filename = 'compression_speed_comparison.jpg'
    fig2.tight_layout()
    plt.savefig(speed_filename, dpi=300)
    plt.close(fig2) # Close the figure to free memory
    print(f"Saved: {speed_filename}")


def cleanup_files():
    print("\n" + "=" * 50)
    print("Cleaning up generated files...")
    for file_name in file_name_list:
        try:
            os.remove(file_name)
            print(f"Deleted: {file_name}")
        except OSError as e:
            if os.path.exists(file_name):
                 print(f"Error deleting file {file_name}: {e}")
            else:
                 # This happens if the file was never created, which is fine
                 pass
    print("Cleanup complete.")

if __name__ == "__main__":
    print("--- Data Compression Benchmarking Tool ---")
    input_files = create_sample_files()
    
    try:
        benchmark_results = run_benchmark(input_files)
        
        if benchmark_results:
            visualize_benchmarks(benchmark_results)

    except Exception as e:
        print(f"\nFATAL ERROR during execution: {e}")
    
    finally:
        cleanup_files()
