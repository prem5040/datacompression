#!/usr/bin/env python3
import os
import sys
import time
import lzma
import zstandard as zstd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import psutil
from datetime import datetime
import json

class CompressionBenchmark:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)
        
        self.system_info = {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def create_test_files(self, data_dir="data"):
        os.makedirs(data_dir, exist_ok=True)
        test_files = []
        
        print("Creating test files...")
        
        # Large text file
        large_text = os.path.join(data_dir, "large_text.txt")
        with open(large_text, "w") as f:
            f.write("A fallstreak hole (also known as a cavum, hole punch cloud, punch hole cloud, skypunch, cloud canal or cloud hole) is a large gap, usually circular or elliptical, that can appear in cirrocumulus or altocumulus clouds. The holes are caused by supercooled water in the clouds suddenly evaporating or freezing, and may be triggered by passing aircraft. Such clouds are not unique to any one geographic area and have been photographed from many places.\nBecause of their rarity and unusual appearance, fallstreak holes have been mistaken for or attributed to unidentified flying objects.\nSuch holes are formed when the water temperature in the clouds is below freezing, but the water, in a supercooled state, has not frozen yet due to the lack of ice nucleation. When ice crystals do form, a domino effect is set off due to the Wegener-Bergeron-Findeisen process, causing the water droplets around the crystals to evaporate; this leaves a large, often circular, hole in the cloud. It is thought that the introduction of large numbers of tiny ice crystals into the cloud layer sets off this domino effect of fusion which creates the hole.\nThe ice crystals can be formed by passing aircraft, which often have a large reduction in pressure behind the wing-tip or propeller-tips. This cools the air very quickly, and can produce a ribbon of ice crystals trailing in the aircraft's wake. These ice crystals find themselves surrounded by droplets, and grow quickly by the Bergeron process, causing the droplets to evaporate and creating a hole with brush-like streaks of ice crystals below it. An early satellite documentation of elongated fallstreak holes over the Florida Panhandle that likely were induced by passing aircraft appeared in Corfidi and Brandli (1986). Fallstreak holes are more routinely seen by the higher resolution satellites of today (e.g., see fourth example image in this article).")

        test_files.append(("Large Text", large_text))
        
        # Random data
        random_file = os.path.join(data_dir, "random_data.bin")
        with open(random_file, "wb") as f:
#            f.write(")3{VQTZ9B£3dV}[M,L|kb$LQ/~D0urI:w5?x.2ai_X:,xqDT?cfB£@+7=8G#k2pjD!e6W6}*2@2Q1d[X2.HLss,}}`fXhodOHDe=D|I-hcmP._X_MbPx088b3s9]1£aI[K)l%cN,e{xo5P~tu9{]50lB%_v:VVRt<4n;'Kc7S`E=1JFq_7C`^`MhY4+]k^<='!@JokP%m£z£iaR7Bk,6sVaA}@£^7K4y@l<!I42OiIC#b9Z}W<LQV--waXNv5O1}~|:Yx%HiI_9bS=!qE{=H[&Rh9CqiS6;%*#sBKB|2$Pf?R5O@6y'sB{2;e;`qC_6y2gZWT0CyMJpxbXlqbwF8;oW=s7e+4dZVyG4AT0x>1'M*£:AQcIO{[p*ogq=`J[5=k{[~H7i6zg4Fe@57vTYRD9nWPmSUg0j4iGtX_[o~+]ftT~0N;8/WS=NC|)yJ~f4Sqy8&//VA^T2NT$!D|9Gv_E£hr#G3u7>hC[kK0Y~=[j;]UgoSP6bH{G;'x53(.!4UrRm!4:F=;d,C4{~-f:i'g'-<~SMDfE9^$uh3O-hS&6k5z?99gE-i}&w}hIpNBm`k6%Nh@#(Ui9_*a4*£x~13RbsC(2Z_5:oRV*3c@/DY(NjseU$#GiY.`FNjY+f$")
            f.write(os.urandom(1024 * 500))
        test_files.append(("Random Binary", random_file))
        
        # Repeating pattern
        repeating_file = os.path.join(data_dir, "repeating_pattern.txt")
        with open(repeating_file, "w") as f:
            f.write("Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.Crazy? I was crazy once, they locked me in a room, a rubber room, a rubber room filled with rats and rats make me crazy.")
        test_files.append(("Repeating Pattern", repeating_file))
        
        print(f"✓ Created {len(test_files)} test files\n")
        return test_files
    
    def benchmark_algorithm(self, file_path, file_type, algorithm_name, 
                           compress_func, decompress_func, level=None):
        try:
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            
            start_time = time.time()
            if level:
                compressed_data = compress_func(original_data, level)
            else:
                compressed_data = compress_func(original_data)
            compression_time = time.time() - start_time
            
            compressed_size = len(compressed_data)
            
            start_time = time.time()
            decompressed_data = decompress_func(compressed_data)
            decompression_time = time.time() - start_time
            
            integrity = (original_data == decompressed_data)
            
            compression_ratio = compressed_size / original_size
            space_saved = 1 - compression_ratio
            
            level_str = f" (Level {level})" if level else ""
            
            result = {
                'File Type': file_type,
                'Algorithm': f"{algorithm_name}{level_str}",
                'Original Size (bytes)': original_size,
                'Compressed Size (bytes)': compressed_size,
                'Compression Ratio': f"{compression_ratio:.4f}",
                'Space Saved': f"{space_saved:.2%}",
                'Compression Time (s)': f"{compression_time:.6f}",
                'Decompression Time (s)': f"{decompression_time:.6f}",
                'Integrity Check': 'PASS' if integrity else 'FAIL',
                'compression_ratio_float': compression_ratio,
                'space_saved_float': space_saved,
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def run_benchmark(self, test_files):
        print("="*80)
        print("STARTING COMPREHENSIVE COMPRESSION BENCHMARK")
        print("="*80)
        print(f"Host: {self.system_info['hostname']}")
        print(f"CPUs: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print("="*80 + "\n")
        
        lzma_presets = [1, 9]
        zstd_levels = [1, 22]
        
        for file_type, file_path in test_files:
            print(f"\nTesting: {file_type}")
            
            for preset in lzma_presets:
                print(f"  LZMA (Preset {preset})...", end=' ')
                result = self.benchmark_algorithm(
                    file_path, file_type, "LZMA",
                    lambda d, p: lzma.compress(d, preset=p),
                    lzma.decompress, preset
                )
                if result:
                    print(f"✓ {result['Space Saved']} saved")
            
            for level in zstd_levels:
                print(f"  Zstandard (Level {level})...", end=' ')
                result = self.benchmark_algorithm(
                    file_path, file_type, "Zstandard",
                    lambda d, l: zstd.ZstdCompressor(level=l).compress(d),
                    lambda d: zstd.ZstdDecompressor().decompress(d), level
                )
                if result:
                    print(f"✓ {result['Space Saved']} saved")
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
    
    def save_results(self):
        df = pd.DataFrame(self.results)
        
        csv_path = os.path.join(self.output_dir, "compression_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        info_path = os.path.join(self.output_dir, "system_info.json")
        with open(info_path, 'w') as f:
            json.dump(self.system_info, f, indent=2)
        
        self.generate_summary(df)
        self.generate_visualizations(df)
    
    def generate_summary(self, df):
        summary_path = os.path.join(self.output_dir, "compression_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA COMPRESSION BENCHMARK SUMMARY\n")
            f.write("Chameleon Cloud Experiment\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {self.system_info['timestamp']}\n")
            f.write(f"Host: {self.system_info['hostname']}\n\n")
            
            for file_type in df['File Type'].unique():
                file_data = df[df['File Type'] == file_type]
                best = file_data.loc[file_data['compression_ratio_float'].idxmin()]
                f.write(f"{file_type}:\n")
                f.write(f"  Best: {best['Algorithm']}\n")
                f.write(f"  Space Saved: {best['Space Saved']}\n\n")
        
        print(f"✓ Summary saved to: {summary_path}")
    
    def generate_visualizations(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        pivot_ratio = df.pivot_table(
            index='File Type', columns='Algorithm',
            values='compression_ratio_float'
        )
        pivot_ratio.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Compression Ratio')
        axes[0,0].set_ylabel('Ratio (Lower = Better)')
        
        pivot_saved = df.pivot_table(
            index='File Type', columns='Algorithm',
            values='space_saved_float'
        )
        pivot_saved.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Space Saved')
        axes[0,1].set_ylabel('Space Saved (Higher = Better)')
        
        heatmap_data = df.pivot_table(
            index='Algorithm', columns='File Type',
            values='compression_ratio_float'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', ax=axes[1,0])
        axes[1,0].set_title('Compression Ratio Heatmap')
        
        algo_summary = df.groupby('Algorithm')['space_saved_float'].mean()
        algo_summary.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Average Space Saved by Algorithm')
        axes[1,1].set_ylabel('Average Space Saved')
        
        plt.tight_layout()
        viz_path = os.path.join(self.output_dir, "analysis.png")
        plt.savefig(viz_path, dpi=300)
        print(f"✓ Visualizations saved to: {viz_path}")
        plt.close()

def main():
    print("\nDATA COMPRESSION BENCHMARK - CHAMELEON CLOUD\n")
    benchmark = CompressionBenchmark(output_dir="results")
    test_files = benchmark.create_test_files(data_dir="data")
    benchmark.run_benchmark(test_files)
    benchmark.save_results()
    print("\n✓ All tasks completed!\n")

if __name__ == "__main__":
    main()
