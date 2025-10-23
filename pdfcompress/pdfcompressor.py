#!/usr/bin/env python3
import os
import sys
import time
import lzma
import argparse
import zstandard as zstd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# COMPRESSION FUNCTIONS
def compress_with_lzma(input_file, output_file, preset=9):
    """
    Compress file using LZMA algorithm
    preset: 0-9, where 9 is maximum compression
    """
    try:
        print(f"  Compressing with LZMA (preset={preset})...")
        with open(input_file, 'rb') as f:
            data = f.read()
        
        start_time = time.time()
        compressed_data = lzma.compress(data, preset=preset)
        compression_time = time.time() - start_time
        
        with open(output_file, 'wb') as f:
            f.write(compressed_data)
        
        return len(data), len(compressed_data), compression_time, True
    
    except Exception as e:
        print(f"  Error in LZMA compression: {e}")
        return None, None, None, False

def compress_with_zstd(input_file, output_file, level=22):
    """
    Compress file using Zstandard algorithm
    level: 1-22, where 22 is maximum compression
    """
    try:
        print(f"  Compressing with Zstandard (level={level})...")
        with open(input_file, 'rb') as f:
            data = f.read()
        
        start_time = time.time()
        cctx = zstd.ZstdCompressor(level=level)
        compressed_data = cctx.compress(data)
        compression_time = time.time() - start_time
        
        with open(output_file, 'wb') as f:
            f.write(compressed_data)
        
        return len(data), len(compressed_data), compression_time, True
    
    except Exception as e:
        print(f"  Error in Zstandard compression: {e}")
        return None, None, None, False

def decompress_lzma(compressed_file):
    """Decompress LZMA file and verify integrity"""
    try:
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
        
        start_time = time.time()
        decompressed_data = lzma.decompress(compressed_data)
        decompression_time = time.time() - start_time
        
        return decompressed_data, decompression_time, True
    except Exception as e:
        print(f"  Error in LZMA decompression: {e}")
        return None, None, False

def decompress_zstd(compressed_file):
    """Decompress Zstandard file and verify integrity"""
    try:
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
        
        start_time = time.time()
        dctx = zstd.ZstdDecompressor()
        decompressed_data = dctx.decompress(compressed_data)
        decompression_time = time.time() - start_time
        
        return decompressed_data, decompression_time, True
    except Exception as e:
        print(f"  Error in Zstandard decompression: {e}")
        return None, None, False

def create_visualizations(df_pdf, pdf_filename):
    """Create and save visualization charts"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. File Size Comparison
    ax1 = plt.subplot(2, 3, 1)
    algorithms = df_pdf['Algorithm'].values
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_pdf['Original Size (MB)'].values, width, 
                    label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df_pdf['Compressed Size (MB)'].values, width, 
                    label='Compressed', color='#2ecc71', alpha=0.8)
    
    ax1.set_ylabel('File Size (MB)', fontsize=11, fontweight='bold')
    ax1.set_title(f'File Size Comparison\n{pdf_filename}', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Compression Ratio
    ax2 = plt.subplot(2, 3, 2)
    ratios = df_pdf['Compression Ratio'].values * 100
    bars = ax2.barh(algorithms, ratios, color=['#3498db', '#9b59b6'][:len(algorithms)], alpha=0.8)
    ax2.set_xlabel('Compression Ratio (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Compression Ratio\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax2.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Original size')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, ratio in zip(bars, ratios):
        ax2.text(ratio + 1, bar.get_y() + bar.get_height()/2,
                f'{ratio:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # 3. Space Saved
    ax3 = plt.subplot(2, 3, 3)
    space_saved = df_pdf['Space Saved'].values * 100
    bars = ax3.bar(algorithms, space_saved, color=['#2ecc71', '#27ae60'][:len(algorithms)], alpha=0.8)
    ax3.set_ylabel('Space Saved (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Space Saved\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Compression Time
    ax4 = plt.subplot(2, 3, 4)
    comp_times = df_pdf['Compression Time (s)'].values
    bars = ax4.bar(algorithms, comp_times, color=['#e67e22', '#d35400'][:len(algorithms)], alpha=0.8)
    ax4.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Compression Time\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Decompression Time
    ax5 = plt.subplot(2, 3, 5)
    decomp_times = df_pdf['Decompression Time (s)'].values
    bars = ax5.bar(algorithms, decomp_times, color=['#16a085', '#1abc9c'][:len(algorithms)], alpha=0.8)
    ax5.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_title('Decompression Time\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Efficiency Score
    ax6 = plt.subplot(2, 3, 6)
    efficiency = (df_pdf['Space Saved'] / df_pdf['Compression Time (s)']).values
    bars = ax6.bar(algorithms, efficiency, color=['#8e44ad', '#9b59b6'][:len(algorithms)], alpha=0.8)
    ax6.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
    ax6.set_title('Compression Efficiency\n(Space Saved per Second)', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    viz_filename = pdf_filename.replace('.pdf', '_compression_visualization.png')
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as '{viz_filename}'")
    
    return viz_filename

def print_detailed_analysis(df_pdf):
    """Print detailed analysis of compression results"""
    print("\n" + "="*80)
    print("DETAILED COMPRESSION ANALYSIS")
    print("="*80)
    
    for idx, row in df_pdf.iterrows():
        print(f"\n{'='*80}")
        print(f"ALGORITHM: {row['Algorithm']}")
        print(f"{'='*80}")
        
        print(f"\nFILE SIZE:")
        print(f"   Original: {row['Original Size (MB)']:.3f} MB ({row['Original Size (bytes)']:,} bytes)")
        print(f"   Compressed: {row['Compressed Size (MB)']:.3f} MB ({row['Compressed Size (bytes)']:,} bytes)")
        print(f"   Saved: {(row['Original Size (MB)'] - row['Compressed Size (MB)']):.3f} MB")
        
        print(f"\nCOMPRESSION METRICS:")
        print(f"   Compression Ratio: {row['Compression Ratio']:.2%}")
        print(f"   Space Saved: {row['Space Saved']:.2%}")
        
        print(f"\nPERFORMANCE:")
        print(f"   Compression Time: {row['Compression Time (s)']:.4f} seconds")
        print(f"   Decompression Time: {row['Decompression Time (s)']:.4f} seconds")
        print(f"   Total Time: {row['Compression Time (s)'] + row['Decompression Time (s)']:.4f} seconds")
        
        efficiency = row['Space Saved'] / row['Compression Time (s)']
        print(f"   Efficiency Score: {efficiency:.2f}")
        
        print(f"\nINTEGRITY:")
        print(f"   Data Verification: {row['Integrity Check']}")
        print(f"   Output File: {row['Output File']}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*80}")
    
    best_compression = df_pdf.loc[df_pdf['Compression Ratio'].idxmin()]
    fastest_comp = df_pdf.loc[df_pdf['Compression Time (s)'].idxmin()]
    fastest_decomp = df_pdf.loc[df_pdf['Decompression Time (s)'].idxmin()]
    most_efficient = df_pdf.loc[(df_pdf['Space Saved'] / df_pdf['Compression Time (s)']).idxmax()]
    
    print(f"\nBEST COMPRESSION RATIO:")
    print(f"   {best_compression['Algorithm']}: {best_compression['Compression Ratio']:.2%} ({best_compression['Space Saved']:.2%} saved)")
    
    print(f"\nFASTEST COMPRESSION:")
    print(f"   {fastest_comp['Algorithm']}: {fastest_comp['Compression Time (s)']:.4f} seconds")
    
    print(f"\nFASTEST DECOMPRESSION:")
    print(f"   {fastest_decomp['Algorithm']}: {fastest_decomp['Decompression Time (s)']:.4f} seconds")
    
    print(f"\nMOST EFFICIENT:")
    eff_score = most_efficient['Space Saved'] / most_efficient['Compression Time (s)']
    print(f"   {most_efficient['Algorithm']}: {eff_score:.2f} (space saved per second)")


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compress PDF files using LZMA and Zstandard algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pdf_compressor.py prem.pdf
  python3 pdf_compressor.py document.pdf
  python3 pdf_compressor.py /path/to/myfile.pdf
        """
    )
    parser.add_argument('pdf_file', help='PDF file to compress')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    pdf_filename = args.pdf_file
    
    print("="*80)
    print("PDF COMPRESSION TOOL - LZMA vs Zstandard")
    print("="*80)
    print(f"\nTarget PDF: {pdf_filename}")
    
    # Check if PDF exists
    if not os.path.exists(pdf_filename):
        print(f"\nError: '{pdf_filename}' not found!")
        print("Please ensure the PDF file exists and the path is correct.")
        sys.exit(1)
    
    # Get original file info
    original_size = os.path.getsize(pdf_filename)
    print(f"\nOriginal PDF Statistics:")
    print(f"   File: {pdf_filename}")
    print(f"   Size: {original_size:,} bytes ({original_size/(1024*1024):.3f} MB)")
    
    # Read original data for integrity check
    with open(pdf_filename, 'rb') as f:
        original_data = f.read()
    
    # Prepare output files
    base_name = pdf_filename.replace('.pdf', '')
    lzma_output = f"{base_name}_compressed.lzma"
    zstd_output = f"{base_name}_compressed.zst"
    
    # Store results
    compression_results = []
    
    # ========================================================================
    # LZMA COMPRESSION
    # ========================================================================
    
    print("\n" + "="*80)
    print("LZMA COMPRESSION")
    print("="*80)
    
    orig_size, comp_size, comp_time, success = compress_with_lzma(pdf_filename, lzma_output)
    
    if success and comp_size:
        print(f"\n  Verifying integrity...")
        decompressed_data, decomp_time, decomp_success = decompress_lzma(lzma_output)
        
        integrity_ok = (decompressed_data == original_data) if decomp_success else False
        
        compression_ratio = comp_size / orig_size
        space_saved = 1 - compression_ratio
        
        compression_results.append({
            'Algorithm': 'LZMA',
            'Original Size (bytes)': orig_size,
            'Compressed Size (bytes)': comp_size,
            'Original Size (MB)': orig_size / (1024 * 1024),
            'Compressed Size (MB)': comp_size / (1024 * 1024),
            'Compression Ratio': compression_ratio,
            'Space Saved': space_saved,
            'Compression Time (s)': comp_time,
            'Decompression Time (s)': decomp_time if decomp_success else None,
            'Integrity Check': 'PASS' if integrity_ok else 'FAIL',
            'Output File': lzma_output
        })
        
        print(f"\n✓ LZMA Compression Complete:")
        print(f"   Original Size:      {orig_size:,} bytes ({orig_size/(1024*1024):.3f} MB)")
        print(f"   Compressed Size:    {comp_size:,} bytes ({comp_size/(1024*1024):.3f} MB)")
        print(f"   Compression Ratio:  {compression_ratio:.2%}")
        print(f"   Space Saved:        {space_saved:.2%}")
        print(f"   Compression Time:   {comp_time:.4f} seconds")
        print(f"   Decompression Time: {decomp_time:.4f} seconds" if decomp_success else "   Decompression: Failed")
        print(f"   Integrity Check:    {'✓ PASS' if integrity_ok else '✗ FAIL'}")
        print(f"   Output File:        {lzma_output}")
    
    # ========================================================================
    # ZSTANDARD COMPRESSION
    # ========================================================================
    
    print("\n" + "="*80)
    print("ZSTANDARD COMPRESSION")
    print("="*80)
    
    orig_size, comp_size, comp_time, success = compress_with_zstd(pdf_filename, zstd_output)
    
    if success and comp_size:
        print(f"\n  Verifying integrity...")
        decompressed_data, decomp_time, decomp_success = decompress_zstd(zstd_output)
        
        integrity_ok = (decompressed_data == original_data) if decomp_success else False
        
        compression_ratio = comp_size / orig_size
        space_saved = 1 - compression_ratio
        
        compression_results.append({
            'Algorithm': 'Zstandard',
            'Original Size (bytes)': orig_size,
            'Compressed Size (bytes)': comp_size,
            'Original Size (MB)': orig_size / (1024 * 1024),
            'Compressed Size (MB)': comp_size / (1024 * 1024),
            'Compression Ratio': compression_ratio,
            'Space Saved': space_saved,
            'Compression Time (s)': comp_time,
            'Decompression Time (s)': decomp_time if decomp_success else None,
            'Integrity Check': 'PASS' if integrity_ok else 'FAIL',
            'Output File': zstd_output
        })
        
        print(f"\n✓ Zstandard Compression Complete:")
        print(f"   Original Size:      {orig_size:,} bytes ({orig_size/(1024*1024):.3f} MB)")
        print(f"   Compressed Size:    {comp_size:,} bytes ({comp_size/(1024*1024):.3f} MB)")
        print(f"   Compression Ratio:  {compression_ratio:.2%}")
        print(f"   Space Saved:        {space_saved:.2%}")
        print(f"   Compression Time:   {comp_time:.4f} seconds")
        print(f"   Decompression Time: {decomp_time:.4f} seconds" if decomp_success else "   Decompression: Failed")
        print(f"   Integrity Check:    {'✓ PASS' if integrity_ok else '✗ FAIL'}")
        print(f"   Output File:        {zstd_output}")
    
    # ========================================================================
    # RESULTS SUMMARY AND VISUALIZATION
    # ========================================================================
    
    if compression_results:
        df_pdf = pd.DataFrame(compression_results)
        
        print("\n" + "="*80)
        print("COMPRESSION COMPARISON SUMMARY")
        print("="*80)
        print(df_pdf[['Algorithm', 'Original Size (MB)', 'Compressed Size (MB)', 
                       'Compression Ratio', 'Space Saved', 'Compression Time (s)',
                       'Decompression Time (s)', 'Integrity Check']].to_string(index=False))
        
        # Save results to CSV
        csv_filename = f"{base_name}_compression_results.csv"
        df_pdf.to_csv(csv_filename, index=False)
        print(f"\n✓ Results saved to '{csv_filename}'")
        
        # Create visualizations
        if not args.no_viz:
            try:
                viz_filename = create_visualizations(df_pdf, pdf_filename)
            except Exception as e:
                print(f"⚠ Could not generate visualizations: {e}")
                viz_filename = None
        else:
            viz_filename = None
            print("\n⚠ Visualization generation skipped")
        
        # Print detailed analysis
        print_detailed_analysis(df_pdf)
        
        print("\n" + "="*80)
        print("✓ PDF COMPRESSION ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nGenerated Files:")
        print(f"  • {lzma_output}")
        print(f"  • {zstd_output}")
        print(f"  • {csv_filename}")
        if viz_filename:
            print(f"  • {viz_filename}")
    else:
        print("\n⚠ No compression results available. Check for errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
