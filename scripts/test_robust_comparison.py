"""
Robust comparison script for prediction outputs with different dimensions.
Handles spatial mismatches by cropping to overlapping area and provides detailed analysis.
"""
import argparse
import numpy as np
import rasterio
from loguru import logger
import matplotlib.pyplot as plt
import sys

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level: <8} | {message}", level="INFO")

def load_raster(filepath):
    """Load raster and return data array and metadata"""
    logger.info(f"Loading {filepath}")
    
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile
        
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Data type: {data.dtype}")
    logger.info(f"  Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    logger.info(f"  Mean: {np.mean(data):.2f}")
    logger.info(f"  No-data pixels (255): {np.sum(data == 255):,}")
    
    return data, profile

def crop_to_common_area(arr1, arr2):
    """Crop both arrays to their common overlapping area"""
    min_height = min(arr1.shape[0], arr2.shape[0])
    min_width = min(arr1.shape[1], arr2.shape[1])
    
    logger.info(f"Cropping to common area: ({min_height}, {min_width})")
    
    arr1_cropped = arr1[:min_height, :min_width]
    arr2_cropped = arr2[:min_height, :min_width]
    
    return arr1_cropped, arr2_cropped

def analyze_differences(arr1, arr2, name1="Image 1", name2="Image 2"):
    """Detailed analysis of differences between two arrays"""
    logger.info("=" * 60)
    logger.info("DIFFERENCE ANALYSIS")
    logger.info("=" * 60)
    
    # Basic statistics
    diff = arr2 - arr1
    abs_diff = np.abs(diff)
    
    # Mask out no-data pixels (value 255) from analysis
    valid_mask = (arr1 != 255) & (arr2 != 255)
    valid_pixels = np.sum(valid_mask)
    
    if valid_pixels == 0:
        logger.warning("No valid pixels found for comparison!")
        return
    
    # Statistics on valid pixels only
    diff_valid = diff[valid_mask]
    abs_diff_valid = abs_diff[valid_mask]
    arr1_valid = arr1[valid_mask]
    arr2_valid = arr2[valid_mask]
    
    logger.info(f"Valid pixels for comparison: {valid_pixels:,} / {arr1.size:,} ({100*valid_pixels/arr1.size:.1f}%)")
    
    logger.info(f"\n{name1} statistics (valid pixels):")
    logger.info(f"  Mean: {np.mean(arr1_valid):.2f}")
    logger.info(f"  Std:  {np.std(arr1_valid):.2f}")
    logger.info(f"  Range: [{np.min(arr1_valid):.2f}, {np.max(arr1_valid):.2f}]")
    
    logger.info(f"\n{name2} statistics (valid pixels):")
    logger.info(f"  Mean: {np.mean(arr2_valid):.2f}")
    logger.info(f"  Std:  {np.std(arr2_valid):.2f}")
    logger.info(f"  Range: [{np.min(arr2_valid):.2f}, {np.max(arr2_valid):.2f}]")
    
    logger.info(f"\nDifference statistics:")
    logger.info(f"  Mean difference: {np.mean(diff_valid):.3f}")
    logger.info(f"  Mean absolute difference: {np.mean(abs_diff_valid):.3f}")
    logger.info(f"  Max absolute difference: {np.max(abs_diff_valid):.3f}")
    logger.info(f"  RMSE: {np.sqrt(np.mean(diff_valid**2)):.3f}")
    
    # Correlation
    correlation = np.corrcoef(arr1_valid, arr2_valid)[0, 1]
    logger.info(f"  Correlation: {correlation:.4f}")
    
    # Percentile analysis
    percentiles = [50, 90, 95, 99]
    logger.info(f"\nAbsolute difference percentiles:")
    for p in percentiles:
        val = np.percentile(abs_diff_valid, p)
        logger.info(f"  {p}th percentile: {val:.3f}")
    
    # Large differences
    large_diff_threshold = 5.0
    large_diffs = np.sum(abs_diff_valid > large_diff_threshold)
    logger.info(f"\nPixels with |difference| > {large_diff_threshold}: {large_diffs:,} ({100*large_diffs/valid_pixels:.2f}%)")
    
    return {
        'correlation': correlation,
        'mean_abs_diff': np.mean(abs_diff_valid),
        'rmse': np.sqrt(np.mean(diff_valid**2)),
        'max_abs_diff': np.max(abs_diff_valid),
        'valid_pixels': valid_pixels,
        'large_diffs': large_diffs
    }

def create_comparison_plots(arr1, arr2, name1="Original", name2="Temporal", output_prefix="comparison"):
    """Create visualization plots comparing the two arrays"""
    logger.info("Creating comparison plots...")
    
    # Mask no-data pixels
    valid_mask = (arr1 != 255) & (arr2 != 255)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Prediction Comparison: {name1} vs {name2}', fontsize=16)
    
    # Original prediction
    im1 = axes[0, 0].imshow(arr1, vmin=0, vmax=100, cmap='RdYlGn')
    axes[0, 0].set_title(f'{name1}\nMean: {np.mean(arr1[valid_mask]):.1f}%')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Temporal prediction  
    im2 = axes[0, 1].imshow(arr2, vmin=0, vmax=100, cmap='RdYlGn')
    axes[0, 1].set_title(f'{name2}\nMean: {np.mean(arr2[valid_mask]):.1f}%')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Difference map
    diff = arr2 - arr1
    diff_masked = np.where(valid_mask, diff, np.nan)
    max_diff = np.nanmax(np.abs(diff_masked))
    im3 = axes[0, 2].imshow(diff_masked, vmin=-max_diff, vmax=max_diff, cmap='RdBu_r')
    axes[0, 2].set_title(f'Difference ({name2} - {name1})\nMAE: {np.nanmean(np.abs(diff_masked)):.2f}%')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Scatter plot
    arr1_flat = arr1[valid_mask]
    arr2_flat = arr2[valid_mask]
    
    # Sample for scatter plot if too many points
    if len(arr1_flat) > 10000:
        indices = np.random.choice(len(arr1_flat), 10000, replace=False)
        arr1_sample = arr1_flat[indices]
        arr2_sample = arr2_flat[indices]
    else:
        arr1_sample = arr1_flat
        arr2_sample = arr2_flat
    
    axes[1, 0].scatter(arr1_sample, arr2_sample, alpha=0.5, s=1)
    axes[1, 0].plot([0, 100], [0, 100], 'r--', label='1:1 line')
    axes[1, 0].set_xlabel(f'{name1} (%)')
    axes[1, 0].set_ylabel(f'{name2} (%)')
    axes[1, 0].set_title(f'Scatter Plot\nR = {np.corrcoef(arr1_flat, arr2_flat)[0,1]:.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Difference histogram
    diff_flat = diff[valid_mask]
    axes[1, 1].hist(diff_flat, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Difference (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Difference Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    stats_text = f"""
    Correlation: {np.corrcoef(arr1_flat, arr2_flat)[0,1]:.4f}
    Mean Abs Diff: {np.mean(np.abs(diff_flat)):.2f}%
    RMSE: {np.sqrt(np.mean(diff_flat**2)):.2f}%
    Max Abs Diff: {np.max(np.abs(diff_flat)):.2f}%
    
    {name1} Mean: {np.mean(arr1_flat):.1f}%
    {name2} Mean: {np.mean(arr2_flat):.1f}%
    """
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_file = f'{output_prefix}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"Comparison plot saved: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Robust comparison of prediction outputs")
    parser.add_argument('file1', help="First prediction file (e.g., original pipeline output)")
    parser.add_argument('file2', help="Second prediction file (e.g., temporal pipeline output)")
    parser.add_argument('--name1', default="Original", help="Name for first dataset")
    parser.add_argument('--name2', default="Temporal", help="Name for second dataset")
    parser.add_argument('--plot', action='store_true', help="Create comparison plots")
    parser.add_argument('--output-prefix', default="comparison", help="Output prefix for plots")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ROBUST PREDICTION COMPARISON")
    logger.info("=" * 60)
    
    # Load both files
    try:
        arr1, profile1 = load_raster(args.file1)
        arr2, profile2 = load_raster(args.file2)
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return 1
    
    # Check if dimensions match
    if arr1.shape == arr2.shape:
        logger.success("Dimensions match perfectly!")
        arr1_analysis = arr1
        arr2_analysis = arr2
    else:
        logger.warning(f"Dimension mismatch: {args.file1} is {arr1.shape}, {args.file2} is {arr2.shape}")
        logger.info("Cropping to common overlapping area...")
        arr1_analysis, arr2_analysis = crop_to_common_area(arr1, arr2)
    
    # Perform detailed analysis
    stats = analyze_differences(arr1_analysis, arr2_analysis, args.name1, args.name2)
    
    # Create plots if requested
    if args.plot:
        plot_file = create_comparison_plots(arr1_analysis, arr2_analysis, 
                                          args.name1, args.name2, args.output_prefix)
    
    # Summary assessment
    logger.info("=" * 60)
    logger.info("ASSESSMENT SUMMARY")
    logger.info("=" * 60)
    
    if stats:
        correlation = stats['correlation']
        mean_abs_diff = stats['mean_abs_diff']
        
        if correlation > 0.95 and mean_abs_diff < 2.0:
            logger.success("🟢 EXCELLENT: Very high correlation and low differences")
        elif correlation > 0.90 and mean_abs_diff < 5.0:
            logger.info("🟡 GOOD: High correlation with moderate differences")
        elif correlation > 0.80:
            logger.warning("🟠 FAIR: Reasonable correlation but notable differences")
        else:
            logger.error("🔴 POOR: Low correlation or large differences")
        
        logger.info(f"Key metrics:")
        logger.info(f"  • Correlation: {correlation:.4f}")
        logger.info(f"  • Mean absolute difference: {mean_abs_diff:.2f}%")
        logger.info(f"  • RMSE: {stats['rmse']:.2f}%")
    
    logger.info("=" * 60)
    return 0

if __name__ == '__main__':
    exit(main()) 