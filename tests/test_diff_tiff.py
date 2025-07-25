"""
A simple script to visualize the difference between two TIFF files.

This script loads two single-band TIFF images, calculates their per-pixel
difference, and displays the two original images and the resulting difference
map side-by-side using matplotlib.

It assumes the two input TIFFs are perfectly aligned and have the same
dimensions, will crop if they don't but interpretability of results will be meh. 
Geospatial information is ignored.
"""
import argparse
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def diff_visualizer(file1_path, file2_path):
    """Loads two TIFFs, computes their difference, and visualizes the result."""

    # --- 1. Validate and Load Input Files ---
    for f in [file1_path, file2_path]:
        if not os.path.exists(f):
            print(f"Error: File not found at '{f}'")
            return

    try:
        with rasterio.open(file1_path) as src1:
            img1 = src1.read(1)
        with rasterio.open(file2_path) as src2:
            img2 = src2.read(1)
    except Exception as e:
        print(f"Error reading TIFF files: {e}")
        return

    # --- 2. Handle Different Dimensions ---
    original_shape1 = img1.shape
    original_shape2 = img2.shape
    
    if img1.shape != img2.shape:
        print(f"Warning: Input images have different dimensions.")
        print(f"  {os.path.basename(file1_path)}: {original_shape1}")
        print(f"  {os.path.basename(file2_path)}: {original_shape2}")
        print("  Cropping both images to the smaller dimensions and aligning to top-left corner.")
        
        # Determine the minimum dimensions
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        
        # Crop both images to the smaller dimensions (top-left alignment)
        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]
        
        print(f"  Cropped both images to: {img1.shape}")
    
    # Compute the difference. Convert to float to handle potential negative values.
    difference = img1.astype(np.float32) - img2.astype(np.float32)

    # --- 3. Visualize the Results ---
    fig = plt.figure(figsize=(18, 10))
    
    # Create a grid layout: 3 plots on top row, 1 histogram spanning bottom
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Top row - original 3 plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Bottom row - histogram spanning all columns
    ax4 = fig.add_subplot(gs[1, :])

    # Define a consistent colormap and range for the reference images
    cmap_ref = 'viridis'
    vmin_ref = 0
    vmax_ref = 100

    # Display Image 1
    im1 = ax1.imshow(img1, cmap=cmap_ref, vmin=vmin_ref, vmax=vmax_ref)
    title1 = os.path.basename(file1_path)
    if original_shape1 != img1.shape:
        title1 += f"\n(Original: {original_shape1}, Cropped: {img1.shape})"
    ax1.set_title(title1)
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1)
    cbar1.set_label("Pixel Value (0-100)")

    # Display Image 2
    im2 = ax2.imshow(img2, cmap=cmap_ref, vmin=vmin_ref, vmax=vmax_ref)
    title2 = os.path.basename(file2_path)
    if original_shape2 != img2.shape:
        title2 += f"\n(Original: {original_shape2}, Cropped: {img2.shape})"
    ax2.set_title(title2)
    ax2.set_xlabel('Width')
    ax2.set_yticklabels([]) # Hide y-axis labels to avoid clutter
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1)
    cbar2.set_label("Pixel Value (0-100)")

    # Display Difference
    # Use a diverging colormap to highlight positive and negative differences
    abs_max = np.max(np.abs(difference))
    im3 = ax3.imshow(difference, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)
    ax3.set_title('Difference (Image 1 - Image 2)')
    ax3.set_xlabel('Width')
    ax3.set_yticklabels([]) # Hide y-axis labels
    cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.1)
    cbar3.set_label('Pixel Value Difference')
    
    # Add histogram of differences at the bottom
    # Flatten the difference array and remove any NaN values
    diff_flat = difference.flatten()
    diff_flat = diff_flat[~np.isnan(diff_flat)]
    
    # Create histogram with bin widths of 1 unit
    min_diff = np.floor(np.min(diff_flat))
    max_diff = np.ceil(np.max(diff_flat))
    bins = np.arange(min_diff, max_diff + 1, 1)  # Bins with width of 1
    counts, bins, patches = ax4.hist(diff_flat, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Color the histogram bars: red for negative differences, blue for positive, green for near-zero
    bin_centers = (bins[:-1] + bins[1:]) / 2
    threshold = abs_max * 0.1  # Consider values within 10% of max as "near zero"
    
    for i, (patch, center) in enumerate(zip(patches, bin_centers)):
        if abs(center) <= threshold:
            patch.set_facecolor('lightgreen')  # Near zero differences
        elif center < 0:
            patch.set_facecolor('lightcoral')  # Negative differences
        else:
            patch.set_facecolor('lightblue')   # Positive differences
    
    ax4.set_xlabel('Pixel Value Difference')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Pixel Differences')
    ax4.grid(True, alpha=0.3)
    
    # Add vertical line at zero for reference
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero difference')
    
    # Add statistics text
    mean_diff = np.mean(diff_flat)
    std_diff = np.std(diff_flat)
    ax4.text(0.02, 0.98, f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}\nPixels: {len(diff_flat):,}', 
             transform=ax4.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.legend()

    fig.suptitle('TIFF Difference Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("Displaying plot... Close the plot window to exit.")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize the difference between two single-band TIFF files."
    )
    parser.add_argument(
        'file1',
        type=str,
        help="Path to the first input TIFF file."
    )
    parser.add_argument(
        'file2',
        type=str,
        help="Path to the second input TIFF file."
    )
    args = parser.parse_args()
    
    diff_visualizer(args.file1, args.file2) 