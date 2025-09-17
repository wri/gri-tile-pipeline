"""
A simple script to visualize the difference between two TIFF files.

This script loads two single-band TIFF images, calculates their per-pixel
difference, and displays the two original images and the resulting difference
map side-by-side using matplotlib.

It assumes the two input TIFFs are perfectly aligned and have the same
dimensions. Geospatial information is ignored.
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

    # --- 2. Check Compatibility and Compute Difference ---
    if img1.shape != img2.shape:
        print(
            "Error: Input images have different dimensions. "
            f"{os.path.basename(file1_path)} is {img1.shape}, but "
            f"{os.path.basename(file2_path)} is {img2.shape}."
        )
        return
    
    # Compute the difference. Convert to float to handle potential negative values.
    difference = img1.astype(np.float32) - img2.astype(np.float32)

    # --- 3. Visualize the Results ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define a consistent colormap and range for the reference images
    cmap_ref = 'viridis'
    vmin_ref = 0
    vmax_ref = 100

    # Display Image 1
    im1 = axes[0].imshow(img1, cmap=cmap_ref, vmin=vmin_ref, vmax=vmax_ref)
    axes[0].set_title(os.path.basename(file1_path))
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1)
    cbar1.set_label("Pixel Value (0-100)")

    # Display Image 2
    im2 = axes[1].imshow(img2, cmap=cmap_ref, vmin=vmin_ref, vmax=vmax_ref)
    axes[1].set_title(os.path.basename(file2_path))
    axes[1].set_xlabel('Width')
    axes[1].set_yticklabels([]) # Hide y-axis labels to avoid clutter
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1)
    cbar2.set_label("Pixel Value (0-100)")

    # Display Difference
    # Use a diverging colormap to highlight positive and negative differences
    abs_max = np.max(np.abs(difference))
    im3 = axes[2].imshow(difference, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)
    axes[2].set_title('Difference (Image 1 - Image 2)')
    axes[2].set_xlabel('Width')
    axes[2].set_yticklabels([]) # Hide y-axis labels
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.1)
    cbar3.set_label('Pixel Value Difference')

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