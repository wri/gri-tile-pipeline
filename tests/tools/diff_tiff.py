"""
A simple script to visualize the difference between two 2D arrays.

Supports inputs as:
- GeoTIFF: .tif/.tiff (reads band 1)
- NumPy: .npy
- Hickle: .hkl/.hlk (expects a NumPy array saved via hickle)

This script loads two inputs, ensures they are 2D arrays (you can specify a
slice to reduce higher-dimensional arrays; otherwise we select the first
band/slice heuristically), calculates their per-pixel difference,
and displays the two originals and the resulting difference map side-by-side
using matplotlib.

It assumes the two inputs are perfectly aligned and have the same dimensions,
will crop if they don't but interpretability of results will be meh. Geospatial
information is ignored.
"""
import argparse
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def _to_2d_array(arr):
    """Ensure a 2D array. If 3D, pick the first band/slice heuristically."""
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr)
        except Exception:
            return None

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Heuristics: treat shape as (bands, rows, cols) if first dim is small
        if arr.shape[0] <= 12 and arr.shape[1] >= 32 and arr.shape[2] >= 32:
            return arr[0]
        # Or (rows, cols, bands)
        if arr.shape[2] <= 12 and arr.shape[0] >= 32 and arr.shape[1] >= 32:
            return arr[:, :, 0]
        # Fallback: first slice on axis 0
        return arr[0]

    # If 1D or other shapes, cannot visualize as image
    return None

def _parse_slice_spec(slice_spec_str, ndim):
    """Parse a Python-like slice spec string (e.g., "0,:,:,0") to a tuple usable in numpy indexing.

    Rules:
    - Comma-separated tokens for each dimension.
    - Token can be an integer (e.g., "0") or ":" meaning full slice.
    - Fewer tokens than ndim: remaining dims get ":".
    - More tokens than ndim: ignored extras.
    """
    if slice_spec_str is None:
        return None

    tokens = [t.strip() for t in str(slice_spec_str).split(',')]
    index_elems = []
    for i in range(min(len(tokens), ndim)):
        tok = tokens[i]
        if tok == ':' or tok == '':
            index_elems.append(slice(None))
        else:
            try:
                index_elems.append(int(tok))
            except ValueError:
                # unsupported token, fallback to full slice
                index_elems.append(slice(None))
    # pad with full slices if needed
    while len(index_elems) < ndim:
        index_elems.append(slice(None))
    return tuple(index_elems)

def _apply_slice(arr, slice_spec_str):
    if slice_spec_str is None:
        return arr
    spec = _parse_slice_spec(slice_spec_str, arr.ndim)
    try:
        sliced = arr[spec]
    except Exception:
        return arr
    return np.squeeze(sliced)

def load_array_from_path(path):
    """Load a NumPy array from .tif/.tiff, .npy, or .hkl/.hlk files.

    - .tif/.tiff: reads band 1 (2D). For multi-band slicing, use formats that load full arrays.
    - .npy/.hkl/.hlk: returns the raw array (can be 2D/3D/4D).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.tif', '.tiff'}:
        with rasterio.open(path) as src:
            arr = src.read(1)
        return arr
    if ext in {'.npy'}:
        arr = np.load(path, allow_pickle=False)
        return arr
    if ext in {'.hkl', '.hlk', '.hickle'}:
        try:
            import hickle as hkl
        except Exception as e:
            print("Error: hickle is required to read .hkl/.hlk files. Install with 'pip install hickle'.")
            print(f"Import error: {e}")
            return None
        arr = hkl.load(path)
        return arr

    print(f"Error: Unsupported file extension '{ext}'. Supported: .tif/.tiff, .npy, .hkl/.hlk")
    return None

def diff_visualizer(file1_path, file2_path, slice_spec=None):
    """Loads two arrays, computes their difference, and visualizes the result."""

    # --- 1. Validate and Load Input Files ---
    for f in [file1_path, file2_path]:
        if not os.path.exists(f):
            print(f"Error: File not found at '{f}'")
            return

    try:
        img1 = load_array_from_path(file1_path)
        img2 = load_array_from_path(file2_path)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Optionally apply user-provided slicing, then coerce to 2D
    if img1 is not None:
        img1 = _apply_slice(img1, slice_spec)
        img1 = _to_2d_array(img1)
    if img2 is not None:
        img2 = _apply_slice(img2, slice_spec)
        img2 = _to_2d_array(img2)

    if img1 is None or img2 is None:
        print("Error: Failed to load one or both inputs as 2D arrays.")
        return

    # --- 2. Handle Different Dimensions ---
    original_shape1 = img1.shape
    original_shape2 = img2.shape
    
    if img1.shape != img2.shape:
        print("Warning: Input images have different dimensions.")
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
    # Robust min/max across both images (2nd-98th percentiles), fallback to min/max
    combined = np.concatenate([img1.astype(np.float32).ravel(), img2.astype(np.float32).ravel()])
    combined = combined[~np.isnan(combined)]
    if combined.size > 0:
        try:
            vmin_ref = float(np.percentile(combined, 2))
            vmax_ref = float(np.percentile(combined, 98))
        except Exception:
            vmin_ref = float(np.nanmin(combined))
            vmax_ref = float(np.nanmax(combined))
    else:
        vmin_ref, vmax_ref = 0.0, 100.0
    if not np.isfinite(vmin_ref) or not np.isfinite(vmax_ref) or vmin_ref == vmax_ref:
        vmin_ref, vmax_ref = 0.0, 100.0

    # Display Image 1
    im1 = ax1.imshow(img1, cmap=cmap_ref, vmin=vmin_ref, vmax=vmax_ref)
    title1 = os.path.basename(file1_path)
    if original_shape1 != img1.shape:
        title1 += f"\n(Original: {original_shape1}, Cropped: {img1.shape})"
    if slice_spec:
        title1 += f"\nSlice: {slice_spec}"
    ax1.set_title(title1)
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1)
    cbar1.set_label("Pixel Value")

    # Display Image 2
    im2 = ax2.imshow(img2, cmap=cmap_ref, vmin=vmin_ref, vmax=vmax_ref)
    title2 = os.path.basename(file2_path)
    if original_shape2 != img2.shape:
        title2 += f"\n(Original: {original_shape2}, Cropped: {img2.shape})"
    if slice_spec:
        title2 += f"\nSlice: {slice_spec}"
    ax2.set_title(title2)
    ax2.set_xlabel('Width')
    ax2.set_yticklabels([]) # Hide y-axis labels to avoid clutter
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1)
    cbar2.set_label("Pixel Value")

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

    fig.suptitle('Array Difference Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("Displaying plot... Close the plot window to exit.")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize the difference between two arrays (.tif/.tiff, .npy, .hkl/.hlk)."
    )
    parser.add_argument(
        'file1',
        type=str,
        help="Path to the first input file (.tif/.tiff, .npy, .hkl/.hlk)."
    )
    parser.add_argument(
        'file2',
        type=str,
        help="Path to the second input file (.tif/.tiff, .npy, .hkl/.hlk)."
    )
    parser.add_argument(
        '--slice',
        dest='slice_spec',
        type=str,
        default=None,
        help=(
            "Optional Python-like slice spec applied to both inputs before visualization. "
            "Example for 4D (T,H,W,C): '0,:,:,0' selects T=0 and C=0."
        )
    )
    args = parser.parse_args()
    
    diff_visualizer(args.file1, args.file2, slice_spec=args.slice_spec) 