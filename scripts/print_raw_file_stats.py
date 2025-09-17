

import numpy as np
import hickle as hkl
import os
import sys
import argparse
import re
import datetime as dt
from loguru import logger

def _p(a, q):
    try:
        return float(np.nanpercentile(a, q))
    except Exception:
        return float("nan")


def load_raw(raw_data_folder: str, tile_id: str):
    P = dict(
        clouds     = f"{raw_data_folder}/clouds/clouds_{tile_id}.hkl",
        cloudmask  = f"{raw_data_folder}/clouds/cloudmask_{tile_id}.hkl",
        s2_10      = f"{raw_data_folder}/s2_10/{tile_id}.hkl",
        s2_20      = f"{raw_data_folder}/s2_20/{tile_id}.hkl",
        s2_dates   = f"{raw_data_folder}/misc/s2_dates_{tile_id}.hkl",
        s1         = f"{raw_data_folder}/s1/{tile_id}.hkl",
        s1_dates   = f"{raw_data_folder}/misc/s1_dates_{tile_id}.hkl",
        dem        = f"{raw_data_folder}/misc/dem_{tile_id}.hkl",
    )
    clouds = hkl.load(P["clouds"]).astype(np.float32) if os.path.exists(P["clouds"]) else None
    cloudmask = hkl.load(P["cloudmask"]).astype(np.uint8) if os.path.exists(P["cloudmask"]) else None
    s2_10_u16 = hkl.load(P["s2_10"])     # (T,H10,W10,4) uint16
    s2_20_u16 = hkl.load(P["s2_20"])     # (T,H20,W20,6) uint16
    s2_dates  = np.array(hkl.load(P["s2_dates"]), dtype=np.int32)
    s1_u16    = hkl.load(P["s1"]) if os.path.exists(P["s1"]) else None
    s1_dates  = np.array(hkl.load(P["s1_dates"])) if os.path.exists(P["s1_dates"]) else None
    dem       = hkl.load(P["dem"]).astype(np.float32) if os.path.exists(P["dem"]) else None
    return clouds, cloudmask, s2_10_u16, s2_20_u16, s2_dates, s1_u16, s1_dates, dem



def log_array_stats(tag: str, arr: np.ndarray, channel_names=None, channel_axis: int = -1, single_channel: bool = False):
    """
    Logs basic stats per channel.

    By default, treats the last dimension as the channel dimension. Set channel_axis=0
    to summarize by the first dimension instead (e.g., time is the first axis).
    """
    if arr is None:
        logger.debug(f"{tag}: <None>")
        return
    a = np.asarray(arr)
    if single_channel:
        a = a.reshape((-1, 1))
    else:
        # Move the requested channel axis to the last position for uniform handling
        if channel_axis != -1 and channel_axis != (a.ndim - 1):
            try:
                a = np.moveaxis(a, channel_axis, -1)
            except Exception:
                logger.debug(f"{tag}: shape={arr.shape} (unable to move axis {channel_axis} to last)")
                return
        # Collapse all but the last (channel) dimension
        if a.ndim >= 2:
            a = a.reshape((-1, a.shape[-1]))
        elif a.ndim == 1:
            a = a.reshape((-1, 1))
        else:
            logger.debug(f"{tag}: shape={arr.shape}")
            return

    C = a.shape[-1]
    logger.debug(f"{tag}: shape={arr.shape}")
    for c in range(C):
        v = a[:, c]
        try:
            nz = v[~np.isnan(v)]
        except Exception:
            nz = v
        name = channel_names[c] if (channel_names and c < len(channel_names)) else f"c{c}"
        logger.debug(
            f"  {name:>8s}: min={np.nanmin(v):.4f} p1={_p(v,1):.4f} p50={_p(v,50):.4f} p99={_p(v,99):.4f} "
            f"max={np.nanmax(v):.4f} mean={np.nanmean(v):.4f} nan%={100.0*(len(v)-len(nz))/len(v):.2f}"
        )

# logging order:
# s2_dates, s1_dates, clouds, cloudmask, s2_10_u16, s2_20_u16, s1_u16, dem
# 
# 

def _find_tile_ids(raw_data_folder: str):
    """Best-effort discovery of tile IDs present under the raw data folder."""
    candidates = set()
    try:
        for name in os.listdir(os.path.join(raw_data_folder, "s2_10")):
            if name.endswith(".hkl"):
                candidates.add(os.path.splitext(name)[0])
    except Exception:
        pass
    try:
        for name in os.listdir(os.path.join(raw_data_folder, "s2_20")):
            if name.endswith(".hkl"):
                candidates.add(os.path.splitext(name)[0])
    except Exception:
        pass
    try:
        for name in os.listdir(os.path.join(raw_data_folder, "s1")):
            if name.endswith(".hkl"):
                candidates.add(os.path.splitext(name)[0])
    except Exception:
        pass
    try:
        for name in os.listdir(os.path.join(raw_data_folder, "clouds")):
            if name.startswith("clouds_") and name.endswith(".hkl"):
                candidates.add(name[len("clouds_"):-4])
    except Exception:
        pass
    try:
        for name in os.listdir(os.path.join(raw_data_folder, "misc")):
            if name.startswith("s2_dates_") and name.endswith(".hkl"):
                candidates.add(name[len("s2_dates_"):-4])
            if name.startswith("dem_") and name.endswith(".hkl"):
                candidates.add(name[len("dem_"):-4])
    except Exception:
        pass
    return sorted(candidates)


def _log_dates_stats(tag: str, arr: np.ndarray):
    if arr is None:
        logger.debug(f"{tag}: <None>")
        return
    a = np.asarray(arr)
    try:
        logger.debug(
            f"{tag}: shape={a.shape} min={np.nanmin(a)} max={np.nanmax(a)} mean={np.nanmean(a):.2f} count={a.size}"
        )
        logger.debug(
            f"  {a}"
        )
    except Exception:
        logger.debug(f"{tag}: shape={a.shape}")


def _plot_hist_for_channels(tag: str, arr: np.ndarray, channel_names=None, bins: int = 256):
    """Plot per-channel histograms over 0..65535 for arrays with channels in the last dim.

    - Supports arrays shaped (..., C). Collapses all but the last axis.
    - Skips if arr is None.
    - Uses lazy import of matplotlib to avoid dependency unless needed.
    """
    if arr is None:
        logger.debug(f"{tag} hist: <None>")
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error(f"Matplotlib not available; cannot plot histograms: {e}")
        return

    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape((-1, 1))
    elif a.ndim >= 2:
        a = a.reshape((-1, a.shape[-1]))
    else:
        logger.debug(f"{tag} hist: unsupported shape {a.shape}")
        return

    num_channels = a.shape[-1]
    num_rows = num_channels
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, max(2, num_rows * 2)), sharex=True)
    if num_rows == 1:
        axes = [axes]
    for c in range(num_channels):
        v = a[:, c].astype(np.float64, copy=False)
        try:
            v = v[~np.isnan(v)]
        except Exception:
            pass
        name = channel_names[c] if (channel_names and c < len(channel_names)) else f"c{c}"
        axes[c].hist(v, bins=bins, range=(0, 65535), color="steelblue", alpha=0.85)
        axes[c].set_ylabel(name)
        axes[c].grid(True, linestyle=":", alpha=0.3)
    axes[-1].set_xlabel("DN (0..65535)")
    fig.suptitle(f"{tag} per-channel histograms")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        import matplotlib.pyplot as plt  # re-import for environments that require top-level reference
        plt.show()
    except Exception:
        pass


def _plot_time_hw_c_hist_and_median_grid(groups, outfile: str = None, bins: int = 256, dates_info: dict = None, base_year: int = None, dem: np.ndarray = None):
    """Create a single-page figure showing per-channel histograms and median images.

    groups: list of tuples (tag: str, arr: np.ndarray, channel_names: list[str] | None)
            Each arr is expected to be shaped (T, H, W, C). If None, it is skipped.
    outfile: If provided, saves a single-page PDF to this path; otherwise shows interactively.
    bins: Histogram bins; range is fixed to (0, 65535).
    """
    # Count total rows (channels)
    total_channels = 0
    for _, arr, _ in groups:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.ndim != 4:
            logger.warning(f"Skipping array with shape {a.shape}: expected (T,H,W,C)")
            continue
        total_channels += a.shape[-1]

    # Add one extra row for DEM if provided
    add_dem_row = dem is not None and np.asarray(dem).ndim == 2
    total_rows = total_channels + (1 if add_dem_row else 0)

    if total_channels == 0:
        logger.debug("No channels available to plot.")
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:
        logger.error(f"Matplotlib not available; cannot plot: {e}")
        return

    fig, axes = plt.subplots(total_rows, 2, figsize=(12, max(3, total_rows * 2)), squeeze=False)
    stats_rows = []
    row = 0
    for tag, arr, channel_names in groups:
        if arr is None:
            continue
        A = np.asarray(arr)
        if A.ndim != 4:
            logger.warning(f"Skipping {tag} with shape {A.shape}: expected (T,H,W,C)")
            continue
        T, H, W, C = A.shape
        for c in range(C):
            # Histogram
            v_full = A[..., c].reshape(-1).astype(np.float64, copy=False)
            try:
                v = v_full[~np.isnan(v_full)]
            except Exception:
                v = v_full
            ch_name = channel_names[c] if (channel_names and c < len(channel_names)) else f"c{c}"
            ax_hist = axes[row, 0]
            ax_hist.hist(v, bins=bins, range=(0, 65535), color="steelblue", alpha=0.85)
            ax_hist.set_ylabel(ch_name)
            ax_hist.set_title(f"{tag} {ch_name} shape={A.shape}")
            ax_hist.grid(True, linestyle=":", alpha=0.3)

            # Median image over time
            median_img = np.nanmedian(A[..., c], axis=0)
            ax_img = axes[row, 1]
            ax_img.imshow(median_img, vmin=0, vmax=65535, cmap="turbo")
            ax_img.set_axis_off()

            # Collect stats for table
            try:
                v_min = float(np.nanmin(v)) if v.size else float("nan")
                v_p1 = float(np.nanpercentile(v, 1)) if v.size else float("nan")
                v_p50 = float(np.nanpercentile(v, 50)) if v.size else float("nan")
                v_p99 = float(np.nanpercentile(v, 99)) if v.size else float("nan")
                v_max = float(np.nanmax(v)) if v.size else float("nan")
                v_mean = float(np.nanmean(v)) if v.size else float("nan")
                nan_pct = 100.0 * (len(v_full) - len(v)) / len(v_full) if len(v_full) else float("nan")
            except Exception:
                v_min = v_p1 = v_p50 = v_p99 = v_max = v_mean = nan_pct = float("nan")
            stats_rows.append([
                tag, ch_name,
                f"{v_min:.1f}", f"{v_p1:.1f}", f"{v_p50:.1f}", f"{v_p99:.1f}", f"{v_max:.1f}", f"{v_mean:.2f}", f"{nan_pct:.2f}%"
            ])

            row += 1

    # Optional DEM row at the bottom
    if add_dem_row:
        D = np.asarray(dem)
        dem_row = total_rows - 1
        v_full = D.reshape(-1).astype(np.float64, copy=False)
        try:
            v = v_full[~np.isnan(v_full)]
        except Exception:
            v = v_full
        ax_hist = axes[dem_row, 0]
        ax_hist.hist(v, bins=bins, color="dimgray", alpha=0.85)
        ax_hist.set_ylabel("DEM")
        ax_hist.set_title(f"dem shape={D.shape}")
        ax_hist.grid(True, linestyle=":", alpha=0.3)

        ax_img = axes[dem_row, 1]
        vmin = float(np.nanmin(D)) if v.size else None
        vmax = float(np.nanmax(D)) if v.size else None
        ax_img.imshow(D, vmin=vmin, vmax=vmax, cmap="terrain")
        ax_img.set_axis_off()

    axes[-1, 0].set_xlabel("DN (0..65535)")
    fig.suptitle("Per-channel histograms and median images")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outfile:
        try:
            with PdfPages(outfile) as pdf:
                pdf.savefig(fig)

                # Table + dates page
                num_rows = len(stats_rows)
                table_height = max(3, 0.4 * (num_rows + 2))
                text_height = 2.2
                total_height = table_height + text_height
                fig2, (ax_table, ax_text) = plt.subplots(
                    2, 1, figsize=(12, total_height), gridspec_kw={"height_ratios": [table_height, text_height]}
                )
                ax_table.axis('off')
                col_labels = ["Dataset", "Channel", "Min", "P1", "P50", "P99", "Max", "Mean", "NaN %"]
                table = ax_table.table(cellText=stats_rows, colLabels=col_labels, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.3)
                ax_table.set_title("Channel summary statistics")

                # Dates pretty print
                ax_text.axis('off')
                def _format_doy_block(name: str, arr: np.ndarray):
                    if arr is None:
                        return f"{name}: <None>"
                    vals = np.asarray(arr).reshape(-1)
                    if vals.size == 0:
                        return f"{name}: <empty>"
                    # Unique sorted DOY
                    try:
                        doys = np.unique(vals.astype(int))
                    except Exception:
                        doys = np.unique(vals)
                    # Determine leap behavior using inferred base year if provided
                    leap = False
                    mmdd = []
                    iso_dates = []
                    if base_year is not None:
                        start = dt.date(base_year, 1, 1)
                        leap = (base_year % 4 == 0 and (base_year % 100 != 0 or base_year % 400 == 0))
                        for d in doys:
                            try:
                                date = start + dt.timedelta(days=int(d) - 1)
                                mmdd.append(date.strftime("%m-%d"))
                                iso_dates.append(date.isoformat())
                            except Exception:
                                mmdd.append("??-??")
                                iso_dates.append("????-??-??")
                    else:
                        # Fallback to month/day approximation if year not known
                        leap = np.max(doys) >= 366
                        days_in_month = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                        for d in doys:
                            di = int(d)
                            if di <= 0:
                                mmdd.append("??-??")
                                iso_dates.append("????-??-??")
                                continue
                            rem = di
                            month = 1
                            for dim in days_in_month:
                                if rem > dim:
                                    rem -= dim
                                    month += 1
                                else:
                                    break
                            if month > 12 or rem <= 0:
                                mmdd.append("??-??")
                                iso_dates.append("????-??-??")
                            else:
                                mmdd.append(f"{month:02d}-{rem:02d}")
                                iso_dates.append(f"????-{month:02d}-{rem:02d}")
                    # Chunk lines for readability
                    def chunk_list(xs, n):
                        for i in range(0, len(xs), n):
                            yield xs[i:i+n]
                    lines = [f"{name}: {len(doys)} dates (DOY):"]
                    for chunk in chunk_list([str(int(x)) for x in doys], 20):
                        lines.append("  " + ", ".join(chunk))
                    lines.append(f"{name} as MM-DD (leap={leap}):")
                    for chunk in chunk_list(mmdd, 20):
                        lines.append("  " + ", ".join(chunk))
                    if base_year is not None:
                        lines.append(f"{name} as ISO using base year {base_year}:")
                        for chunk in chunk_list(iso_dates, 8):
                            lines.append("  " + ", ".join(chunk))
                    return "\n".join(lines)

                s2_text = _format_doy_block("s2_dates", (dates_info or {}).get("s2_dates"))
                s1_text = _format_doy_block("s1_dates", (dates_info or {}).get("s1_dates"))
                full_text = s2_text + "\n\n" + s1_text
                ax_text.text(0.01, 0.98, full_text, va='top', ha='left', family='monospace', fontsize=9)

                fig2.tight_layout()
                pdf.savefig(fig2)
                plt.close(fig2)

            logger.info(f"Saved plots to {outfile}")
        except Exception as e:
            logger.error(f"Failed to save PDF to {outfile}: {e}")
        finally:
            plt.close(fig)
    else:
        try:
            plt.show()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Log basic stats for raw tile arrays under a directory.")
    parser.add_argument("--directory", required=True, help="Path to raw data folder containing subdirs like s2_10, s2_20, clouds, misc, s1")
    parser.add_argument("--hist", action="store_true", help="Show per-channel histograms for s2_10, s2_20, and s1 (0..65535 x-axis)")
    parser.add_argument("--outfile", help="Write plots to a single-page PDF at this path")
    args = parser.parse_args()

    # Ensure debug logs are visible
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    raw_dir = args.directory
    if not os.path.isdir(raw_dir):
        logger.error(f"Provided --directory does not exist or is not a directory: {raw_dir}")
        sys.exit(2)

    tile_ids = _find_tile_ids(raw_dir)
    if not tile_ids:
        logger.error(f"Could not infer any tile_id under directory: {raw_dir}")
        sys.exit(3)
    tile_id = tile_ids[0]
    if len(tile_ids) > 1:
        logger.warning(f"Multiple tile IDs found {tile_ids}; proceeding with the first: {tile_id}")

    logger.info(f"Logging stats for tile_id={tile_id} under directory={raw_dir}")

    clouds, cloudmask, s2_10_u16, s2_20_u16, s2_dates, s1_u16, s1_dates, dem = load_raw(raw_dir, tile_id)

    # Log in the specified order
    _log_dates_stats("s2_dates", s2_dates)
    _log_dates_stats("s1_dates", s1_dates)
    log_array_stats("clouds", clouds, channel_axis=0)
    log_array_stats("cloudmask", cloudmask, channel_axis=0)
    log_array_stats("s2_10_u16", s2_10_u16, channel_names=["B02", "B03", "B04", "B08"])
    log_array_stats("s2_20_u16", s2_20_u16, channel_names=["B05", "B06", "B07", "B8A", "B11", "B12"])
    log_array_stats("s1_u16", s1_u16, channel_names=["VV", "VH"])
    log_array_stats("dem", dem, single_channel=True)

    if args.hist or args.outfile:
        groups = [
            ("s2_10_u16", s2_10_u16, ["B02", "B03", "B04", "B08"]),
            ("s2_20_u16", s2_20_u16, ["B05", "B06", "B07", "B8A", "B11", "B12"]),
            ("s1_u16", s1_u16, ["VV", "VH"]),
        ]
        # Try to infer base year from directory path (any 4-digit year component)
        base_year = None
        try:
            years = [int(y) for y in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", os.path.abspath(raw_dir))]
            if years:
                base_year = years[0]
        except Exception:
            base_year = None

        _plot_time_hw_c_hist_and_median_grid(
            groups,
            outfile=args.outfile,
            bins=256,
            dates_info={
                "s2_dates": s2_dates,
                "s1_dates": s1_dates,
            },
            base_year=base_year,
            dem=dem,
        )


if __name__ == "__main__":
    main()
