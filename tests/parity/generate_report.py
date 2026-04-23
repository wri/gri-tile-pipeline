#!/usr/bin/env python3
"""Generate a self-contained HTML parity report for golden tiles.

Usage:
    uv run python tests/parity/generate_report.py
    uv run python tests/parity/generate_report.py --save-intermediates

Outputs: temp/parity_report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "loaders"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from conftest import GOLDEN_DIR, GOLDEN_RAW, GOLDEN_TILES, MODEL_DIR
from parity.metrics import aggregate_golden_report, compare_predictions


def load_golden_tile(tile_name: str) -> dict:
    import hickle as hkl
    result = {
        "s2_10": hkl.load(str(GOLDEN_RAW / "s2_10" / f"{tile_name}.hkl")),
        "s2_20": hkl.load(str(GOLDEN_RAW / "s2_20" / f"{tile_name}.hkl")),
        "s1": hkl.load(str(GOLDEN_RAW / "s1" / f"{tile_name}.hkl")),
        "dem": hkl.load(str(GOLDEN_RAW / "misc" / f"dem_{tile_name}.hkl")),
        "clouds": hkl.load(str(GOLDEN_RAW / "clouds" / f"clouds_{tile_name}.hkl")),
        "s2_dates": hkl.load(str(GOLDEN_RAW / "misc" / f"s2_dates_{tile_name}.hkl")),
    }
    clm_path = GOLDEN_RAW / "clouds" / f"cloudmask_{tile_name}.hkl"
    if clm_path.exists():
        result["clm"] = hkl.load(str(clm_path))
    return result


def load_reference_tif(tile_name: str) -> np.ndarray:
    import rasterio
    with rasterio.open(str(GOLDEN_DIR / f"{tile_name}_FINAL.tif")) as src:
        return src.read(1)


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return b64


def _turbo_png_b64(arr: np.ndarray, vmin: float = 0, vmax: float = 100) -> str:
    """Render a 2D uint8 array as a pixel-perfect turbo-colormapped PNG.

    Nodata (==255) becomes transparent so overlays blink cleanly without
    the nodata pixels flickering.
    """
    from matplotlib import cm
    from PIL import Image

    valid = arr != 255
    vals = np.clip(arr.astype(np.float32), vmin, vmax)
    norm = (vals - vmin) / (vmax - vmin)
    rgba = cm.turbo(norm)  # (H, W, 4) float in [0, 1]
    rgba[~valid] = 0  # transparent nodata

    img = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_id(s: str) -> str:
    """Sanitize a tile label for use as an HTML id."""
    import re as _re
    return _re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def make_tile_section(tile_name: str, pred: np.ndarray, ref: np.ndarray,
                      stats: dict, intermediates: dict | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    h = min(pred.shape[0], ref.shape[0])
    w = min(pred.shape[1], ref.shape[1])
    pred_c = pred[:h, :w].astype(np.float32)
    ref_c = ref[:h, :w].astype(np.float32)

    valid = (ref_c != 255) & (pred_c != 255)
    pred_disp = np.where(valid, pred_c, np.nan)
    ref_disp = np.where(valid, ref_c, np.nan)
    diff = np.where(valid, pred_c - ref_c, np.nan)
    abs_diff = np.abs(diff)

    images = {}

    # 0. Blink comparator — pixel-perfect turbo PNGs stacked so the overlay
    #    lines up exactly. Toggle button / spacebar flips the overlay on/off.
    #    Valid pixels only; nodata transparent so flicker is minimal.
    ref_u8 = ref[:h, :w].astype(np.uint8)
    pred_u8 = pred[:h, :w].astype(np.uint8)
    images["blink_ref"] = _turbo_png_b64(ref_u8)
    images["blink_pred"] = _turbo_png_b64(pred_u8)

    # 1. Side-by-side maps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(ref_disp, vmin=0, vmax=100, cmap="viridis")
    ax1.set_title("Reference")
    ax1.axis("off")
    im2 = ax2.imshow(pred_disp, vmin=0, vmax=100, cmap="viridis")
    ax2.set_title("Our Prediction")
    ax2.axis("off")
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.7, label="Tree Cover %")
    fig.suptitle(f"{tile_name} — Side-by-Side", fontsize=14)
    images["sidebyside"] = fig_to_base64(fig)
    plt.close(fig)

    # 2. Difference heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(np.nanmax(np.abs(diff[np.isfinite(diff)])), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im1 = ax1.imshow(diff, cmap="RdBu_r", norm=norm)
    ax1.set_title("Signed Diff (ours - ref)")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, shrink=0.7)
    im2 = ax2.imshow(abs_diff, cmap="hot_r", vmin=0, vmax=50)
    ax2.set_title("Absolute Diff")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, shrink=0.7)
    fig.suptitle(f"{tile_name} — Difference Maps", fontsize=14)
    images["diff"] = fig_to_base64(fig)
    plt.close(fig)

    # 3. Outlier map
    fig, ax = plt.subplots(figsize=(7, 6))
    overlay = np.full((*pred_disp.shape, 3), 0.9)
    valid_px = np.isfinite(pred_disp)
    overlay[valid_px] = plt.cm.viridis(pred_disp[valid_px] / 100.0)[:, :3]
    gt10 = np.isfinite(abs_diff) & (abs_diff > 10) & (abs_diff <= 30)
    gt30 = np.isfinite(abs_diff) & (abs_diff > 30)
    overlay[gt10] = [1, 1, 0]  # yellow
    overlay[gt30] = [1, 0, 0]  # red
    ax.imshow(overlay)
    ax.set_title(f"{tile_name} — Outliers (yellow >10DN, red >30DN)")
    ax.axis("off")
    images["outlier"] = fig_to_base64(fig)
    plt.close(fig)

    # 4. Histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    r_vals = ref_c[valid].flatten()
    o_vals = pred_c[valid].flatten()
    ax1.hist(r_vals, bins=100, alpha=0.6, label="Reference", color="steelblue", range=(0, 100))
    ax1.hist(o_vals, bins=100, alpha=0.6, label="Ours", color="coral", range=(0, 100))
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Value Distribution")
    ax1.legend()
    d_vals = abs_diff[np.isfinite(abs_diff)].flatten()
    ax2.hist(d_vals, bins=100, color="gray", range=(0, 100))
    ax2.axvline(x=1, color="green", linestyle="--", alpha=0.7, label="1 DN")
    ax2.axvline(x=10, color="orange", linestyle="--", alpha=0.7, label="10 DN")
    ax2.axvline(x=30, color="red", linestyle="--", alpha=0.7, label="30 DN")
    ax2.set_xlabel("Absolute Difference (DN)")
    ax2.set_ylabel("Count")
    ax2.set_title("Difference Distribution")
    ax2.legend()
    ax2.set_yscale("log")
    fig.suptitle(f"{tile_name} — Histograms", fontsize=14)
    images["hist"] = fig_to_base64(fig)
    plt.close(fig)

    # 5. Intermediates (if available)
    intermed_html = ""
    if intermediates:
        fig, axes = plt.subplots(1, min(4, len(intermediates)), figsize=(16, 4))
        if not hasattr(axes, "__len__"):
            axes = [axes]
        for ax, (name, arr) in zip(axes, list(intermediates.items())[:4]):
            if arr.ndim == 4:
                # Show mean across time and bands
                vis = np.mean(arr, axis=(0, -1))
            elif arr.ndim == 3:
                vis = np.mean(arr, axis=0)
            else:
                vis = arr
            ax.imshow(vis, cmap="viridis")
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{tile_name} — Intermediates", fontsize=14)
        images["intermed"] = fig_to_base64(fig)
        plt.close(fig)
        intermed_html = f'<img src="data:image/png;base64,{images["intermed"]}">'

    # Build metrics card
    pass_1dn = stats["pct_within_1"] > 70
    pass_10dn = stats["pct_within_10"] > 95
    corr_excl = stats.get("correlation_excl_outliers", stats["correlation"])
    pass_corr = corr_excl > 0.75

    def badge(ok):
        return '<span style="color:green">PASS</span>' if ok else '<span style="color:red">FAIL</span>'

    tile_id = _safe_id(tile_name)
    return f"""
    <div class="tile-section">
      <h2 onclick="this.parentElement.classList.toggle('collapsed')"
          style="cursor:pointer">{tile_name}
        <span class="metrics-badge">
          %&le;1DN: {stats['pct_within_1']:.1f}% {badge(pass_1dn)} |
          %&le;10DN: {stats['pct_within_10']:.1f}% {badge(pass_10dn)} |
          Corr: {corr_excl:.4f} {badge(pass_corr)}
        </span>
      </h2>
      <div class="tile-content">
        <div class="blink-container" id="blink-{tile_id}">
          <div class="blink-controls">
            <button class="blink-toggle">Show new prediction</button>
            <button class="blink-auto">Auto-blink</button>
            <label class="blink-speed-label">Speed
              <input type="range" class="blink-speed" min="100" max="1500" step="50" value="400">
              <span class="blink-speed-val">400ms</span>
            </label>
            <span class="blink-hint">(spacebar toggles first tile)</span>
          </div>
          <div class="blink-viewport">
            <img class="blink-base" src="data:image/png;base64,{images['blink_ref']}"
                 alt="Reference">
            <img class="blink-overlay" src="data:image/png;base64,{images['blink_pred']}"
                 alt="New prediction">
          </div>
          <div class="blink-legend">
            <span>0</span>
            <div class="blink-gradient"></div>
            <span>100</span>
            <span class="blink-legend-label">tree cover %</span>
          </div>
          <div class="blink-label">Currently: <b>Reference</b></div>
        </div>
        <table class="metrics-table">
          <tr><th>Metric</th><th>Value</th><th>Baseline</th><th>Improved</th><th>Target</th></tr>
          <tr><td>%&le;1DN</td><td><b>{stats['pct_within_1']:.1f}%</b></td>
              <td>&gt;40%</td><td>&gt;70%</td><td>&gt;95%</td></tr>
          <tr><td>%&le;5DN</td><td><b>{stats['pct_within_5']:.1f}%</b></td>
              <td>-</td><td>&gt;90%</td><td>&gt;99%</td></tr>
          <tr><td>%&le;10DN</td><td><b>{stats['pct_within_10']:.1f}%</b></td>
              <td>&gt;75%</td><td>&gt;95%</td><td>-</td></tr>
          <tr><td>Mean Abs Diff</td><td><b>{stats['mean_abs_diff']:.2f}</b></td>
              <td>-</td><td>-</td><td>&lt;1.0</td></tr>
          <tr><td>Correlation</td><td><b>{stats['correlation']:.4f}</b></td>
              <td>-</td><td>-</td><td>&gt;0.95</td></tr>
          <tr><td>Corr (excl outliers)</td><td><b>{corr_excl:.4f}</b></td>
              <td>-</td><td>&gt;0.75</td><td>-</td></tr>
          <tr><td>Outliers (&gt;30DN)</td><td><b>{stats['pct_outlier_30dn']:.1f}%</b></td>
              <td>-</td><td>-</td><td>-</td></tr>
          <tr><td>Our Mean</td><td>{stats['our_mean']:.1f}</td>
              <td colspan="3">Ref Mean: {stats['ref_mean']:.1f}</td></tr>
        </table>
        <img src="data:image/png;base64,{images['sidebyside']}">
        <img src="data:image/png;base64,{images['diff']}">
        <img src="data:image/png;base64,{images['outlier']}">
        <img src="data:image/png;base64,{images['hist']}">
        {intermed_html}
      </div>
    </div>
    """


def make_summary_section(all_stats: dict) -> str:
    agg = aggregate_golden_report(all_stats)
    rows = ""
    for tile, s in sorted(all_stats.items()):
        corr_excl = s.get("correlation_excl_outliers", s["correlation"])
        rows += f"""<tr>
          <td>{tile}</td>
          <td>{s['mean_abs_diff']:.2f}</td>
          <td>{s['pct_within_1']:.1f}%</td>
          <td>{s['pct_within_5']:.1f}%</td>
          <td>{s['pct_within_10']:.1f}%</td>
          <td>{s['correlation']:.4f}</td>
          <td>{corr_excl:.4f}</td>
          <td>{s['pct_outlier_30dn']:.1f}%</td>
        </tr>"""

    return f"""
    <div class="summary-section">
      <h2>Summary</h2>
      <table class="metrics-table">
        <tr><th>Tile</th><th>MeanDiff</th><th>%&le;1DN</th><th>%&le;5DN</th>
            <th>%&le;10DN</th><th>Corr</th><th>CorrExcl</th><th>Outlier%</th></tr>
        {rows}
        <tr class="agg-row">
          <td><b>AGGREGATE</b></td>
          <td>-</td>
          <td><b>{agg['mean_pct_within_1']:.1f}%</b></td>
          <td>-</td>
          <td><b>{agg['mean_pct_within_10']:.1f}%</b></td>
          <td><b>{agg['mean_correlation']:.4f}</b></td>
          <td><b>{agg['mean_correlation_excl_outliers']:.4f}</b></td>
          <td>-</td>
        </tr>
      </table>
      <p>Worst tile (1DN): <b>{agg['worst_tile_1dn']}</b> ({agg['min_pct_within_1']:.1f}%)</p>
    </div>
    """


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Parity Report — GRI Tile Pipeline</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .summary-section {{ background: white; padding: 20px; border-radius: 8px;
                      margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .tile-section {{ background: white; padding: 20px; border-radius: 8px;
                   margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .tile-section.collapsed .tile-content {{ display: none; }}
  .tile-section h2 {{ margin-top: 0; font-size: 18px; }}
  .metrics-badge {{ font-size: 13px; font-weight: normal; margin-left: 12px; }}
  .metrics-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 6px 10px;
                                          text-align: right; font-size: 13px; }}
  .metrics-table th {{ background: #f0f0f0; text-align: center; }}
  .metrics-table td:first-child {{ text-align: left; }}
  .agg-row {{ background: #f8f8f0; }}
  img {{ max-width: 100%; margin: 8px 0; }}
  .timestamp {{ color: #888; font-size: 13px; }}

  /* --- Blink comparator --- */
  .blink-container {{ margin: 12px 0 20px 0; padding: 12px; border: 1px solid #ddd;
                      border-radius: 6px; background: #fafafa; }}
  .blink-controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
                     margin-bottom: 10px; font-size: 13px; }}
  .blink-controls button {{ padding: 6px 12px; font-size: 13px; cursor: pointer;
                            border: 1px solid #888; background: white; border-radius: 4px; }}
  .blink-controls button:hover {{ background: #eef; }}
  .blink-controls button.active {{ background: #2a6; color: white; border-color: #2a6; }}
  .blink-speed-label {{ display: flex; align-items: center; gap: 6px; color: #555; }}
  .blink-speed {{ width: 140px; }}
  .blink-hint {{ color: #888; font-size: 12px; }}
  .blink-viewport {{ position: relative; max-width: 900px; margin: 0 auto;
                     background: #222; border: 1px solid #444;
                     image-rendering: pixelated; image-rendering: crisp-edges; }}
  .blink-viewport img {{ display: block; width: 100%; height: auto; margin: 0;
                         image-rendering: pixelated; image-rendering: crisp-edges; }}
  .blink-overlay {{ position: absolute; top: 0; left: 0; opacity: 0;
                    transition: opacity 0.04s linear; pointer-events: none; }}
  .blink-overlay.on {{ opacity: 1; }}
  .blink-label {{ text-align: center; font-size: 13px; margin-top: 6px; color: #444; }}
  .blink-legend {{ display: flex; align-items: center; gap: 8px; max-width: 900px;
                   margin: 8px auto 0 auto; font-size: 12px; color: #555; }}
  .blink-gradient {{ flex: 1; height: 12px; border: 1px solid #888; border-radius: 2px;
                     background: linear-gradient(to right,
                       #30123b 0%, #4454c4 12.5%, #3ba3f7 25%, #22d1a7 37.5%,
                       #75f147 50%, #cbe333 62.5%, #f79a3a 75%, #e13c27 87.5%,
                       #7a0403 100%); }}
  .blink-legend-label {{ font-weight: 500; margin-left: 4px; }}
</style>
</head>
<body>
<h1>GRI Tile Pipeline — Parity Report</h1>
<p class="timestamp">Generated: {timestamp}</p>
{summary}
{tiles}
<script>
(function() {{
  const containers = Array.from(document.querySelectorAll('.blink-container'));
  containers.forEach((container) => {{
    const overlay = container.querySelector('.blink-overlay');
    const toggleBtn = container.querySelector('.blink-toggle');
    const autoBtn = container.querySelector('.blink-auto');
    const speed = container.querySelector('.blink-speed');
    const speedVal = container.querySelector('.blink-speed-val');
    const label = container.querySelector('.blink-label');
    let interval = null;

    function currentlyNew() {{ return overlay.classList.contains('on'); }}
    function updateLabels() {{
      label.innerHTML = 'Currently: <b>' + (currentlyNew() ? 'New prediction' : 'Reference') + '</b>';
      toggleBtn.textContent = currentlyNew() ? 'Show reference' : 'Show new prediction';
    }}
    function flip() {{ overlay.classList.toggle('on'); updateLabels(); }}

    toggleBtn.addEventListener('click', flip);
    updateLabels();

    function startAuto() {{
      if (interval) return;
      interval = setInterval(flip, parseInt(speed.value, 10));
      autoBtn.classList.add('active');
      autoBtn.textContent = 'Stop auto-blink';
    }}
    function stopAuto() {{
      if (!interval) return;
      clearInterval(interval); interval = null;
      autoBtn.classList.remove('active');
      autoBtn.textContent = 'Auto-blink';
    }}
    autoBtn.addEventListener('click', () => {{ interval ? stopAuto() : startAuto(); }});

    speed.addEventListener('input', () => {{
      speedVal.textContent = speed.value + 'ms';
      if (interval) {{ stopAuto(); startAuto(); }}
    }});
  }});

  // Spacebar flips the first blink container that's in view.
  document.addEventListener('keydown', (e) => {{
    if (e.code !== 'Space' || e.target.tagName === 'INPUT' ||
        e.target.tagName === 'TEXTAREA') return;
    const firstVisible = containers.find((c) => {{
      const r = c.getBoundingClientRect();
      return r.bottom > 0 && r.top < window.innerHeight;
    }});
    if (firstVisible) {{
      e.preventDefault();
      firstVisible.querySelector('.blink-toggle').click();
    }}
  }});
}})();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate parity report")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Include intermediate visualizations")
    args = parser.parse_args()

    from predict_tile import predict_tile_from_arrays

    all_stats = {}
    tile_sections = []

    for tile_name in GOLDEN_TILES:
        print(f"Processing {tile_name}...")
        t0 = time.time()

        arrays = load_golden_tile(tile_name)
        intermediates = {} if args.save_intermediates else None
        pred = predict_tile_from_arrays(
            **arrays, model_path=str(MODEL_DIR), seed=42,
            intermediates=intermediates,
        )
        ref = load_reference_tif(tile_name)

        h = min(pred.shape[0], ref.shape[0])
        w = min(pred.shape[1], ref.shape[1])
        stats = compare_predictions(pred[:h, :w], ref[:h, :w])
        all_stats[tile_name] = stats

        section = make_tile_section(tile_name, pred, ref, stats, intermediates)
        tile_sections.append(section)

        elapsed = time.time() - t0
        print(f"  {tile_name}: %<=1DN={stats['pct_within_1']:.1f}%, "
              f"%<=10DN={stats['pct_within_10']:.1f}%, "
              f"corr_excl={stats.get('correlation_excl_outliers', 0):.4f} "
              f"({elapsed:.1f}s)")

    summary = make_summary_section(all_stats)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        summary=summary,
        tiles="\n".join(tile_sections),
    )

    out_dir = REPO_ROOT / "temp"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "parity_report.html"
    out_path.write_text(html)
    print(f"\nReport written to {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
