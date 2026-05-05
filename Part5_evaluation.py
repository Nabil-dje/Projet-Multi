"""
Part 5 — Evaluation & Visualisation (25%)

5a — Quality Metrics:
    - Compression ratio (global and per-frame)
    - Frame-type breakdown (I-frames vs P-frames)
    - PSNR per frame
    - Compression ratio vs Quantization Factor plot
    - Compression ratio vs GOP size plot

5b — Pipeline Visualisation:
    1. Original frames sequence
    2. Y, Cb, Cr channels of one frame
    3. One 8×8 block: raw pixels → DCT coefficients → quantised → reconstructed
    4. Motion vectors overlaid on a P-frame
    5. Residual maps alongside reconstructed frames
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Part1_preprocessing import (
    load_frames, preprocess_frames,
    bgr_to_ycbcr, chroma_subsample_420,
    chroma_upsample, ycbcr_to_bgr
)
from Part2_Iframe import (
    encode_iframe, decode_iframe,
    dct2d, idct2d, get_quant_matrix, BLOCK_SIZE
)
from Part3_Pframe import encode_gop, decode_all_frames, MACROBLOCK_SIZE
from Part4_entropy import encode_to_bin, decode_from_bin, compute_original_size


# ─────────────────────────────────────────────────────────────────────────────
# 5a — QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray,
                 max_val: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images/channels.
    Higher PSNR = better quality (> 30 dB is generally acceptable).
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)


def reconstruct_frame_bgr(decoded_frame: dict) -> np.ndarray:
    """
    Reconstruct a BGR image from a decoded frame dict.
    """
    Y = decoded_frame['Y']
    Cb_sub = decoded_frame['Cb_sub']
    Cr_sub = decoded_frame['Cr_sub']
    orig_shape = decoded_frame['original_shape']

    Cb_up, Cr_up = chroma_upsample(Cb_sub, Cr_sub, orig_shape)
    bgr = ycbcr_to_bgr(Y, Cb_up, Cr_up)
    return bgr


def evaluate_pipeline(original_frames: list, encoded_frames: list,
                       decoded_frames: list, bin_path: str) -> dict:
    """
    Compute all quality metrics for the full pipeline.

    Returns a dict with:
        - psnr_per_frame      : list of PSNR values (dB)
        - frame_types         : list of 'I' or 'P'
        - num_i_frames        : int
        - num_p_frames        : int
        - compression_ratio   : float (global)
        - original_size_bytes : int
        - compressed_size_bytes: int
        - avg_psnr            : float
        - min_psnr            : float
        - max_psnr            : float
    """
    bin_size = os.path.getsize(bin_path)
    orig_size = compute_original_size(encoded_frames)
    compression_ratio = orig_size / bin_size

    psnr_values = []
    frame_types = []
    num_i = 0
    num_p = 0

    for orig_bgr, enc, dec in zip(original_frames, encoded_frames, decoded_frames):
        ftype = enc['type']
        frame_types.append(ftype)
        if ftype == 'I':
            num_i += 1
        else:
            num_p += 1

        recon_bgr = reconstruct_frame_bgr(dec)

        # Resize if needed (border effects)
        h, w = orig_bgr.shape[:2]
        recon_bgr = recon_bgr[:h, :w]

        psnr = compute_psnr(orig_bgr, recon_bgr)
        psnr_values.append(psnr)

    return {
        'psnr_per_frame': psnr_values,
        'frame_types': frame_types,
        'num_i_frames': num_i,
        'num_p_frames': num_p,
        'compression_ratio': compression_ratio,
        'original_size_bytes': orig_size,
        'compressed_size_bytes': bin_size,
        'avg_psnr': np.mean(psnr_values),
        'min_psnr': np.min(psnr_values),
        'max_psnr': np.max(psnr_values),
    }


def print_metrics_report(metrics: dict):
    """Print a formatted metrics summary to stdout."""
    print("\n" + "=" * 55)
    print("          PIPELINE EVALUATION REPORT")
    print("=" * 55)
    print(f"  Frames total     : {len(metrics['frame_types'])}")
    print(f"  I-frames         : {metrics['num_i_frames']}")
    print(f"  P-frames         : {metrics['num_p_frames']}")
    print(f"  Original size    : {metrics['original_size_bytes']:,} bytes")
    print(f"  Compressed size  : {metrics['compressed_size_bytes']:,} bytes")
    print(f"  Compression ratio: {metrics['compression_ratio']:.2f}x")
    print(f"  PSNR avg         : {metrics['avg_psnr']:.2f} dB")
    print(f"  PSNR min         : {metrics['min_psnr']:.2f} dB")
    print(f"  PSNR max         : {metrics['max_psnr']:.2f} dB")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Experimental plots: compression ratio vs QF and GOP size
# ─────────────────────────────────────────────────────────────────────────────

def plot_compression_vs_qf(preprocessed_frames: list,
                            qf_values: list = None,
                            gop_size: int = 8,
                            output_dir: str = ".") -> plt.Figure:
    """
    Plot compression ratio vs quantization factor (QF).
    Also shows average PSNR vs QF on a secondary axis.
    """
    if qf_values is None:
        qf_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    ratios = []
    psnrs = []

    for qf in qf_values:
        print(f"  [QF sweep] Testing qf={qf}...")
        enc_frames = encode_gop(preprocessed_frames, gop_size=gop_size,
                                qf=qf, search_window=4)
        dec_frames = decode_all_frames(enc_frames)
        tmp_path = os.path.join(output_dir, f"_tmp_qf{qf}.bin")
        encode_to_bin(enc_frames, tmp_path, {'qf': qf})

        orig_size = compute_original_size(enc_frames)
        bin_size = os.path.getsize(tmp_path)
        ratios.append(orig_size / bin_size)

        # PSNR on Y channel only (fast)
        frame_psnrs = []
        for orig_prep, dec in zip(preprocessed_frames, dec_frames):
            p = compute_psnr(orig_prep['Y'], dec['Y'])
            frame_psnrs.append(p)
        psnrs.append(np.mean(frame_psnrs))
        os.remove(tmp_path)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax1.set_facecolor('#0f0f1a')

    color_ratio = '#00e5ff'
    color_psnr  = '#ff6e6e'

    ax1.plot(qf_values, ratios, 'o-', color=color_ratio, lw=2.5,
             markersize=8, markerfacecolor='white', label='Compression Ratio')
    ax1.set_xlabel('Quantization Factor (QF)', color='white', fontsize=12)
    ax1.set_ylabel('Compression Ratio (x)', color=color_ratio, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_ratio)
    ax1.tick_params(axis='x', colors='white')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color(color_ratio)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.label.set_color(color_ratio)

    ax2 = ax1.twinx()
    ax2.set_facecolor('#0f0f1a')
    ax2.plot(qf_values, psnrs, 's--', color=color_psnr, lw=2,
             markersize=7, markerfacecolor='white', label='Avg PSNR (Y)')
    ax2.set_ylabel('Average PSNR (dB)', color=color_psnr, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_psnr)
    ax2.spines['right'].set_color(color_psnr)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['left'].set_color(color_ratio)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
               facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    ax1.set_title('Compression Ratio & PSNR vs Quantization Factor',
                  color='white', fontsize=14, pad=15)
    ax1.grid(True, color='#2a2a3e', linestyle='--', alpha=0.7)

    plt.tight_layout()
    out = os.path.join(output_dir, "plot_compression_vs_qf.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"[Part 5] Saved: {out}")
    return fig


def plot_compression_vs_gop(preprocessed_frames: list,
                             gop_values: list = None,
                             qf: float = 1.0,
                             output_dir: str = ".") -> plt.Figure:
    """
    Plot compression ratio vs GOP size.
    """
    if gop_values is None:
        n = len(preprocessed_frames)
        gop_values = [g for g in [1, 2, 4, 8, 11, 16, 32] if g <= n]

    ratios = []
    for gop in gop_values:
        print(f"  [GOP sweep] Testing gop_size={gop}...")
        enc_frames = encode_gop(preprocessed_frames, gop_size=gop,
                                qf=qf, search_window=4)
        tmp_path = f"_tmp_gop{gop}.bin"
        encode_to_bin(enc_frames, tmp_path, {'gop_size': gop})
        orig_size = compute_original_size(enc_frames)
        bin_size = os.path.getsize(tmp_path)
        ratios.append(orig_size / bin_size)
        os.remove(tmp_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    bars = ax.bar(range(len(gop_values)), ratios, color='#7c3aed',
                  edgecolor='#a855f7', linewidth=1.5)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{ratio:.2f}x', ha='center', va='bottom',
                color='white', fontsize=10)

    ax.set_xticks(range(len(gop_values)))
    ax.set_xticklabels([f'G={g}' for g in gop_values], color='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel('GOP Size (G)', color='white', fontsize=12)
    ax.set_ylabel('Compression Ratio (x)', color='white', fontsize=12)
    ax.set_title('Compression Ratio vs GOP Size', color='white',
                 fontsize=14, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, axis='y', color='#2a2a3e', linestyle='--', alpha=0.7)

    plt.tight_layout()
    out = os.path.join(output_dir, "plot_compression_vs_gop.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"[Part 5] Saved: {out}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5b — PIPELINE VISUALISATION (single figure)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline(original_frames: list,
                        preprocessed_frames: list,
                        encoded_frames: list,
                        decoded_frames: list,
                        frame_idx: int = 0,
                        p_frame_idx: int = None,
                        output_dir: str = ".") -> plt.Figure:
    """
    Produce a comprehensive single-figure pipeline visualisation with 5 sections:
      1. Original frame sequence (up to 6 frames)
      2. Y, Cb, Cr channels of one frame
      3. One 8×8 block: raw → DCT → quantised → reconstructed
      4. Motion vectors overlaid on a P-frame
      5. Residual maps alongside reconstructed frames
    """

    # ── Find a P-frame for sections 4 & 5 ────────────────────────────────
    if p_frame_idx is None:
        p_frame_idx = next(
            (i for i, e in enumerate(encoded_frames) if e['type'] == 'P'),
            None
        )

    has_p_frame = p_frame_idx is not None

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24), facecolor='#0d0d1a')
    gs_main = gridspec.GridSpec(5, 1, figure=fig, hspace=0.5,
                                 top=0.97, bottom=0.03, left=0.04, right=0.96)

    TITLE_STYLE = dict(color='#e0e0ff', fontsize=13, fontweight='bold',
                       pad=8, loc='left')
    LABEL_STYLE = dict(color='#aaaacc', fontsize=9)

    def section_title(ax, text):
        ax.set_title(text, **TITLE_STYLE)

    def remove_axes(ax):
        ax.axis('off')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  SECTION 1 — Original frame sequence                                ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, min(6, len(original_frames)), subplot_spec=gs_main[0], wspace=0.05)

    for j, frame_bgr in enumerate(original_frames[:6]):
        ax = fig.add_subplot(gs1[0, j])
        frame_rgb = frame_bgr[:, :, ::-1]
        ax.imshow(frame_rgb)
        ax.set_xlabel(f'Frame {j}', **LABEL_STYLE)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        if j == 0:
            section_title(ax, '① Original Frame Sequence')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  SECTION 2 — Y, Cb, Cr channels                                     ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1],
                                            wspace=0.06)
    prep = preprocessed_frames[frame_idx]
    orig_rgb = original_frames[frame_idx][:, :, ::-1]
    Y_ch  = prep['Y']
    Cb_ch = prep['Cb_sub']
    Cr_ch = prep['Cr_sub']

    channel_data = [
        (orig_rgb, 'Original (RGB)', None),
        (Y_ch,    'Y  (Luminance)', 'gray'),
        (Cb_ch,   'Cb (Blue diff)', 'PuBu'),
        (Cr_ch,   'Cr (Red diff)',  'OrRd'),
    ]
    for j, (data, label, cmap) in enumerate(channel_data):
        ax = fig.add_subplot(gs2[0, j])
        ax.imshow(data, cmap=cmap)
        ax.set_xlabel(label, **LABEL_STYLE)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        if j == 0:
            section_title(ax, '② Color Space Decomposition (Frame 0)')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  SECTION 3 — 8×8 block through DCT pipeline                         ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2],
                                            wspace=0.12)

    Y = preprocessed_frames[frame_idx]['Y']
    # Pick a block near center for interesting content
    bh = min(BLOCK_SIZE, Y.shape[0])
    bw = min(BLOCK_SIZE, Y.shape[1])
    br = (Y.shape[0] // 2 // BLOCK_SIZE) * BLOCK_SIZE
    bc = (Y.shape[1] // 2 // BLOCK_SIZE) * BLOCK_SIZE
    raw_block = Y[br:br+bh, bc:bc+bw].astype(np.float32) - 128

    dct_block  = dct2d(raw_block)
    qm         = get_quant_matrix('Y', 1.0)[:bh, :bw]
    quant_block = np.round(dct_block / qm).astype(np.int16)
    recon_block = idct2d(quant_block.astype(np.float32) * qm) + 128

    block_stages = [
        (raw_block + 128, 'Raw Pixels\n(8×8 block)', 'gray', False),
        (dct_block,       'DCT Coefficients',         'RdBu_r', True),
        (quant_block,     'Quantised Coefficients',   'RdBu_r', True),
        (recon_block,     'Reconstructed Block',      'gray', False),
    ]
    for j, (data, label, cmap, show_vals) in enumerate(block_stages):
        ax = fig.add_subplot(gs3[0, j])
        im = ax.imshow(data, cmap=cmap, interpolation='nearest',
                       aspect='equal')
        ax.set_xlabel(label, **LABEL_STYLE)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

        if show_vals:
            for rr in range(data.shape[0]):
                for cc in range(data.shape[1]):
                    val = int(data[rr, cc])
                    ax.text(cc, rr, str(val), ha='center', va='center',
                            fontsize=5.5, color='white' if abs(val) > 30 else '#cccccc')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=7, colors='#aaaacc')

        if j == 0:
            section_title(ax, '③ DCT & Quantisation Pipeline (8×8 block)')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  SECTION 4 — Motion vectors overlaid on P-frame                     ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[3],
                                            wspace=0.08)
    ax4a = fig.add_subplot(gs4[0, 0])
    ax4b = fig.add_subplot(gs4[0, 1])

    if has_p_frame:
        penc = encoded_frames[p_frame_idx]
        pdec = decoded_frames[p_frame_idx]
        recon_rgb = reconstruct_frame_bgr(pdec)[:, :, ::-1]

        # Show reconstructed P-frame with motion vectors overlaid
        ax4a.imshow(recon_rgb)
        mv = penc['motion_vectors']
        MB = MACROBLOCK_SIZE
        h_img, w_img = recon_rgb.shape[:2]

        for bi in range(mv.shape[0]):
            for bj in range(mv.shape[1]):
                dy, dx = mv[bi, bj]
                cy = bi * MB + MB // 2
                cx = bj * MB + MB // 2
                if cy < h_img and cx < w_img and (dy != 0 or dx != 0):
                    ax4a.annotate(
                        '', xy=(cx + dx, cy + dy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='#00ff88',
                                        lw=1.2, mutation_scale=10)
                    )
        ax4a.set_title(f'Motion Vectors (P-frame {p_frame_idx})',
                       color='#aaaacc', fontsize=10)
        ax4a.set_xticks([])
        ax4a.set_yticks([])
        for spine in ax4a.spines.values():
            spine.set_edgecolor('#333355')

        # Motion magnitude heatmap
        mag = np.sqrt(mv[:, :, 0].astype(float)**2 + mv[:, :, 1].astype(float)**2)
        im4b = ax4b.imshow(mag, cmap='hot', interpolation='nearest')
        ax4b.set_title('Motion Magnitude Map', color='#aaaacc', fontsize=10)
        ax4b.set_xticks([])
        ax4b.set_yticks([])
        plt.colorbar(im4b, ax=ax4b, fraction=0.046, pad=0.04,
                     label='pixels').ax.tick_params(labelsize=7, colors='#aaaacc')
        for spine in ax4b.spines.values():
            spine.set_edgecolor('#333355')
    else:
        for ax in [ax4a, ax4b]:
            remove_axes(ax)
            ax.text(0.5, 0.5, 'No P-frames in this GOP', ha='center',
                    va='center', color='#aaaacc', fontsize=11,
                    transform=ax.transAxes)

    section_title(ax4a, '④ Motion Vectors & Magnitude (P-frame)')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  SECTION 5 — Residuals and reconstruction                           ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    n_cols = 4
    gs5 = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs_main[4],
                                            wspace=0.06)

    # Pick one I-frame and one P-frame for comparison
    i_idx = next((i for i, e in enumerate(encoded_frames) if e['type'] == 'I'), 0)
    disp_indices = [i_idx]
    if has_p_frame:
        disp_indices.append(p_frame_idx)

    show_items = []
    for idx in disp_indices:
        orig_bgr = original_frames[idx]
        dec = decoded_frames[idx]
        recon_bgr = reconstruct_frame_bgr(dec)
        h, w = orig_bgr.shape[:2]
        recon_bgr = recon_bgr[:h, :w]

        # Residual = absolute difference (Y channel for clarity)
        orig_Y, _, _ = bgr_to_ycbcr(orig_bgr)
        residual = np.abs(orig_Y - dec['Y'][:h, :w])
        show_items.append((recon_bgr[:, :, ::-1], residual,
                           encoded_frames[idx]['type'], idx))

    col = 0
    for recon_rgb, residual, ftype, fidx in show_items:
        if col >= n_cols:
            break
        ax_r = fig.add_subplot(gs5[0, col])
        ax_r.imshow(recon_rgb)
        ax_r.set_xlabel(f'Reconstructed ({ftype}-frame {fidx})', **LABEL_STYLE)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_edgecolor('#333355')
        if col == 0:
            section_title(ax_r, '⑤ Residuals & Reconstructed Frames')
        col += 1

        if col < n_cols:
            ax_res = fig.add_subplot(gs5[0, col])
            im_res = ax_res.imshow(residual, cmap='inferno',
                                   interpolation='nearest')
            ax_res.set_xlabel(f'Residual Map ({ftype}-frame {fidx})',
                              **LABEL_STYLE)
            ax_res.set_xticks([])
            ax_res.set_yticks([])
            plt.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04,
                         label='|diff|').ax.tick_params(
                labelsize=7, colors='#aaaacc')
            for spine in ax_res.spines.values():
                spine.set_edgecolor('#333355')
            col += 1

    # Fill remaining columns if needed
    while col < n_cols:
        ax_empty = fig.add_subplot(gs5[0, col])
        remove_axes(ax_empty)
        col += 1

    # ── Super title ──────────────────────────────────────────────────────
    fig.suptitle('MPEG-4 Encoder Pipeline — Full Visualisation',
                 color='white', fontsize=17, fontweight='bold', y=0.995)

    out = os.path.join(output_dir, "pipeline_visualisation.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"[Part 5] Saved: {out}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PSNR per frame bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_psnr_per_frame(metrics: dict, output_dir: str = ".") -> plt.Figure:
    """Bar chart of PSNR per frame, coloured by frame type."""
    psnrs = metrics['psnr_per_frame']
    ftypes = metrics['frame_types']
    indices = list(range(len(psnrs)))

    colors = ['#00b4d8' if t == 'I' else '#f4a261' for t in ftypes]

    fig, ax = plt.subplots(figsize=(max(10, len(psnrs) * 0.6), 5))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    bars = ax.bar(indices, psnrs, color=colors, edgecolor='#1a1a2e',
                  linewidth=0.8)

    avg_line = ax.axhline(metrics['avg_psnr'], color='#ff6b6b',
                          linestyle='--', lw=1.8, label=f"Avg PSNR: {metrics['avg_psnr']:.1f} dB")

    patch_i = mpatches.Patch(color='#00b4d8', label='I-frame')
    patch_p = mpatches.Patch(color='#f4a261', label='P-frame')
    ax.legend(handles=[patch_i, patch_p, avg_line],
              facecolor='#1a1a2e', edgecolor='#444', labelcolor='white',
              fontsize=9)

    ax.set_xlabel('Frame Index', color='white', fontsize=11)
    ax.set_ylabel('PSNR (dB)', color='white', fontsize=11)
    ax.set_title('PSNR per Frame (I-frames vs P-frames)',
                 color='white', fontsize=13, pad=10)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, axis='y', color='#2a2a3e', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, "plot_psnr_per_frame.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"[Part 5] Saved: {out}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run full Part 5 evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(frames_folder: str,
                         output_dir: str = "output",
                         gop_size: int = 8,
                         qf: float = 1.0,
                         search_window: int = 8):
    """
    Run the complete pipeline and produce all Part 5 outputs.

    Parameters
    ----------
    frames_folder  : path to folder with PNG/JPG frames
    output_dir     : where to save all output files
    gop_size       : Group of Pictures size (every G-th frame is I-frame)
    qf             : quantization factor (1.0 = standard JPEG quality)
    search_window  : motion search window ±S pixels
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "encoded_video.bin")

    print("\n[Part 5] ── Loading & preprocessing frames...")
    original_frames = load_frames(frames_folder)
    preprocessed    = preprocess_frames(original_frames)

    print("[Part 5] ── Encoding GOP...")
    encoded = encode_gop(preprocessed, gop_size=gop_size,
                          qf=qf, search_window=search_window)

    print("[Part 5] ── Writing .bin file...")
    encode_to_bin(encoded, bin_path,
                  {'gop_size': gop_size, 'qf': qf, 'search_window': search_window})

    print("[Part 5] ── Decoding all frames...")
    decoded = decode_all_frames(encoded)

    # ── 5a: Metrics ───────────────────────────────────────────────────────
    print("[Part 5] ── Computing metrics...")
    metrics = evaluate_pipeline(original_frames, encoded, decoded, bin_path)
    print_metrics_report(metrics)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("[Part 5] ── Generating PSNR per frame plot...")
    plot_psnr_per_frame(metrics, output_dir)

    print("[Part 5] ── Generating compression vs QF plot...")
    plot_compression_vs_qf(preprocessed, output_dir=output_dir,
                            gop_size=gop_size)

    print("[Part 5] ── Generating compression vs GOP plot...")
    plot_compression_vs_gop(preprocessed, qf=qf, output_dir=output_dir)

    # ── 5b: Pipeline visualisation ────────────────────────────────────────
    print("[Part 5] ── Generating pipeline visualisation...")
    visualize_pipeline(original_frames, preprocessed, encoded, decoded,
                       frame_idx=0, output_dir=output_dir)

    print(f"\n[Part 5] ✅  All outputs saved to '{output_dir}/'")
    print("   - encoded_video.bin")
    print("   - pipeline_visualisation.png")
    print("   - plot_psnr_per_frame.png")
    print("   - plot_compression_vs_qf.png")
    print("   - plot_compression_vs_gop.png")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Quick synthetic self-test (no real frames needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Part 5 — Evaluation & Visualisation")
    parser.add_argument("--frames", type=str, default=None,
                        help="Path to folder with PNG/JPG frames")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--gop",    type=int, default=8)
    parser.add_argument("--qf",     type=float, default=1.0)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    if args.frames:
        # ── Real frames mode ─────────────────────────────────────────────
        run_full_evaluation(
            frames_folder="my_Frames",
            output_dir=args.output,
            gop_size=args.gop,
            qf=args.qf,
            search_window=args.window,
        )
    else:
        # ── Synthetic test mode ──────────────────────────────────────────
        print("[Part 5] No --frames provided — running synthetic self-test...")
        import cv2

        TMP_DIR = "/tmp/synthetic_frames"
        os.makedirs(TMP_DIR, exist_ok=True)

        # Generate 12 synthetic frames with slight motion
        for i in range(12):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            # Moving rectangle to create meaningful motion vectors
            x = (i * 4) % 48
            frame[16:48, x:x+16] = [80 + i*10, 120, 200 - i*8]
            frame[8:24, 8:24] = [200, 80, 80]
            noise = np.random.randint(0, 20, (64, 64, 3), dtype=np.uint8)
            frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(TMP_DIR, f"frame_{i:03d}.png"), frame)

        run_full_evaluation(
            frames_folder=TMP_DIR,
            output_dir="output_synthetic",
            gop_size=4,
            qf=1.0,
            search_window=4,
        )

        print("\nPart 5 self-test OK")