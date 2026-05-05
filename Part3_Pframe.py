"""
Part 3 — Inter-frame Coding / P-frames (25%)
- GOP structure: every G-th frame is an I-frame, others are P-frames
- Block matching (motion estimation) on 16×16 macroblocks
- Residual coding with DCT + quantization
"""

import numpy as np
from Part2_Iframe import (
    encode_channel, decode_channel,
    get_quant_matrix, BLOCK_SIZE, pad_to_multiple, dct2d, idct2d
)

MACROBLOCK_SIZE = 16


# ── Motion Estimation ────────────────────────────────────────────────────────

def block_matching(current_Y: np.ndarray, reference_Y: np.ndarray,
                   search_window: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-search block matching on 16×16 macroblocks.
    Returns:
        motion_vectors – shape (num_blocks_h, num_blocks_w, 2) [dy, dx]
        prediction     – motion-compensated prediction for current_Y
    """
    h, w = current_Y.shape
    MB = MACROBLOCK_SIZE
    S  = search_window

    num_h = (h + MB - 1) // MB
    num_w = (w + MB - 1) // MB

    motion_vectors = np.zeros((num_h, num_w, 2), dtype=np.int16)
    prediction     = np.zeros_like(current_Y, dtype=np.float32)

    # Pad reference to handle border search
    pad = S + MB
    ref_padded = np.pad(reference_Y, pad, mode='edge')

    for bi in range(num_h):
        for bj in range(num_w):
            # Current macroblock position (may be smaller at border)
            r0 = bi * MB
            c0 = bj * MB
            r1 = min(r0 + MB, h)
            c1 = min(c0 + MB, w)
            cur_block = current_Y[r0:r1, c0:c1].astype(np.float32)
            bh, bw = cur_block.shape

            best_sad = np.inf
            best_dy, best_dx = 0, 0

            # Search window in reference (padded coords)
            ref_r0 = r0 + pad
            ref_c0 = c0 + pad

            for dy in range(-S, S + 1):
                for dx in range(-S, S + 1):
                    rr = ref_r0 + dy
                    rc = ref_c0 + dx
                    ref_block = ref_padded[rr:rr+bh, rc:rc+bw].astype(np.float32)
                    sad = np.sum(np.abs(cur_block - ref_block))
                    if sad < best_sad:
                        best_sad = sad
                        best_dy, best_dx = dy, dx

            motion_vectors[bi, bj] = [best_dy, best_dx]

            # Fill prediction
            rr = ref_r0 + best_dy
            rc = ref_c0 + best_dx
            prediction[r0:r1, c0:c1] = ref_padded[rr:rr+bh, rc:rc+bw]

    return motion_vectors, prediction


# ── Residual encode / decode ─────────────────────────────────────────────────

def encode_residual(residual: np.ndarray, qf: float) -> tuple[np.ndarray, tuple]:
    """DCT + quantize residual channel (same as I-frame channel encode)."""
    qm = get_quant_matrix('Y', qf)
    h, w = residual.shape
    padded = pad_to_multiple(residual)
    ph, pw = padded.shape
    coeffs = np.zeros((ph, pw), dtype=np.int16)

    for r in range(0, ph, BLOCK_SIZE):
        for c in range(0, pw, BLOCK_SIZE):
            block = padded[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE].astype(np.float32)
            dct_b = dct2d(block)
            coeffs[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = np.round(dct_b / qm).astype(np.int16)

    return coeffs, (h, w)


def decode_residual(coeffs: np.ndarray, orig_shape: tuple, qf: float) -> np.ndarray:
    """Dequantize + IDCT residual channel."""
    qm = get_quant_matrix('Y', qf)
    ph, pw = coeffs.shape
    recon = np.zeros((ph, pw), dtype=np.float32)

    for r in range(0, ph, BLOCK_SIZE):
        for c in range(0, pw, BLOCK_SIZE):
            q_b = coeffs[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE].astype(np.float32)
            recon[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = idct2d(q_b * qm)

    oh, ow = orig_shape
    return recon[:oh, :ow]


# ── P-frame encode / decode ──────────────────────────────────────────────────

def encode_pframe(frame_data: dict, ref_decoded: dict,
                  qf: float = 1.0, search_window: int = 8) -> dict:
    """
    Encode a P-frame given the current frame and the decoded reference frame.
    frame_data  – dict from preprocess_frames()
    ref_decoded – dict returned by decode_iframe() or decode_pframe()
    """
    Y_cur  = frame_data['Y']
    Cb_cur = frame_data['Cb_sub']
    Cr_cur = frame_data['Cr_sub']

    Y_ref  = ref_decoded['Y']
    Cb_ref = ref_decoded['Cb_sub']
    Cr_ref = ref_decoded['Cr_sub']

    # ── Motion estimation on Y channel ──────────────────────────────────────
    motion_vectors, Y_pred = block_matching(Y_cur, Y_ref, search_window)

    # ── Residuals ────────────────────────────────────────────────────────────
    Y_residual = Y_cur - Y_pred

    # For chroma: use motion vectors scaled by 0.5 (4:2:0)
    Cb_pred = _apply_motion(Cb_ref, motion_vectors, scale=0.5)
    Cr_pred = _apply_motion(Cr_ref, motion_vectors, scale=0.5)
    Cb_residual = Cb_cur - Cb_pred
    Cr_residual = Cr_cur - Cr_pred

    # ── Encode residuals ─────────────────────────────────────────────────────
    Y_coeffs,  Y_shape  = encode_residual(Y_residual,  qf)
    Cb_coeffs, Cb_shape = encode_residual(Cb_residual, qf)
    Cr_coeffs, Cr_shape = encode_residual(Cr_residual, qf)

    return {
        'type': 'P',
        'index': frame_data['index'],
        'qf': qf,
        'original_shape': frame_data['original_shape'],
        'motion_vectors': motion_vectors,
        'Y_coeffs':  Y_coeffs,  'Y_shape':  Y_shape,
        'Cb_coeffs': Cb_coeffs, 'Cb_shape': Cb_shape,
        'Cr_coeffs': Cr_coeffs, 'Cr_shape': Cr_shape,
    }


def decode_pframe(encoded: dict, ref_decoded: dict) -> dict:
    """
    Reconstruct a P-frame from its encoded data and the reference decoded frame.
    """
    qf = encoded['qf']
    motion_vectors = encoded['motion_vectors']

    Y_ref  = ref_decoded['Y']
    Cb_ref = ref_decoded['Cb_sub']
    Cr_ref = ref_decoded['Cr_sub']

    # Reconstruct predictions
    Y_pred  = _apply_motion_from_vectors(Y_ref,  motion_vectors, scale=1.0)
    Cb_pred = _apply_motion(Cb_ref, motion_vectors, scale=0.5)
    Cr_pred = _apply_motion(Cr_ref, motion_vectors, scale=0.5)

    # Decode residuals
    Y_res  = decode_residual(encoded['Y_coeffs'],  encoded['Y_shape'],  qf)
    Cb_res = decode_residual(encoded['Cb_coeffs'], encoded['Cb_shape'], qf)
    Cr_res = decode_residual(encoded['Cr_coeffs'], encoded['Cr_shape'], qf)

    # Reconstruct
    oh, ow = encoded['Y_shape']
    Y  = np.clip(Y_pred[:oh, :ow]  + Y_res,  0, 255).astype(np.float32)
    Cb = np.clip(Cb_pred + Cb_res, 0, 255).astype(np.float32)
    Cr = np.clip(Cr_pred + Cr_res, 0, 255).astype(np.float32)

    return {
        'type': 'P',
        'index': encoded['index'],
        'Y': Y,
        'Cb_sub': Cb,
        'Cr_sub': Cr,
        'original_shape': encoded['original_shape'],
    }


# ── Helper: apply motion vectors to build prediction ────────────────────────

def _apply_motion_from_vectors(ref: np.ndarray, motion_vectors: np.ndarray,
                                scale: float = 1.0) -> np.ndarray:
    """Build a prediction image by copying blocks from ref using motion vectors."""
    h, w = ref.shape
    MB = int(MACROBLOCK_SIZE * scale)
    MB = max(MB, 1)
    pad = int(MACROBLOCK_SIZE) + 1
    ref_padded = np.pad(ref, pad, mode='edge')
    pred = np.zeros_like(ref, dtype=np.float32)

    num_h, num_w = motion_vectors.shape[:2]
    for bi in range(num_h):
        for bj in range(num_w):
            dy, dx = motion_vectors[bi, bj]
            r0 = bi * MB
            c0 = bj * MB
            r1 = min(r0 + MB, h)
            c1 = min(c0 + MB, w)
            bh, bw = r1 - r0, c1 - c0

            rr = r0 + pad + int(dy * scale)
            rc = c0 + pad + int(dx * scale)
            rr = np.clip(rr, 0, ref_padded.shape[0] - bh)
            rc = np.clip(rc, 0, ref_padded.shape[1] - bw)
            pred[r0:r1, c0:c1] = ref_padded[rr:rr+bh, rc:rc+bw]

    return pred


def _apply_motion(ref: np.ndarray, motion_vectors: np.ndarray,
                  scale: float = 0.5) -> np.ndarray:
    return _apply_motion_from_vectors(ref, motion_vectors, scale)


# ── GOP encoder ──────────────────────────────────────────────────────────────

def encode_gop(preprocessed_frames: list, gop_size: int = 8,
               qf: float = 1.0, search_window: int = 8) -> list:
    """
    Encode all frames using GOP structure.
    Every gop_size-th frame (0, G, 2G, …) is an I-frame; others are P-frames.
    Returns list of encoded frame dicts.
    """
    from Part2_Iframe import encode_iframe, decode_iframe

    encoded_frames = []
    ref_decoded    = None

    for frame_data in preprocessed_frames:
        i = frame_data['index']

        if i % gop_size == 0:
            enc = encode_iframe(frame_data, qf=qf)
            ref_decoded = decode_iframe(enc)
            frame_type  = 'I'
        else:
            enc = encode_pframe(frame_data, ref_decoded, qf=qf,
                                search_window=search_window)
            ref_decoded = decode_pframe(enc, ref_decoded)
            frame_type  = 'P'

        encoded_frames.append(enc)
        print(f"[Part 3] Frame {i:03d} encoded as {frame_type}-frame")

    return encoded_frames


def decode_all_frames(encoded_frames: list) -> list:
    """
    Decode all encoded frames in order, returning list of decoded frame dicts.
    """
    from Part2_Iframe import decode_iframe

    decoded = []
    ref     = None

    for enc in encoded_frames:
        if enc['type'] == 'I':
            dec = decode_iframe(enc)
        else:
            dec = decode_pframe(enc, ref)
        decoded.append(dec)
        ref = dec

    return decoded


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from Part1_preprocessing import bgr_to_ycbcr, chroma_subsample_420
    import numpy as np

    frames_raw = []
    for i in range(10):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Y, Cb, Cr = bgr_to_ycbcr(bgr)
        Cb_s, Cr_s = chroma_subsample_420(Cb, Cr)
        frames_raw.append({
            'index': i, 'Y': Y, 'Cb_sub': Cb_s,
            'Cr_sub': Cr_s, 'original_shape': bgr.shape
        })

    encoded = encode_gop(frames_raw, gop_size=4, qf=1.0, search_window=4)
    decoded = decode_all_frames(encoded)
    print(f"\nEncoded {len(encoded)} frames, decoded {len(decoded)} frames")
    print("Part 3 OK")