import numpy as np
from scipy.fft import dctn, idctn

QUANT_MATRIX_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

QUANT_MATRIX_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

BLOCK_SIZE = 8


def get_quant_matrix(channel: str, qf: float = 1.0) -> np.ndarray:
    
    base = QUANT_MATRIX_LUMA if channel == 'Y' else QUANT_MATRIX_CHROMA
    return np.clip(base * qf, 1, 255).astype(np.float32)


# ── Padding helpers ──────────────────────────────────────────────────────────

def pad_to_multiple(channel: np.ndarray, block: int = BLOCK_SIZE) -> np.ndarray:
    h, w = channel.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    return np.pad(channel, ((0, ph), (0, pw)), mode='edge')


# ── DCT / IDCT on a single 8×8 block ────────────────────────────────────────

def dct2d(block: np.ndarray) -> np.ndarray:
    return dctn(block, type=2, norm='ortho')


def idct2d(block: np.ndarray) -> np.ndarray:
    return idctn(block, type=2, norm='ortho')


# ── Encode one channel ───────────────────────────────────────────────────────

def encode_channel(channel: np.ndarray, quant_matrix: np.ndarray) -> tuple[np.ndarray, tuple]:
    
    orig_shape = channel.shape
    padded = pad_to_multiple(channel)
    h, w = padded.shape
    coeffs = np.zeros_like(padded, dtype=np.float32)

    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            block = padded[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE].astype(np.float32) - 128
            dct_block = dct2d(block)
            q_block = np.round(dct_block / quant_matrix).astype(np.int16)
            coeffs[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = q_block

    return coeffs.astype(np.int16), orig_shape


# ── Decode one channel ───────────────────────────────────────────────────────

def decode_channel(coeffs: np.ndarray, quant_matrix: np.ndarray,
                   orig_shape: tuple) -> np.ndarray:
    
    h, w = coeffs.shape
    recon = np.zeros_like(coeffs, dtype=np.float32)

    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            q_block = coeffs[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE].astype(np.float32)
            dct_block = q_block * quant_matrix
            block = idct2d(dct_block) + 128
            recon[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = block

    oh, ow = orig_shape
    recon = recon[:oh, :ow]
    return np.clip(recon, 0, 255).astype(np.float32)


# ── Full I-frame encode / decode ─────────────────────────────────────────────

def encode_iframe(frame_data: dict, qf: float = 1.0) -> dict:
    
    Y = frame_data['Y']
    Cb = frame_data['Cb_sub']
    Cr = frame_data['Cr_sub']

    qm_y  = get_quant_matrix('Y', qf)
    qm_c  = get_quant_matrix('C', qf)

    Y_coeffs,  Y_shape  = encode_channel(Y,  qm_y)
    Cb_coeffs, Cb_shape = encode_channel(Cb, qm_c)
    Cr_coeffs, Cr_shape = encode_channel(Cr, qm_c)

    return {
        'type': 'I',
        'index': frame_data['index'],
        'qf': qf,
        'original_shape': frame_data['original_shape'],
        'Y_coeffs':  Y_coeffs,  'Y_shape':  Y_shape,
        'Cb_coeffs': Cb_coeffs, 'Cb_shape': Cb_shape,
        'Cr_coeffs': Cr_coeffs, 'Cr_shape': Cr_shape,
    }


def decode_iframe(encoded: dict) -> dict:
    
    qf = encoded['qf']
    qm_y = get_quant_matrix('Y', qf)
    qm_c = get_quant_matrix('C', qf)

    Y  = decode_channel(encoded['Y_coeffs'],  qm_y, encoded['Y_shape'])
    Cb = decode_channel(encoded['Cb_coeffs'], qm_c, encoded['Cb_shape'])
    Cr = decode_channel(encoded['Cr_coeffs'], qm_c, encoded['Cr_shape'])

    return {
        'type': 'I',
        'index': encoded['index'],
        'Y': Y,
        'Cb_sub': Cb,
        'Cr_sub': Cr,
        'original_shape': encoded['original_shape'],
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from part1_preprocessing import bgr_to_ycbcr, chroma_subsample_420
    import cv2

    dummy_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Y, Cb, Cr = bgr_to_ycbcr(dummy_bgr)
    Cb_s, Cr_s = chroma_subsample_420(Cb, Cr)

    frame_data = {
        'index': 0, 'Y': Y, 'Cb_sub': Cb_s,
        'Cr_sub': Cr_s, 'original_shape': dummy_bgr.shape
    }

    encoded = encode_iframe(frame_data, qf=1.0)
    decoded = decode_iframe(encoded)

    print(f"Y original range:       [{Y.min():.1f}, {Y.max():.1f}]")
    print(f"Y reconstructed range:  [{decoded['Y'].min():.1f}, {decoded['Y'].max():.1f}]")

    mse = np.mean((Y - decoded['Y'])**2)
    print(f"MSE (Y channel): {mse:.4f}")
    print("Part 2 OK")