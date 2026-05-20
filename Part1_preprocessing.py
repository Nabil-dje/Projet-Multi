import cv2
import numpy as np
import os


def bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    frame_float = frame_bgr.astype(np.float32)
    B = frame_float[:, :, 0]
    G = frame_float[:, :, 1]
    R = frame_float[:, :, 2]

    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128

    return Y, Cb, Cr


def chroma_subsample_420(Cb: np.ndarray, Cr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Cb_sub, Cr_sub


def chroma_upsample(Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                    original_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    
    h, w = original_shape[:2]
    Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_NEAREST)
    Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_NEAREST)
    return Cb_up, Cr_up


def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    bgr = np.stack([B, G, R], axis=2)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr


def load_frames(folder_path: str) -> list[np.ndarray]:
    
    supported = ('.png', '.jpg', '.jpeg')
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(supported)
    ])
    if not files:
        raise ValueError(f"No image frames found in: {folder_path}")

    frames = []
    for fname in files:
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Could not read image: {path}")
        frames.append(img)

    print(f"[Part 1] Loaded {len(frames)} frames from '{folder_path}'")
    return frames


def preprocess_frames(frames: list[np.ndarray]) -> list[dict]:
    
    processed = []
    for i, frame in enumerate(frames):
        Y, Cb, Cr = bgr_to_ycbcr(frame)
        Cb_sub, Cr_sub = chroma_subsample_420(Cb, Cr)
        processed.append({
            'index': i,
            'Y': Y,
            'Cb_sub': Cb_sub,
            'Cr_sub': Cr_sub,
            'original_shape': frame.shape,
        })

    print(f"[Part 1] Pre-processed {len(processed)} frames (YCbCr + 4:2:0 subsampling)")
    return processed


if __name__ == "__main__":

    dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Y, Cb, Cr = bgr_to_ycbcr(dummy)
    Cb_s, Cr_s = chroma_subsample_420(Cb, Cr)
    Cb_u, Cr_u = chroma_upsample(Cb_s, Cr_s, dummy.shape)
    recon = ycbcr_to_bgr(Y, Cb_u, Cr_u)
    print(f"Input shape:  {dummy.shape}")
    print(f"Y shape:      {Y.shape}")
    print(f"Cb_sub shape: {Cb_s.shape}")
    print(f"Recon shape:  {recon.shape}")
    print("Part 1 OK")