import zlib
import pickle
import struct
import os
import numpy as np




MAGIC = b'MP4S'




def _compress(obj) -> bytes:

    raw = pickle.dumps(obj, protocol=4)
    return zlib.compress(raw, level=9)


def _decompress(data: bytes):

    raw = zlib.decompress(data)
    return pickle.loads(raw)


def encode_to_bin(encoded_frames: list, output_path: str,
                  metadata: dict = None) -> int:
    
    if metadata is None:
        metadata = {}


    frame_headers = []
    frame_data_list = []

    for enc in encoded_frames:
        header = {k: v for k, v in enc.items()
                  if not k.endswith('_coeffs')}
        data = {k: v for k, v in enc.items()
                if k.endswith('_coeffs') or k == 'motion_vectors'}
        frame_headers.append(header)
        frame_data_list.append(data)

    global_meta = {
        'metadata': metadata,
        'frame_headers': frame_headers,
        'num_frames': len(encoded_frames),
    }

    compressed_meta = _compress(global_meta)
    compressed_frames = [_compress(fd) for fd in frame_data_list]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:

        f.write(MAGIC)
        f.write(struct.pack('>I', len(encoded_frames)))
        f.write(struct.pack('>I', len(compressed_meta)))
        f.write(compressed_meta)


        for cdata in compressed_frames:
            f.write(struct.pack('>I', len(cdata)))
            f.write(cdata)

    size = os.path.getsize(output_path)
    print(f"[Part 4] Written {len(encoded_frames)} frames → '{output_path}' ({size:,} bytes)")
    return size



def decode_from_bin(input_path: str) -> tuple[list, dict]:
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid file format: expected {MAGIC}, got {magic}")

        num_frames = struct.unpack('>I', f.read(4))[0]
        meta_len   = struct.unpack('>I', f.read(4))[0]

        global_meta = _decompress(f.read(meta_len))

        compressed_frames = []
        for _ in range(num_frames):
            clen  = struct.unpack('>I', f.read(4))[0]
            cdata = f.read(clen)
            compressed_frames.append(cdata)

    frame_headers = global_meta['frame_headers']
    metadata      = global_meta['metadata']

    encoded_frames = []
    for header, cdata in zip(frame_headers, compressed_frames):
        frame_data = _decompress(cdata)
        enc = {**header, **frame_data}
        encoded_frames.append(enc)

    print(f"[Part 4] Loaded {len(encoded_frames)} frames from '{input_path}'")
    return encoded_frames, metadata



def compute_original_size(encoded_frames: list) -> int:
    
    total = 0
    for enc in encoded_frames:
        h, w, c = enc['original_shape']
        total += h * w * c
    return total


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from part1_preprocessing import bgr_to_ycbcr, chroma_subsample_420
    from part3_pframe import encode_gop

    # Synthetic test
    frames_raw = []
    for i in range(8):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Y, Cb, Cr = bgr_to_ycbcr(bgr)
        Cb_s, Cr_s = chroma_subsample_420(Cb, Cr)
        frames_raw.append({
            'index': i, 'Y': Y, 'Cb_sub': Cb_s,
            'Cr_sub': Cr_s, 'original_shape': bgr.shape
        })

    encoded = encode_gop(frames_raw, gop_size=4, qf=1.0, search_window=4)

    out_path = '/tmp/test_output.bin'
    meta = {'gop_size': 4, 'qf': 1.0, 'search_window': 4}
    bin_size = encode_to_bin(encoded, out_path, meta)

    loaded, loaded_meta = decode_from_bin(out_path)

    orig_size = compute_original_size(encoded)
    ratio = orig_size / bin_size
    print(f"Original size:    {orig_size:,} bytes")
    print(f"Compressed size:  {bin_size:,} bytes")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Metadata: {loaded_meta}")
    print("Part 4 OK")