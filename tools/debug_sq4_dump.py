import argparse
import struct
import numpy as np

def load_nibbles(buf: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    out[0::2] = b[: (n + 1) // 2] & 0x0F
    out[1::2] = b[: n // 2] >> 4
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    args = ap.parse_args()

    with open(args.bin, "rb") as f:
        # header: 7*i32 + group_size*i32
        hdr = f.read(7 * 4)
        dim, hidden, nlayer, nhead, nkv, vocab_signed, seqlen = struct.unpack("<7i", hdr)
        group_size, = struct.unpack("<i", f.read(4))
        vocab = abs(vocab_signed)

        print("header:", dict(dim=dim, hidden=hidden, nlayer=nlayer, nhead=nhead, nkv=nkv,
                             vocab_signed=vocab_signed, seqlen=seqlen, group_size=group_size))

        # first block desc
        desc = f.read(32)
        b_type, rows, cols, q_size, s_size, z_size, gsz, _ = struct.unpack("<8i", desc)
        print("desc:", dict(type=b_type, rows=rows, cols=cols, q_size=q_size, s_size=s_size, z_size=z_size, gsz=gsz))

        q_bytes = f.read(q_size)
        s_bytes = f.read(s_size)
        z_bytes = f.read(z_size)

        assert cols % gsz == 0
        groups_per_row = cols // gsz
        group_count = rows * groups_per_row

        q4 = load_nibbles(q_bytes, rows * cols).reshape(rows, cols)
        z4 = load_nibbles(z_bytes, group_count).reshape(rows, groups_per_row)
        s = np.frombuffer(s_bytes, dtype=np.float32).reshape(rows, groups_per_row)

        z0 = int(z4[0, 0])
        s0 = float(s[0, 0])
        print("row0 group0:", dict(z0=z0, s0=s0))

        for i in range(16):
            q = int(q4[0, i])
            w = (q - z0) * s0
            print(f"w(0,{i}): q4={q} deq={w}")

if __name__ == "__main__":
    main()