#!/usr/bin/env python3
# NumPy reference for same-size sub-word casts (the byte-copy pairs the SubwordCopy
# kernel now intercepts). Emits "key\tsha256(result_bytes)" so the C# twin can prove
# its strided cross-cast is BIT-EXACT with NumPy across all 8 layouts.
#   Writes to benchmark/poc/_xref/np_hashes.tsv (abs path; bash/.NET /tmp differ).
import hashlib, os
import numpy as np

R = C = 130                       # even width so [:, ::2] is exact; small
OUT = r"K:\source\NumSharp\benchmark\poc\_xref"
os.makedirs(OUT, exist_ok=True)

# NumSharp char == 16-bit unsigned -> map to uint16 for byte-identical comparison.
DT = {"bool": np.bool_, "u8": np.uint8, "i8": np.int8,
      "i16": np.int16, "u16": np.uint16, "char": np.uint16, "f16": np.float16}

# pairs the kernel intercepts (same size; same-type, int<->int, bool->int; no Half cross, no X->bool)
PAIRS = [
    ("bool","bool"),("u8","u8"),("i8","i8"),("bool","u8"),("bool","i8"),("u8","i8"),("i8","u8"),
    ("i16","i16"),("u16","u16"),("char","char"),("f16","f16"),
    ("i16","u16"),("i16","char"),("u16","i16"),("u16","char"),("char","i16"),("char","u16"),
    # 2B-int -> 1B narrowing (low-byte truncate) + ->bool (!=0): SubwordNarrow kernel
    ("i16","i8"),("i16","u8"),("u16","i8"),("u16","u8"),("char","i8"),("char","u8"),
    ("i16","bool"),("u16","bool"),("char","bool"),
    # 1B-int -> 2B widening (sign-extend i8 / zero-extend u8,bool): SubwordWiden kernel
    ("i8","i16"),("i8","u16"),("i8","char"),("u8","i16"),("u8","u16"),("u8","char"),
    ("bool","i16"),("bool","u16"),("bool","char"),
]
LAY = ["C","F","T","sliced","negrow","negcol","strided","bcast"]

def layout(b, l):
    if l=="C": return b
    if l=="F": return np.asfortranarray(b)
    if l=="T": return b.T
    if l=="sliced": return b[1:R-1, 1:C-1]
    if l=="negrow": return b[::-1, :]
    if l=="negcol": return b[:, ::-1]
    if l=="strided": return b[:, ::2]
    if l=="bcast": return np.broadcast_to(b[0:1, :], (R, C))
    raise ValueError(l)

lines = []
for sn, dn in PAIRS:
    base = (np.arange(R*C) % 65521).astype(DT[sn]).reshape(R, C)   # incl 0 + diverse high bytes
    for l in LAY:
        v = layout(base, l)
        r = v.astype(DT[dn], copy=True)
        # contiguous C-order bytes of the logical result (astype yields a fresh array)
        h = hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
        lines.append(f"{sn}|{l}|{dn}\t{h}\t{r.size}")

with open(os.path.join(OUT, "np_hashes.tsv"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} reference hashes to {OUT}\\np_hashes.tsv")
