#!/usr/bin/env python3
# NumPy 8-layout reference for i64/u64 -> f16 (bucket A). Values span -inf / finite / +inf / 0
# so every stride path exercises all three result classes. Emits key\tsha256(result_bytes).
import hashlib, os
import numpy as np

R = C = 130
OUT = r"K:\source\NumSharp\benchmark\poc\_xref"
os.makedirs(OUT, exist_ok=True)
LAY = ["C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast"]

def layout(b, l):
    if l == "C": return b
    if l == "F": return np.asfortranarray(b)
    if l == "T": return b.T
    if l == "sliced": return b[1:R-1, 1:C-1]
    if l == "negrow": return b[::-1, :]
    if l == "negcol": return b[:, ::-1]
    if l == "strided": return b[:, ::2]
    if l == "bcast": return np.broadcast_to(b[0:1, :], (R, C))
    raise ValueError(l)

lines = []
# i64: span -200000..~16.7M  -> covers -inf, finite, +inf
base_i = (np.arange(R*C, dtype=np.int64) * 991 - 200000).reshape(R, C)
# u64: 0..~16.7M -> covers 0, finite, +inf
base_u = (np.arange(R*C, dtype=np.uint64) * 991).reshape(R, C)

for tag, base in [("i64", base_i), ("u64", base_u)]:
    for l in LAY:
        v = layout(base, l)
        r = v.astype(np.float16, copy=True)
        h = hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
        lines.append(f"{tag}|{l}\t{h}\t{r.size}")

with open(os.path.join(OUT, "i64u64_half_layout.tsv"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} hashes")
