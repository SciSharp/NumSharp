#!/usr/bin/env python3
# NumPy 8-layout reference for {f32,f64,c128} -> u64 (bucket B). Values exercise the modular
# wrap (negatives), the 2^63 sentinel (NaN/+-inf/overflow), and normal truncation, so every
# new load path (negcol reverse, strided deinterleave, c128 negcol deinterleave-reverse) is
# checked. Emits key\tsha256(result_bytes).
import hashlib, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

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

# f64 base: negatives (wrap) + fractions (trunc) + scattered specials (inf/nan/overflow -> 2^63)
flat = (np.arange(R*C, dtype=np.float64) * 7.3 - 3000.0)
for k, val in enumerate([np.inf, -np.inf, np.nan, 1e20, -1e20, 2e19, 9.3e18, -5.0, 0.0]):
    flat[(k * 911 + 13) % flat.size] = val
base64 = flat.reshape(R, C)
base32 = base64.astype(np.float32)
basec = base64.astype(np.complex128)

lines = []
for tag, base in [("f32", base32), ("f64", base64), ("c128", basec)]:
    for l in LAY:
        v = layout(base, l)
        r = v.astype(np.uint64, copy=True)
        h = hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
        lines.append(f"{tag}|{l}\t{h}\t{r.size}")

with open(os.path.join(OUT, "float_u64_layout.tsv"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} hashes")
