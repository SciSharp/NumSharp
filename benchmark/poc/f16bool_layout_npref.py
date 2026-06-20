#!/usr/bin/env python3
# NumPy 8-layout reference for f16 -> bool (Wave 17 bucket E). Special values (+-0, NaN, +-inf,
# subnormal, 65504) exercise the half-truthiness `(bits & 0x7fff) != 0`. key\tsha256.
import hashlib, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
R = C = 130; OUT = r"K:\source\NumSharp\benchmark\poc\_xref"; os.makedirs(OUT, exist_ok=True)
def layout(b, l):
    return {"C":b,"F":np.asfortranarray(b),"T":b.T,"sliced":b[1:R-1,1:C-1],"negrow":b[::-1,:],
            "negcol":b[:,::-1],"strided":b[:,::2],"bcast":np.broadcast_to(b[0:1,:],(R,C))}[l]
flat = (np.arange(R*C) % 17).astype(np.float16)
specials = [np.float16(x) for x in (0.0,-0.0,np.nan,np.inf,-np.inf,6e-8,65504.0,-1.0,1e-7)]
for k, val in enumerate(specials): flat[(k*911+13)%flat.size] = val
base = flat.reshape(R, C)
lines = []
for l in ["C","F","T","sliced","negrow","negcol","strided","bcast"]:
    r = layout(base, l).astype(np.bool_, copy=True)
    lines.append(f"f16|{l}\t{hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()}\t{r.size}")
open(os.path.join(OUT,"f16bool_layout.tsv"),"w").write("\n".join(lines)+"\n")
print(f"wrote {len(lines)} hashes")
