import numpy as np, tempfile, os, struct

d = tempfile.gettempdir()
sizes = [1, 7, 8, 9, 100, 128, 129, 200, 256, 1000, 3163, 10000, 100000]
for n in sizes:
    a64 = np.fromfile(os.path.join(d, f"pw_f64_{n}.bin"), dtype=np.float64)
    a32 = np.fromfile(os.path.join(d, f"pw_f32_{n}.bin"), dtype=np.float32)
    r64 = np.add.reduce(a64)                 # == np.sum; the pairwise reduce path
    r32 = np.add.reduce(a32)
    b64 = struct.unpack('<Q', struct.pack('<d', float(r64)))[0]
    b32 = struct.unpack('<I', struct.pack('<f', np.float32(r32)))[0]
    print(f"f64 {n} {b64:016X}")
    print(f"f32 {n} {b32:08X}")
