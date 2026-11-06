"""Python TensorFlow 2.21 + NumPy 2.4.2 twin of the C# probe's timing section: the same kind of 1M random texts
(lengths 1-40 code points, one in ten from a non-ASCII / astral palette), crossing NumPy <-> TF string tensors the ways
Python offers. Best of 5 after 2 warm-ups, in ms and ns per element.

Run: python -I py_timing.py [n]
"""
import random
import sys
import time

import numpy as np
import tensorflow as tf

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = random.Random(12345)
ascii_palette = [chr(c) for c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "]
palette = ascii_palette + [chr(0xE9), chr(0xDF), chr(0x65E5), chr(0x1F600)]
texts = []
for _ in range(n):
    length = rng.randint(1, 40)
    texts.append("".join(rng.choice(palette) if rng.randrange(10) == 0 else rng.choice(ascii_palette) for _ in range(length)))

u = np.array(texts)
s = np.strings.encode(u)
t = np.array(texts, dtype=np.dtypes.StringDType())
o = np.array(texts, dtype=object)
print(f"U itemsize {u.dtype.itemsize}, S itemsize {s.dtype.itemsize}")


def best(label, fn):
    """Time fn: 2 untimed warm-ups, then the best of 5."""
    for _ in range(2):
        fn()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    ms = min(times) * 1e3
    print(f"TIME  {label:<58} {ms:9.1f} ms  {ms * 1e6 / n:7.1f} ns/elem", flush=True)


best("export U   tf.constant(U array)", lambda: tf.constant(u))
best("export S   tf.constant(S array)", lambda: tf.constant(s))
best("export T   tf.constant(StringDType array)  [refused]", lambda: None)
best("export T   tf.constant(T.astype(object))", lambda: tf.constant(t.astype(object)))
best("export obj tf.constant(object array of str)", lambda: tf.constant(o))
source = tf.constant(u)
best("import     tensor.numpy() -> object array of bytes", lambda: source.numpy())
best("import ->T tensor.numpy().astype(StringDType())", lambda: source.numpy().astype(np.dtypes.StringDType()))
best("import ->S tensor.numpy().astype('S')", lambda: source.numpy().astype("S"))
best("import ->U np.strings.decode(tensor.numpy().astype('S'))", lambda: np.strings.decode(source.numpy().astype("S")))
