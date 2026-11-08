"""NumPy 2.4.2's speed for ``np.strings.encode`` / ``decode`` with codecs NumSharp refuses, next to the UTF-8 C fast path
(``docs/plans/strings-codecs.md``, section 5): the floor a NumSharp port must beat by 1.5x.

Three data sets: 100K elements of 12 random CJK ideographs, 100K short ASCII elements, and one element of 100K CJK
ideographs (the per-character cost without the per-element call). Each cell is the best of 5 single calls after one
untimed call. Pin the process to two CPUs as the benchmark convention asks (on this host, from cmd):

    start /b /wait /affinity 3 python -I docs/plans/strings-codecs/bench_numpy.py
"""
import sys
import timeit

import numpy as np

CODECS = ("utf-8", "gbk", "shift_jis", "big5", "utf-7", "unicode_escape")


def datasets():
    """The three inputs.

    :returns: ``[(label, U array)]``.
    """
    rng = np.random.default_rng(0)
    cjk = np.array(["".join(chr(int(c)) for c in rng.integers(0x4E00, 0x4E00 + 3000, 12)) for _ in range(100_000)])
    ascii_ = np.array(["hello world %d" % i for i in range(100_000)])
    long_cjk = np.array(["".join(chr(int(c)) for c in rng.integers(0x4E00, 0x4E00 + 3000, 100_000))])
    return [("cjk12 x 100K", cjk), ("ascii x 100K", ascii_), ("cjk 1 x 100K", long_cjk)]


def best(fn):
    """The best of 5 single calls, in seconds.

    :param fn: the call.
    :returns: the minimum time.
    """
    return min(timeit.repeat(fn, number=1, repeat=5))


def main():
    """Prints one line per data set and codec; a codec that cannot encode the data (shift_jis / big5 lack many of the
    random ideographs) is reported as raising."""
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"NumPy {np.__version__}, CPython {sys.version.split()[0]}")
    for label, a in datasets():
        for codec in CODECS:
            try:
                e = np.strings.encode(a, codec)
            except UnicodeEncodeError:
                print(f"{label:13s} {codec:15s} encode raises UnicodeEncodeError")
                continue
            np.strings.decode(e, codec)
            te = best(lambda: np.strings.encode(a, codec))
            td = best(lambda: np.strings.decode(e, codec))
            print(f"{label:13s} {codec:15s} encode {te * 1e3:9.2f} ms ({te / a.size * 1e9:9.0f} ns/elem)  "
                  f"decode {td * 1e3:9.2f} ms ({td / a.size * 1e9:9.0f} ns/elem)")


if __name__ == "__main__":
    main()
