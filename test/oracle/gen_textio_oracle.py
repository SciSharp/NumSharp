#!/usr/bin/env python3
"""NumPy 2.4.2 oracle for NumSharp's text readers and writers: ``np.fromstring``, ``np.fromfile``, ``np.loadtxt``,
``np.savetxt`` and ``ndarray.tofile`` over every dtype (strings phase 1, the validation pass of P1.7c).

Writes ``test/NumSharp.Tests.Oracle/TextIO/corpus/textio_<section>.jsonl``, one file per section, replayed by
``test/NumSharp.Tests.Oracle/TextIO/TextIOOracleTests.cs`` (category ``TextIOOracle``; no Python at test time):

* ``fromtext`` -- ``np.fromstring`` (a ``str`` and a ``bytes`` argument) and ``np.fromfile``'s text mode (a file name,
  an open file and the file's position afterwards): NumPy's ``array_from_text`` with its separators and counts, and its
  C scanners (CPython's ``PyOS_strtol`` / ``PyOS_strtoul`` and NumPy's ``NumPyOS_ascii_strtod`` for a string, glibc's
  ``fscanf`` for a file), over a vocabulary of signs, digits, overflows, NaN / inf spellings, complex forms and junk.
* ``frombin`` -- ``np.fromfile``'s binary mode: counts, offsets, partial trailing items, the position afterwards.
* ``fromedges`` -- a fixed grid for both text readers: every scanner token (integer magnitudes at every width's limit in
  every sign, malformed integers, NaN / inf spellings, float limits, complex forms) alone, after a clean element and
  before one; every separator in seven placements; bytes at or past 0x80 where NumPy's signed ``char`` loses a 0xFF.
  A random text usually fails on an early token, so the rules for a value that IS read (an overflow's saturation, a
  lost byte) need this grid to decide outcomes.
* ``loadtxt`` -- ``np.loadtxt`` with every option (delimiters, comments, quoting, ``skiprows``, ``max_rows``,
  ``usecols``, ``ndmin``, ``unpack``, encodings, converters) over the four sources it reads (a list of lines, a text
  stream, a byte stream, a file name) and every dtype, ``StringDType`` included.
* ``ltfields`` -- numeric fields: float16 (Python's ``float()``), float32 / float64 / complex128 and the integer
  dtypes. Fields NumPy accepts are grouped 64 to a case; every refused field is its own case.
* ``lttokens`` -- the tokenizer's edges, read as text dtypes so every field's characters are visible: Unicode
  whitespace, ``U+001C..U+001F``, newlines inside lines, comments, quotes, empty fields and fields longer than the
  tokenizer's 16-character scalar scan.
* ``savetxt`` / ``tofile`` -- the writers: ``%`` formats (valid and broken), delimiters, newlines, headers, footers,
  comments and encodings into a text stream, a byte stream and a file; ``ndarray.tofile``'s binary and text modes;
  every layout of the written array.

Each file starts with a header record (``{"header": {...}}``: the generator, NumPy and Python versions, the seed and
the case count, which the replay checks), then one case per line.

**Run it with Linux NumPy 2.4.2** (WSL), from a directory outside ``refs/numpy`` (whose sources would shadow the
installed package)::

    wsl -e bash -lc 'cd ~ && ~/np242/bin/python /mnt/k/source/NumSharp/test/oracle/gen_textio_oracle.py'

then ``dotnet build`` the oracle project to copy the corpus. ``--sections fromtext,loadtxt`` regenerates a subset;
``--out DIR`` writes somewhere else (a dry run).

**Why Linux.** NumSharp pins the C ``long`` to 64 bits and ``fscanf`` to glibc's (Linux / macOS NumPy, as the random
module pins its legacy integers): Windows NumPy's 32-bit ``long`` and MSVC's ``fscanf`` parse 247 of 20,000 random
integer texts differently. The host check refuses any other platform. A file NAME opened without an encoding is UTF-8
here and in NumSharp (NumPy's locale encoding on Linux / macOS).

**Newlines.** ``np.savetxt`` into a file NAME writes through Python's text mode, which turns every ``\\n`` into
``os.linesep`` before encoding, and NumSharp does the same, so the bytes differ between Windows and POSIX hosts. Each
file-target case therefore records a second outcome, ``crlf``: the same call written through
``io.TextIOWrapper(io.BytesIO(), encoding=..., newline="\\r\\n")``, which is what Windows' text mode does. The
generator first proves the wrapper reproduces the real file on this host (``newline="\\n"``), and drops a case whose
wrapper and file disagree (it reports the count, which must be zero). The replay picks ``crlf`` on a Windows host.

**What the corpus leaves out**, because NumSharp documents the divergence and unit-tests it instead: results NumPy
builds with zero-width string elements (NumSharp cannot hold one); a non-native dtype's TEXT read, which NumPy
byte-swaps after its scanner wrote a native value (the case records NumPy's answer for the native dtype, which is
NumSharp's); and ``ndarray.tofile`` of an element whose text is not ASCII, which crashes NumPy 2.4.2's process.

Strings travel as JSON strings, or as ``{"u16": hex}`` (UTF-16LE) when they hold a lone surrogate (System.Text.Json
refuses one); bytes as ``{"utf8": str}`` when they are strict UTF-8, else ``{"hex": ...}``. The files are raw UTF-8.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import locale
import os
import platform
import random
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

GENERATOR_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "test" / "NumSharp.Tests.Oracle" / "TextIO" / "corpus"

# The seed and the number of cases of each section. ltfields' count is the number of field texts drawn (the case count
# follows from how many NumPy refuses); fromedges is a fixed grid (its count is unused); the others are case counts.
SECTIONS = {
    "fromtext": (8101, 10000),
    "frombin": (8102, 3000),
    "fromedges": (8108, 0),
    "loadtxt": (8103, 8000),
    "ltfields": (8104, 24000),
    "lttokens": (8105, 8000),
    "savetxt": (8106, 6000),
    "tofile": (8107, 3000),
}

# GitHub warns above 50 MiB per file; a section past this size is an error (shrink its count).
MAX_SECTION_BYTES = 40 * 1024 * 1024

# Inline payloads up to these sizes; larger ones travel as their SHA-256 and length.
INLINE_BYTES = 4096
INLINE_TEXT = 8192


# ---------------------------------------------------------------------------------------------------------------------
# Host and encodings
# ---------------------------------------------------------------------------------------------------------------------

def check_host():
    """Refuses to run anywhere but NumSharp's pinned platform model.

    :raises SystemExit: NumPy is not 2.4.2, Python is not 3.12, the C ``long`` is not 64 bits, the C library is not
        glibc, the locale encoding is not UTF-8, or ``os.linesep`` is not ``"\\n"``.
    """
    problems = []
    if np.__version__ != "2.4.2":
        problems.append(f"NumPy {np.__version__}, not 2.4.2")
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {sys.version.split()[0]}, not 3.12 (CPython's int() / float() texts and grammar)")
    if ctypes.sizeof(ctypes.c_long) != 8:
        problems.append("a 32-bit C long (NumSharp pins LP64)")
    if platform.libc_ver()[0] != "glibc":
        problems.append(f"C library {platform.libc_ver()[0] or 'unknown'}, not glibc (fscanf's behavior)")
    if locale.getpreferredencoding(False).lower().replace("-", "").replace("_", "") != "utf8":
        problems.append(f"locale encoding {locale.getpreferredencoding(False)}, not UTF-8")
    if os.linesep != "\n":
        problems.append("os.linesep is not \\n")
    if problems:
        raise SystemExit("gen_textio_oracle.py must run with Linux NumPy 2.4.2: " + "; ".join(problems))


def S(s):
    """A ``str`` in the corpus: itself, or ``{"u16": hex}`` when it holds a lone surrogate.

    :param s: the string, or None.
    :returns: the JSON value (None stays None).
    """
    if s is None:
        return None
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        return {"u16": s.encode("utf-16-le", "surrogatepass").hex()}


def B(b):
    """Bytes in the corpus: ``{"utf8": str}`` when they are strict UTF-8 (a quarter of hex's size for text), else hex.

    :param b: the bytes.
    :returns: the JSON object.
    """
    try:
        return {"utf8": b.decode("utf-8")}
    except UnicodeDecodeError:
        return {"hex": b.hex()}


def u16(s):
    """A str as UTF-16LE hex, lone surrogates kept (the converter-log keys and StringDType cells use it).

    :param s: the string.
    :returns: the hex text.
    """
    return s.encode("utf-16-le", "surrogatepass").hex()


def bytes_out(b):
    """An output's bytes: inline up to INLINE_BYTES, else their SHA-256 and length.

    :param b: the bytes.
    :returns: the JSON object.
    """
    if len(b) <= INLINE_BYTES:
        return B(b)
    return {"sha": hashlib.sha256(b).hexdigest(), "len": len(b)}


def text_out(t):
    """An output's text: inline up to INLINE_TEXT characters, else the SHA-256 of its UTF-16LE code units.

    :param t: the text.
    :returns: the JSON object.
    """
    if len(t) <= INLINE_TEXT:
        return {"text": S(t)}
    return {"sha16": hashlib.sha256(t.encode("utf-16-le", "surrogatepass")).hexdigest(), "len": len(t)}


def error_of(e):
    """NumPy's exception as the corpus records it.

    :param e: the exception.
    :returns: ``{"type": class name, "msg": str(e)}`` (the text through :func:`S`: a message can quote a field).
    """
    return {"type": type(e).__name__, "msg": S(str(e))}


def array_out(r, nread=None):
    """A numeric or fixed-width result: its dtype name, shape and C-order bytes (of the written prefix ``nread``).

    :param r: the array (a non-native one is reported as its native twin, which is NumSharp's).
    :param nread: the number of leading elements NumPy wrote (np.fromstring with a count), or None for all.
    :returns: the JSON object, or None for a result NumSharp cannot hold (zero-width string elements).
    """
    if r.dtype.kind in "SU" and r.dtype.itemsize == 0:
        return None
    if r.dtype.kind in "biufc" and r.dtype.itemsize > 1 and not r.dtype.isnative:
        r = r.astype(r.dtype.newbyteorder("="))
    out = {"dtype": str(r.dtype), "shape": list(r.shape)}
    if nread is not None and r.dtype.kind in "biufc":
        out["nread"] = nread
        payload = np.ascontiguousarray(r).reshape(-1)[:nread].tobytes()
    else:
        payload = np.ascontiguousarray(r).tobytes()
    out.update(bytes_out(payload) if len(payload) > INLINE_BYTES else {"hex": payload.hex()})
    return out


def cell(x):
    """One StringDType element: ``null`` for None, ``{"nan": true}`` for a NaN NA object, the text through :func:`S`,
    or ``{"o": repr}`` for any other object.

    :param x: the element as ``tolist()`` returns it.
    :returns: the JSON value.
    """
    if x is None:
        return None
    if isinstance(x, float) and x != x:
        return {"nan": True}
    if isinstance(x, str):
        return S(x)
    return {"o": repr(x)}


def cells_out(r):
    """A StringDType result as its elements (:func:`cell`).

    :param r: the array.
    :returns: the JSON object.
    """
    return {"dtype": str(r.dtype), "shape": list(r.shape), "cells": [cell(x) for x in r.ravel().tolist()]}


def result_out(r):
    """Any loadtxt result: cells for StringDType, bytes otherwise (None when NumSharp cannot hold it).

    :param r: the array.
    :returns: the JSON object or None.
    """
    if isinstance(r.dtype, np.dtypes.StringDType):
        return cells_out(r)
    return array_out(r)


# ---------------------------------------------------------------------------------------------------------------------
# Section fromtext / frombin: np.fromstring and np.fromfile
# ---------------------------------------------------------------------------------------------------------------------

FS_NUMERIC = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
              "float16", "float32", "float64", "complex128"]
FS_OTHER = ["U3", "S2", "U", "S", "T", ">i4", "<i4", ">f8", "=f8", "|b1", ">c16", "<u2"]
FS_INTS = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "<u2", ">i4", "<i4")
FS_WS = [" ", "  ", "\t", "\n", "\r", "\v", "\f", "\r\n", " \t ", "\x1c", "\xa0", "　", "\x00"]
FS_SEPS = [" ", ",", ", ", " , ", "  ", "\t", "\n", "x", "ab", ";", "::", "\xe9", "a b", " ,", ", ,", "\x1c", "-",
           "+", ".", "e", "j", "0", "1", "\x00", "", "\ud800", "　"]


def fs_digits(rng, lo=1, hi=22):
    """A run of decimal digits.

    :param rng: the section's generator.
    :param lo: the fewest digits.
    :param hi: the most digits.
    :returns: the digits.
    """
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(lo, hi)))


def fs_int_token(rng):
    """An integer text: signs (doubled, spaced), bases, limits past every width, long runs, Unicode digits.

    :param rng: the section's generator.
    :returns: the text.
    """
    r = rng.random()
    sign = rng.choice(["", "", "", "-", "+", "+-", "-+", "- ", "+ ", "--"])
    if r < 0.15:
        return sign + rng.choice(["0", "00", "0x10", "0X1f", "0b11", "0o7", "010", "1_000", "١٢",
                                  "１２"])
    if r < 0.3:
        return sign + rng.choice(["127", "128", "255", "256", "32767", "32768", "65535", "65536", "2147483647",
                                  "2147483648", "4294967295", "4294967296", "9223372036854775807",
                                  "9223372036854775808", "18446744073709551615", "18446744073709551616",
                                  "99999999999999999999999999"])
    return sign + fs_digits(rng, 1, rng.choice([3, 5, 12, 22]))


def fs_float_token(rng):
    """A float text: NaN / inf spellings, limits, broken exponents and points, long mantissas, ordinary decimals.

    :param rng: the section's generator.
    :returns: the text.
    """
    r = rng.random()
    sign = rng.choice(["", "", "", "-", "+", "+-", "- "])
    if r < 0.2:
        return sign + rng.choice(["inf", "INF", "Inf", "infinity", "Infinity", "INFINITY", "infin", "in", "i",
                                  "nan", "NaN", "NAN", "nan(123)", "nan(abc_9)", "nan(", "nan()", "nan(x y)",
                                  "nanx", "n", "na", "infinityx", "infinit"])
    if r < 0.3:
        return sign + rng.choice(["1e400", "1e-400", "1e308", "2e-308", "4.9e-324", "2.5e-324",
                                  "1.7976931348623157e308", "1.7976931348623159e308", "65504", "65520", "6e-8",
                                  "3.4028235e38", "3.4028236e38", "0x1p3", "1_0", "1e", "1e+", "1e-", "e5", ".",
                                  ".e5", "5.", ".5", "0.", "-.5e-3", "1.5e3.2", "1..2", "1e5e5", "1E5", "1e05",
                                  "1e0005"])
    if r < 0.35:
        # a mantissa past fromfile's 120-character number buffer, and a correctly rounded tie
        return sign + fs_digits(rng, 100, 140) + rng.choice(["", ".5", "e-100"])
    mant = fs_digits(rng, 1, 12)
    if rng.random() < 0.6:
        mant += "." + fs_digits(rng, 0, 20)
    if rng.random() < 0.3:
        mant = "." + fs_digits(rng, 1, 8)
    if rng.random() < 0.4:
        mant += rng.choice(["e", "E"]) + rng.choice(["", "+", "-"]) + fs_digits(rng, 1, 3)
    return sign + mant


def fs_complex_token(rng):
    """A complex text: NumPy's forms, the near misses (``J``, parentheses, spaces, doubled signs) and real parts.

    :param rng: the section's generator.
    :returns: the text.
    """
    r = rng.random()
    if r < 0.25:
        return rng.choice(["1+2j", "1-2j", "2j", "j", "+j", "-j", "1+j", "1-j", "1+2J", "(1+2j)", "1e5+1e-5j",
                           "nan+nanj", "inf-infj", "1+2", "1+2jx", "1+-2j", "1-+2j", "+1+2j", "-1-2j", "1j+2",
                           "1 +2j", "1+ 2j", "1+2 j", "0j", "-0-0j", "1e+2j", "1e+2+3j", "infj", "nanj",
                           "1+infj", "1-nanj", "1+2jj", "jj", "1.5", "-2.5e3"])
    a = fs_float_token(rng)
    if rng.random() < 0.5:
        return a
    b = fs_float_token(rng).lstrip("+-")
    return a + rng.choice(["+", "-"]) + b + rng.choice(["j", "j", "J", ""])


def fs_junk_token(rng):
    """A token no scanner accepts as a whole (or only in part).

    :param rng: the section's generator.
    :returns: the text.
    """
    return rng.choice(["abc", "", "x", "\x00", "\xe9", "1,2", "12abc", "1.5abc", "--1", "++1", "0x", " ", " ",
                       "١", "True", "False", "true", "None", "1e5j", "(1)", "[1]", "1/2", "\ud800"])


def fs_kind(dt):
    """The scanner family a dtype reads with.

    :param dt: the dtype name.
    :returns: ``int``, ``float`` or ``complex``.
    """
    if dt in FS_INTS:
        return "int"
    if dt.startswith("complex") or "c16" in dt:
        return "complex"
    return "float"


def fs_token(rng, dt):
    """One token for a dtype: mostly its own family, sometimes another family or junk.

    :param rng: the section's generator.
    :param dt: the dtype name.
    :returns: the text.
    """
    if rng.random() < 0.08:
        return fs_junk_token(rng)
    kind = fs_kind(dt)
    if dt in ("bool", "|b1"):
        kind = rng.choice(["int", "float"])
    if rng.random() < 0.15:
        kind = rng.choice(["int", "float", "complex"])
    return {"int": fs_int_token, "float": fs_float_token, "complex": fs_complex_token}[kind](rng)


def fs_clean_token(rng, dt):
    """A token every reader of the dtype's family accepts (for long texts across the file reader's chunk boundaries).

    :param rng: the section's generator.
    :param dt: the dtype name.
    :returns: the text.
    """
    kind = fs_kind(dt)
    if kind == "int":
        return str(rng.randint(0, 10 ** rng.randint(1, 18)))
    if kind == "complex":
        return f"{rng.uniform(-1e6, 1e6)!r}{rng.choice('+-')}{abs(rng.uniform(-1e6, 1e6))!r}j"
    return repr(rng.uniform(-1e9, 1e9) * 10 ** rng.randint(-30, 30))


def fs_long_text(rng, dt, sep):
    """A long, clean text of one dtype's tokens, 66 to 72 KB, so the C FILE model refills its 64 KB chunk once, at a
    random point (inside a number or a separator), optionally with one bad token after the refill.

    :param rng: the section's generator.
    :param dt: the dtype name.
    :param sep: the separator (a space when empty).
    :returns: the text.
    """
    joiner = sep if sep else " "
    target = rng.randint(66_000, 72_000)
    toks, size = [], 0
    while size < target:
        t = fs_clean_token(rng, dt)
        toks.append(t)
        size += len(t) + len(joiner)
    if rng.random() < 0.3:
        toks[rng.randrange(len(toks) // 2, len(toks))] = "zz"
    return joiner.join(toks)


def fs_text(rng, dt, sep):
    """A short text: a few tokens with separators, whitespace runs, doubled or missing separators, and leading or
    trailing separators.

    :param rng: the section's generator.
    :param dt: the dtype name.
    :param sep: the separator.
    :returns: the text.
    """
    n = rng.choice([0, 1, 1, 2, 3, 3, 4, 5, 8])
    toks = [fs_token(rng, dt) for _ in range(n)]
    core = sep.strip() if sep.strip() else " "
    parts = []
    for i, t in enumerate(toks):
        lead = rng.choice(["", "", "", rng.choice(FS_WS)])
        parts.append(lead + t)
        if i != n - 1:
            r = rng.random()
            if r < 0.7:
                parts.append(sep if rng.random() < 0.6
                             else rng.choice(FS_WS) + core + rng.choice(["", rng.choice(FS_WS)]))
            elif r < 0.8:
                parts.append(rng.choice(FS_WS))
            elif r < 0.9:
                parts.append(core + core)
            else:
                parts.append("")
    s = "".join(parts)
    r = rng.random()
    if r < 0.15:
        s += sep
    elif r < 0.25:
        s += rng.choice(FS_WS)
    elif r < 0.3:
        s = sep + s
    elif r < 0.35:
        s += sep + sep
    return s


def fs_nonnative(dt):
    """Whether a dtype is a sized numeric one whose byte order is not the host's.

    :param dt: the dtype name.
    :returns: True for e.g. ``>i4`` on a little-endian host.
    """
    try:
        d = np.dtype(dt)
    except TypeError:
        return False
    return d.kind in "biufc" and d.itemsize > 1 and not d.isnative


def fs_native(dt):
    """The native twin of a dtype name (``>i4`` -> ``<i4``).

    :param dt: the dtype name.
    :returns: the native dtype's ``str``.
    """
    return np.dtype(dt).newbyteorder("=").str


def fs_written(fn, nbytes):
    """Runs a counted np.fromstring twice over NumPy's small-allocation cache primed with two fill bytes, so the
    elements NumPy never wrote (uninitialized memory) differ between the runs.

    :param fn: the call.
    :param nbytes: the result's size in bytes (the cache is primed with a block of that size).
    :returns: ``(result, nread)``: the first run's result and the length of its written prefix (None when the result
        is not a 1-D numeric array).
    :raises Exception: whatever the call raises.
    """
    results = []
    for fill in (0xAB, 0xCD):
        if nbytes > 0:
            prime = np.full(nbytes, fill, np.uint8)
            del prime
        results.append(fn())
    a, b = results
    if a.dtype.kind not in "biufc" or a.shape != b.shape or a.ndim != 1:
        return a, None
    isz = a.dtype.itemsize
    ba, bb = a.tobytes(), b.tobytes()
    nread = 0
    while nread < a.shape[0] and ba[nread * isz:(nread + 1) * isz] == bb[nread * isz:(nread + 1) * isz]:
        nread += 1
    return a, nread


def gen_fromstring(rng, case):
    """One np.fromstring case (a str or a bytes argument).

    :param rng: the section's generator.
    :param case: the case dict to fill (its id is set).
    :returns: the case, or None when NumSharp cannot hold NumPy's result.
    """
    dt = rng.choice(FS_NUMERIC * 4 + FS_OTHER)
    sep = rng.choice(FS_SEPS)
    count = rng.choice([-1, -1, -1, -2, 0, 1, 2, 3, 5, 9])
    text = fs_text(rng, dt, sep if sep else " ")
    case.update({"fn": "fromstring", "dtype": dt, "count": count, "sep": S(sep)})
    if rng.random() < 0.15:
        data = text.encode("utf-8", "surrogatepass")
        if rng.random() < 0.2:
            data += bytes([rng.randint(0x80, 0xff)])
        case["bytes"] = B(data)
        arg = data
    else:
        case["text"] = S(text)
        arg = text
    return record_fromstring(case, arg, dt, count, sep)


def record_fromstring(case, arg, dt, count, sep):
    """Calls np.fromstring and records NumPy's outcome into the case.

    :param case: the case dict (its id, ``fn`` and argument recorded; the dtype, count and separator are added here).
    :param arg: the ``str`` or ``bytes`` argument.
    :param dt: the dtype name.
    :param count: the count.
    :param sep: the separator.
    :returns: the case, or None when NumSharp cannot hold NumPy's result.
    """
    case.update({"dtype": dt, "count": count, "sep": S(sep)})
    # NumPy's text readers write a NATIVE value into a non-native array and then byte-swap it (a NumPy bug); NumSharp
    # reads the text's value into the native dtype, which is NumPy's answer for the native dtype.
    call_dt = fs_native(dt) if fs_nonnative(dt) else dt
    try:
        try:
            nbytes = max(count, 0) * np.dtype(call_dt).itemsize if dt != "T" else 0
        except TypeError:
            nbytes = 0
        r, nread = fs_written(lambda: np.fromstring(arg, dtype=call_dt, count=count, sep=sep), nbytes)
        out = array_out(r, nread)
        if out is None:
            return None
        case["expect"] = out
    except Exception as e:
        case["error"] = error_of(e)
    return case


def gen_fromfile_text(rng, case, tmpdir, long_text):
    """One np.fromfile text case: a file name, an open file (positioned past a lead), the position afterwards.

    :param rng: the section's generator.
    :param case: the case dict to fill.
    :param tmpdir: the generator's temporary directory.
    :param long_text: whether to read a long text across the C FILE model's chunk boundaries.
    :returns: the case, or None when NumSharp cannot hold NumPy's result.
    """
    dt = rng.choice(FS_NUMERIC * 4 + FS_OTHER)
    sep = rng.choice([s for s in FS_SEPS if s != ""])
    count = rng.choice([-1, -1, -1, -2, 0, 1, 2, 3, 5, 9])
    if long_text:
        dt = rng.choice(FS_NUMERIC)
        sep = rng.choice([" ", ",", ", ", "\n", " , "])
        count = rng.choice([-1, -1, 5000])
        text = fs_long_text(rng, dt, sep)
    else:
        text = fs_text(rng, dt, sep)
    data = text.encode("utf-8", "surrogatepass")
    # A trailing byte no UTF-8 sequence holds (the separator test's signed-char comparison); not on a long text, whose
    # bytes would then travel as hex at twice the size.
    if not long_text and rng.random() < 0.1:
        data += bytes([rng.randint(0x80, 0xff)])
    lead = b"" if rng.random() < 0.8 else rng.choice([b"XY", b"1 ", b"\xef\xbb\xbf"])
    offset = rng.choice([0, 0, 0, 0, 1, -1, 3]) if rng.random() < 0.2 else 0
    mode = rng.choice(["name", "name", "file"])
    case["fn"] = "fromfile_text"
    return record_fromfile(case, tmpdir, lead + data, len(lead), dt, count, sep, offset, mode)


def record_fromfile(case, tmpdir, contents, skip, dt, count, sep, offset, mode):
    """Writes a file, calls np.fromfile over it (by name, or through an open file positioned at ``skip``, whose position
    afterwards is recorded) and records NumPy's outcome into the case.

    :param case: the case dict (its id and ``fn`` set).
    :param tmpdir: the generator's temporary directory.
    :param contents: the file's bytes.
    :param skip: where an open file is positioned before the call (it may lie past the end).
    :param dt: the dtype name.
    :param count: the count.
    :param sep: the separator (None: the call passes no sep).
    :param offset: the offset.
    :param mode: ``name`` or ``file``.
    :returns: the case, or None when NumSharp cannot hold NumPy's result.
    """
    case.update({"dtype": dt, "count": count, "sep": S(sep), "data": B(contents), "skip": skip, "offset": offset,
                 "mode": mode})
    path = os.path.join(tmpdir, "ff.dat")
    with open(path, "wb") as f:
        f.write(contents)
    kw = {} if sep is None else {"sep": sep}
    # A separator makes the call a TEXT read, where NumPy writes a native value into a non-native array (see
    # record_fromstring): the expectation is NumPy's answer for the native dtype, which is NumSharp's.
    call_dt = fs_native(dt) if sep and fs_nonnative(dt) else dt
    try:
        if mode == "name":
            out = array_out(np.fromfile(path, dtype=call_dt, count=count, offset=offset, **kw))
        else:
            with open(path, "rb") as f:
                f.seek(skip)
                try:
                    out = array_out(np.fromfile(f, dtype=call_dt, count=count, offset=offset, **kw))
                finally:
                    case["tell"] = f.tell()
        if out is None:
            os.remove(path)
            return None
        case["expect"] = out
    except Exception as e:
        case["error"] = error_of(e)
    os.remove(path)
    return case


def gen_fromtext(rng, tmpdir, n):
    """Section fromtext: np.fromstring and np.fromfile's text mode, half and half; about 0.2% of the file cases read a
    long text across the chunk boundaries (about ten).

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory.
    :param n: the number of cases.
    :returns: the cases.
    """
    cases = []
    while len(cases) < n:
        case = {"id": f"fromtext/{len(cases)}"}
        if rng.random() < 0.5:
            c = gen_fromstring(rng, case)
        else:
            c = gen_fromfile_text(rng, case, tmpdir, long_text=rng.random() < 0.002)
        if c is not None:
            cases.append(c)
    return cases


def gen_frombin(rng, tmpdir, n):
    """Section frombin: np.fromfile's binary mode over random bytes, counts, offsets and positions.

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory.
    :param n: the number of cases.
    :returns: the cases.
    """
    cases = []
    while len(cases) < n:
        dt = rng.choice(FS_NUMERIC * 3 + FS_OTHER)
        size = rng.choice([0, 1, 2, 3, 7, 8, 16, 17, 33])
        data = bytes(rng.randint(0, 255) for _ in range(size))
        count = rng.choice([-1, -1, -2, 0, 1, 2, 3, 100])
        offset = rng.choice([0, 0, 0, 1, 2, 5, 40, -1, -5])
        sep = rng.choice(["", "", "", None]) if rng.random() < 0.95 else " "
        lead = rng.choice([0, 0, 0, 3])
        mode = rng.choice(["name", "file"])
        case = {"id": f"frombin/{len(cases)}", "fn": "fromfile_bin"}
        # An open file starts `lead` bytes in, past the end of a short file too (NumPy's negative count).
        c = record_fromfile(case, tmpdir, data, lead, dt, count, sep, offset, mode)
        if c is not None:
            cases.append(c)
    return cases


# The edge grid's tokens. A random text usually fails on an early token, so the scanners' rules for a value that is
# read (an overflow's saturation, a sign without digits, a NaN spelling, a byte the C FILE model loses) rarely decide a
# random case's outcome; here each token is the whole text, follows a clean element, or precedes one.
FE_INT_MAGNITUDES = ["0", "1", "127", "128", "255", "256", "32767", "32768", "65535", "65536", "2147483647",
                     "2147483648", "4294967295", "4294967296", "9223372036854775807", "9223372036854775808",
                     "18446744073709551615", "18446744073709551616", "99999999999999999999"]
FE_INT_MALFORMED = ["--1", "+-1", "- 1", "+ 1", "-", "+", "0x10", "1_0", "010", "1.5", "1e3", "١٢", " 7", "7 ",
                    "-0", "+0"]
FE_FLOAT_TOKENS = ["inf", "INF", "Inf", "infinity", "Infinity", "INFINITY", "infin", "in", "i", "nan", "NaN", "NAN",
                   "nan(123)", "nan(abc_9)", "nan(", "nan()", "nan(x y)", "nanx", "n", "na", "infinityx", "infinit",
                   "1e400", "1e-400", "1e308", "2e-308", "4.9e-324", "2.5e-324", "1.7976931348623157e308",
                   "1.7976931348623159e308", "65504", "65520", "6e-8", "3.4028235e38", "3.4028236e38", "0x1p3", "1_0",
                   "1e", "1e+", "1e-", "e5", ".", ".e5", "5.", ".5", "0.", "-.5e-3", "1.5e3.2", "1..2", "1e5e5", "1E5",
                   "1e05", "1e0005", "0", "-0", "2.5"]
FE_COMPLEX_TOKENS = ["1+2j", "1-2j", "2j", "j", "+j", "-j", "1+j", "1-j", "1+2J", "(1+2j)", "1e5+1e-5j", "nan+nanj",
                     "inf-infj", "1+2", "1+2jx", "1+-2j", "1-+2j", "+1+2j", "-1-2j", "1j+2", "1 +2j", "1+ 2j", "1+2 j",
                     "0j", "-0-0j", "1e+2j", "1e+2+3j", "infj", "nanj", "1+infj", "1-nanj", "1+2jj", "jj", "1.5",
                     "-2.5e3", "1+", "1-", "1+j2", "2.5j"]
FE_SHAPES = ["{t}", "{t} 5", "5 {t}"]
FE_SEP_SHAPES = ["1{s}2", "1{s}2{s}", "{s}1{s}2", "1{s}{s}2", " 1 {s} 2 ", "1{s} 2", "1 {s}2"]
FE_HIGH_BYTES = [b"1\xff", b"1\xff2", b"1 \xff", b"\xff1", b"1\x80", b"1\xc3\xa9", b"1\xc3\xa92", b"1 \xc3\xa9 2",
                 b"1j\xff", b"1+2j\xff", b"1+2\xff", b"1+\xff", b"1\xff\xff", b"-\xff", b"1.5e\xff"]


def fe_tokens():
    """The (dtype, token) pairs of the edge grid: integer magnitudes in every sign and malformed integers for the
    integer dtypes and bool, the float spellings in two signs for the float dtypes and bool, the complex forms.

    :returns: the list of pairs.
    """
    ints = [sign + m for m in FE_INT_MAGNITUDES for sign in ("", "-", "+")] + FE_INT_MALFORMED
    floats = [sign + t for t in FE_FLOAT_TOKENS for sign in ("", "-")]
    pairs = []
    for dt in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"):
        pairs += [(dt, t) for t in ints]
    for dt in ("float16", "float32", "float64", "bool"):
        pairs += [(dt, t) for t in floats]
    pairs += [("complex128", t) for t in FE_COMPLEX_TOKENS]
    return pairs


def gen_fromedges(rng, tmpdir, n):
    """Section fromedges: the scanners' edge grid, deterministic (``rng`` and ``n`` are unused).

    * every token of :func:`fe_tokens` as the whole text, after a clean element and before one, through
      np.fromstring and np.fromfile by name (count -1), and after a clean element through an open file (its position
      afterwards) and np.fromstring with count 3 (NumPy's never-shrinking result);
    * the separators of the random section around two elements in seven placements, for int64, float64 and
      complex128, with count -1 and 1;
    * texts with bytes at or past 0x80 (a 0xFF that a signed char turns into EOF, a UTF-8 separator NumPy compares
      byte by byte against a signed char) read by name and through an open file.

    :param rng: unused (the grid is fixed).
    :param tmpdir: the generator's temporary directory.
    :param n: unused.
    :returns: the cases.
    """
    cases = []

    def add(c):
        if c is not None:
            cases.append(c)

    def fresh():
        return {"id": f"fromedges/{len(cases)}"}

    for dt, tok in fe_tokens():
        for shape in FE_SHAPES:
            text = shape.format(t=tok)
            c = fresh()
            c.update({"fn": "fromstring", "text": S(text)})
            add(record_fromstring(c, text, dt, -1, " "))
            c = fresh()
            c["fn"] = "fromfile_text"
            add(record_fromfile(c, tmpdir, text.encode("utf-8"), 0, dt, -1, " ", 0, "name"))
        text = "5 " + tok
        c = fresh()
        c["fn"] = "fromfile_text"
        add(record_fromfile(c, tmpdir, text.encode("utf-8"), 0, dt, -1, " ", 0, "file"))
        c = fresh()
        c.update({"fn": "fromstring", "text": S(text)})
        add(record_fromstring(c, text, dt, 3, " "))
    for dt in ("int64", "float64", "complex128"):
        for sep in FS_SEPS:
            for shape in FE_SEP_SHAPES:
                text = shape.format(s=sep)
                for count in (-1, 1):
                    c = fresh()
                    c.update({"fn": "fromstring", "text": S(text)})
                    add(record_fromstring(c, text, dt, count, sep))
                    if sep != "":
                        c = fresh()
                        c["fn"] = "fromfile_text"
                        add(record_fromfile(c, tmpdir, text.encode("utf-8", "surrogatepass"), 0, dt, count, sep, 0,
                                            "name"))
    for dt in ("int64", "uint8", "float64", "complex128"):
        for data in FE_HIGH_BYTES:
            for sep in (" ", ",", "\xe9", "\xff"):
                for mode in ("name", "file"):
                    c = fresh()
                    c["fn"] = "fromfile_text"
                    add(record_fromfile(c, tmpdir, data, 0, dt, -1, sep, 0, mode))
    return cases


# ---------------------------------------------------------------------------------------------------------------------
# Section loadtxt: every option and source
# ---------------------------------------------------------------------------------------------------------------------

def _as_text(s):
    """A converter's argument as text (bytes under ``encoding='bytes'`` decode as Latin-1).

    :param s: the field (str or bytes).
    :returns: the str.
    """
    return s.decode("latin1") if isinstance(s, bytes) else s


# The converter menu: NAME -> function over the field. A Python function cannot travel in the corpus, so each case
# logs every call (field text -> result or exception) and the C# replay answers from that log.
CONVERTERS = {
    "float3": lambda s: float(_as_text(s)) / 3,
    "int": lambda s: int(_as_text(s)),
    "upper": lambda s: _as_text(s).upper(),
    "len": lambda s: len(_as_text(s)),
    "none": lambda s: None,
    "same": lambda s: s,
    "strip": lambda s: _as_text(s).strip(),
    "big": lambda s: 10 ** 30,
    "neg": lambda s: -len(_as_text(s)),
    "half": lambda s: np.float16(0.5),
    "cplx": lambda s: complex(1, 2),
    "raise": lambda s: 1 // 0,
    "boolish": lambda s: _as_text(s) == "1",
    "nan": lambda s: float("nan"),
}


def conv_value_out(v):
    """A converter's Python result in the form the C# twin rebuilds.

    :param v: the result.
    :returns: the JSON object.
    :raises TypeError: a result kind the replay cannot rebuild.
    """
    if v is None:
        return {"t": "n"}
    if isinstance(v, bool):
        return {"t": "bool", "v": v}
    if isinstance(v, int):
        return {"t": "i", "v": str(v)}
    if isinstance(v, float):
        return {"t": "f", "v": np.float64(v).view(np.uint64).item()}
    if isinstance(v, complex):
        return {"t": "c", "re": np.float64(v.real).view(np.uint64).item(),
                "im": np.float64(v.imag).view(np.uint64).item()}
    if isinstance(v, np.float16):
        return {"t": "h", "v": int(v.view(np.uint16))}
    if isinstance(v, bytes):
        return {"t": "b", "v": v.hex()}
    if isinstance(v, str):
        return {"t": "s", "v": u16(v)}
    raise TypeError(f"unencodable converter result {v!r}")


def recording(name, log):
    """The menu converter ``name``, logging each call's field and result (or exception) into ``log``.

    :param name: the menu name.
    :param log: the case's call log, ``{name: {u16(field): result}}``.
    :returns: the converter.
    """
    fn = CONVERTERS[name]

    def wrapper(s):
        key = u16(_as_text(s))
        try:
            v = fn(s)
        except Exception as e:
            log.setdefault(name, {})[key] = {"t": "x", "type": type(e).__name__, "msg": S(str(e))}
            raise
        log.setdefault(name, {})[key] = conv_value_out(v)
        return v
    return wrapper


def conv_arg(spec, log):
    """The ``converters=`` argument of a converter spec: one menu name, or ``{column: name or None}``.

    :param spec: the spec as the corpus records it.
    :param log: the case's call log.
    :returns: NumPy's argument.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return recording(spec, log)
    return {int(k): (recording(v, log) if v is not None else None) for k, v in spec.items()}


def lt_digits(rng, lo, hi):
    """A run of decimal digits.

    :param rng: the section's generator.
    :param lo: the fewest digits.
    :param hi: the most digits.
    :returns: the digits.
    """
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(lo, hi)))


def lt_field_int(rng):
    """An integer field: limits past every width, signs, bases, padding, junk.

    :param rng: the section's generator.
    :returns: the text.
    """
    sign = rng.choice(["", "", "", "-", "+"])
    if rng.random() < 0.15:
        return rng.choice(["0", "-0", "+0", "007", "1_0", "0x1f", "1.0", "1e3", "", " 5 ", "\t7", "127", "128",
                           "-128", "-129", "255", "256", "32767", "32768", "65535", "65536", "2147483647",
                           "2147483648", "4294967295", "4294967296", "9223372036854775807", "9223372036854775808",
                           "-9223372036854775808", "-9223372036854775809", "18446744073709551615",
                           "18446744073709551616", "١٢", "x", "--1", "+-1", "1 2"])
    return sign + lt_digits(rng, 1, rng.choice([2, 4, 10, 19]))


def lt_field_float(rng):
    """A float field: NaN / inf spellings, limits, underscores, Unicode digits, padding, junk, ordinary decimals.

    :param rng: the section's generator.
    :returns: the text.
    """
    sign = rng.choice(["", "", "", "-", "+"])
    if rng.random() < 0.2:
        return sign + rng.choice(["inf", "INF", "Infinity", "infinity", "nan", "NaN", "-nan", "nan(1)", "1e400",
                                  "1e-400", "4.9e-324", "65504", "65520", "6e-8", "3.4028236e38", "1e", "1e+", ".",
                                  "5.", ".5", "1_0", "1_000.5", "0x1p3", "١٫٥", "١٢",
                                  " 1.5 ", "1.5\t", "1,5", "1e5e5", "", "abc", "1.5j", "  ", "+.5e-3"])
    m = lt_digits(rng, 1, 10)
    if rng.random() < 0.6:
        m += "." + lt_digits(rng, 0, 17)
    if rng.random() < 0.3:
        m += rng.choice("eE") + rng.choice(["", "+", "-"]) + lt_digits(rng, 1, 3)
    return sign + m


def lt_field_complex(rng):
    """A complex field: NumPy's forms (parenthesized included) and the near misses.

    :param rng: the section's generator.
    :returns: the text.
    """
    if rng.random() < 0.4:
        return rng.choice(["1+2j", "1-2j", "2j", "j", "1+j", "(1+2j)", "( 1+2j )", "(1+2j", "1+2J", "1+2", "1",
                           "-1.5e3-2.5e-3j", "nan+nanj", "inf-infj", "1+-2j", "1-+2j", " 3j ", "1 + 2j", "",
                           "1+2jx", "(1)", "()", "infj", "-j", "+j", "0j"])
    a = lt_field_float(rng)
    return a + rng.choice(["", "+" + lt_field_float(rng).lstrip("+-") + "j", "j"])


def lt_field_bool(rng):
    """A bool field (NumPy reads it through int64).

    :param rng: the section's generator.
    :returns: the text.
    """
    return rng.choice(["0", "1", "-1", "2", "True", "False", "true", "", " 1", "1.0", "0x1", "+1", "00"])


def lt_field_text(rng):
    """A text field: quotes, comment characters, separators, non-ASCII, astral characters.

    :param rng: the section's generator.
    :returns: the text.
    """
    if rng.random() < 0.3:
        return rng.choice(["abc", "", " ", "NA", "None", "nan", "h\xe9llo", "日本", "a b", "x,y", '"q"',
                           "'s'", 'a"b', "tab\there", "#hash", "%pct", "\x1c", "\U0001f600", "\xa0", "  pad  "])
    alphabet = "abcXYZ019 ,;#%\"'\t\xe9\xe9日"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))


def lt_field(rng, dt):
    """One field for a dtype name.

    :param rng: the section's generator.
    :param dt: the corpus dtype name.
    :returns: the text.
    """
    if dt in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
        return lt_field_int(rng)
    if dt in ("float16", "float32", "float64"):
        return lt_field_float(rng)
    if dt == "complex128":
        return lt_field_complex(rng)
    if dt == "bool":
        return lt_field_bool(rng)
    return lt_field_text(rng)


LT_DTYPES = ["float64", "float64", "float32", "float16", "int8", "uint8", "int16", "uint16", "int32", "uint32",
             "int64", "uint64", "bool", "complex128", "U5", "U", "S3", "S", "T", "T_nanNA", "T_strNA", "T_nocoerce"]


def lt_dtype(name):
    """The NumPy dtype of a corpus dtype name (``T_*`` are StringDType variants).

    :param name: the corpus name.
    :returns: the dtype.
    """
    if name == "T":
        return np.dtypes.StringDType()
    if name == "T_nanNA":
        return np.dtypes.StringDType(na_object=np.nan)
    if name == "T_strNA":
        return np.dtypes.StringDType(na_object="NA")
    if name == "T_nocoerce":
        return np.dtypes.StringDType(coerce=False)
    return np.dtype(name)


def lt_lines(rng, dt, delim, comments, quote):
    """The lines of one source: rows of fields (some rows short or long), blank lines, comment lines, padding and
    trailing comments.

    :param rng: the section's generator.
    :param dt: the corpus dtype name.
    :param delim: the one-character delimiter the content is built with (None: whitespace runs).
    :param comments: the comment marker(s), or None.
    :param quote: the quote character the content uses, or None.
    :returns: the lines (without line ends).
    """
    nrows = rng.choice([0, 1, 1, 2, 3, 4, 6])
    ncols = rng.choice([1, 1, 2, 3, 4])
    lines = []
    for _ in range(nrows):
        r = rng.random()
        if r < 0.07:
            lines.append("")
            continue
        if r < 0.12 and comments:
            c = comments if isinstance(comments, str) else rng.choice(
                [x for x in comments if isinstance(x, str)] or ["#"])
            lines.append(c + " comment " + lt_field(rng, dt))
            continue
        k = ncols if rng.random() < 0.85 else rng.choice([ncols - 1, ncols + 1, 0])
        fields = []
        for _ in range(max(k, 0)):
            f = lt_field(rng, dt)
            if quote and rng.random() < 0.25:
                f = quote + f.replace(quote, quote + quote) + quote
            fields.append(f)
        sep = delim if delim is not None else rng.choice([" ", "  ", "\t", " \t "])
        line = sep.join(fields)
        if rng.random() < 0.1:
            line = rng.choice([" ", "\t", "  "]) + line
        if rng.random() < 0.1:
            line += rng.choice([" ", "\t"])
        if comments and rng.random() < 0.1:
            c = comments if isinstance(comments, str) else "#"
            line += " " + c + " trailing"
        lines.append(line)
    return lines


def lt_call(case, src, dtype, kw):
    """Calls np.loadtxt and records its outcome into the case.

    :param case: the case dict.
    :param src: the source.
    :param dtype: the dtype.
    :param kw: the keyword arguments.
    :returns: the case, or None when NumSharp cannot hold NumPy's result.
    """
    try:
        out = result_out(np.loadtxt(src, dtype=dtype, **kw))
        if out is None:
            return None
        case["expect"] = out
    except Exception as e:
        case["error"] = error_of(e)
    return case


def lt_source(rng, case, source, lines, eol, encoding, tmpdir):
    """Builds one loadtxt source and records it into the case.

    :param rng: the section's generator.
    :param case: the case dict.
    :param source: ``lines``, ``reader``, ``stream`` or ``file``.
    :param lines: the lines.
    :param eol: the line end the text sources use.
    :param encoding: the case's ``encoding=`` (decides the bytes of a byte source).
    :param tmpdir: the generator's temporary directory.
    :returns: NumPy's source argument.
    """
    text = eol.join(lines) + (eol if lines and rng.random() < 0.7 else "")
    if source == "lines":
        items = [ln + (eol if rng.random() < 0.5 else "") for ln in lines]
        case["lines"] = [S(x) for x in items]
        return items
    if source == "reader":
        case["text"] = S(text)
        return io.StringIO(text)
    codec = encoding if encoding not in (None, "bytes") else ("latin1" if source == "stream" else "utf-8")
    try:
        data = text.encode(codec)
    except (UnicodeEncodeError, LookupError):
        data = text.encode("utf-8", "surrogatepass")
    case["data"] = B(data)
    if source == "stream":
        return io.BytesIO(data)
    path = os.path.join(tmpdir, "lt.txt")
    with open(path, "wb") as f:
        f.write(data)
    return path


def gen_loadtxt(rng, tmpdir, n):
    """Section loadtxt: random calls over every option, source and dtype.

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory.
    :param n: the number of cases.
    :returns: the cases.
    """
    cases = []
    while len(cases) < n:
        dt = rng.choice(LT_DTYPES)
        delim = rng.choice([None, None, ",", ",", ";", " ", "\t", "|", "ab", "\n", "", ",", "\xe9"])
        comments = rng.choice(["#", "#", "#", None, "//", ["#", "%"], ("#",), "", ["#", None], [None], [1], "%", "a",
                               ["//", ";"], "\xe9"])
        quote = rng.choice([None, None, None, '"', '"', "'", "ab", ","])
        real_quote = quote if quote in ('"', "'") else None
        content_delim = delim if isinstance(delim, str) and len(delim) == 1 and delim not in "\n" else None
        lines = lt_lines(rng, dt, content_delim, comments if isinstance(comments, (str, list, tuple)) else None,
                         real_quote)
        eol = rng.choice(["\n", "\n", "\r\n", "\r"])
        source = rng.choice(["lines", "lines", "reader", "stream", "file"])
        kw = {}
        if rng.random() < 0.9:
            kw["delimiter"] = delim
        if rng.random() < 0.8:
            kw["comments"] = comments
        if rng.random() < 0.6:
            kw["quotechar"] = quote
        if rng.random() < 0.25:
            kw["skiprows"] = rng.choice([0, 1, 2, 3, -1])
        if rng.random() < 0.3:
            kw["max_rows"] = rng.choice([None, 0, 1, 2, 5, -1])
        if rng.random() < 0.3:
            kw["usecols"] = rng.choice([0, [0], [1, 0], [-1], [2], [0, 0], 5, [-5], 1])
        if rng.random() < 0.2:
            kw["ndmin"] = rng.choice([0, 1, 2, 3, -1])
        if rng.random() < 0.15:
            kw["unpack"] = True
        if rng.random() < 0.25:
            kw["encoding"] = rng.choice([None, "utf-8", "latin1", "bytes", "ascii", "cp1252", "utf-16"])
        log = {}
        params = json.loads(json.dumps(kw))
        if rng.random() < 0.15:
            spec = rng.choice(list(CONVERTERS)) if rng.random() < 0.5 else {
                str(rng.choice([0, 1, -1, 3])): rng.choice(list(CONVERTERS) + [None])}
            params["converters"] = spec
            kw["converters"] = conv_arg(spec, log)
        case = {"id": f"loadtxt/{len(cases)}", "dtype": dt, "source": source, "params": params}
        src = lt_source(rng, case, source, lines, eol, kw.get("encoding"), tmpdir)
        c = lt_call(case, src, lt_dtype(dt), kw)
        if isinstance(src, str) and os.path.exists(src):
            os.remove(src)
        if c is not None:
            c["conv_calls"] = log
            cases.append(c)
    return cases


# ---------------------------------------------------------------------------------------------------------------------
# Section ltfields: numeric fields, grouped where NumPy accepts them
# ---------------------------------------------------------------------------------------------------------------------

def lf_number(rng):
    """A float field for the numeric parsers: Python float()'s grammar (underscores, Unicode digits and spaces) and
    its near misses, float16's rounding edges, long mantissas and wide exponents.

    :param rng: the section's generator.
    :returns: the text.
    """
    r = rng.random()
    sign = rng.choice(["", "", "-", "+"])
    if r < 0.1:
        return sign + rng.choice(["inf", "INF", "Infinity", "nan", "NaN", "infinity", "Inf", "infinit", "na",
                                  "nan(1)"])
    if r < 0.2:
        return sign + rng.choice(["65504", "65519.99", "65520", "65536", "6e-8", "5.96e-8", "2.98e-8",
                                  "2.9802322387695312e-08", "2.9802322387695313e-08", "6.103515625e-05",
                                  "6.1035156e-05", "1e-400", "1e400", "0", "0.0", "-0", ".5", "5.", "1_0", "1__0",
                                  "_1", "1_", "1_0.5_5", "1e1_0", "0x10", "1e", "1e+", "١٢",
                                  "１２", "1.5　", "　1.5", " 1.5 ", "\t1.5\t", "1.5\x1c",
                                  "\x1c1.5"])
    m = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 22)))
    if rng.random() < 0.6:
        m = m[:rng.randint(0, len(m))] + "." + m[rng.randint(0, len(m)):]
        if m == ".":
            m = "0."
    if rng.random() < 0.4:
        m += rng.choice("eE") + rng.choice(["", "+", "-"]) + str(rng.randint(0, 340))
    pad = rng.choice(["", "", "", " ", "\t", "  "])
    return pad + sign + m + rng.choice(["", "", "", " ", "\t"])


def lf_field(rng, dt):
    """A field for one ltfields dtype.

    :param rng: the section's generator.
    :param dt: the dtype name.
    :returns: the text.
    """
    if dt == "complex128":
        if rng.random() < 0.5:
            return lf_number(rng)
        return lt_field_complex(rng)
    if dt in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
        return lt_field_int(rng)
    if dt == "bool":
        return lt_field_bool(rng)
    return lf_number(rng)


LF_DTYPES = ["float16"] * 6 + ["float32"] * 3 + ["float64"] * 3 + ["complex128"] * 2 + \
            ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"]

# A field travels on its own line with delimiter ',' so whitespace around it stays in the field; texts that would
# split, end the line or start a comment / quote go to the loadtxt section instead.
LF_EXCLUDED = (",", "\n", "\r", "#", '"')


def gen_ltfields(rng, tmpdir, n):
    """Section ltfields: ``n`` field texts. Each is first read alone; the ones NumPy accepts are read again in groups
    of 64 lines (one case per group, one value per line), and every refused one is its own case.

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory (unused: every source is a list of lines).
    :param n: the number of field texts.
    :returns: the cases.
    """
    accepted = {}
    cases = []
    drawn = 0
    while drawn < n:
        dt = rng.choice(LF_DTYPES)
        f = lf_field(rng, dt)
        if any(x in f for x in LF_EXCLUDED):
            continue
        drawn += 1
        line = f + "\n"
        try:
            np.loadtxt([line], dtype=dt, delimiter=",")
            accepted.setdefault(dt, []).append(line)
        except Exception as e:
            cases.append({"id": None, "dtype": dt, "source": "lines", "params": {"delimiter": ","},
                          "lines": [S(line)], "error": error_of(e), "conv_calls": {}})
    for dt in sorted(accepted):
        group = accepted[dt]
        for i in range(0, len(group), 64):
            chunk = group[i:i + 64]
            case = {"id": None, "dtype": dt, "source": "lines", "params": {"delimiter": ","},
                    "lines": [S(x) for x in chunk], "conv_calls": {}}
            r = np.loadtxt(chunk, dtype=dt, delimiter=",")
            case["expect"] = array_out(r)
            cases.append(case)
    for i, c in enumerate(cases):
        c["id"] = f"ltfields/{i}"
    return cases


# ---------------------------------------------------------------------------------------------------------------------
# Section lttokens: the tokenizer's edges
# ---------------------------------------------------------------------------------------------------------------------

TOK_WS = [" ", "\t", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\xa0", " ", " ", " ",
          " ", " ", " ", " ", " ", "　"]
TOK_PLAIN = list("abcXYZ019.-+_") + ["\xe9", "日", "\xff", "Ā", "​", "﻿"]


def tok_field(rng, quote, comment):
    """A field: usually short, sometimes 15 to 40 characters (around and past the 16-character scalar scan), with a
    quote or comment character inside now and then.

    :param rng: the section's generator.
    :param quote: the quote character, or None.
    :param comment: the comment character, or None.
    :returns: the text.
    """
    r = rng.random()
    if r < 0.05:
        return ""
    n = rng.randint(15, 40) if r < 0.15 else rng.randint(1, 8)
    chars = [rng.choice(TOK_PLAIN) for _ in range(n)]
    if quote and rng.random() < 0.1:
        chars.insert(rng.randint(0, len(chars)), quote)
    if comment and rng.random() < 0.05:
        chars.insert(rng.randint(0, len(chars)), comment)
    return "".join(chars)


def tok_sep(rng, delim):
    """A separator: whitespace runs for ``delimiter=None``, else the delimiter with whitespace around it now and then.

    :param rng: the section's generator.
    :param delim: the delimiter, or None.
    :returns: the text.
    """
    if delim is None:
        return "".join(rng.choice(TOK_WS) for _ in range(rng.choice([1, 1, 1, 2, 3])))
    s = delim
    if rng.random() < 0.15:
        s = rng.choice(TOK_WS) + s
    if rng.random() < 0.15:
        s += rng.choice(TOK_WS)
    return s


def tok_line(rng, delim, quote, comment, nfields):
    """One line: fields and separators, padding, a trailing comment, whitespace-only lines and newlines inside.

    :param rng: the section's generator.
    :param delim: the delimiter, or None.
    :param quote: the quote character, or None.
    :param comment: the comment character, or None.
    :param nfields: the number of fields, or None for a random count.
    :returns: the line.
    """
    fields = [tok_field(rng, quote, comment) for _ in range(nfields or rng.choice([1, 2, 2, 3, 4]))]
    out = fields[0]
    for f in fields[1:]:
        out += tok_sep(rng, delim) + f
    if rng.random() < 0.15:
        out = "".join(rng.choice(TOK_WS) for _ in range(rng.randint(1, 3))) + out
    if rng.random() < 0.15:
        out += "".join(rng.choice(TOK_WS) for _ in range(rng.randint(1, 3)))
    if comment and rng.random() < 0.1:
        out += rng.choice(["", " "]) + comment + rng.choice(["", " x", "\ry"])
    if rng.random() < 0.03:
        out = rng.choice(["", " ", "\t", "\x1c", "　"])
    if rng.random() < 0.03:
        k = rng.randint(0, len(out))
        out = out[:k] + rng.choice(["\r", "\n", "\r\n"]) + out[k:]
    return out


def gen_lttokens(rng, tmpdir, n):
    """Section lttokens: text dtypes over lines built to stress the tokenizer's stop rules.

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory.
    :param n: the number of cases.
    :returns: the cases.
    """
    cases = []
    while len(cases) < n:
        delim = rng.choice([None, None, None, ",", " ", "\t", ";", "\xe9", "　", "\x1c"])
        comment = rng.choice(["#", "#", None, "\xe9", "x", ";", "　"])
        if comment == delim:
            comment = "#"
        quote = rng.choice([None, None, None, '"', "'"])
        dt = rng.choice(["U", "U", "S", "U8", "S8", "U3", "T", "T", "T_strNA"])
        nf = rng.choice([1, 2, 2, 3, 4]) if rng.random() < 0.85 else None
        lines = [tok_line(rng, delim, quote, comment, nf) for _ in range(rng.randint(1, 6))]
        source = rng.choice(["lines", "lines", "lines", "reader", "file"])
        params = {"delimiter": delim, "comments": comment, "quotechar": quote}
        if rng.random() < 0.1:
            params["max_rows"] = rng.choice([1, 2, 3])
        if rng.random() < 0.1:
            params["skiprows"] = rng.choice([1, 2])
        if rng.random() < 0.1:
            params["usecols"] = rng.choice([0, [0], [1, 0], [-1]])
        eol = rng.choice(["\n", "\n", "\r\n", "\r"])
        case = {"id": f"lttokens/{len(cases)}", "dtype": dt, "source": source, "params": params, "conv_calls": {}}
        if source == "lines":
            items = [ln + rng.choice(["", "", "\n", "\r\n", "\r"]) for ln in lines]
            case["lines"] = [S(x) for x in items]
            src = items
        elif source == "reader":
            text = eol.join(lines) + (eol if rng.random() < 0.7 else "")
            case["text"] = S(text)
            src = io.StringIO(text)
        else:
            text = eol.join(lines) + (eol if rng.random() < 0.7 else "")
            data = text.encode("utf-8", "surrogatepass")
            case["data"] = B(data)
            params["encoding"] = "utf-8"
            src = os.path.join(tmpdir, "tok.txt")
            with open(src, "wb") as f:
                f.write(data)
        c = lt_call(case, src, lt_dtype(dt), dict(params))
        if source == "file" and os.path.exists(src):
            os.remove(src)
        if c is not None:
            cases.append(c)
    return cases


# ---------------------------------------------------------------------------------------------------------------------
# Sections savetxt / tofile: the writers
# ---------------------------------------------------------------------------------------------------------------------

ST_NUMERIC = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32",
              "float64", "complex128"]
ST_TEXT = ["U4", "S3", "T"]


def st_specials(dt):
    """The special values of a numeric dtype: limits, signed zeros, NaN / inf, subnormals, awkward decimals.

    :param dt: the dtype name.
    :returns: the values.
    """
    d = np.dtype(dt)
    if d.kind == "b":
        return [True, False]
    if d.kind in "iu":
        info = np.iinfo(d)
        return [0, 1, -1 if d.kind == "i" else 2, info.min, info.max, info.max // 3, 7, 42]
    if d.kind == "f":
        fi = np.finfo(d)
        return [0.0, -0.0, 1.5, -2.25, np.nan, np.inf, -np.inf, fi.max, fi.tiny, fi.smallest_subnormal, 1 / 3, 1e10,
                123456.789, 0.1, 5e-324 if d.itemsize == 8 else 1e-40]
    return [0j, 1 + 2j, -1.5 - 0.5j, complex(np.nan, 1), complex(1, np.inf), complex(-0.0, -0.0), 1e300 + 1e-300j]


def st_base(rng, dt, shape):
    """A random base array of a dtype and shape.

    :param rng: the section's generator.
    :param dt: the dtype name (``T`` is StringDType with a random NA object).
    :param shape: the shape.
    :returns: ``(array, na kind)``; the NA kind is None except for StringDType.
    """
    n = int(np.prod(shape)) if shape else 1
    if dt == "T":
        vals = [rng.choice(["", "a", "abc", "h\xe9llo", "日本", "x y", "NA", "nan", "%s", "100", "1.5", "\t"])
                for _ in range(n)]
        if rng.random() < 0.3 and n > 0:
            vals[rng.randrange(n)] = None
        na = rng.choice([None, "nan", "str"])
        if na is None:
            return np.array([v if v is not None else "None" for v in vals],
                            dtype=np.dtypes.StringDType()).reshape(shape), "plain"
        if na == "nan":
            return np.array([np.nan if v is None else v for v in vals],
                            dtype=np.dtypes.StringDType(na_object=np.nan)).reshape(shape), "nan"
        return np.array(["NA" if v is None else v for v in vals],
                        dtype=np.dtypes.StringDType(na_object="NA")).reshape(shape), "strNA"
    if dt in ("U4", "S3"):
        pool = ["", "a", "ab", "abc", "abcd", "\xe9", "日", "x", "%", "1"]
        vals = [rng.choice(pool) for _ in range(n)]
        if dt == "S3":
            vals = [v.encode("utf-8")[:3] if v.isascii() else b"q" for v in vals]
        return np.array(vals, dtype=dt).reshape(shape), None
    d = np.dtype(dt)
    specials = st_specials(dt)
    out = np.empty(n, dtype=d)
    for i in range(n):
        if rng.random() < 0.4:
            out[i] = specials[rng.randrange(len(specials))]
        elif d.kind in "iu":
            info = np.iinfo(d)
            out[i] = rng.randint(int(info.min), int(info.max))
        elif d.kind == "f":
            out[i] = rng.uniform(-1e6, 1e6) * 10 ** rng.randint(-8, 8)
        elif d.kind == "c":
            out[i] = complex(rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3))
        else:
            out[i] = rng.random() < 0.5
    return out.reshape(shape), None


def st_view(arr, recipe):
    """A view of the base array: C, F, transposed, stepped, reversed, offset or broadcast.

    :param arr: the base array.
    :param recipe: the recipe name.
    :returns: the view.
    :raises ValueError: an unknown recipe.
    """
    if recipe == "c":
        return arr
    if recipe == "f":
        return np.asfortranarray(arr)
    if recipe == "t":
        return arr.T
    if recipe == "s0":
        return arr[::2]
    if recipe == "s1":
        return arr[:, ::2]
    if recipe == "r0":
        return arr[::-1]
    if recipe == "r1":
        return arr[:, ::-1]
    if recipe == "o":
        return arr[1:]
    if recipe == "b":
        return np.broadcast_to(arr[:1], (3,) + arr.shape[1:])
    raise ValueError(recipe)


def st_array(rng, dt, ndims):
    """Draws a base array and a view recipe.

    :param rng: the section's generator.
    :param dt: the dtype name.
    :param ndims: the rank choices.
    :returns: ``(base, na kind, recipe, view)``.
    """
    ndim = rng.choice(ndims)
    shape = tuple(rng.choice([1, 2, 3, 4]) for _ in range(ndim))
    if rng.random() < 0.05 and ndim > 0:
        shape = (0,) + shape[1:]
    if rng.random() < 0.003 and ndim in (1, 2):
        # a large array: the writers' batched flushes
        shape = (700,) + shape[1:]
    base, na = st_base(rng, dt, shape)
    recipes = ["c"]
    if ndim >= 1 and base.shape[0] > 1:
        recipes += ["s0", "r0", "o"]
    if ndim >= 2:
        recipes += ["f", "t", "s1", "r1"] + (["b"] if base.shape[0] > 0 else [])
    recipe = rng.choice(recipes)
    return base, na, recipe, st_view(base, recipe)


def st_array_out(arr, na, dt):
    """The base array as the replay rebuilds it (C order).

    :param arr: the base array.
    :param na: the StringDType NA kind, or None.
    :param dt: the dtype name.
    :returns: the JSON object.
    """
    out = {"dtype": dt, "shape": list(arr.shape)}
    if dt == "T":
        out["cells"] = [cell(x) for x in arr.ravel().tolist()]
        out["na"] = na
    else:
        out["hex"] = np.ascontiguousarray(arr).tobytes().hex()
    return out


def st_spec(rng, allow_bad=True):
    """A random ``%`` conversion: flags, width, precision, a length modifier and a conversion (now and then broken).

    :param rng: the section's generator.
    :param allow_bad: whether an invalid conversion character may be drawn.
    :returns: the spec.
    """
    flags = "".join(rng.choice(["-", "+", " ", "#", "0"]) for _ in range(rng.choice([0, 0, 0, 1, 2])))
    width = rng.choice(["", "", "", str(rng.randint(0, 14)), "*"]) if rng.random() < 0.7 else ""
    prec = rng.choice(["", "", "." + str(rng.randint(0, 20)), ".", ".*"]) if rng.random() < 0.6 else ""
    length = rng.choice(["", "", "", "", "h", "l", "L", "ll"])
    conv = rng.choice("dioxXeEfFgGcrsa" + ("%" if rng.random() < 0.05 else ""))
    if allow_bad and rng.random() < 0.05:
        conv = rng.choice(["y", "k", " ", "\xe9", "", "("])
    return "%" + flags + width + prec + length + conv


def st_fmt(rng, ncol):
    """A savetxt ``fmt``: one spec, a fixed menu, a multi-spec string (with or without the right count), or a list.

    :param rng: the section's generator.
    :param ncol: the written array's column count.
    :returns: the fmt (a str or a list of str).
    """
    r = rng.random()
    if r < 0.45:
        return st_spec(rng)
    if r < 0.6:
        return rng.choice(["%.18e", "%d", "%s", "%r", "%g", "%10.5f", "%x", "%c", "%5s", "%-5d", "%+.3e", "%#x",
                           "%.0f", "%e", "%f", "%%", "%(x)s", "x", "", "%s%s", "%d,%d", "%.2f %.2f %.2f"])
    if r < 0.8:
        k = rng.choice([ncol, ncol, ncol, max(ncol - 1, 0), ncol + 1])
        lits = ["", " ", ",", "|", "[", "]", "%%", "\xe9"]
        return rng.choice(lits) + "".join(st_spec(rng, False) + rng.choice(lits) for _ in range(k))
    k = rng.choice([ncol, ncol, ncol, ncol + 1, max(ncol - 1, 0)])
    return [st_spec(rng, rng.random() < 0.3) for _ in range(k)]


def st_handle_write(view, kw, newline):
    """np.savetxt through ``io.TextIOWrapper(io.BytesIO(), encoding=..., newline=newline)``: the bytes Python's text
    mode writes to a file NAME on a host whose line separator is ``newline``.

    :param view: the array.
    :param kw: savetxt's keyword arguments (``encoding`` decides the wrapper's codec; None and ``'locale'`` are UTF-8,
        this host's and NumSharp's locale encoding).
    :param newline: the translation of ``"\\n"``.
    :returns: ``(bytes, error)``.
    """
    enc = kw.get("encoding")
    codec = "utf-8" if enc in (None, "locale") else enc
    bio = io.BytesIO()
    tw = io.TextIOWrapper(bio, encoding=codec, newline=newline)
    error = None
    try:
        np.savetxt(tw, view, **kw)
    except Exception as e:
        error = e
    tw.flush()
    data = bio.getvalue()
    tw.detach()
    return data, error


def same_error(a, b):
    """Whether two outcomes raised the same exception (class and text), or neither raised.

    :param a: an exception or None.
    :param b: an exception or None.
    :returns: True when they agree.
    """
    if a is None or b is None:
        return a is None and b is None
    return type(a) is type(b) and str(a) == str(b)


class StStats:
    """Counts the savetxt file cases dropped because the text-mode wrapper did not reproduce the real file."""
    dropped = 0


def gen_savetxt(rng, tmpdir, n):
    """Section savetxt: random calls into a text stream, a byte stream and a file name (with the CRLF twin).

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory.
    :param n: the number of cases.
    :returns: the cases.
    """
    cases = []
    while len(cases) < n:
        dt = rng.choice(ST_NUMERIC * 3 + ST_TEXT)
        base, na, recipe, view = st_array(rng, dt, [1, 2, 2, 2, 0, 3])
        ncol = view.shape[1] if view.ndim == 2 else 1
        fmt = st_fmt(rng, ncol) if rng.random() < 0.85 else "%.18e"
        kw = {"fmt": fmt}
        if rng.random() < 0.4:
            kw["delimiter"] = rng.choice([" ", ",", "\t", "", "\xe9", "%", "%%", ", "])
        if rng.random() < 0.3:
            kw["newline"] = rng.choice(["\n", "\r\n", "", ";", "\xe9", "%"])
        if rng.random() < 0.3:
            kw["header"] = rng.choice(["", "h", "line1\nline2", "\xe9", "%d"])
        if rng.random() < 0.3:
            kw["footer"] = rng.choice(["", "f", "a\nb", "日"])
        if rng.random() < 0.3:
            kw["comments"] = rng.choice(["# ", "", "%", "//", "\xe9"])
        if rng.random() < 0.25:
            kw["encoding"] = rng.choice([None, "utf-8", "latin1", "ascii", "utf-16", "utf-8-sig", "cp1252"])
        target = rng.choice(["writer", "writer", "stream", "file"])
        params = {k: (v if k == "encoding" else (S(v) if isinstance(v, str) else [S(x) for x in v]))
                  for k, v in kw.items()}
        case = {"id": f"savetxt/{len(cases)}", "fn": "savetxt", "array": st_array_out(base, na, dt), "view": recipe,
                "params": params, "fmt_is_list": isinstance(fmt, list), "target": target}
        error = None
        if target == "writer":
            buf = io.StringIO()
            try:
                np.savetxt(buf, view, **kw)
            except Exception as e:
                error = e
            case["out"] = text_out(buf.getvalue())
        elif target == "stream":
            buf = io.BytesIO()
            try:
                np.savetxt(buf, view, **kw)
            except Exception as e:
                error = e
            case["out"] = bytes_out(buf.getvalue())
        else:
            path = os.path.join(tmpdir, "st.txt")
            try:
                np.savetxt(path, view, **kw)
            except Exception as e:
                error = e
            with open(path, "rb") as f:
                real = f.read()
            os.remove(path)
            lf, lf_error = st_handle_write(view, kw, "\n")
            if lf != real or not same_error(lf_error, error):
                StStats.dropped += 1
                continue
            crlf, crlf_error = st_handle_write(view, kw, "\r\n")
            case["out"] = bytes_out(real)
            case["crlf"] = {"out": bytes_out(crlf), "error": None if crlf_error is None else error_of(crlf_error)}
        case["error"] = None if error is None else error_of(error)
        cases.append(case)
    return cases


def st_tofile_crashes(view, kw):
    """Whether NumPy 2.4.2's ndarray.tofile would crash on this call: a text-mode element whose ``str`` / ``format``
    text is not ASCII (``PyUnicode_AsASCIIString``'s NULL is not checked).

    :param view: the array.
    :param kw: tofile's keyword arguments.
    :returns: True when the call must not run.
    """
    if kw.get("sep", "") == "":
        return False
    fmt = kw.get("format", "%s")
    for x in view.ravel().tolist() if view.size else []:
        try:
            s = str(x) if fmt == "" else fmt % (x,)
        except Exception:
            return False
        if not s.isascii():
            return True
    return False


def gen_tofile(rng, tmpdir, n):
    """Section tofile: ndarray.tofile's binary and text modes over every layout.

    :param rng: the section's generator.
    :param tmpdir: the generator's temporary directory.
    :param n: the number of cases.
    :returns: the cases.
    """
    cases = []
    while len(cases) < n:
        dt = rng.choice(ST_NUMERIC * 3 + ST_TEXT)
        base, na, recipe, view = st_array(rng, dt, [0, 1, 2, 2, 3])
        kw = {}
        if rng.random() < 0.85:
            kw["sep"] = rng.choice(["", "", " ", ",", "\n", ", ", "\xe9", "|", "\x00", "\ud800", "%s"])
        if rng.random() < 0.7 and kw.get("sep", "") != "":
            kw["format"] = rng.choice(["%s", "%s", "%d", "%.3f", "%r", "%x", "%c", "%5s", "", "%e", "%g", "%%",
                                       st_spec(rng)])
        if st_tofile_crashes(view, kw):
            continue
        case = {"id": f"tofile/{len(cases)}", "fn": "tofile", "array": st_array_out(base, na, dt), "view": recipe,
                "params": {k: S(v) for k, v in kw.items()}, "target": "file"}
        path = os.path.join(tmpdir, "tf.bin")
        error = None
        try:
            view.tofile(path, **kw)
        except Exception as e:
            error = e
        if os.path.exists(path):
            with open(path, "rb") as f:
                case["out"] = bytes_out(f.read())
            os.remove(path)
        else:
            case["out"] = {"missing": True}
        case["error"] = None if error is None else error_of(error)
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------------------------

GENERATORS = {
    "fromtext": gen_fromtext,
    "frombin": gen_frombin,
    "fromedges": gen_fromedges,
    "loadtxt": gen_loadtxt,
    "ltfields": gen_ltfields,
    "lttokens": gen_lttokens,
    "savetxt": gen_savetxt,
    "tofile": gen_tofile,
}


def write_section(name, out_dir, tmpdir):
    """Generates one section and writes its file (header record first).

    :param name: the section name.
    :param out_dir: the corpus directory.
    :param tmpdir: the generator's temporary directory.
    :returns: the number of cases written.
    :raises SystemExit: the file is larger than MAX_SECTION_BYTES.
    """
    seed, n = SECTIONS[name]
    rng = random.Random(seed)
    cases = GENERATORS[name](rng, tmpdir, n)
    header = {"header": {"section": name, "generator": "gen_textio_oracle.py", "version": GENERATOR_VERSION,
                         "numpy": np.__version__, "python": sys.version.split()[0], "seed": seed,
                         "cases": len(cases)}}
    path = out_dir / f"textio_{name}.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        # Raw UTF-8 (a third of the escaped size for the tokenizer's Unicode whitespace): every string that could hold a
        # lone surrogate went through S(), so each one encodes.
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        for c in cases:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    size = path.stat().st_size
    if size > MAX_SECTION_BYTES:
        raise SystemExit(f"{path.name} is {size:,} bytes, past {MAX_SECTION_BYTES:,}: lower the section's count")
    return len(cases), size


def main():
    """Parses the arguments, checks the host and writes the requested sections."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sections", default=",".join(SECTIONS),
                        help="comma-separated sections to write (default: all)")
    parser.add_argument("--out", default=str(CORPUS_DIR), help="the corpus directory (default: the committed one)")
    args = parser.parse_args()
    check_host()
    # Expected noise: loadtxt warns on a source without data, and random float16 / float32 bases overflow to inf.
    warnings.filterwarnings("ignore", message="loadtxt: input contained no data", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"Input line \d+ contained no data", category=UserWarning)
    warnings.filterwarnings("ignore", message="overflow encountered in cast", category=RuntimeWarning)
    names = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in names if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown sections {unknown}; known: {list(SECTIONS)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="textio_oracle_")
    try:
        for name in names:
            count, size = write_section(name, out_dir, tmpdir)
            print(f"{name:10s} {count:7d} cases {size:12,d} bytes")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if StStats.dropped:
        raise SystemExit(f"{StStats.dropped} savetxt file cases dropped: the text-mode wrapper did not reproduce the "
                         "file, so their Windows twin cannot be derived; investigate before committing")


if __name__ == "__main__":
    main()
