"""
gen_random_oracle.py — the random-API oracle: every public member of NumSharp's random world, by exact C# overload,
against NumPy 2.4.2.

Plan and state: docs/plans/random-oracle-coverage.md. Replayed by test/NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests.RandomApi.cs
(harness in Fuzz/RandomApi/). No Python runs at test time: this script records NumPy's answers into the committed corpus.

How a case is made
------------------
1. The inventory test/oracle/random_surface.json (reflected by Fuzz/RandomApi/RandomApiSurface.cs) lists every public member
   with ONE canonical signature. Every member must be claimed by a family handler below or by an exemption, and every
   handler signature must exist in the inventory — generation refuses otherwise, so an overload cannot be forgotten.
2. A family handler knows the NumPy counterpart of a member (method name, parameter mapping, value domains, which calls
   raise). For each overload it emits cases: receiver (engine x seed) x argument variant, each naming the overload's
   exact signature, its arguments by parameter name (an omitted optional argument is the C# default path, and NumPy is
   called WITHOUT it, so a default that differs from NumPy's shows as a divergence), and NumPy's observation.
3. Observations are canonical (see `obs`): arrays as (dtype, shape, bytes), Python scalars by value, objects by type +
   state. After every call the receiver's full state text is recorded too (stream position, buffered half, gauss cache).

Tiers (files under test/NumSharp.Tests.Oracle/Fuzz/corpus/):
    random_api.jsonl        libm-free results (bits, integers, uniform, permutations, raw words, state, seeding)
    random_api_host.jsonl   transform/rejection samplers that consume libm (win-amd64 CRT authored)
    random_api_mvn.jsonl    multivariate_normal (byte parity only through the pinned OpenBLAS; threads=1)
    random_api_lp64.jsonl   outcomes that depend on C long being 64-bit (authored with Linux NumPy, WSL)

Usage:
    python test/oracle/gen_random_oracle.py                      # the committed corpus (10 fixed seeds; LP64 via WSL)
    python test/oracle/gen_random_oracle.py --seeds 11,22 --out DIR   # soak mode: fresh seeds into DIR
    python test/oracle/gen_random_oracle.py --lp64-only --out DIR     # (run under Linux NumPy) the LP64 tier only
    python test/oracle/gen_random_oracle.py --seeds S --out DIR --lp64-rows ROWS  # soak: LP64 rows from a Linux job

The nightly soak (.github/workflows/fuzz-soak.yml, jobs random-api-lp64 + random-api-soak) draws 10 fresh seeds, runs
the --lp64-only mode on a Linux runner, then this script on windows-latest with --lp64-rows, and replays DIR through
FuzzCorpusTests.RandomApiSoak (NUMSHARP_RANDOM_SOAK_DIR). A soak DIR also gets manifest.json (seeds, NumPy, counts).
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import warnings

# The multivariate_normal tier is byte-exact only against NumPy's own scipy-openblas at one thread (NumSharp pins the
# same binary); set before numpy loads its BLAS.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layout_catalog import describe  # noqa: E402
from numpy.random.bit_generator import SeedlessSeedSequence  # noqa: E402

CORPUS_DIR = os.path.normpath(os.path.join(HERE, "..", "NumSharp.Tests.Oracle", "Fuzz", "corpus"))
SURFACE_PATH = os.path.join(HERE, "random_surface.json")

# The 10 fixed seeds (plan §5): all valid for the legacy RandomState(int) (< 2**32) and for every engine, both ends of the
# 32-bit range included.
FIXED_SEEDS = [0, 1, 7, 42, 1234, 65535, 2147483647, 2147483648, 987654321, 4294967295]
ENGINES = ["MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"]
ENGINE_CLASS = {name: getattr(np.random, name) for name in ENGINES}
TIERS = ("random_api", "random_api_host", "random_api_mvn", "random_api_lp64")

# Legacy members whose win-amd64 NumPy result is the C long (int32 there) and which NumSharp returns as int64, its
# LP64 model for EVERY legacy long (NumPyRandom.LegacyLong, 9aac6ee9); recorded widened, exactly as the random_parity
# tiers do (gen_oracle._RND_INT64_CAST). A family whose result is a long only on some calls (randint's default dtype,
# choice's indices, permutation of an int) decides `widen` per case; this set names the families whose cases the LP64
# sub-run regenerates under Linux NumPy.
LEGACY_INT64_CAST = {"poisson", "zipf", "logseries", "hypergeometric", "geometric", "binomial", "negative_binomial",
                     "randint", "random_integers", "permutation", "choice", "multinomial", "tomaxint"}


# ======================================================================================================================
# Encoding: arguments
# ======================================================================================================================

class Omit:
    """An optional argument left out: the C# default path, and NumPy is called without it."""

    def __repr__(self):
        return "OMIT"


OMIT = Omit()


def f64_bits(x):
    """The IEEE-754 bit pattern of a Python float as 16 lowercase hex digits (big-endian reading)."""
    return "%016x" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


class Arg:
    """One argument of a case: the C# parameter it binds (name + canonical type), its JSON encoding, and the Python value
    the NumPy call receives (None for OMIT)."""

    __slots__ = ("name", "ctype", "json", "py", "omit", "operand")

    def __init__(self, name, ctype, json_value, py, omit=False, operand=None):
        self.name, self.ctype, self.json, self.py, self.omit, self.operand = name, ctype, json_value, py, omit, operand


class Operands:
    """The NDArray arguments of one case, serialized in order (the JSON references them by index)."""

    def __init__(self):
        self.items = []

    def add(self, base, view):
        self.items.append(describe(base, view))
        return len(self.items) - 1


def encode(name, ctype, value, ops):
    """Encode one argument value for the C# parameter (name, ctype); `value` is the Python value NumPy receives, or OMIT.

    NDArray arguments arrive as (base, view) pairs or plain ndarrays and become operands. Object-typed arguments arrive as
    `Obj` wrappers carrying both the C# runtime type and the Python value.
    """
    if value is OMIT:
        return Arg(name, ctype, {"n": name, "omit": True}, None, omit=True)
    j = {"n": name, "t": ctype}
    base = ctype.rstrip("?")
    nullable = ctype.endswith("?")
    if value is None and (nullable or base in ("NDArray", "DType", "string", "object", "ISeedSequence", "BitGenerator",
                                               "Generator", "NumPyRandom", "NumPyRandom.State", "BitGeneratorState",
                                               "SeedSequence")
                          # every array spelling (T[], T[,]) and the engines' typed State classes are references
                          or base.endswith("]") or base.startswith("NDArray") or base.endswith(".State")):
        j["null"] = True
        return Arg(name, ctype, j, None)
    if base == "double" or base == "float":
        j["bits"] = f64_bits(value)
        py = float(value)
    elif base in ("int", "long", "uint", "ulong", "BigInteger", "UInt128", "short", "byte"):
        j["v"] = str(int(value))
        py = int(value)
    elif base == "bool":
        j["v"] = bool(value)
        py = bool(value)
    elif base == "string":
        j["v"] = value
        py = value
    elif base == "Shape":
        if isinstance(value, ShapeNone):
            j["none"] = True
            py = None
        else:
            dims = [int(d) for d in value]
            j["dims"] = [str(d) for d in dims]
            py = tuple(dims)
    elif base == "DType":
        j["name"] = value
        py = np.dtype(value)
    elif base.startswith("NDArray"):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], np.ndarray):
            b, v = value
        else:
            # np.array(copy=True) keeps a 0-d array 0-d; np.ascontiguousarray would silently make it (1,).
            b = np.array(value, copy=True, order="C")
            v = b
        j["op"] = ops.add(b, v)
        py = v
    elif base in ("int[]", "long[]", "uint[]", "ulong[]", "params long[]"):
        j["v"] = [str(int(x)) for x in value]
        py = list(int(x) for x in value)
    elif base == "double[]":
        j["v"] = [f64_bits(x) for x in value]
        py = [float(x) for x in value]
    elif base == "double[,]":
        j["rows"] = [[f64_bits(x) for x in row] for row in value]
        py = np.array(value, dtype=np.float64)
    elif isinstance(value, Obj):
        j["obj"] = value.json
        py = value.py
    elif base == "object" and isinstance(value, (np.ndarray, tuple)):
        # An ndarray passed where C# takes `object` (a seed, a spawn key, Philox's counter/key): an operand.
        if isinstance(value, tuple):
            b, v = value
        else:
            b = np.array(value, copy=True, order="C")
            v = b
        j["op"] = ops.add(b, v)
        py = v
    else:
        raise TypeError(f"cannot encode {name}: {ctype} from {value!r}")
    return Arg(name, ctype, j, py)


class Alias:
    """An NDArray argument that IS another argument of the same call (the same operand, e.g. permuted(x, out=x)): both
    sides pass one array object to both parameters."""

    def __init__(self, name):
        self.name = name


class ShapeNone:
    """The C# `default(Shape)` — NumPy's size=None."""

    def __repr__(self):
        return "None"


SHAPE_NONE = ShapeNone()


class Obj:
    """An argument of a non-primitive C# type (object, ISeedSequence, BitGenerator, Generator, NumPyRandom, state objects):
    `json` tells the C# harness how to build its value, `py` is the Python value NumPy receives.

    A stateful value (an engine, a state dict) is given as `make`, a factory called once per case, so a case never
    inherits an object an earlier case drew from or mutated.
    """

    def __init__(self, json_value, py=None, make=None):
        self.json, self._py, self.make = json_value, py, make

    @property
    def py(self):
        return self.make() if self.make is not None else self._py


def bitgen_obj(engine, seed, prime_how="none"):
    """A BitGenerator argument: ENGINE(seed), primed."""
    def make():
        bg = make_engine(engine, seed)
        prime(bg, prime_how)
        return bg
    return Obj({"bitgen": {"engine": engine, "seed": str(seed), "prime": prime_how}}, make=make)


def bgstate_obj(engine, seed, prime_how="none"):
    """A BitGeneratorState argument: the `state` of ENGINE(seed) after priming (NumPy: the bit generator's state dict)."""
    def make():
        bg = make_engine(engine, seed)
        prime(bg, prime_how)
        return bg.state
    return Obj({"bgstate": {"engine": engine, "seed": str(seed), "prime": prime_how}}, make=make)


def legacy_state_of(recv):
    """A NativeRandomState argument: the legacy tuple of another receiver (its get_state())."""
    spec = recv.spec()
    spec["prime"] = recv.prime
    return Obj({"legacy_state_of": spec}, make=lambda: recv.build().get_state())


def legacy_state_explicit(key, pos, has_gauss=0, gauss=0.0, algorithm="MT19937"):
    """A NativeRandomState argument spelled out: ('MT19937', key, pos, has_gauss, cached_gaussian)."""
    j = {"algorithm": algorithm, "key": None if key is None else [str(int(k)) for k in key], "pos": str(pos),
         "has_gauss": str(has_gauss), "gauss": f64_bits(gauss)}
    return Obj({"legacy_state": j},
               make=lambda: (algorithm, None if key is None else np.array(key, dtype=np.uint32), pos, has_gauss, gauss))


def rs_dict_of(recv):
    """A NumPyRandom.State argument: the dict state of another receiver (its get_state(legacy=False))."""
    spec = recv.spec()
    spec["prime"] = recv.prime
    return Obj({"rs_dict_of": spec}, make=lambda: recv.build().get_state(legacy=False))


def rs_dict_parts(state_obj, has_gauss, gauss):
    """A NumPyRandom.State argument built from parts, NumPy's dict {**bitgen_state, 'has_gauss', 'gauss'};
    `state_obj` is a bgstate_obj or None (C#'s unset state)."""
    def make():
        d = dict(state_obj.py) if state_obj is not None else {}
        d["has_gauss"], d["gauss"] = has_gauss, gauss
        return d
    return Obj({"rs_dict_parts": {"state": None if state_obj is None else state_obj.json["bgstate"],
                                  "has_gauss": str(has_gauss), "gauss": f64_bits(gauss)}}, make=make)


def entropy_spec(e):
    """The JSON of a seed-sequence entropy / object seed value: an int, a flat or nested list of ints, a uint32 array
    (C#: uint[] — NumSharp's stand-in for NumPy's uint32 ndarray), a str, a float, a bool, or None (OS entropy)."""
    if e is None:
        return {"none": True}
    if isinstance(e, bool):
        return {"bool": e}
    if isinstance(e, int):
        return {"int": str(e)}
    if isinstance(e, float):
        return {"float": f64_bits(e)}
    if isinstance(e, str):
        return {"str": e}
    if isinstance(e, np.ndarray) and e.dtype == np.uint32:
        return {"uint32": [str(int(x)) for x in e]}
    if isinstance(e, (list, tuple)):
        return {"list": [entropy_spec(x) for x in e]}
    raise TypeError(f"no entropy spec for {e!r}")


def entropy_py(spec):
    """The Python value an entropy spec stands for (the inverse of entropy_spec)."""
    if "none" in spec:
        return None
    if "bool" in spec:
        return spec["bool"]
    if "int" in spec:
        return int(spec["int"])
    if "float" in spec:
        return struct.unpack("<d", struct.pack("<Q", int(spec["float"], 16)))[0]
    if "str" in spec:
        return spec["str"]
    if "uint32" in spec:
        return np.array([int(x) for x in spec["uint32"]], dtype=np.uint32)
    return [entropy_py(x) for x in spec["list"]]


def seedseq_spec(entropy, spawn_key=None, pool_size=None, n_children_spawned=None):
    """A SeedSequence description both sides build from: the entropy spec and the optional keywords (absent = NumPy's
    default)."""
    spec = {"entropy": entropy_spec(entropy)}
    if spawn_key is not None:
        spec["spawn_key"] = entropy_spec(list(spawn_key) if not isinstance(spawn_key, (str, int)) else spawn_key)
    if pool_size is not None:
        spec["pool_size"] = str(pool_size)
    if n_children_spawned is not None:
        spec["n_children_spawned"] = str(n_children_spawned)
    return spec


def build_seedseq(spec):
    """np.random.SeedSequence from a seedseq_spec (C#: new SeedSequence(object entropy, object spawn_key, long
    pool_size, uint n_children_spawned) with the same defaults)."""
    kw = {}
    if "spawn_key" in spec:
        kw["spawn_key"] = entropy_py(spec["spawn_key"])
    if "pool_size" in spec:
        kw["pool_size"] = int(spec["pool_size"])
    if "n_children_spawned" in spec:
        kw["n_children_spawned"] = int(spec["n_children_spawned"])
    return np.random.SeedSequence(entropy_py(spec["entropy"]), **kw)


def seedseq_obj(entropy, spawn_key=None, pool_size=None, n_children_spawned=None):
    """An ISeedSequence/SeedSequence argument."""
    spec = seedseq_spec(entropy, spawn_key, pool_size, n_children_spawned)
    return Obj({"seedseq": spec}, make=lambda: build_seedseq(spec))


def seedless_obj():
    """A SeedlessSeedSequence argument."""
    return Obj({"seedless": True}, make=SeedlessSeedSequence)


def generator_obj(engine, seed, prime_how="none"):
    """A Generator argument: Generator(ENGINE(seed)), primed."""
    def make():
        g = np.random.Generator(make_engine(engine, seed))
        prime(g, prime_how)
        return g
    return Obj({"generator": {"engine": engine, "seed": str(seed), "prime": prime_how}}, make=make)


def randomstate_obj(recv):
    """A NumPyRandom (RandomState) argument: another receiver, built and primed."""
    spec = recv.spec()
    spec["prime"] = recv.prime
    return Obj({"randomstate": spec}, make=recv.build)


def value_obj(v):
    """An `object` argument holding a plain Python value (int, float, bool, str, list, uint32 array, None)."""
    return Obj({"value": entropy_spec(v)}, make=lambda: entropy_py(entropy_spec(v)))


def bgstate_explicit(engine, **fields):
    """A bit generator state spelled out (the engine's typed State in C#, NumPy's state dict): unspecified fields take
    the parameterless State's values in C# — the generator records them explicitly on both sides."""
    if engine == "MT19937":
        d = {"bit_generator": "MT19937", "state": {"key": fields.get("key"), "pos": fields.get("pos", 0)}}
    elif engine in ("PCG64", "PCG64DXSM"):
        d = {"bit_generator": engine, "state": {"state": fields.get("state", 0), "inc": fields.get("inc", 0)},
             "has_uint32": fields.get("has_uint32", 0), "uinteger": fields.get("uinteger", 0)}
    elif engine == "Philox":
        d = {"bit_generator": "Philox", "state": {"counter": fields.get("counter"), "key": fields.get("key")},
             "buffer": fields.get("buffer"), "buffer_pos": fields.get("buffer_pos", 0),
             "has_uint32": fields.get("has_uint32", 0), "uinteger": fields.get("uinteger", 0)}
    else:
        d = {"bit_generator": "SFC64", "state": {"state": fields.get("state")},
             "has_uint32": fields.get("has_uint32", 0), "uinteger": fields.get("uinteger", 0)}

    def js(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple, np.ndarray)):
            return [str(int(x)) for x in v]
        return str(int(v))

    def make():
        out = json.loads(json.dumps(d, default=lambda o: [int(x) for x in o]))
        core = out["state"]
        for k in ("key", "counter", "state"):
            if k in core and isinstance(core[k], list):
                core[k] = np.array(core[k], dtype=np.uint32 if engine == "MT19937" else np.uint64)
        if isinstance(out.get("buffer"), list):
            out["buffer"] = np.array(out["buffer"], dtype=np.uint64)
        return out
    flat = {"engine": engine}
    for k, v in fields.items():
        flat[k] = js(v)
    return Obj({"bgstate_explicit": flat}, make=make)


def pyint_obj(v):
    """An `object` argument holding a Python int (C#: a boxed long)."""
    return Obj({"pyint": str(v)}, py=v)


def pystr_obj(v):
    """An `object` argument holding a Python str (C#: a string)."""
    return Obj({"str": v}, py=v)


# ======================================================================================================================
# Receivers and priming
# ======================================================================================================================

def make_engine(engine, seed):
    """ENGINE(seed) — a Python int seed goes through SeedSequence, as `new ENGINE((long)seed)` does in C#."""
    return ENGINE_CLASS[engine](seed)


class Recv:
    """How to build the object a member is invoked on, identically on both sides.

    Stream receivers: `RandomState` (legacy-seeded when `engine` is None, else RandomState(ENGINE(seed))), `Generator`,
    `BitGenerator`. Object receivers wrap an inner stream receiver: `legacy_tuple` is its `get_state()` tuple (C#: the
    NativeRandomState struct), `rs_dict` its `get_state(legacy=False)` dict (C#: NumPyRandom.State), `bgstate` a bit
    generator's `state` (C#: the engine's typed State), `rs_bitgen` a RandomState's `_bit_generator`. `seedseq` is a
    SeedSequence built from `extra['ss']` (see seedseq_spec), `seedless` a SeedlessSeedSequence. `none` is for
    constructors and static members.
    """

    def __init__(self, kind, engine=None, seed=None, prime="none", extra=None, inner=None):
        self.kind, self.engine, self.seed, self.prime, self.extra = kind, engine, seed, prime, extra or {}
        self.inner = inner

    def spec(self):
        s = {"k": self.kind}
        if self.engine is not None:
            s["engine"] = self.engine
        if self.seed is not None:
            s["seed"] = str(self.seed)
        if self.inner is not None:
            inner = self.inner.spec()
            inner["prime"] = self.inner.prime
            s["of"] = inner
        s.update(self.extra)
        return s

    def tag(self):
        """The receiver's part of a case id (stable: kind, engine, seed, priming, the seed-sequence spec, and the inner
        receiver)."""
        t = f"{self.kind}:{self.engine or '-'}:s{self.seed}:{self.prime}"
        if "ss" in self.extra:
            t += ":" + json.dumps(self.extra["ss"], separators=(",", ":"), sort_keys=True)
        return f"{t}({self.inner.tag()})" if self.inner is not None else t

    def build(self):
        """The NumPy receiver, primed."""
        if self.kind == "none":
            return None
        if self.kind == "legacy_tuple":
            return self.inner.build().get_state()
        if self.kind == "rs_dict":
            return self.inner.build().get_state(legacy=False)
        if self.kind == "bgstate":
            return self.inner.build().state
        if self.kind == "rs_bitgen":
            return self.inner.build()._bit_generator
        if self.kind == "seedseq":
            return build_seedseq(self.extra["ss"])
        if self.kind == "seedless":
            return SeedlessSeedSequence()
        if self.kind == "RandomState":
            r = np.random.RandomState(self.seed) if self.engine is None else np.random.RandomState(make_engine(self.engine, self.seed))
        elif self.kind == "Generator":
            r = np.random.Generator(make_engine(self.engine, self.seed))
        elif self.kind == "BitGenerator":
            r = make_engine(self.engine, self.seed)
        else:
            raise ValueError(f"unknown receiver kind {self.kind}")
        prime(r, self.prime)
        return r


def prime(r, how):
    """Put a receiver into a non-trivial state before the observed call (mirrored by the C# harness).

    raw3  — three raw words from the bit generator;
    u32   — one 32-bit draw (leaves a buffered half on a 64-bit engine): integers/randint of one full-range uint32;
    gauss — one legacy normal (RandomState's Gaussian cache full).
    """
    if how == "none":
        return
    if how == "raw3":
        bg = r if isinstance(r, np.random.BitGenerator) else (r.bit_generator if isinstance(r, np.random.Generator) else r._bit_generator)
        bg.random_raw(3)
    elif how == "u32":
        if isinstance(r, np.random.Generator):
            r.integers(0, 2 ** 32, size=1, dtype=np.uint32)
        else:
            r.randint(0, 2 ** 32, size=1, dtype=np.uint32)
    elif how == "gauss":
        r.standard_normal()
    else:
        raise ValueError(f"unknown prime {how}")


# ======================================================================================================================
# Observations
# ======================================================================================================================

def arr_obs(a):
    a = np.asarray(a)
    return {"k": "array", "dtype": a.dtype.name, "shape": [int(d) for d in a.shape],
            "hex": np.ascontiguousarray(a).tobytes().hex()}


def _words(v):
    """Comma-joined decimal words of an integer sequence; `null` for an unset (None) one."""
    return "null" if v is None else ",".join(str(int(x)) for x in v)


def bitgen_state_text(st):
    """Canonical text of a bit generator state dict (+ has_gauss/gauss when it is a RandomState's). A field the dict
    leaves unset (None) reads `null` — the parameterless C# State objects start that way."""
    name = st["bit_generator"]
    core = st["state"]
    if name == "MT19937":
        key = "null" if core["key"] is None else hashlib.sha256(np.asarray(core["key"], dtype=np.uint32).tobytes()).hexdigest()
        text = f"MT19937|pos={int(core['pos'])}|key={key}"
    elif name in ("PCG64", "PCG64DXSM"):
        text = (f"{name}|state={int(core['state'])}|inc={int(core['inc'])}"
                f"|has_uint32={int(st['has_uint32'])}|uinteger={int(st['uinteger'])}")
    elif name == "Philox":
        text = ("Philox|counter=" + _words(core["counter"]) + "|key=" + _words(core["key"]) +
                "|buffer=" + _words(st["buffer"]) +
                f"|buffer_pos={int(st['buffer_pos'])}|has_uint32={int(st['has_uint32'])}|uinteger={int(st['uinteger'])}")
    elif name == "SFC64":
        text = ("SFC64|state=" + _words(core["state"]) +
                f"|has_uint32={int(st['has_uint32'])}|uinteger={int(st['uinteger'])}")
    else:
        raise ValueError(f"unknown bit generator {name}")
    if "has_gauss" in st:
        text += f"|has_gauss={int(st['has_gauss'])}|gauss={f64_bits(st['gauss'])}"
    return text


def legacy_tuple_text(t):
    """Canonical text of a legacy state tuple ('MT19937', key, pos, has_gauss, cached_gaussian) — C#'s NativeRandomState."""
    key = "null" if t[1] is None else hashlib.sha256(np.asarray(t[1], dtype=np.uint32).tobytes()).hexdigest()
    return (f"tuple|{t[0]}|key={key}|pos={int(t[2])}|has_gauss={int(t[3])}|gauss={f64_bits(t[4])}")


def rsdict_text(d):
    """Canonical text of RandomState's dict state (get_state(legacy=False)) — C#'s NumPyRandom.State. An unset bit
    generator state (C#'s parameterless State) reads `null`."""
    if d.get("state") is None:
        return f"rsdict|null|has_gauss={int(d.get('has_gauss', 0))}|gauss={f64_bits(d.get('gauss', 0.0))}"
    return "rsdict|" + bitgen_state_text(d)


def seedseq_text(ss):
    """Canonical text of a seed sequence: NumPy's repr (entropy, spawn key, pool size, child counter as it prints them —
    NumSharp's ToString reproduces it) plus the mixed pool's hash and the child counter; `SeedlessSeedSequence` for the
    seedless one; None for none. The repr is the LAST field (it contains no '|'), so masks can blank it."""
    if ss is None:
        return None
    if isinstance(ss, SeedlessSeedSequence):
        return "SeedlessSeedSequence"
    pool = np.asarray(ss.pool, dtype=np.uint32)
    return (f"SeedSequence|n_children_spawned={int(ss.n_children_spawned)}"
            f"|pool={hashlib.sha256(pool.tobytes()).hexdigest()}|repr={ss!r}")


def recv_state_text(r):
    """The receiver's full state after the call, canonical ('' for receivers without one)."""
    if r is None:
        return ""
    if isinstance(r, np.random.Generator):
        return bitgen_state_text(r.bit_generator.state)
    if isinstance(r, np.random.RandomState):
        return bitgen_state_text(r.get_state(legacy=False))
    if isinstance(r, np.random.BitGenerator):
        return bitgen_state_text(r.state)
    if isinstance(r, tuple):
        return legacy_tuple_text(r)
    if isinstance(r, dict):
        # A RandomState dict carries the Gaussian cache; a bit generator's own state dict does not.
        return rsdict_text(r) if "has_gauss" in r else "bgstate|" + bitgen_state_text(r)
    if isinstance(r, (np.random.SeedSequence, SeedlessSeedSequence)):
        return seedseq_text(r)
    return ""


def mask_text(text, fields):
    """Replaces the named `field=value` entries of a canonical state text by `field=*` (the entropy-seeded words that
    neither side can reproduce; the C# comparator applies the same mask)."""
    for f in fields:
        text = re.sub(r"(^|\|)" + re.escape(f) + r"=[^|]*", r"\g<1>" + f + "=*", text)
    return text


def mask_obs(o, fields):
    """`mask_text` applied to every state text inside an observation tree (keys `state` and the text kinds' `v`)."""
    if not fields:
        return o
    if isinstance(o, dict):
        return {k: (mask_text(v, fields) if k in ("state", "v", "seed_seq") and isinstance(v, str) else mask_obs(v, fields))
                for k, v in o.items()}
    if isinstance(o, list):
        return [mask_obs(x, fields) for x in o]
    return o


def obs(v, ctype):
    """The canonical observation of a NumPy result against the C# return type `ctype` of the overload under test."""
    if ctype == "void":
        return {"k": "none"}
    if ctype in ("BitGenerator",) + tuple(ENGINES):
        return {"k": "bitgen", "type": type(v).__name__, "state": bitgen_state_text(v.state),
                "seed_seq": seedseq_text(v.seed_seq) or "None"}
    if ctype == "NumPyRandom":
        return {"k": "rs", "str": str(v), "state": bitgen_state_text(v.get_state(legacy=False))}
    if ctype == "Generator":
        return {"k": "gen", "str": str(v), "state": bitgen_state_text(v.bit_generator.state),
                "seed_seq": seedseq_text(v.bit_generator.seed_seq) or "None"}
    if ctype in ("ISeedSequence", "SeedSequence", "SeedlessSeedSequence", "ISpawnableSeedSequence"):
        return {"k": "none"} if v is None else {"k": "seedseq", "v": seedseq_text(v)}
    if ctype == "BigInteger[]":
        # SeedSequence.spawn_key: NumPy's tuple of Python ints, compared by its repr.
        return {"k": "text", "v": repr(tuple(int(x) for x in v))}
    if ctype == "Dictionary<string,object>":
        # SeedSequence.state: NumPy's dict, compared by its repr (NumPy's key order).
        return {"k": "text", "v": repr(v)}
    if ctype == "object" and not isinstance(v, (tuple, dict)):
        # SeedSequence.entropy: the int / list / array NumPy keeps, compared by its repr.
        return {"k": "text", "v": repr(v)}
    if ctype in ("NativeRandomState", "NumPyRandom.State", "object") and isinstance(v, (tuple, dict)):
        # By the VALUE: get_state() answers a non-MT19937 RandomState with the dict (and a warning) although the C#
        # overload is typed as the tuple — recorded as NumPy's answer; the divergence is classified on the C# side.
        if isinstance(v, tuple):
            return {"k": "legacy_state", "v": legacy_tuple_text(v)}
        return {"k": "rsdict", "v": rsdict_text(v)}
    if ctype == "BitGeneratorState" or (ctype.endswith(".State") and ctype != "NumPyRandom.State"):
        # The base class and every engine's typed State: NumPy's state dict of that engine.
        return {"k": "bgstate", "v": bitgen_state_text(v)}
    if ctype.endswith("[]") and ctype[:-2] in ("Generator", "BitGenerator", "SeedSequence", "NumPyRandom",
                                               "ISpawnableSeedSequence") + tuple(ENGINES):
        return {"k": "seq", "items": [obs(x, ctype[:-2]) for x in v]}
    if ctype in ("uint[]", "ulong[]", "long[]", "int[]"):
        dt = {"uint[]": np.uint32, "ulong[]": np.uint64, "long[]": np.int64, "int[]": np.int32}[ctype]
        return arr_obs(np.asarray(v, dtype=dt))
    if ctype.startswith("NDArray") or ctype == "T":
        if v is None:
            # random_raw(output=False): NumPy returns None, NumSharp a null NDArray.
            return {"k": "none"}
        if isinstance(v, (bytes, bytearray)):
            return arr_obs(np.frombuffer(bytes(v), dtype=np.uint8))
        if isinstance(v, bool):
            return {"k": "bool", "v": v}
        if isinstance(v, float):
            return {"k": "float", "bits": f64_bits(v)}
        if isinstance(v, int):
            return {"k": "int", "v": str(v)}
        return arr_obs(np.asarray(v))
    if ctype == "string":
        return {"k": "text", "v": v}
    if ctype in ("int", "long", "uint", "ulong", "UInt128", "BigInteger"):
        return {"k": "int", "v": str(int(v))}
    if ctype == "double":
        return {"k": "float", "bits": f64_bits(v)}
    if ctype == "bool":
        return {"k": "bool", "v": bool(v)}
    raise NotImplementedError(f"observation for C# type {ctype}")


# ======================================================================================================================
# Case emission
# ======================================================================================================================

class Case:
    """Accumulates the corpus rows per tier.

    Ids are STABLE — built from the signature, receiver, priming and variant tag, never a running counter — so adding a
    family does not renumber the others, and the LP64 sub-run (which generates a subset) produces the same id for the same
    case. A repeated id gets a `#k` suffix in generation order.
    """

    def __init__(self):
        self.rows = {t: [] for t in TIERS}
        self.ids = set()
        self.lp64_ids = set()   # legacy int-sampler cases whose Windows answer may depend on C long being 32-bit
        self.n = 0

    def unique(self, ident):
        if ident not in self.ids:
            return ident
        k = 2
        while f"{ident}#{k}" in self.ids:
            k += 1
        return f"{ident}#{k}"

    def add(self, tier, row, lp64=False):
        if row["id"] in self.ids:
            raise ValueError(f"duplicate case id {row['id']}")
        self.ids.add(row["id"])
        if lp64:
            self.lp64_ids.add(row["id"])
        self.rows[tier].append(row)
        self.n += 1


def run_numpy(recv, call, calls):
    """Build the receiver, perform the call `calls` times, return (result, receiver)."""
    r = recv.build()
    result = None
    for _ in range(calls):
        result = call(r)
    return result, r


def emit(out, tier, member, sig, returns, recv, args, call, calls=1, widen=False, variant="base", ops=None,
         state=True, lp64=False, targs=None, watch=(), state_mask=()):
    """Run NumPy for one case and record it (value or error).

    :param out: the Case accumulator.
    :param tier: the corpus tier.
    :param member: NumPy-level member (`Generator.normal`) — the coverage join's parameter.
    :param sig: the exact C# overload.
    :param returns: the overload's canonical C# return type (drives the observation).
    :param recv: the receiver.
    :param args: the Arg list (every parameter of the overload, in order; OMIT ones included).
    :param call: fn(receiver) -> NumPy result.
    :param calls: how many times the call repeats (the last result is recorded).
    :param widen: record an int32 result widened to int64 (NumSharp's LP64 model for some legacy samplers).
    :param variant: a short tag naming the argument variant (part of the id).
    :param ops: the Operands of the NDArray arguments.
    :param state: whether to record the receiver's post-call state.
    :param lp64: the case belongs to a legacy family whose Windows answer can depend on C long being 32-bit; the driver
        re-runs it under Linux NumPy and keeps the LP64 answer where it differs in that way (see `merge_lp64`).
    :param targs: canonical C# names of a generic member's type arguments (`randn<T>`).
    :param watch: the NDArray Args the member mutates in place (shuffle): their contents after the call are recorded
        as `expected.after`, in order, and the C# side observes the same operands.
    :param state_mask: state-text fields drawn from OS entropy (see `mask_text`), masked in the recorded state and in
        every state text inside the result observation.
    :returns: True when NumPy returned a value, False when it raised.
    """
    params = {"member": member, "sig": sig, "recv": recv.spec(), "prime": recv.prime,
              "args": [a.json for a in args]}
    if calls != 1:
        params["calls"] = calls
    if targs:
        params["targs"] = list(targs)
    if watch:
        params["watch"] = [a.json["op"] for a in watch]
    if state_mask:
        params["state_mask"] = list(state_mask)
    ident = out.unique(f"random_api/{sig}/{recv.tag()}/{variant}")
    operands = ops.items if ops is not None else []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result, r = run_numpy(recv, call, calls)
        except Exception as e:  # noqa: BLE001 — NumPy's exception IS the recorded contract
            out.add(tier, {"id": ident, "op": "random_api", "params": params, "operands": operands,
                           "expected": {"kind": "random_api"}, "expects_throw": True,
                           "error": {"type": type(e).__name__, "text": str(e).strip()},
                           "layout": f"random_api/{recv.kind}", "valueclass": "error"}, lp64=lp64)
            return False
    if widen and isinstance(result, (np.ndarray, np.generic)) and result.dtype == np.int32:
        # The C long NumSharp models as int64 (a numpy int32 SCALAR too: randint(dtype='l') with size=None).
        result = np.asarray(result).astype(np.int64)
        params["widened"] = True
    exp = {"kind": "random_api", "result": mask_obs(obs(result, returns), state_mask)}
    if state:
        exp["state"] = mask_text(recv_state_text(r), state_mask)
    if watch:
        exp["after"] = [arr_obs(a.py) for a in watch]
    out.add(tier, {"id": ident, "op": "random_api", "params": params, "operands": operands, "expected": exp,
                   "layout": f"random_api/{recv.kind}", "valueclass": "stream"}, lp64=lp64)
    return True


# ======================================================================================================================
# The inventory and family dispatch
# ======================================================================================================================

def load_surface():
    with open(SURFACE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


FAMILIES = {}      # (declaring type, member name) -> handler(out, member_dict, seeds)
EXEMPT = {}        # sig -> reason (no NumPy counterpart); mirrored by the C# exemption table


def family(type_name, *names):
    """Register a handler for the named members of a declaring type."""
    def deco(fn):
        for n in names:
            FAMILIES[(type_name, n)] = fn
        return fn
    return deco


def exempt(sig, reason):
    EXEMPT[sig] = reason


# ---- shared argument-variant machinery -------------------------------------------------------------------------------

def size_values(ctype):
    """The size-like values every overload's size parameter takes, by C# type. First = the base variant."""
    base = ctype.rstrip("?")
    if base == "Shape":
        # (3,) base; None (the scalar path), () (a GIVEN 0-d size), (2,3), an empty (0,).
        vals = [(3,), SHAPE_NONE, (), (2, 3), (0,)]
        if ctype.endswith("?"):
            vals[1] = None
        return vals
    if base in ("int", "long"):
        # One npy_intp dimension (the single-integer size overloads).
        return [3, 1, 0]
    if base in ("int[]", "long[]"):
        # null last: NumPy's size=None (one draw) — every reference parameter takes null somewhere.
        return [[3], [], [2, 3], [0], None]
    if base == "params long[]":
        return [[3], [], [2, 3], [0]]
    raise ValueError(f"no size domain for {ctype}")


def legacy_receivers(seeds, engines=True):
    """RandomState receivers: the legacy-seeded MT19937 and RandomState(ENGINE(seed)) for every engine."""
    for s in seeds:
        yield Recv("RandomState", None, s)
        if engines:
            for e in ENGINES:
                yield Recv("RandomState", e, s)


def generator_receivers(seeds):
    for s in seeds:
        for e in ENGINES:
            yield Recv("Generator", e, s)


def bitgen_receivers(seeds, engines=ENGINES):
    for s in seeds:
        for e in engines:
            yield Recv("BitGenerator", e, s)


def overloads(surface, type_name, name):
    return [m for m in surface["members"] if m["type"] == type_name and m["name"] == name]


def param_names(m):
    return [p["name"] for p in m["params"]]


# ======================================================================================================================
# Families: samplers (shared by NumPyRandom and Generator)
# ======================================================================================================================

NODEFAULT = object()   # a NumPy parameter without a default (a C# null NDArray then means Python's None)

# Each sampler: NumPy parameter names in order, NumPy's defaults, the base values, extra scalar variants (dicts of
# overrides, some of which NumPy rejects, recorded with NumPy's error), and the tier. Values reach every internal branch.
NAN = float("nan")
INF = float("inf")
HOST = "random_api_host"
PORTABLE = "random_api"

# "variants" run on both APIs; "legacy"/"gen" hold API-only variants (branches only one API has, or inputs on which the
# other API's NumPy never returns — legacy zipf(a >= 1025) and legacy vonmises(kappa >= 2**511) loop forever).
SAMPLERS = {
    "normal": dict(params=["loc", "scale"], defaults=[0.0, 1.0], base=[0.5, 2.0], tier=HOST,
                   variants=[{"loc": -3.0}, {"scale": 0.0}, {"scale": -1.0}, {"scale": -0.0}, {"loc": NAN},
                             {"scale": NAN}, {"loc": INF}]),
    "uniform": dict(params=["low", "high"], defaults=[0.0, 1.0], base=[-3.0, 7.0], tier=PORTABLE,
                    variants=[{"low": 5.0, "high": 5.0}, {"low": 2.0, "high": 1.0}, {"low": -1e308, "high": 1e308},
                              {"high": INF}, {"low": NAN}, {"low": -0.0, "high": 0.0}]),
    "beta": dict(params=["a", "b"], defaults=[NODEFAULT, NODEFAULT], base=[2.0, 3.0], tier=HOST,
                 variants=[{"a": 0.5, "b": 0.5}, {"a": 1e-3, "b": 1e-3}, {"a": 1e-105, "b": 1e-105}, {"a": 0.5, "b": 2.0},
                           {"a": 0.0}, {"b": -1.0}, {"a": NAN}, {"a": 1.0, "b": 1.0}]),
    "binomial": dict(params=["n", "p"], defaults=[NODEFAULT, NODEFAULT], base=[10, 0.35], tier=HOST, ints={"n"},
                     variants=[{"n": 100, "p": 0.4}, {"n": 1000, "p": 0.7}, {"n": 20, "p": 0.9}, {"n": 0}, {"p": 0.0},
                               {"p": 1.0}, {"n": -1}, {"p": 1.5}, {"p": -0.1}, {"p": NAN}, {"n": 5000, "p": 0.05}]),
    "chisquare": dict(params=["df"], defaults=[NODEFAULT], base=[3.0], tier=HOST,
                      variants=[{"df": 0.5}, {"df": 0.0}, {"df": -1.0}, {"df": 5e-324}, {"df": NAN}, {"df": 1e-300}]),
    "exponential": dict(params=["scale"], defaults=[1.0], base=[2.5], tier=HOST,
                        variants=[{"scale": 0.0}, {"scale": -1.0}, {"scale": -0.0}, {"scale": NAN}, {"scale": INF}]),
    "f": dict(params=["dfnum", "dfden"], defaults=[NODEFAULT, NODEFAULT], base=[5.0, 7.0], tier=HOST,
              variants=[{"dfnum": 0.5, "dfden": 0.5}, {"dfnum": 0.0}, {"dfden": -1.0}, {"dfnum": NAN}, {"dfden": 5e-324}]),
    "gamma": dict(params=["shape", "scale"], defaults=[NODEFAULT, 1.0], base=[2.0, 3.0], tier=HOST,
                  variants=[{"shape": 0.5, "scale": 2.0}, {"shape": 1.0}, {"shape": 0.0}, {"shape": -1.0},
                            {"scale": -1.0}, {"scale": 0.0}, {"shape": -0.0}, {"shape": NAN}]),
    "geometric": dict(params=["p"], defaults=[NODEFAULT], base=[0.35], tier=HOST, step=0.2,
                      variants=[{"p": 0.1}, {"p": 1e-5}, {"p": 1.0}, {"p": 0.0}, {"p": 1.5}, {"p": NAN}, {"p": 5e-324}]),
    "gumbel": dict(params=["loc", "scale"], defaults=[0.0, 1.0], base=[0.5, 2.0], tier=HOST,
                   variants=[{"scale": 0.0}, {"scale": -1.0}, {"loc": -5.0}, {"scale": NAN}]),
    "hypergeometric": dict(params=["ngood", "nbad", "nsample"], defaults=[NODEFAULT] * 3, base=[10, 7, 8], tier=HOST,
                           ints={"ngood", "nbad", "nsample"},
                           variants=[{"ngood": 100, "nbad": 200, "nsample": 50}, {"ngood": 300, "nbad": 100, "nsample": 250},
                                     {"ngood": 1000, "nbad": 1000, "nsample": 1995}, {"ngood": 15, "nbad": 15, "nsample": 20},
                                     {"ngood": 0, "nbad": 5, "nsample": 3}, {"ngood": 5, "nbad": 0, "nsample": 3},
                                     {"nsample": 0}, {"ngood": 5, "nbad": 5, "nsample": 11}, {"ngood": -1},
                                     {"nbad": -1}, {"nsample": -1}],
                           gen=[{"ngood": 10 ** 9}, {"nbad": 10 ** 9}]),
    "laplace": dict(params=["loc", "scale"], defaults=[0.0, 1.0], base=[0.5, 1.5], tier=HOST,
                    variants=[{"scale": 0.0}, {"scale": -1.0}, {"loc": NAN}]),
    "logistic": dict(params=["loc", "scale"], defaults=[0.0, 1.0], base=[0.5, 1.5], tier=HOST,
                     variants=[{"scale": 0.0}, {"scale": -1.0}, {"loc": NAN}]),
    "lognormal": dict(params=["mean", "sigma"], defaults=[0.0, 1.0], base=[1.0, 0.5], tier=HOST,
                      variants=[{"sigma": 0.0}, {"sigma": -1.0}, {"mean": NAN}, {"mean": 800.0}]),
    "logseries": dict(params=["p"], defaults=[NODEFAULT], base=[0.6], tier=HOST, step=0.1,
                      variants=[{"p": 0.99}, {"p": 0.1}, {"p": 0.0}, {"p": 1.0}, {"p": -0.1}, {"p": NAN},
                                {"p": 0.9999999999999999}]),
    "negative_binomial": dict(params=["n", "p"], defaults=[NODEFAULT, NODEFAULT], base=[5.0, 0.4], tier=HOST,
                              variants=[{"n": 0.5, "p": 0.5}, {"p": 1.0}, {"n": 0.0}, {"p": 1.5}, {"p": 0.0}, {"n": NAN},
                                        {"p": NAN}, {"n": 50.0, "p": 0.9}],
                              gen=[{"n": 1e18, "p": 1e-18}]),
    "noncentral_chisquare": dict(params=["df", "nonc"], defaults=[NODEFAULT, NODEFAULT], base=[3.0, 1.5], tier=HOST,
                                 variants=[{"df": 0.5, "nonc": 1.5}, {"df": 3.0, "nonc": 0.0}, {"df": 0.5, "nonc": 30.0},
                                           {"nonc": -1.0}, {"df": 0.0}, {"nonc": NAN}, {"df": 5e-324, "nonc": 5e-324}]),
    "noncentral_f": dict(params=["dfnum", "dfden", "nonc"], defaults=[NODEFAULT] * 3, base=[5.0, 7.0, 1.5], tier=HOST,
                         variants=[{"dfnum": 0.5}, {"nonc": 0.0}, {"dfnum": 0.0}, {"nonc": -1.0}, {"dfden": 0.0},
                                   {"nonc": NAN}]),
    "pareto": dict(params=["a"], defaults=[NODEFAULT], base=[3.0], tier=HOST,
                   variants=[{"a": 0.5}, {"a": 0.0}, {"a": -1.0}, {"a": NAN}, {"a": 1e300}]),
    "poisson": dict(params=["lam"], defaults=[1.0], base=[3.5], tier=HOST,
                    variants=[{"lam": 0.0}, {"lam": 15.0}, {"lam": 100.0}, {"lam": 1e6}, {"lam": -1.0}, {"lam": NAN},
                              {"lam": 1e19}, {"lam": -0.0}]),
    "power": dict(params=["a"], defaults=[NODEFAULT], base=[2.5], tier=HOST,
                  variants=[{"a": 0.3}, {"a": 0.0}, {"a": -1.0}, {"a": NAN}, {"a": 1e-3}]),
    "rayleigh": dict(params=["scale"], defaults=[1.0], base=[1.5], tier=HOST,
                     variants=[{"scale": 0.0}, {"scale": -1.0}, {"scale": NAN}]),
    "standard_gamma": dict(params=["shape"], defaults=[NODEFAULT], base=[2.0], tier=HOST,
                           variants=[{"shape": 0.5}, {"shape": 1.0}, {"shape": 0.0}, {"shape": -1.0}, {"shape": NAN},
                                     {"shape": -0.0}]),
    "standard_t": dict(params=["df"], defaults=[NODEFAULT], base=[3.5], tier=HOST,
                       variants=[{"df": 0.5}, {"df": 0.0}, {"df": 5e-324}, {"df": -1.0}, {"df": NAN}]),
    "triangular": dict(params=["left", "mode", "right"], defaults=[NODEFAULT] * 3, base=[0.0, 3.0, 10.0], tier=PORTABLE,
                       variants=[{"left": 0.0, "mode": 0.0, "right": 1.0}, {"left": 0.0, "mode": 1.0, "right": 1.0},
                                 {"left": 5.0}, {"mode": 11.0}, {"left": 0.0, "mode": 0.0, "right": 0.0},
                                 {"left": NAN}, {"left": -1e300, "right": 1e300}]),
    "vonmises": dict(params=["mu", "kappa"], defaults=[NODEFAULT, NODEFAULT], base=[0.5, 2.0], tier=HOST,
                     variants=[{"mu": 0.0, "kappa": 1e-9}, {"mu": 1.0, "kappa": 1e-6}, {"mu": -2.0, "kappa": 1e7},
                               {"mu": 3.0, "kappa": 50.0}, {"kappa": -1.0}, {"kappa": 0.0}, {"mu": NAN}, {"kappa": NAN}],
                     gen=[{"kappa": INF}]),
    "wald": dict(params=["mean", "scale"], defaults=[NODEFAULT, NODEFAULT], base=[3.0, 2.0], tier=HOST,
                 variants=[{"mean": 0.5, "scale": 10.0}, {"mean": 0.0}, {"scale": -1.0}, {"scale": 0.0}, {"mean": NAN}]),
    "weibull": dict(params=["a"], defaults=[NODEFAULT], base=[1.79], tier=HOST,
                    variants=[{"a": 0.3}, {"a": 0.0}, {"a": -1.0}, {"a": NAN}, {"a": -0.0}]),
    "zipf": dict(params=["a"], defaults=[NODEFAULT], base=[3.0], tier=HOST,
                 variants=[{"a": 1.5}, {"a": 1.05}, {"a": 1.0}, {"a": 0.5}, {"a": NAN}],
                 gen=[{"a": 1025.0}, {"a": 2000.0}]),
}


def sampler_scalar_values(name, variant):
    spec = SAMPLERS[name]
    vals = dict(zip(spec["params"], spec["base"]))
    vals.update(variant)
    return vals


def variant_receivers(api, seeds):
    """The receivers the NON-base variants run on: a few seeds, rotating engines (the base variant runs on all)."""
    s = (seeds * 3)[:3]
    if api == "legacy":
        return [Recv("RandomState", None, s[0]), Recv("RandomState", "PCG64", s[1]), Recv("RandomState", "Philox", s[2])]
    return [Recv("Generator", "PCG64", s[0]), Recv("Generator", "MT19937", s[1]), Recv("Generator", "SFC64", s[2])]


def emit_sampler_overload(out, m, api, name, seeds):
    """Every case for one sampler overload `m` (NumPyRandom or Generator).

    Base values on every receiver (all engines x all seeds); then, on the rotating variant receivers: every size value
    of the overload's size type, every optional parameter omitted, a null for every nullable/array parameter, every
    scalar variant (incl. NumPy's rejections), priming (buffered half, gauss cache, raw draws) and a repeated call, and
    for NDArray overloads the array layouts (0-d, 1-D, column x row, strided).
    """
    spec = SAMPLERS[name]
    member = ("RandomState." if api == "legacy" else "Generator.") + name
    pnames = param_names(m)
    ptypes = {p["name"]: p["type"] for p in m["params"]}
    popt = {p["name"]: p.get("optional", False) for p in m["params"]}
    np_default = dict(zip(spec["params"], spec["defaults"]))
    ints = spec.get("ints", set())
    receivers = list(legacy_receivers(seeds)) if api == "legacy" else list(generator_receivers(seeds))
    vrecv = variant_receivers(api, seeds)
    # Only a parameter NAMED size is the output size here (gamma's 'shape' is a distribution parameter).
    size_param = "size" if "size" in pnames else None
    array_params = [p for p in spec["params"] if p in ptypes and ptypes[p].startswith("NDArray")]

    def build(values, size, recv, variant, calls=1):
        ops = Operands()
        args = []
        kwargs = {}
        for p in pnames:
            ct = ptypes[p]
            v = size if p == size_param else values.get(p, OMIT)
            if v is OMIT and not popt[p]:
                raise ValueError(f"{m['sig']}: required parameter {p} has no value")
            if ct.startswith("NDArray") and v is not OMIT and v is not None and not isinstance(v, (np.ndarray, tuple)):
                v = np.array(v, dtype=np.int64 if p in ints else np.float64)
            a = encode(p, ct, v, ops)
            args.append(a)
            if a.omit:
                continue
            if p == size_param:
                kwargs["size"] = a.py
            elif a.py is None and ct.startswith("NDArray"):
                # A null NDArray is NumPy's default where NumPy has one ("not passed"), else Python's None.
                if np_default.get(p, NODEFAULT) is NODEFAULT:
                    kwargs[p] = None
            else:
                kwargs[p] = a.py
        widen = api == "legacy" and name in LEGACY_INT64_CAST
        return emit(out, spec["tier"], member, m["sig"], m["returns"], recv, args,
                    lambda r: getattr(r, name)(**kwargs), calls=calls, widen=widen, variant=variant, ops=ops,
                    lp64=widen)

    def build_on(values, size, recvs, variant, calls=1):
        """Emit on each receiver; once NumPy raises, the rest would raise identically (validation precedes the draws,
        and nothing about it depends on the engine or seed), so one error case suffices."""
        for recv in recvs:
            if not build(values, size, recv, variant, calls):
                return

    size_vals = size_values(ptypes[size_param]) if size_param else [None]
    base_size = size_vals[0]
    base_vals = sampler_scalar_values(name, {})
    # Base: all seeds x all receivers. An NDArray overload's base is the broadcast path (the first parameter 1-D of
    # three, the rest 0-d), so the engine x seed sweep exercises the array machinery rather than the scalar path.
    base_call_vals = dict(sampler_array_variants(name, array_params))["oned"] if array_params else base_vals
    for recv in receivers:
        build(base_call_vals, base_size, recv, "base")
    # Sizes.
    for i, sz in enumerate(size_vals[1:], 1):
        build_on(base_vals, sz, vrecv[:2], f"size{i}")
    # Every optional parameter omitted (the C# default path == NumPy's default).
    for p in pnames:
        if popt[p] and p != size_param:
            vals = dict(base_vals)
            vals[p] = OMIT
            build_on(vals, base_size, vrecv[:2], f"omit_{p}")
    if size_param and popt[size_param]:
        build_on(base_vals, OMIT, vrecv[:2], "omit_size")
    # A null for every NDArray parameter (NumPy's default, or None where NumPy has no default).
    for p in array_params:
        vals = dict(base_vals)
        vals[p] = None
        build_on(vals, base_size, vrecv[:1], f"null_{p}")
    # Scalar variants — every internal branch, and every input NumPy rejects (recorded with NumPy's error).
    for i, var in enumerate(spec["variants"] + spec.get(api, [])):
        build_on(sampler_scalar_values(name, var), base_size, vrecv[:2], f"var{i}")
    # Priming + a repeated call: stream advancement from non-trivial states.
    for recv0 in vrecv[:2]:
        for pr in (["u32", "gauss", "raw3"] if api == "legacy" else ["u32", "raw3"]):
            build(base_vals, base_size, Recv(recv0.kind, recv0.engine, recv0.seed, prime=pr), f"prime_{pr}")
        build(base_vals, base_size, recv0, "calls2", calls=2)
    # Array layouts for the NDArray overloads (the scalar variants' values ride the 1-D form too).
    if array_params:
        broadcast_size = SHAPE_NONE if ptypes[size_param] == "Shape" else None
        for tag, vals in sampler_array_variants(name, array_params):
            build_on(vals, broadcast_size if tag in ("zerod", "bcast2d") else base_size, vrecv[:2], f"arr_{tag}")
        for i, var in enumerate(spec["variants"] + spec.get(api, [])):
            vals = sampler_scalar_values(name, var)
            first = array_params[0]
            arr = dict(vals)
            arr[first] = np.array([vals[first]] * 3, dtype=np.int64 if first in ints else np.float64)
            build_on(arr, base_size, vrecv[:1], f"arrvar{i}")


def sampler_array_variants(name, array_params):
    """Array-parameter variants for an NDArray overload: (tag, {param: value}) with (base, view) pairs for views.

    0-d everywhere (NumPy's scalar path), the first parameter 1-D, a column against a row, a strided view, and the dtypes
    the 'safe' conversion gate sees: an int64 parameter where float64 is expected, a float32 one, a bool one, and a
    float where an integer is expected (NumPy's TypeError).
    """
    spec = SAMPLERS[name]
    ints = spec.get("ints", set())
    base = dict(zip(spec["params"], spec["base"]))

    def arr(p, values):
        return np.array(values, dtype=np.int64 if p in ints else np.float64)

    out = []
    first = array_params[0]
    b0 = base[first]
    # The spacing of the 1-D/strided forms: 1 by default; a probability sampler's own step keeps every element inside
    # its domain, so the engine x seed base sweep records streams instead of NumPy's domain error.
    step = 1 if first in ints else spec.get("step", 1.0)
    out.append(("zerod", {p: arr(p, base[p]) for p in array_params}))
    one = {p: arr(p, base[p]) for p in array_params}
    one[first] = arr(first, [b0, b0 + step, b0 + 2 * step])
    out.append(("oned", one))
    if len(array_params) >= 2:
        col = {p: arr(p, base[p]) for p in array_params}
        a0, a1 = array_params[0], array_params[1]
        s0 = 1 if a0 in ints else 1.0
        col[a0] = arr(a0, [[base[a0]], [base[a0] + s0]])
        col[a1] = arr(a1, [[base[a1], base[a1], base[a1]]])
        out.append(("bcast2d", col))
    strided = {p: arr(p, base[p]) for p in array_params}
    src = arr(first, [b0, b0, b0 + step, b0, b0 + 2 * step, b0])
    strided[first] = (src, src[::2])
    out.append(("strided", strided))
    if first in ints:
        conv = {p: arr(p, base[p]) for p in array_params}
        conv[first] = np.array([float(b0), float(b0) + 1.0], dtype=np.float64)   # NumPy: TypeError, 'safe' refuses
        out.append(("float_for_int", conv))
        u32 = {p: arr(p, base[p]) for p in array_params}
        u32[first] = np.array([b0, b0 + 1], dtype=np.uint32)
        out.append(("uint32", u32))
    else:
        i64 = {p: arr(p, base[p]) for p in array_params}
        i64[first] = np.array([int(b0) + 1, int(b0) + 2, int(b0) + 3], dtype=np.int64)
        out.append(("int64", i64))
        f32 = {p: arr(p, base[p]) for p in array_params}
        f32[first] = np.array([b0, b0 + step, b0 + 2.0 * step], dtype=np.float32)
        out.append(("float32", f32))
        cplx = {p: arr(p, base[p]) for p in array_params}
        cplx[first] = np.array([b0 + 0j, b0 + 1j], dtype=np.complex128)   # NumPy: TypeError, 'safe' refuses complex
        out.append(("complex", cplx))
    boolean = {p: arr(p, base[p]) for p in array_params}
    boolean[first] = np.array([True, False, True])
    out.append(("bool", boolean))
    return out


def fam_legacy_samplers(out, surface, name, seeds):
    for m in overloads(surface, "NumPyRandom", name):
        emit_sampler_overload(out, m, "legacy", name, seeds)


def fam_generator_samplers(out, surface, name, seeds):
    for m in overloads(surface, "Generator", name):
        emit_sampler_overload(out, m, "gen", name, seeds)


# Every table sampler, on both APIs; Generator.standard_gamma has dtype/out parameters and gets its own family below.
for _name in SAMPLERS:
    FAMILIES[("NumPyRandom", _name)] = fam_legacy_samplers
    if _name != "standard_gamma":
        FAMILIES[("Generator", _name)] = fam_generator_samplers


# ======================================================================================================================
# Per-overload emission for the non-table members
# ======================================================================================================================

def emit_m(out, tier, member, m, recv, values, variant, np_fn, calls=1, widen=False, lp64=False, state=True,
           targs=None, watch=(), state_mask=()):
    """One case of overload `m`: `values` maps C# parameter names to Python values (OMIT = left out; a no-argument
    callable is called for a fresh dict); `np_fn(numpy_receiver, py)` performs NumPy's counterpart, where `py` maps the
    passed parameters' names to the values NumPy receives. `widen`/`lp64` may be callables of the values dict. A
    constructor's observation is of the object it builds (its declaring type), not of `void`.
    """
    values = values() if callable(values) else values
    for p in m["params"]:
        if not fits(p["type"], values.get(p["name"], OMIT)):
            return None   # not expressible through this overload (e.g. 2**64 for a ulong): not a case
    ops = Operands()
    args, py = [], {}
    for p in m["params"]:
        name, ct = p["name"], p["type"]
        v = values.get(name, OMIT)
        if v is OMIT and not p.get("optional", False):
            raise ValueError(f"{m['sig']}: required parameter {name} has no value")
        if isinstance(v, Alias):
            # The same operand index and the same Python object as the aliased (earlier) argument.
            src = next(a for a in args if a.name == v.name)
            a = Arg(name, ct, {"n": name, "t": ct, "op": src.json["op"]}, src.py)
        else:
            a = encode(name, ct, v, ops)
        args.append(a)
        if not a.omit:
            py[name] = a.py
    w = widen(values) if callable(widen) else widen
    lp = lp64(values) if callable(lp64) else lp64
    watch_args = [a for a in args if a.name in watch and not a.omit and a.py is not None
                  and "op" in a.json and a.json["op"] not in [w.json.get("op") for w in args[:args.index(a)] if w.name in watch]]
    returns = m["type"] if m["kind"] == "ctor" else m["returns"]
    return emit(out, tier, member, m["sig"], returns, recv, args, lambda r: np_fn(r, py), calls=calls, widen=w,
                variant=variant, ops=ops, state=state, lp64=lp, targs=targs, watch=watch_args, state_mask=state_mask)


INT_RANGES = {"sbyte": (-2 ** 7, 2 ** 7 - 1), "byte": (0, 2 ** 8 - 1), "short": (-2 ** 15, 2 ** 15 - 1),
              "ushort": (0, 2 ** 16 - 1), "int": (-2 ** 31, 2 ** 31 - 1), "uint": (0, 2 ** 32 - 1),
              "long": (-2 ** 63, 2 ** 63 - 1), "ulong": (0, 2 ** 64 - 1), "UInt128": (0, 2 ** 128 - 1)}


def fits(ctype, value):
    """Whether a Python value is representable as an argument of the C# parameter type (integers and integer arrays by
    range; everything else is accepted — its encoder decides)."""
    if value is OMIT or value is None:
        return True
    base = ctype.rstrip("?").replace("params ", "")
    if base in INT_RANGES and isinstance(value, int) and not isinstance(value, bool):
        lo, hi = INT_RANGES[base]
        return lo <= value <= hi
    if base.endswith("[]") and base[:-2] in INT_RANGES and isinstance(value, (list, tuple)):
        lo, hi = INT_RANGES[base[:-2]]
        return all(lo <= int(x) <= hi for x in value)
    return True


def sweep(out, api, tier, member, m, seeds, base, variants, np_fn, primes=None, calls2=True, base_all=True, prefix="",
          **kw):
    """The standard case set of one overload of either API (`legacy`: RandomState, `gen`: Generator): `base` on every
    receiver of the API (legacy: the legacy-seeded MT19937 and RandomState(ENGINE) x 5; gen: Generator(ENGINE) x 5;
    x every seed) — or only the two variant receivers when `base_all` is false; each (tag, values) of `variants` on the
    two variant receivers (stopping at the first NumPy error: validation precedes the draws and depends on neither
    engine nor seed); then the base values from primed receivers (the API's primings unless `primes` is given) and a
    repeated call. `prefix` goes in front of the sweep's own tags (base, prime_*, calls2) when one overload is swept
    more than once; variant tags are the caller's.
    """
    if primes is None:
        primes = ("u32", "gauss", "raw3") if api == "legacy" else ("u32", "raw3")
    vrecv = variant_receivers(api, seeds)
    all_recvs = legacy_receivers(seeds) if api == "legacy" else generator_receivers(seeds)
    for recv in (all_recvs if base_all else vrecv[:2]):
        emit_m(out, tier, member, m, recv, base, prefix + "base", np_fn, **kw)
    for tag, vals in variants:
        for recv in vrecv[:2]:
            # None (not expressible) or False (NumPy raised): the rest of the receivers would add nothing.
            if not emit_m(out, tier, member, m, recv, vals, tag, np_fn, **kw):
                break
    for recv0 in vrecv[:2]:
        for pr in primes:
            emit_m(out, tier, member, m, Recv(recv0.kind, recv0.engine, recv0.seed, prime=pr), base,
                   f"{prefix}prime_{pr}", np_fn, **kw)
        if calls2:
            emit_m(out, tier, member, m, recv0, base, prefix + "calls2", np_fn, calls=2, **kw)


def legacy_sweep(out, tier, member, m, seeds, base, variants, np_fn, **kw):
    """`sweep` over the legacy RandomState receivers."""
    return sweep(out, "legacy", tier, member, m, seeds, base, variants, np_fn, **kw)


def gen_sweep(out, tier, member, m, seeds, base, variants, np_fn, **kw):
    """`sweep` over the Generator receivers."""
    return sweep(out, "gen", tier, member, m, seeds, base, variants, np_fn, **kw)


def dims_variants(pn, ctype):
    """The shape variants of a dimensions/size parameter, by its C# type: (tag, values) pairs, base first."""
    base = ctype.rstrip("?")
    if base in ("params long[]", "long[]", "int[]"):
        # A null array is Python's None: size=None, i.e. no dimensions (for rand/randn, the no-argument call).
        return [("base", {pn: [3]}), ("empty", {pn: []}), ("d2x3", {pn: [2, 3]}), ("zero", {pn: [0]}),
                ("d2x0x3", {pn: [2, 0, 3]}), ("d1", {pn: [1]}), ("neg", {pn: [-1]}), ("neg2", {pn: [3, -2]}),
                ("null", {pn: None})]
    if base == "Shape":
        return [("base", {pn: (3,)}), ("none", {pn: None if ctype.endswith("?") else SHAPE_NONE}),
                ("scalar", {pn: ()}), ("d2x3", {pn: (2, 3)}), ("zero", {pn: (0,)}), ("d1", {pn: (1,)})]
    if base in ("int", "long"):
        return [("base", {pn: 3}), ("one", {pn: 1}), ("zero", {pn: 0}), ("neg", {pn: -1})]
    raise ValueError(f"no dims domain for {ctype}")


def derived(base, fn):
    """(base, fn(base)): a view derived from a 1-D C-contiguous base — the layout_catalog convention (`describe`
    serializes the base in C order and the view by its memory offset and strides, so the base must be 1-D and C)."""
    return base, fn(base)


def size_kw(ctype, value):
    """NumPy's `size` argument for a C# size value of the given type: a null or a default Shape is None, a Shape is
    itself (an empty one is size=()), a params list is a tuple (none = None), a typed array a tuple."""
    base = ctype.rstrip("?")
    if value is None:
        return None
    if base == "params long[]":
        # The loose-dimensions spelling: no dimensions at all is NumPy's size=None (a scalar draw).
        return tuple(value) if len(value) else None
    if base in ("long[]", "int[]"):
        # A typed size array is a tuple: an empty one is size=() (a 0-d draw), not None.
        return tuple(value)
    return value


# ======================================================================================================================
# Families: NumPyRandom (the legacy RandomState) — everything but the distribution table
# ======================================================================================================================

LEGACY_PRIMES = ("u32", "gauss", "raw3")


@family("NumPyRandom", "rand", "randn")
def fam_rand(out, surface, name, seeds):
    """rand(d0, …, dn) / randn(d0, …, dn): the loose dimensions; rand(Shape) spreads the shape (NumPy has no
    size=None for rand: a default Shape is rand() itself); randn<T>() is NumPy's cast of one randn() draw to T."""
    tier = PORTABLE if name == "rand" else HOST
    for m in overloads(surface, "NumPyRandom", name):
        if m["generic"]:
            fam_randn_generic(out, m, seeds)
            continue
        pn, ct = m["params"][0]["name"], m["params"][0]["type"]
        if ct == "params long[]":
            def np_fn(r, py, pn=pn):
                # A null params array is no dimensions at all: rand() / randn().
                return getattr(r, name)(*(py.get(pn) or []))
        else:
            def np_fn(r, py, pn=pn):
                return getattr(r, name)(*(py.get(pn) or ()))
        vs = dims_variants(pn, ct)
        legacy_sweep(out, tier, f"RandomState.{name}", m, seeds, vs[0][1], vs[1:], np_fn)


# randn<T>: every NumPy dtype NumSharp has (Char and Decimal have no NumPy dtype).
RANDN_TARGS = [("double", "float64"), ("float", "float32"), ("Half", "float16"), ("int", "int32"), ("long", "int64"),
               ("short", "int16"), ("sbyte", "int8"), ("byte", "uint8"), ("ushort", "uint16"), ("uint", "uint32"),
               ("ulong", "uint64"), ("bool", "bool"), ("Complex", "complex128")]


def fam_randn_generic(out, m, seeds):
    """randn<T>(): one legacy normal (the Gaussian cache consumed or refilled exactly as randn()), converted to T the
    way NumPy casts a float64 to that dtype. double runs on every receiver; the other T on the variant receivers, from
    fresh and primed states."""
    vrecv = variant_receivers("legacy", seeds)
    for cs, npd in RANDN_TARGS:
        def np_fn(r, py, npd=npd):
            return np.array(r.randn(), dtype=np.float64).astype(npd)
        recvs = list(legacy_receivers(seeds)) if cs == "double" else vrecv[:2]
        for recv in recvs:
            emit_m(out, HOST, "RandomState.randn", m, recv, {}, f"T={cs}", np_fn, targs=[cs])
        for recv0 in vrecv[:2]:
            for pr in LEGACY_PRIMES:
                emit_m(out, HOST, "RandomState.randn", m, Recv(recv0.kind, recv0.engine, recv0.seed, prime=pr), {},
                       f"T={cs}/prime_{pr}", np_fn, targs=[cs])
            emit_m(out, HOST, "RandomState.randn", m, recv0, {}, f"T={cs}/calls2", np_fn, calls=2, targs=[cs])


@family("NumPyRandom", "random_sample", "random", "ranf", "sample")
def fam_random_sample(out, surface, name, seeds):
    """random_sample and its aliases. `ranf`/`sample` are NumPy module functions over the singleton's random_sample;
    the instance counterpart is random_sample itself. The params-array spelling passes the dimensions as ONE size
    tuple (none = size=None); the Shape spelling is size itself (default = None, () = a 0-d array)."""
    np_name = "random" if name == "random" else "random_sample"
    for m in overloads(surface, "NumPyRandom", name):
        pn, ct = m["params"][0]["name"], m["params"][0]["type"]

        def np_fn(r, py, pn=pn, ct=ct):
            return getattr(r, np_name)(size_kw(ct, py.get(pn)))
        vs = dims_variants(pn, ct)
        legacy_sweep(out, PORTABLE, f"RandomState.{name}", m, seeds, vs[0][1], vs[1:], np_fn)


@family("NumPyRandom", "standard_cauchy", "standard_exponential", "standard_normal")
def fam_standard(out, surface, name, seeds):
    """The parameterless samplers: () is size=None; (Shape) takes the size."""
    for m in overloads(surface, "NumPyRandom", name):
        if not m["params"]:
            legacy_sweep(out, HOST, f"RandomState.{name}", m, seeds, {}, [], lambda r, py: getattr(r, name)())
            continue
        pn, ct = m["params"][0]["name"], m["params"][0]["type"]
        vs = dims_variants(pn, ct) + [("d5x7", {pn: (5, 7)})]
        legacy_sweep(out, HOST, f"RandomState.{name}", m, seeds, vs[0][1], vs[1:],
                     lambda r, py, pn=pn: getattr(r, name)(size=py.get(pn)))


@family("NumPyRandom", "tomaxint")
def fam_tomaxint(out, surface, name, seeds):
    """tomaxint: bounded draws in [0, LONG_MAX] — the C long's width decides the values (LP64 tier)."""
    for m in overloads(surface, "NumPyRandom", name):
        if not m["params"]:
            legacy_sweep(out, PORTABLE, "RandomState.tomaxint", m, seeds, {}, [], lambda r, py: r.tomaxint(),
                         widen=True, lp64=True)
            continue
        pn, ct = m["params"][0]["name"], m["params"][0]["type"]
        vs = dims_variants(pn, ct)
        legacy_sweep(out, PORTABLE, "RandomState.tomaxint", m, seeds, vs[0][1], vs[1:],
                     lambda r, py, pn=pn: r.tomaxint(size=py.get(pn)), widen=True, lp64=True)


@family("NumPyRandom", "bernoulli")
def fam_bernoulli(out, surface, name, seeds):
    """bernoulli is NumSharp's own (NumPy has none): documented as one uniform per value, `U < p`, as float64 — so its
    oracle is that NumPy composition over the same stream."""
    def np_fn(r, py):
        p = py["p"]
        size = py.get("size")
        if size is None:
            return 1.0 if r.random_sample() < p else 0.0
        return (r.random_sample(size) < p).astype(np.float64)
    pvals = [("p0", 0.0), ("p1", 1.0), ("pneg", -0.5), ("pbig", 1.5), ("pnan", NAN), ("phalf", 0.5)]
    for m in overloads(surface, "NumPyRandom", name):
        if len(m["params"]) == 1:
            legacy_sweep(out, PORTABLE, "RandomState.bernoulli", m, seeds, {"p": 0.3},
                         [(t, {"p": v}) for t, v in pvals], np_fn)
            continue
        vs = dims_variants("size", m["params"][1]["type"])
        variants = [(t, dict(v, p=0.3)) for t, v in vs[1:]] + [(t, {"p": v, "size": (4,)}) for t, v in pvals]
        legacy_sweep(out, PORTABLE, "RandomState.bernoulli", m, seeds, {"p": 0.3, "size": (5,)}, variants, np_fn)


@family("NumPyRandom", "bytes")
def fam_bytes(out, surface, name, seeds):
    """bytes(length): whole uint32 words drawn, sliced like Python's tobytes()[:length] (C division for the count)."""
    for m in overloads(surface, "NumPyRandom", name):
        variants = [(f"len{n}", {"length": n}) for n in (0, 1, 3, 4, 5, 7, 8, 13, 16, 100, -1, -2, -3, -4, -5, -6,
                                                          -7, -8, -9, -100)]
        legacy_sweep(out, PORTABLE, "RandomState.bytes", m, seeds, {"length": 10}, variants,
                     lambda r, py: r.bytes(py["length"]))


def choice_np(r, py):
    """NumPy's RandomState.choice with the C# arguments (an omitted optional stays NumPy's default; a null p is None)."""
    kw = {}
    if "size" in py:
        kw["size"] = py["size"]
    if "replace" in py:
        kw["replace"] = py["replace"]
    if "p" in py:
        kw["p"] = py["p"]
    return r.choice(py["a"], **kw)


@family("NumPyRandom", "choice")
def fam_choice(out, surface, name, seeds):
    """choice over an integer population (the int/long overloads, a 0-d integer array) and over 1-D arrays: size forms,
    replace, p (valid, invalid, 0-d, 2-D), and every rejection NumPy has. Indices are the C long (widened; LP64 tier
    where the width decides)."""
    p5 = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    common = [("size_none", {"size": SHAPE_NONE}), ("size_scalar", {"size": ()}), ("size_2x3", {"size": (2, 3)}),
              ("size_zero", {"size": (0,)}), ("omit_size", {"size": OMIT}),
              ("noreplace", {"size": (3,), "replace": False}), ("noreplace_all", {"size": (5,), "replace": False}),
              ("noreplace_too_many", {"size": (6,), "replace": False}), ("omit_replace", {"size": (3,), "replace": OMIT}),
              ("p", {"size": (4,), "p": p5}), ("p_none", {"size": (4,), "p": None}), ("omit_p", {"size": (4,), "p": OMIT}),
              ("p_scalar_draw", {"size": SHAPE_NONE, "p": p5}),
              ("p_noreplace", {"size": (3,), "replace": False, "p": p5}),
              ("p_noreplace_zeros", {"size": (3,), "replace": False, "p": np.array([0.5, 0.5, 0.0, 0.0, 0.0])}),
              ("p_wrong_len", {"size": (3,), "p": np.array([0.5, 0.5])}),
              ("p_neg", {"size": (3,), "p": np.array([-0.1, 0.3, 0.3, 0.3, 0.2])}),
              ("p_nan", {"size": (3,), "p": np.array([NAN, 0.3, 0.3, 0.2, 0.2])}),
              ("p_sum", {"size": (3,), "p": np.array([0.1, 0.1, 0.1, 0.1, 0.1])}),
              ("p_0d", {"size": (3,), "p": np.array(0.5)}),
              ("p_2d", {"size": (3,), "p": np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])}),
              ("p_strided", {"size": (3,), "p": derived(np.array([0.1, 9, 0.2, 9, 0.3, 9, 0.25, 9, 0.15, 9]),
                                                          lambda b: b[::2])})]
    for m in overloads(surface, "NumPyRandom", name):
        at = m["params"][0]["type"]
        if at in ("int", "long"):
            base = {"a": 5, "size": (3,)}
            variants = [(t, dict({"a": 5}, **v)) for t, v in common]
            variants += [("a1", {"a": 1, "size": (3,)}), ("a0", {"a": 0, "size": (3,)}), ("a0_empty", {"a": 0, "size": (0,)}),
                         ("aneg", {"a": -3, "size": (3,)}), ("a_big", {"a": 1000003, "size": (4,)})]
            if at == "long":
                variants += [("a_2e31", {"a": 2 ** 31, "size": (4,)}), ("a_2e40", {"a": 2 ** 40, "size": (4,)})]
            legacy_sweep(out, PORTABLE, "RandomState.choice", m, seeds, base, variants, choice_np, widen=True, lp64=True)
            continue
        # NDArray a: 1-D populations (the drawn elements; never widened), a 0-d integer population (indices; widened),
        # and the rejections.
        pop = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        base = {"a": pop, "size": (3,)}
        variants = [(t, dict({"a": pop}, **v)) for t, v in common]
        variants += [("a_f64", {"a": np.array([0.5, 1.5, 2.5, 3.5, 4.5]), "size": (4,)}),
                     ("a_i32", {"a": np.array([1, 2, 3, 4, 5], dtype=np.int32), "size": (4,)}),
                     ("a_u8", {"a": np.array([1, 2, 3, 4, 5], dtype=np.uint8), "size": (4,)}),
                     ("a_bool", {"a": np.array([True, False, True, False, True]), "size": (4,)}),
                     ("a_c128", {"a": np.array([1 + 2j, 3 - 1j, 0j, 5j, -1 + 0j]), "size": (4,)}),
                     ("a_f16", {"a": np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float16), "size": (4,)}),
                     ("a_strided", {"a": derived(np.arange(10, dtype=np.int64), lambda b: b[::2]), "size": (4,)}),
                     ("a_negstride", {"a": derived(np.arange(5, dtype=np.int64), lambda b: b[::-1]), "size": (4,)}),
                     ("a_0d_int", {"a": np.array(7, dtype=np.int64), "size": (4,)}),
                     ("a_0d_int32", {"a": np.array(7, dtype=np.int32), "size": (4,)}),
                     ("a_0d_float", {"a": np.array(7.0), "size": (4,)}),
                     ("a_0d_zero", {"a": np.array(0, dtype=np.int64), "size": (3,)}),
                     ("a_0d_u64_big", {"a": np.array(2 ** 63 + 5, dtype=np.uint64), "size": (3,)}),
                     ("a_2d", {"a": np.arange(6, dtype=np.int64).reshape(2, 3), "size": (3,)}),
                     ("a_empty", {"a": np.array([], dtype=np.int64), "size": (3,)}),
                     ("a_empty_nosample", {"a": np.array([], dtype=np.int64), "size": (0,)}),
                     ("a_empty_scalar", {"a": np.array([], dtype=np.int64), "size": SHAPE_NONE}),
                     ("a_null", {"a": None, "size": (3,)})]

        def widen_fn(values):
            a = values.get("a")
            a = a[1] if isinstance(a, tuple) else a
            return isinstance(a, np.ndarray) and a.ndim == 0 and a.dtype.kind in "iu"
        legacy_sweep(out, PORTABLE, "RandomState.choice", m, seeds, base, variants, choice_np, widen=widen_fn,
                     lp64=widen_fn)


@family("NumPyRandom", "permutation")
def fam_permutation(out, surface, name, seeds):
    """permutation(int) is a shuffled arange in the C long (widened; LP64 where the width decides); permutation(array)
    a shuffled copy along the first axis, any layout, never widened."""
    for m in overloads(surface, "NumPyRandom", name):
        at = m["params"][0]["type"]
        if at in ("int", "long"):
            variants = [("x0", {"x": 0}), ("x1", {"x": 1}), ("xneg", {"x": -3}), ("x2", {"x": 2}), ("x100", {"x": 100})]
            if at == "long":
                variants += [("x_2e62", {"x": 2 ** 62}), ("x_max", {"x": 2 ** 63 - 1}), ("x_band", {"x": 2 ** 63 - 256})]
            legacy_sweep(out, PORTABLE, "RandomState.permutation", m, seeds, {"x": 10}, variants,
                         lambda r, py: r.permutation(py["x"]), widen=True, lp64=True)
            continue
        base = {"x": np.arange(10, dtype=np.int64)}
        variants = [("f64", {"x": np.linspace(0.0, 1.0, 7)}),
                    ("i32", {"x": np.arange(6, dtype=np.int32)}),
                    ("bool", {"x": np.array([True, False, True, True])}),
                    ("c128", {"x": np.array([1 + 1j, 2 - 2j, 3j])}),
                    ("d2", {"x": np.arange(12, dtype=np.int64).reshape(4, 3)}),
                    ("d3", {"x": np.arange(24, dtype=np.float64).reshape(2, 3, 4)}),
                    ("fortran", {"x": derived(np.arange(12, dtype=np.int64), lambda b: b.reshape(3, 4).T)}),
                    ("transposed3", {"x": derived(np.arange(24, dtype=np.float64),
                                                  lambda b: b.reshape(2, 3, 4).transpose(1, 0, 2))}),
                    ("strided", {"x": derived(np.arange(20, dtype=np.int64), lambda b: b[::3])}),
                    ("negstride", {"x": derived(np.arange(8, dtype=np.int64), lambda b: b[::-1])}),
                    ("cols_sliced", {"x": derived(np.arange(20, dtype=np.int64), lambda b: b.reshape(4, 5)[:, 1:4])}),
                    ("rows_strided", {"x": derived(np.arange(30, dtype=np.int64), lambda b: b.reshape(6, 5)[::2])}),
                    ("empty", {"x": np.array([], dtype=np.float64)}),
                    ("empty_rows", {"x": np.zeros((0, 3))}),
                    ("one", {"x": np.array([42], dtype=np.int64)}),
                    ("zerod", {"x": np.array(5, dtype=np.int64)}),
                    ("null", {"x": None})]
        legacy_sweep(out, PORTABLE, "RandomState.permutation", m, seeds, base, variants,
                     lambda r, py: r.permutation(py["x"]))


@family("NumPyRandom", "shuffle")
def fam_shuffle(out, surface, name, seeds):
    """shuffle(x) in place: the result is None and the operand's contents after the call are the observation."""
    def fresh(make):
        return lambda: {"x": make()}

    def view(make_base, fn):
        return lambda: {"x": derived(make_base(), fn)}

    variants = [("f64", fresh(lambda: np.linspace(0.0, 1.0, 9))),
                ("i32", fresh(lambda: np.arange(7, dtype=np.int32))),
                ("bool", fresh(lambda: np.array([True, False, True, False, False]))),
                ("c128", fresh(lambda: np.array([1 + 1j, 2 - 2j, 3j, -4 + 0j]))),
                ("f16", fresh(lambda: np.arange(6, dtype=np.float16))),
                ("d2", fresh(lambda: np.arange(12, dtype=np.int64).reshape(4, 3))),
                ("d3", fresh(lambda: np.arange(24, dtype=np.float64).reshape(3, 2, 4))),
                ("fortran", view(lambda: np.arange(12, dtype=np.int64), lambda b: b.reshape(3, 4).T)),
                ("transposed3", view(lambda: np.arange(24, dtype=np.float64),
                                     lambda b: b.reshape(2, 3, 4).transpose(1, 0, 2))),
                ("strided", view(lambda: np.arange(20, dtype=np.int64), lambda b: b[::3])),
                ("negstride", view(lambda: np.arange(8, dtype=np.int64), lambda b: b[::-1])),
                ("cols_sliced", view(lambda: np.arange(20, dtype=np.int64), lambda b: b.reshape(4, 5)[:, 1:4])),
                ("rows_strided", view(lambda: np.arange(30, dtype=np.int64), lambda b: b.reshape(6, 5)[::2])),
                ("empty", fresh(lambda: np.array([], dtype=np.float64))),
                ("one", fresh(lambda: np.array([42], dtype=np.int64))),
                ("two", fresh(lambda: np.array([1, 2], dtype=np.int64))),
                ("zerod", fresh(lambda: np.array(5, dtype=np.int64))),
                ("null", lambda: {"x": None})]
    for m in overloads(surface, "NumPyRandom", name):
        legacy_sweep(out, PORTABLE, "RandomState.shuffle", m, seeds, fresh(lambda: np.arange(10, dtype=np.int64)),
                     variants, lambda r, py: r.shuffle(py["x"]), watch=("x",))


def size_only_kw(py, ctype):
    """{'size': NumPy's size} when the C# call passes one (see size_kw), else {}."""
    return {"size": size_kw(ctype, py["size"])} if "size" in py else {}


@family("NumPyRandom", "multinomial")
def fam_multinomial(out, surface, name, seeds):
    """multinomial(n, pvals, size): the binomial chain per row (host libm); counts in the C long (widened; LP64)."""
    for m in overloads(surface, "NumPyRandom", name):
        pt = m["params"][1]["type"]
        st = m["params"][2]["type"] if len(m["params"]) > 2 else None
        pv = np.array([0.2, 0.3, 0.5]) if pt.startswith("NDArray") else [0.2, 0.3, 0.5]

        def arr(v):
            return np.array(v, dtype=np.float64) if pt.startswith("NDArray") else list(v)
        base = {"n": 10, "pvals": pv, "size": (3,) if st and st.rstrip("?") == "Shape" else ([3] if st in ("int[]",) else 3)}
        variants = [("n0", dict(base, n=0)), ("nneg", dict(base, n=-1)), ("n_big", dict(base, n=10 ** 6)),
                    ("n_2e40", dict(base, n=2 ** 40)),
                    ("p_one", dict(base, pvals=arr([1.0]))), ("p_zero_one", dict(base, pvals=arr([0.0, 1.0]))),
                    ("p_nan", dict(base, pvals=arr([NAN, 0.5, 0.5]))), ("p_neg", dict(base, pvals=arr([-0.1, 0.6, 0.5]))),
                    ("p_sum", dict(base, pvals=arr([0.6, 0.6, 0.1]))), ("p_last_slack", dict(base, pvals=arr([0.2, 0.2, 0.9]))),
                    ("p_big", dict(base, pvals=arr([1.5, 0.1]))), ("p_empty", dict(base, pvals=arr([]))),
                    ("p_null", dict(base, pvals=None))]
        if pt.startswith("NDArray"):
            variants += [("p_2d", dict(base, pvals=np.array([[0.2, 0.3, 0.5]]))),
                         ("p_0d", dict(base, pvals=np.array(0.5))),
                         ("p_f32", dict(base, pvals=np.array([0.2, 0.3, 0.5], dtype=np.float32))),
                         ("p_strided", dict(base, pvals=derived(np.array([0.2, 9, 0.3, 9, 0.5, 9]), lambda b: b[::2])))]
        if st is not None:
            variants += [(f"size_{t}", dict(base, **v)) for t, v in dims_variants("size", st)[1:]]
        if st is not None and m["params"][2].get("optional"):
            variants.append(("omit_size", dict(base, size=OMIT)))
        legacy_sweep(out, HOST, "RandomState.multinomial", m, seeds, base, variants,
                     lambda r, py, st=st: r.multinomial(py["n"], py["pvals"], **size_only_kw(py, st or "Shape?")),
                     widen=True, lp64=True)


@family("NumPyRandom", "dirichlet")
def fam_dirichlet(out, surface, name, seeds):
    """dirichlet(alpha, size): one standard_gamma per component, normalized (host libm)."""
    for m in overloads(surface, "NumPyRandom", name):
        at = m["params"][0]["type"]
        st = m["params"][1]["type"]

        def arr(v):
            return np.array(v, dtype=np.float64) if at.startswith("NDArray") else list(v)
        sb = dims_variants("size", st)[0][1]["size"]
        base = {"alpha": arr([1.0, 2.0, 3.0]), "size": sb}
        variants = [("small", dict(base, alpha=arr([0.1, 0.1]))), ("tiny", dict(base, alpha=arr([1e-3, 1e-3, 1e-3]))),
                    ("one", dict(base, alpha=arr([2.5]))), ("zero", dict(base, alpha=arr([1.0, 0.0]))),
                    ("neg", dict(base, alpha=arr([1.0, -1.0]))), ("nan", dict(base, alpha=arr([1.0, NAN]))),
                    ("empty", dict(base, alpha=arr([]))), ("null", dict(base, alpha=None)),
                    ("large", dict(base, alpha=arr([1e3, 2e3])))]
        if at.startswith("NDArray"):
            variants += [("a2d", dict(base, alpha=np.array([[1.0, 2.0]]))), ("a0d", dict(base, alpha=np.array(1.5))),
                         ("a_i64", dict(base, alpha=np.array([1, 2, 3], dtype=np.int64))),
                         ("a_strided", dict(base, alpha=derived(np.array([1.0, 9, 2.0, 9, 3.0, 9]), lambda b: b[::2])))]
        variants += [(f"size_{t}", dict(base, **v)) for t, v in dims_variants("size", st)[1:]]
        if m["params"][1].get("optional"):
            variants.append(("omit_size", dict(base, size=OMIT)))

        def np_fn(r, py):
            kw = {}
            if "size" in py:
                kw["size"] = size_kw(st, py["size"]) if st.rstrip("?") not in ("int", "long") else py["size"]
            return r.dirichlet(py["alpha"], **kw)
        legacy_sweep(out, HOST, "RandomState.dirichlet", m, seeds, base, variants, np_fn)


# randint's dtype names: NumPy's name on both sides (the C# harness resolves them with np.dtype). 'int' is np.dtype(int),
# int64 on NumPy 2.x everywhere. The explicit C-long char 'l' is NOT here: np.dtype('l') is the HOST's C long in both
# libraries (NumSharp's dtype system registers it per platform, as NumPy does: int32 on Windows, int64 on Linux), so its
# answer is host-dependent by design and has no single expectation a portable tier could hold. The omitted dtype — the
# legacy C long NumSharp models as the LP64 int64 on every host — is covered (widened, with the LP64 tier).
RANDINT_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "int"]


def randint_long(values):
    """Whether a randint/random_integers case returns the C long (dtype omitted, null or 'l')."""
    d = values.get("dtype", OMIT)
    return d is OMIT or d is None or d == "l"


@family("NumPyRandom", "randint")
def fam_randint(out, surface, name, seeds):
    """randint(low, high, size, dtype): the one-argument form, bounds at every dtype's edges, NumPy's rejections, and
    every dtype (the default is the C long)."""
    for m in overloads(surface, "NumPyRandom", name):
        first = m["params"][0]["type"]
        unsigned = first == "ulong"

        def np_fn(r, py):
            kw = {}
            if py.get("high") is not None:
                kw["high"] = py["high"]
            if "size" in py:
                kw["size"] = py["size"]
            if py.get("dtype") is not None:
                kw["dtype"] = py["dtype"]
            return r.randint(py["low"], **kw)
        if first == "NDArray":
            # The array-bounds overload (NumPy's _rand_<dtype>_broadcast with the masked sampler). The C long cases take
            # the 32-bit path on Windows and the 64-bit one on Linux — the LP64 merge keeps NumSharp's (Linux) answer.
            legacy_sweep(out, PORTABLE, "RandomState.randint", m, seeds,
                         {"low": np.array([0, 10, -5], dtype=np.int64), "high": np.array([5, 20, 3], dtype=np.int64)},
                         integer_array_variants(gen=False), np_fn, widen=randint_long, lp64=randint_long)
            continue
        base = {"low": 0 if unsigned else -5, "high": 17, "size": (6,)}
        variants = [("one_arg", {"low": 10, "high": OMIT, "size": (5,)}), ("high_null", {"low": 10, "high": None, "size": (5,)}),
                    ("empty_range", {"low": 5, "high": 5, "size": (3,)}), ("inverted", {"low": 6, "high": 5, "size": (3,)}),
                    ("one_arg_zero", {"low": 0, "high": OMIT, "size": (3,)}),
                    ("size_none", {"low": 0, "high": 100}), ("size_scalar", {"low": 0, "high": 100, "size": ()}),
                    ("size_2x3", {"low": 0, "high": 100, "size": (2, 3)}), ("size_zero", {"low": 0, "high": 100, "size": (0,)}),
                    ("range1", {"low": 7, "high": 8, "size": (4,)}), ("r_2e31", {"low": 0, "high": 2 ** 31, "size": (4,)}),
                    ("r_2e32", {"low": 0, "high": 2 ** 32, "size": (4,)}), ("r_2e40", {"low": 0, "high": 2 ** 40, "size": (4,)}),
                    ("r_max", {"low": 0, "high": 2 ** 63 - 1, "size": (4,)})]
        if not unsigned:
            variants += [("neg_one_arg", {"low": -3, "high": OMIT, "size": (3,)}),
                         ("r_min", {"low": -2 ** 63, "high": 2 ** 63 - 1, "size": (4,)}),
                         ("r_min31", {"low": -2 ** 31, "high": 2 ** 31, "size": (4,)})]
        else:
            variants += [("u_2e63", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,), "dtype": "uint64"}),
                         ("u_max", {"low": 0, "high": 2 ** 64 - 1, "size": (4,), "dtype": "uint64"}),
                         ("u_2e63_long", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,)})]
        for dt in RANDINT_DTYPES:
            lo, hi = (0, 2) if dt == "bool" else (0 if unsigned or dt.startswith("u") else -5, 17)
            variants.append((f"dt_{dt}", {"low": lo, "high": hi, "size": (6,), "dtype": dt}))
            info_hi = 2 if dt == "bool" else (np.iinfo(dt).max + 1 if dt not in ("l", "int") else None)
            if info_hi is not None:
                variants.append((f"dt_{dt}_full", {"low": 0 if unsigned or dt.startswith("u") or dt == "bool" else int(np.iinfo(dt).min),
                                                  "high": int(info_hi), "size": (6,), "dtype": dt}))
                variants.append((f"dt_{dt}_over", {"low": 0, "high": int(info_hi) + 1, "size": (3,), "dtype": dt}))
                if not unsigned and not dt.startswith("u") and dt != "bool":
                    variants.append((f"dt_{dt}_under", {"low": int(np.iinfo(dt).min) - 1, "high": 0, "size": (3,), "dtype": dt}))
                if not unsigned and (dt.startswith("u") or dt == "bool"):
                    variants.append((f"dt_{dt}_neg", {"low": -1, "high": 1, "size": (3,), "dtype": dt}))
            variants.append((f"dt_{dt}_scalar", {"low": 1, "high": 2, "dtype": dt}))
        variants += [("dt_float", {"low": 0, "high": 5, "size": (3,), "dtype": "float64"}),
                     ("dt_null", {"low": 0, "high": 5, "size": (3,), "dtype": None}),
                     ("omit_dtype", {"low": 0, "high": 5, "size": (3,), "dtype": OMIT}),
                     ("omit_size", {"low": 0, "high": 5, "size": OMIT})]
        if unsigned:
            variants = [(t, v) for t, v in variants if isinstance(v.get("low"), int) and v["low"] >= 0
                        and (v.get("high") in (OMIT, None) or v["high"] >= 0)]
        if first == "BigInteger":
            # Every value either integer overload takes, plus the Python ints past 64 bits only this one can spell
            # (the endpoint forms of the shared extras do not exist for randint).
            variants += [("u_2e63", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,), "dtype": "uint64"}),
                         ("u_max", {"low": 0, "high": 2 ** 64 - 1, "size": (4,), "dtype": "uint64"}),
                         ("u_2e63_long", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,)})]
            variants += [(t, v) for t, v in integer_big_extra() if "endpoint" not in v]
        legacy_sweep(out, PORTABLE, "RandomState.randint", m, seeds, base, variants, np_fn, widen=randint_long,
                     lp64=randint_long)


@family("NumPyRandom", "random_integers")
def fam_random_integers(out, surface, name, seeds):
    """random_integers(low, high, size) = randint(low, high + 1) in the C long (deprecated in NumPy, same stream)."""
    for m in overloads(surface, "NumPyRandom", name):
        def np_fn(r, py):
            kw = {}
            if py.get("high") is not None:
                kw["high"] = py["high"]
            if "size" in py:
                kw["size"] = py["size"]
            return r.random_integers(py["low"], **kw)
        variants = [("one_arg", {"low": 5, "high": OMIT, "size": (4,)}), ("high_null", {"low": 5, "high": None, "size": (4,)}),
                    ("neg", {"low": -3, "high": 3, "size": (5,)}), ("equal", {"low": 4, "high": 4, "size": (3,)}),
                    ("inverted", {"low": 5, "high": 4, "size": (3,)}), ("one_arg_zero", {"low": 0, "high": OMIT, "size": (3,)}),
                    ("r_2e31", {"low": 0, "high": 2 ** 31 - 1, "size": (4,)}), ("r_2e40", {"low": 0, "high": 2 ** 40, "size": (4,)}),
                    ("r_max", {"low": 0, "high": 2 ** 63 - 1, "size": (4,)}), ("size_none", {"low": 1, "high": 6}),
                    ("size_scalar", {"low": 1, "high": 6, "size": ()}), ("size_2x3", {"low": 1, "high": 6, "size": (2, 3)}),
                    ("size_zero", {"low": 1, "high": 6, "size": (0,)}), ("omit_size", {"low": 1, "high": 6, "size": OMIT})]
        legacy_sweep(out, PORTABLE, "RandomState.random_integers", m, seeds, {"low": 1, "high": 6, "size": (6,)}, variants,
                     np_fn, widen=True, lp64=True)


@family("NumPyRandom", "get_bit_generator", "_bit_generator")
def fam_get_bitgen(out, surface, name, seeds):
    """The engine a RandomState draws from (NumPy: the `_bit_generator` attribute; the module's get_bit_generator is the
    singleton's)."""
    for m in overloads(surface, "NumPyRandom", name):
        legacy_sweep(out, PORTABLE, f"RandomState.{name}", m, seeds, {}, [], lambda r, py: r._bit_generator, calls2=False)


def np_set_bitgen(r, py):
    """NumPy's set_bit_generator is `_rand._initialize_bit_generator(bitgen)`; on an instance that is exactly what
    `RandomState.__init__(bitgen)` runs. A None goes through the module function itself (the singleton is restored),
    which is where NumPy raises."""
    bg = py["bitgen"]
    if bg is None:
        saved = np.random.get_bit_generator()
        try:
            np.random.set_bit_generator(None)
        finally:
            np.random.set_bit_generator(saved)
        return None
    r.__init__(bg)
    return None


@family("NumPyRandom", "set_bit_generator")
def fam_set_bitgen(out, surface, name, seeds):
    """set_bit_generator(bitgen): the receiver draws from `bitgen` afterwards, its Gaussian cache discarded."""
    for m in overloads(surface, "NumPyRandom", name):
        variants = [(f"to_{e}", {"bitgen": bitgen_obj(e, 99)}) for e in ENGINES]
        variants += [("to_primed", {"bitgen": bitgen_obj("PCG64", 5, "raw3")}), ("null", {"bitgen": None})]
        legacy_sweep(out, PORTABLE, "RandomState.set_bit_generator", m, seeds, {"bitgen": bitgen_obj("Philox", 7)},
                     variants, np_set_bitgen, calls2=False)


@family("NumPyRandom", "_poisson_lam_max", "ToString")
def fam_rs_attrs(out, surface, name, seeds):
    """_poisson_lam_max (a class constant) and str(RandomState)."""
    for m in overloads(surface, "NumPyRandom", name):
        fn = (lambda r, py: r._poisson_lam_max) if name == "_poisson_lam_max" else (lambda r, py: str(r))
        legacy_sweep(out, PORTABLE, f"RandomState.{name}", m, seeds, {}, [], fn, primes=(), calls2=False)


@family("NumPyRandom", "get_state")
def fam_get_state(out, surface, name, seeds):
    """get_state(): the legacy tuple (MT19937 only — NumPy warns and returns the dict on another engine);
    get_state(legacy): the tuple or the dict. Priming makes the Gaussian cache and a buffered half visible."""
    for m in overloads(surface, "NumPyRandom", name):
        if not m["params"]:
            legacy_sweep(out, PORTABLE, "RandomState.get_state", m, seeds, {}, [], lambda r, py: r.get_state(),
                         calls2=False)
            continue
        legacy_sweep(out, PORTABLE, "RandomState.get_state", m, seeds, {"legacy": False}, [("legacy", {"legacy": True})],
                     lambda r, py: r.get_state(legacy=py["legacy"]), calls2=False)
        # The tuple form from every engine (a warning and the dict where the engine is not MT19937).
        for recv in legacy_receivers(seeds[:2]):
            emit_m(out, PORTABLE, "RandomState.get_state", m, recv, {"legacy": True}, "legacy_all",
                   lambda r, py: r.get_state(legacy=True))


KEY624 = [(i * 2654435761 + 12345) % 2 ** 32 for i in range(624)]


@family("NumPyRandom", "set_state")
def fam_set_state(out, surface, name, seeds):
    """set_state from the legacy tuple, the dict, a bare bit generator state (object overload), and NumPy's rejections:
    another algorithm, another engine's state, a short key, a position past 624, non-state objects."""
    for m in overloads(surface, "NumPyRandom", name):
        pt = m["params"][0]["type"]
        src_mt = Recv("RandomState", None, 777, prime="gauss")
        if pt == "NativeRandomState":
            base = {"state": legacy_state_of(src_mt)}
            variants = [("fresh", {"state": legacy_state_of(Recv("RandomState", None, 3))}),
                        ("explicit", {"state": legacy_state_explicit(KEY624, 5, 1, 0.25)}),
                        ("pos0", {"state": legacy_state_explicit(KEY624, 0)}),
                        ("pos624", {"state": legacy_state_explicit(KEY624, 624)}),
                        ("pos625", {"state": legacy_state_explicit(KEY624, 625)}),
                        ("posneg", {"state": legacy_state_explicit(KEY624, -1)}),
                        ("zeros", {"state": legacy_state_explicit([0] * 624, 624)}),
                        ("short_key", {"state": legacy_state_explicit(KEY624[:623], 5)}),
                        ("long_key", {"state": legacy_state_explicit(KEY624 + [1], 5)}),
                        ("algo", {"state": legacy_state_explicit(KEY624, 5, algorithm="PCG64")}),
                        ("gauss_flag2", {"state": legacy_state_explicit(KEY624, 5, 2, -1.5)})]
        elif pt == "NumPyRandom.State":
            base = {"state": rs_dict_of(src_mt)}
            variants = [(f"from_{e}", {"state": rs_dict_of(Recv("RandomState", e, 11, prime="u32"))}) for e in ENGINES]
            variants += [("parts", {"state": rs_dict_parts(bgstate_obj("MT19937", 4), 1, 0.75)}),
                         ("parts_nostate", {"state": rs_dict_parts(None, 0, 0.0)}),
                         ("null", {"state": None})]
        else:
            base = {"state": legacy_state_of(src_mt)}
            variants = [("dict", {"state": rs_dict_of(Recv("RandomState", None, 5, prime="u32"))}),
                        ("bare_mt", {"state": bgstate_obj("MT19937", 6, "raw3")}),
                        ("bare_pcg", {"state": bgstate_obj("PCG64", 6, "u32" if False else "raw3")}),
                        ("int", {"state": pyint_obj(5)}), ("str", {"state": pystr_obj("abc")}),
                        ("null", {"state": None})]
        # A state from each engine onto each engine's RandomState: matching engines restore, others are rejected.
        cross = [(f"onto_{e}", Recv("RandomState", e, seeds[0])) for e in ENGINES]
        legacy_sweep(out, PORTABLE, "RandomState.set_state", m, seeds, base, variants, lambda r, py: r.set_state(py["state"]),
                     calls2=False)
        for tag, recv in cross:
            emit_m(out, PORTABLE, "RandomState.set_state", m, recv, base, tag, lambda r, py: r.set_state(py["state"]))
        if pt == "NumPyRandom.State":
            # The receiver's own engine: restored on every engine x seed (a dict from another seed and a primed state).
            for recv in legacy_receivers(seeds):
                src = Recv("RandomState", recv.engine, 23, prime="raw3")
                emit_m(out, PORTABLE, "RandomState.set_state", m, recv, {"state": rs_dict_of(src)},
                       f"same_engine_{recv.engine or 'legacy'}", lambda r, py: r.set_state(py["state"]))


@family("NumPyRandom", "seed")
def fam_seed(out, surface, name, seeds):
    """seed(...) re-seeds the MT19937 (legacy seeding: int or init_by_array) and clears the Gaussian cache; seed() takes
    OS entropy (key masked); any other engine refuses (the instance method's rule)."""
    variants_by_type = {
        "int": [("s1", 1), ("s42", 42), ("smax", 2 ** 31 - 1), ("sneg", -1)],
        "uint": [("s1", 1), ("smax", 2 ** 32 - 1)],
        "long": [("s1", 1), ("smax", 2 ** 32 - 1), ("s2e32", 2 ** 32), ("sneg", -1), ("s2e40", 2 ** 40)],
        "ulong": [("s1", 1), ("smax", 2 ** 32 - 1), ("s2e32", 2 ** 32), ("s2e64", 2 ** 64 - 1)],
        "int[]": [("a123", [1, 2, 3]), ("a0", [0]), ("aempty", []), ("aneg", [-1]), ("amax", [2 ** 31 - 1]),
                  ("along", list(range(700)))],
        "long[]": [("a123", [1, 2, 3]), ("amax", [2 ** 32 - 1]), ("a2e32", [2 ** 32]), ("aneg", [5, -5]), ("aempty", []),
                   ("a2e40", [2 ** 40])],
        "uint[]": [("a123", [1, 2, 3]), ("amax", [2 ** 32 - 1, 0]), ("aempty", []), ("along", list(range(700)))],
    }
    for m in overloads(surface, "NumPyRandom", name):
        if not m["params"]:
            legacy_sweep(out, PORTABLE, "RandomState.seed", m, seeds, {}, [], lambda r, py: r.seed(), calls2=False,
                         state_mask=("key",))
            continue
        pt = m["params"][0]["type"]
        base_v = [1234, 5, 6] if pt.endswith("[]") else 1234
        legacy_sweep(out, PORTABLE, "RandomState.seed", m, seeds, {"seed": base_v},
                     [(t, {"seed": v}) for t, v in variants_by_type[pt]], lambda r, py: r.seed(py["seed"]), calls2=False)
        if pt.endswith("[]"):
            # A null array is NumPy's seed(None): fresh OS entropy, so the key is masked.
            for recv in variant_receivers("legacy", seeds)[:2]:
                emit_m(out, PORTABLE, "RandomState.seed", m, recv, {"seed": None}, "anull", lambda r, py: r.seed(None),
                       state_mask=("key",))


@family("NumPyRandom", "RandomState")
def fam_rs_factory(out, surface, name, seeds):
    """The RandomState factories (np.random.RandomState(...)): legacy int and array seeding, a bit generator, the
    legacy state tuple (NumPy: a RandomState restored with set_state), and OS entropy (key masked)."""
    recv = Recv("RandomState", None, 0)
    for m in overloads(surface, "NumPyRandom", name):
        pt = m["params"][0]["type"] if m["params"] else None

        def one(tag, values, fn, mask=()):
            # The receiver's state is recorded: the factory must not draw from the RandomState it is called on.
            emit_m(out, PORTABLE, "RandomState", m, recv, values, tag, fn, state_mask=mask)
        if pt is None:
            one("entropy", {}, lambda r, py: np.random.RandomState(), mask=("key",))
        elif pt in ("int", "long"):
            vals = [0, 1, 7, 42, 2 ** 31 - 1, -1] + ([2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 40] if pt == "long" else [])
            vals += [s for s in seeds if s < 2 ** 31 or pt == "long"]
            for v in dict.fromkeys(vals):
                one(f"s{v}", {"seed": v}, lambda r, py: np.random.RandomState(py["seed"]))
        elif pt.endswith("[]"):
            arrs = [[1, 2, 3], [0], [2 ** 31 - 1], [], [-1], list(range(700))]
            if pt != "int[]":
                arrs += [[2 ** 32 - 1]]
            if pt == "long[]":
                arrs += [[2 ** 32], [2 ** 40]]
            if pt == "uint[]":
                arrs = [a for a in arrs if all(x >= 0 for x in a)]
            for i, a in enumerate(arrs):
                one(f"arr{i}", {"seed": a}, lambda r, py: np.random.RandomState(py["seed"]))
            one("null", {"seed": None}, lambda r, py: np.random.RandomState(None), mask=("key",))
        elif pt == "BitGenerator":
            for e in ENGINES:
                for s in seeds[:3]:
                    one(f"{e}_s{s}", {"seed": bitgen_obj(e, s)}, lambda r, py: np.random.RandomState(py["seed"]))
            one("primed", {"seed": bitgen_obj("SFC64", 3, "raw3")},
                lambda r, py: np.random.RandomState(py["seed"]))
            one("null", {"seed": None}, lambda r, py: np.random.RandomState(None), mask=("key",))
        elif pt == "NativeRandomState":
            def restore(r, py):
                rs = np.random.RandomState()
                rs.set_state(py["state"])
                return rs
            one("from_state", {"state": legacy_state_of(Recv("RandomState", None, 31, prime="gauss"))}, restore)
            one("explicit", {"state": legacy_state_explicit(KEY624, 17, 1, 0.5)}, restore)
            one("algo", {"state": legacy_state_explicit(KEY624, 5, algorithm="Philox")}, restore)
            one("short_key", {"state": legacy_state_explicit(KEY624[:10], 5)}, restore)
            one("pos625", {"state": legacy_state_explicit(KEY624, 625)}, restore)
        else:
            raise ValueError(f"unhandled RandomState factory {m['sig']}")


@family("NumPyRandom", "Seed")
def fam_seed_prop(out, surface, name, seeds):
    for m in overloads(surface, "NumPyRandom", name):
        exempt(m["sig"], "NumSharp bookkeeping of the last legacy integer seed; no NumPy attribute (plan §7)")



# ---- multivariate normal (both APIs): svd/eigh/cholesky + dot through NumPy's own OpenBLAS (the mvn tier) ------------

MVN = "random_api_mvn"
MVN_MEAN = [0.5, -1.0]
MVN_COV = [[2.0, 0.3], [0.3, 1.0]]
MVN_BAD = [[1.0, 2.0], [2.0, 1.0]]          # symmetric, not positive semi-definite
MVN_SINGULAR = [[1.0, 1.0], [1.0, 1.0]]


def mvn_values(mt, ct, mean, cov):
    """Mean/cov in the overload's spelling (NDArray or double[]/double[,])."""
    if mt.startswith("NDArray"):
        return np.array(mean, dtype=np.float64), np.array(cov, dtype=np.float64)
    return list(mean), [list(r) for r in cov]


def mvn_variants(m, api):
    pn = param_names(m)
    ptypes = {p["name"]: p["type"] for p in m["params"]}
    mean, cov = mvn_values(ptypes["mean"], ptypes["cov"], MVN_MEAN, MVN_COV)
    st = ptypes.get("size")
    size_base = dims_variants("size", st)[0][1]["size"] if st else OMIT
    base = {"mean": mean, "cov": cov, "size": size_base}
    v = []
    if st:
        v += [(f"size_{t}", dict(base, **vals)) for t, vals in dims_variants("size", st)[1:]]
        if [p for p in m["params"] if p["name"] == "size" and p.get("optional")]:
            v.append(("omit_size", dict(base, size=OMIT)))
    def mc(meanv, covv):
        return mvn_values(ptypes["mean"], ptypes["cov"], meanv, covv)
    one_mean, one_cov = mc([3.0], [[4.0]])
    three_mean, three_cov = mc([0.0, 1.0, 2.0], [[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 3.0]])
    _, bad = mc(MVN_MEAN, MVN_BAD)
    _, sing = mc(MVN_MEAN, MVN_SINGULAR)
    v += [("dim1", dict(base, mean=one_mean, cov=one_cov)), ("dim3", dict(base, mean=three_mean, cov=three_cov)),
          ("singular", dict(base, cov=sing)), ("zero_cov", dict(base, cov=mc(MVN_MEAN, [[0.0, 0.0], [0.0, 0.0]])[1])),
          ("len_mismatch", dict(base, mean=mc([1.0, 2.0, 3.0], MVN_COV)[0]))]
    if ptypes["mean"].startswith("NDArray"):
        v += [("mean_2d", dict(base, mean=np.array([[0.5, -1.0]]))), ("cov_1d", dict(base, cov=np.array([2.0, 1.0]))),
              ("cov_nonsquare", dict(base, cov=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))),
              ("mean_i64", dict(base, mean=np.array([1, 2], dtype=np.int64))),
              ("cov_strided", dict(base, cov=derived(np.array([2.0, 9.0, 0.3, 9.0, 0.3, 9.0, 1.0, 9.0]),
                                                     lambda b: b[::2].reshape(2, 2))))]
    # Null (NumPy's None) for mean and cov in every overload: NumPy's np.array(None) is a 0-d object array.
    v += [("null_mean", dict(base, mean=None)), ("null_cov", dict(base, cov=None))]
    if "check_valid" in pn:
        v += [("cv_raise_bad", dict(base, cov=bad, check_valid="raise")), ("cv_ignore_bad", dict(base, cov=bad, check_valid="ignore")),
              ("cv_warn_bad", dict(base, cov=bad, check_valid="warn")), ("cv_raise_ok", dict(base, check_valid="raise")),
              ("cv_bogus", dict(base, check_valid="bogus")), ("omit_cv", dict(base, cov=bad, check_valid=OMIT)),
              ("cv_null", dict(base, check_valid=None))]
    if "tol" in pn:
        v += [("tol_tiny", dict(base, cov=sing, check_valid="raise", tol=1e-300)), ("tol_big", dict(base, cov=bad, check_valid="raise", tol=10.0)),
              ("omit_tol", dict(base, tol=OMIT))]
    if "method" in pn:
        v += [("m_eigh", dict(base, method="eigh")), ("m_cholesky", dict(base, method="cholesky")),
              ("m_cholesky_singular", dict(base, cov=sing, method="cholesky")), ("m_eigh_bad", dict(base, cov=bad, method="eigh")),
              ("m_bogus", dict(base, method="bogus")), ("omit_method", dict(base, method=OMIT)),
              ("m_svd", dict(base, method="svd")), ("m_null", dict(base, method=None))]
    return base, v


def mvn_np(r, py, st):
    kw = {}
    if "size" in py:
        kw["size"] = size_kw(st, py["size"]) if st.rstrip("?") not in ("int", "long") else py["size"]
    for k in ("check_valid", "tol", "method"):
        if k in py:
            kw[k] = py[k]
    return r.multivariate_normal(py["mean"], py["cov"], **kw)


@family("NumPyRandom", "multivariate_normal")
def fam_legacy_mvn(out, surface, name, seeds):
    """RandomState.multivariate_normal: svd(cov) then standard_normal @ (sqrt(s)[:, None] * v) + mean; check_valid/tol
    over non-PSD covariances; NumPy's shape errors."""
    for m in overloads(surface, "NumPyRandom", name):
        st = next(p["type"] for p in m["params"] if p["name"] == "size")
        base, variants = mvn_variants(m, "legacy")
        legacy_sweep(out, MVN, "RandomState.multivariate_normal", m, seeds, base, variants,
                     lambda r, py, st=st: mvn_np(r, py, st))


@family("Generator", "multivariate_normal")
def fam_gen_mvn(out, surface, name, seeds):
    """Generator.multivariate_normal: method svd/eigh/cholesky, check_valid/tol, NumPy's errors."""
    for m in overloads(surface, "Generator", name):
        st = next(p["type"] for p in m["params"] if p["name"] == "size")
        base, variants = mvn_variants(m, "gen")
        gen_sweep(out, MVN, "Generator.multivariate_normal", m, seeds, base, variants,
                  lambda r, py, st=st: mvn_np(r, py, st))


# ---- Generator: fillers with size / dtype / method / out ---------------------------------------------------------------

def out_variants(dtype="float64", n=3):
    """The `out=` layouts NumPy distinguishes: contiguous (the result IS out), F-ordered 2-D, a strided view (rejected:
    not contiguous), the wrong dtype (rejected), read-only (a broadcast view; rejected), and a size that disagrees."""
    return [("out", lambda: {"out": np.zeros(n, dtype=dtype), "size": OMIT}),
            ("out_2d_f", lambda: {"out": derived(np.zeros(6, dtype=dtype), lambda b: b.reshape(3, 2).T), "size": OMIT}),
            ("out_strided", lambda: {"out": derived(np.zeros(2 * n, dtype=dtype), lambda b: b[::2]), "size": OMIT}),
            ("out_wrong_dtype", lambda: {"out": np.zeros(n, dtype="float32" if dtype == "float64" else "float64"), "size": OMIT}),
            ("out_readonly", lambda: {"out": derived(np.zeros(1, dtype=dtype), lambda b: np.broadcast_to(b, (n,))), "size": OMIT}),
            ("out_size_match", lambda: {"out": np.zeros(n, dtype=dtype), "size": (n,)}),
            ("out_size_mismatch", lambda: {"out": np.zeros(n, dtype=dtype), "size": (n + 1,)}),
            ("out_empty", lambda: {"out": np.zeros(0, dtype=dtype), "size": OMIT})]


def gen_kw(py, keys):
    """NumPy keyword arguments for the passed C# arguments among `keys` (a null DType is NumPy's default: omitted)."""
    kw = {}
    for k in keys:
        if k in py and not (k == "dtype" and py[k] is None):
            kw[k] = py[k]
    return kw


@family("Generator", "random", "standard_normal", "standard_exponential", "standard_cauchy")
def fam_gen_fill(out, surface, name, seeds):
    """The Generator fillers: size forms, float32/float64 (and rejected dtypes), method (standard_exponential), and
    out= in every layout NumPy distinguishes (the result is `out`; its contents are observed after the call)."""
    tier = PORTABLE if name == "random" else HOST
    member = f"Generator.{name}"
    for m in overloads(surface, "Generator", name):
        pn = param_names(m)
        np_fn = (lambda r, py: getattr(r, name)(**gen_kw(py, ("size", "dtype", "method", "out"))))
        vs = [(t, v) for t, v in dims_variants("size", "Shape")[1:]] + [("omit_size", {"size": OMIT})]
        variants = [(t, dict(v)) for t, v in vs]
        if "dtype" in pn:
            variants += [("f32", {"size": (4,), "dtype": "float32"}), ("f64", {"size": (4,), "dtype": "float64"}),
                         ("f32_scalar", {"size": SHAPE_NONE, "dtype": "float32"}), ("dt_null", {"size": (4,), "dtype": None}),
                         ("dt_i32", {"size": (4,), "dtype": "int32"}), ("dt_f16", {"size": (4,), "dtype": "float16"}),
                         ("f32_big", {"size": (257,), "dtype": "float32"})]
        if "method" in pn:
            variants += [("inv", {"size": (5,), "method": "inv"}), ("zig", {"size": (5,), "method": "zig"}),
                         ("inv_f32", {"size": (5,), "method": "inv", "dtype": "float32"}), ("bogus", {"size": (5,), "method": "bogus"}),
                         ("omit_method", {"size": (5,), "method": OMIT}), ("method_null", {"size": (5,), "method": None})]
        if "out" in pn:
            variants += out_variants("float64")
            variants += [("out_f32", lambda: {"out": np.zeros(4, dtype="float32"), "dtype": "float32", "size": OMIT}),
                         ("out_null", {"size": (3,), "out": None})]
            if "method" in pn:
                variants += [("out_inv", lambda: {"out": np.zeros(4), "method": "inv", "size": OMIT})]
        variants += [("big", {"size": (1000,)})]
        gen_sweep(out, tier, member, m, seeds, {"size": (6,)}, variants, np_fn, watch=("out",))


@family("Generator", "standard_gamma")
def fam_gen_standard_gamma(out, surface, name, seeds):
    """Generator.standard_gamma(shape, size, dtype, out): the scalar and array overloads, float32 (its own sampler),
    shape at the branches (< 1, == 1, > 1) and the rejections, out= layouts."""
    for m in overloads(surface, "Generator", name):
        arr = m["params"][0]["type"].startswith("NDArray")

        def sh(v):
            return np.array(v, dtype=np.float64) if arr else v
        np_fn = (lambda r, py: r.standard_gamma(py["shape"], **gen_kw(py, ("size", "dtype", "out"))))
        base = {"shape": sh(2.5), "size": (5,)}
        variants = [(t, dict(base, **v)) for t, v in dims_variants("size", "Shape")[1:]] + [("omit_size", dict(base, size=OMIT))]
        variants += [("s_small", dict(base, shape=sh(0.3))), ("s_one", dict(base, shape=sh(1.0))), ("s_big", dict(base, shape=sh(1e4))),
                     ("s_zero", dict(base, shape=sh(0.0))), ("s_neg", dict(base, shape=sh(-1.0))), ("s_nan", dict(base, shape=sh(NAN))),
                     ("f32", dict(base, dtype="float32")), ("f32_small", dict(base, shape=sh(0.3), dtype="float32")),
                     ("f32_one", dict(base, shape=sh(1.0), dtype="float32")), ("dt_i32", dict(base, dtype="int32")),
                     ("dt_null", dict(base, dtype=None)), ("f64", dict(base, dtype="float64")),
                     ("dt_f16", dict(base, dtype="float16"))]
        variants += [(t, (lambda f: (lambda: dict(f(), shape=sh(2.5))))(f)) for t, f in out_variants("float64")]
        variants += [("out_f32", lambda: {"shape": sh(2.5), "out": np.zeros(4, dtype="float32"), "dtype": "float32", "size": OMIT})]
        if arr:
            variants += [("a_1d", dict(base, shape=np.array([0.5, 1.0, 2.5, 7.0]), size=OMIT)),
                         ("a_bcast", dict(base, shape=np.array([[0.5], [3.0]]), size=(2, 3))),
                         ("a_strided", dict(base, shape=derived(np.array([0.5, 9, 1.5, 9, 4.0, 9]), lambda b: b[::2]), size=OMIT)),
                         ("a_i64", dict(base, shape=np.array([1, 2, 3], dtype=np.int64), size=OMIT)),
                         ("a_neg", dict(base, shape=np.array([1.0, -1.0]), size=OMIT)),
                         ("a_null", dict(base, shape=None)),
                         ("a_out", lambda: {"shape": np.array([0.5, 2.0, 5.0]), "out": np.zeros(3), "size": OMIT})]
        gen_sweep(out, HOST, "Generator.standard_gamma", m, seeds, base, variants, np_fn, watch=("out",))


def integer_array_variants(gen):
    """The array-bounds overload's variants (`integers(NDArray low, NDArray high, ...)` / `randint(...)`): NumPy's
    `_rand_<dtype>` with array-like bounds — the scalar path when both are 0-d (Python's int() of each: floats truncate,
    NaN/inf/complex/None raise), else `_rand_<dtype>_broadcast`: per-position bounds and draws, the bounds' input dtypes
    (the safe-cast skip, the forced casts), every output dtype at its full range and past it (8/16-bit and bool positions
    sharing 32-bit words), NumPy's rejections in its check order, and its two quirks (a size smaller than the bounds'
    broadcast; 64-bit float bounds in a non-C layout scrambled by the element-wise conversion). `gen` adds endpoint forms."""
    I = lambda *v: np.array(v, dtype=np.int64)   # noqa: E731
    F = lambda *v: np.array(v, dtype=np.float64)   # noqa: E731
    z = np.array
    V = [("arr", {"low": I(0, 10, -5), "high": I(5, 20, 3)}),
         ("arr_size", {"low": I(0, 10, -5), "high": I(5, 20, 3), "size": (3,)}),
         ("arr_size_bigger", {"low": I(0, 10, -5), "high": I(5, 20, 3), "size": (2, 3)}),
         ("arr_size_smaller", {"low": np.zeros((2, 3), dtype=np.int64), "high": z(10), "size": (3,)}),
         ("arr_size_smaller_2d", {"low": np.arange(12, dtype=np.int64).reshape(4, 3), "high": z(100), "size": (2, 3)}),
         ("arr_size_mismatch", {"low": I(0, 0, 0), "high": z(10), "size": (2,)}),
         ("arr_col_row", {"low": np.array([[0], [100]], dtype=np.int64), "high": I(5, 10, 200)}),
         ("arr_high_0d", {"low": I(0, 5, 9), "high": z(20)}),
         ("arr_low_0d", {"low": z(3), "high": I(5, 10, 20)}),
         ("one_arg", {"low": I(5, 10, 1000), "high": OMIT}),
         ("one_arg_null", {"low": I(5, 10), "high": None}),
         ("one_arg_zero", {"low": I(0, 10), "high": OMIT}),
         ("zd_both", {"low": z(3), "high": z(9), "size": (4,)}),
         ("zd_both_none", {"low": z(3), "high": z(9)}),
         ("zd_one_arg", {"low": z(7), "high": OMIT, "size": (3,)}),
         ("zd_float", {"low": z(0.5), "high": z(5.9), "size": (3,)}),
         ("zd_neg_float", {"low": z(-2.5), "high": z(3.5), "size": (3,)}),
         ("zd_nan", {"low": z(NAN), "high": z(5)}),
         ("zd_inf", {"low": z(0), "high": z(INF)}),
         ("zd_bool", {"low": z(True), "high": z(5), "size": (3,)}),
         ("zd_complex", {"low": z(1 + 0j), "high": z(5)}),
         ("zd_huge_float", {"low": z(0), "high": z(1e300)}),
         ("float_arr", {"low": F(0.5, 1.7, -2.5), "high": F(5.9, 10.2, 3.99)}),
         ("float_nan", {"low": F(NAN, 1.0), "high": z(5)}),
         ("float_inf", {"low": F(0.0, 1.0), "high": F(INF, 5.0)}),
         ("float_neg_inf_low", {"low": F(-INF, 1.0), "high": z(5)}),
         ("low_ge_high", {"low": I(5, 0), "high": I(5, 10)}),
         ("low_zero_bad", {"low": I(0, 0), "high": I(0, 10)}),
         ("bounds_mismatch", {"low": I(0, 0, 0), "high": I(5, 6)}),
         ("empty", {"low": np.zeros(0, dtype=np.int64), "high": z(5)}),
         ("empty_2d", {"low": np.zeros((2, 0), dtype=np.int64), "high": z(5)}),
         ("size_zero_bad", {"low": I(5), "high": I(1), "size": (0,)}),
         ("low_null", {"low": None, "high": I(5, 6)}),
         ("low_null_i32", {"low": None, "high": I(5, 6), "dtype": "int32"}),
         ("high_null_low_null", {"low": None, "high": OMIT}),
         ("u64_in", {"low": np.array([2 ** 63], dtype=np.uint64), "high": np.array([2 ** 63 + 5], dtype=np.uint64),
                     "dtype": "uint64"}),
         ("u64_to_i64", {"low": np.array([2 ** 63], dtype=np.uint64), "high": np.array([2 ** 63 + 5], dtype=np.uint64)}),
         ("u64_small_i32", {"low": np.array([0, 5], dtype=np.uint64), "high": np.array([10, 9], dtype=np.uint64),
                            "dtype": "int32"}),
         ("i8_in_i16", {"low": np.array([-5, 5], dtype=np.int8), "high": z(50), "dtype": "int16"}),
         ("i32_in", {"low": np.array([-5, 5], dtype=np.int32), "high": np.array([5, 50], dtype=np.int32)}),
         ("u32_in", {"low": np.array([0, 5], dtype=np.uint32), "high": np.array([7, 50], dtype=np.uint32), "dtype": "uint32"}),
         ("f32_in", {"low": np.array([0.5, 1.5], dtype=np.float32), "high": z(7)}),
         ("complex_in_i32", {"low": np.array([1 + 0j, 2]), "high": z(5), "dtype": "int32"}),
         ("complex_in_i64", {"low": np.array([1 + 0j, 2]), "high": z(5)}),
         ("i64_neg_to_u64", {"low": I(-1, 0), "high": z(5), "dtype": "uint64"}),
         ("f_order_int", {"low": derived(np.arange(6, dtype=np.int64), lambda b: b.reshape(3, 2).T), "high": z(100)}),
         ("f_order_float", {"low": derived(np.arange(6, dtype=np.float64), lambda b: b.reshape(3, 2).T), "high": z(100)}),
         ("f_order_float_high", {"low": z(0), "high": derived(np.arange(10, 16, dtype=np.float64),
                                                              lambda b: b.reshape(3, 2).T)}),
         ("bcast_view_float", {"low": derived(np.arange(3, dtype=np.float64), lambda b: np.broadcast_to(b, (2, 3))),
                               "high": z(100)}),
         ("bcast_view_int", {"low": derived(np.arange(3, dtype=np.int64), lambda b: np.broadcast_to(b, (2, 3))),
                             "high": z(100)}),
         ("permuted3d_float", {"low": derived(np.arange(24, dtype=np.float64), lambda b: b.reshape(2, 3, 4).transpose(1, 0, 2)),
                               "high": z(100)}),
         ("negstride_float", {"low": derived(np.arange(6, dtype=np.float64), lambda b: b[::-1]), "high": z(100)}),
         ("strided_int", {"low": derived(np.arange(12, dtype=np.int64), lambda b: b[::3]), "high": z(100)}),
         ("f_order_i32", {"low": z(0), "high": derived(np.arange(1, 7, dtype=np.int64), lambda b: b.reshape(3, 2).T),
                          "dtype": "int32"}),
         ("big", {"low": np.arange(0, 1000, dtype=np.int64), "high": z(2000)}),
         ("omit_size", {"low": I(0, 10), "high": I(5, 20), "size": OMIT}),
         ("omit_dtype", {"low": I(0, 10), "high": I(5, 20), "dtype": OMIT}),
         ("dt_null", {"low": I(0, 10), "high": I(5, 20), "dtype": None}),
         ("dt_float", {"low": I(0, 10), "high": I(5, 20), "dtype": "float64"})]
    if gen:
        V += [("endpoint", {"low": I(5, 0, -3), "high": I(5, 10, -3), "endpoint": True}),
              ("endpoint_one_arg", {"low": I(0, 7), "high": OMIT, "endpoint": True}),
              ("endpoint_low_gt", {"low": I(6, 0), "high": I(5, 10), "endpoint": True}),
              ("endpoint_zero_bad", {"low": I(0, 0), "high": I(-1, 10), "endpoint": True}),
              ("omit_endpoint", {"low": I(0, 10), "high": I(5, 20), "endpoint": OMIT}),
              ("endpoint_bool", {"low": np.array([False, True]), "high": np.array([True, True]), "endpoint": True})]
    else:
        V += [("bool_in", {"low": np.array([False, False]), "high": np.array([True, True])})]
    def A(*vals):
        """Bounds as an array whose dtype holds them exactly: int64 when they fit, else uint64, else float64 (past
        2**64; the callers pick values a float represents exactly, so Python's int() of each gives the intended bound)."""
        if all(-2 ** 63 <= v < 2 ** 63 for v in vals):
            return np.array(vals, dtype=np.int64)
        if all(0 <= v < 2 ** 64 for v in vals):
            return np.array(vals, dtype=np.uint64)
        return np.array([float(v) for v in vals], dtype=np.float64)

    for dt in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
        vmin = 0 if dt == "bool" else int(np.iinfo(dt).min)
        vmax = 1 if dt == "bool" else int(np.iinfo(dt).max)
        signed = dt.startswith("int")
        # Past the dtype by one, except where float64 (the only array that holds it) cannot tell it from the bound:
        # 2**64 + 4096 and -2**63 - 2048 are exact floats, 2**64 + 1 and -2**63 - 1 are not.
        over = 2 ** 64 + 4096 if dt == "uint64" else vmax + 2
        under = -2 ** 63 - 2048 if dt == "int64" else vmin - 1
        V += [(f"dt_{dt}", {"low": I(0, 0) if not signed else I(-5, 3), "high": I(2, 1) if dt == "bool" else I(17, 9),
                            "dtype": dt}),
              (f"dt_{dt}_many", {"low": np.zeros(70, dtype=np.int64),
                                 "high": np.full(70, 2 if dt == "bool" else min(vmax + 1, 200), dtype=np.int64),
                                 "dtype": dt}),
              (f"dt_{dt}_mixed_ranges", {"low": I(0, 0, 0, 0), "high": I(1, 2, 3, 2) if dt == "bool" else I(1, 7, 100, 3),
                                         "dtype": dt}),
              (f"dt_{dt}_single", {"low": A(vmax, 0), "high": A(vmax, 0) if gen else A(vmax, 1), "dtype": dt,
                                   **({"endpoint": True} if gen else {})}),
              (f"dt_{dt}_scalar_path", {"low": z(1), "high": z(2), "dtype": dt}),
              (f"dt_{dt}_full", {"low": A(vmin, 0), "high": A(vmax + 1, 5), "dtype": dt}),
              (f"dt_{dt}_over", {"low": A(0, 0), "high": A(over, 5), "dtype": dt})]
        if gen:
            V += [(f"dt_{dt}_full_ep", {"low": A(vmin, 0), "high": A(vmax, 5), "dtype": dt, "endpoint": True}),
                  (f"dt_{dt}_over_ep", {"low": A(0, 0), "high": A(2 ** 64 if dt == "uint64" else vmax + 1, 5), "dtype": dt,
                                        "endpoint": True})]
        if signed:
            V.append((f"dt_{dt}_under", {"low": A(under, 0), "high": A(0, 5), "dtype": dt}))
        else:
            V.append((f"dt_{dt}_neg", {"low": A(-1, 0), "high": A(1, 5), "dtype": dt}))
    return V


def integer_big_extra(unsigned_ok=True):
    """Extra BigInteger-overload variants: Python ints past the long/ulong range (NumPy's full-range idiom with an
    EXCLUSIVE 2**64, and values past every dtype, clamped by NumSharp at 2**100 to the same verdicts)."""
    return [("big_u64_excl", {"low": 0, "high": 2 ** 64, "size": (4,), "dtype": "uint64"}),
            ("big_u64_one_arg", {"low": 2 ** 64, "high": OMIT, "size": (4,), "dtype": "uint64"}),
            ("big_over_i64", {"low": 0, "high": 2 ** 100, "size": (3,)}),
            ("big_under_i64", {"low": -2 ** 200, "high": 2 ** 200, "size": (3,)}),
            ("big_both_huge", {"low": 2 ** 200, "high": 2 ** 201, "size": (3,)}),
            ("big_high_null", {"low": 5, "high": None, "size": (3,)}),
            ("big_i64_edges", {"low": -2 ** 63, "high": 2 ** 63, "size": (4,)}),
            ("big_i64_over_by_one", {"low": -2 ** 63 - 1, "high": 0, "size": (3,)})]


@family("Generator", "integers")
def fam_gen_integers(out, surface, name, seeds):
    """Generator.integers(low, high, size, dtype, endpoint): the one-argument form, endpoint on and off, every dtype
    at its full range and just past it, NumPy's rejections (default dtype int64 — no C long involved). The array-bounds
    overload takes integer_array_variants; the arbitrary-precision one the scalar variants plus Python ints past 64 bits."""
    for m in overloads(surface, "Generator", name):
        first = m["params"][0]["type"]
        unsigned = first == "ulong"

        def np_fn(r, py):
            kw = {}
            if py.get("high") is not None:
                kw["high"] = py["high"]
            kw.update(gen_kw(py, ("size", "dtype", "endpoint")))
            return r.integers(py["low"], **kw)
        if first == "NDArray":
            gen_sweep(out, PORTABLE, "Generator.integers", m, seeds,
                      {"low": np.array([0, 10, -5], dtype=np.int64), "high": np.array([5, 20, 3], dtype=np.int64)},
                      integer_array_variants(gen=True), np_fn)
            continue
        base = {"low": 0 if unsigned else -5, "high": 17, "size": (6,)}
        variants = [("one_arg", {"low": 10, "high": OMIT, "size": (5,)}), ("high_null", {"low": 10, "high": None, "size": (5,)}),
                    ("endpoint", {"low": 5, "high": 9, "size": (6,), "endpoint": True}),
                    ("endpoint_one_arg", {"low": 5, "high": OMIT, "size": (6,), "endpoint": True}),
                    ("equal", {"low": 5, "high": 5, "size": (3,)}), ("equal_endpoint", {"low": 5, "high": 5, "size": (3,), "endpoint": True}),
                    ("inverted", {"low": 6, "high": 5, "size": (3,)}), ("inverted_endpoint", {"low": 6, "high": 5, "size": (3,), "endpoint": True}),
                    ("one_arg_zero", {"low": 0, "high": OMIT, "size": (3,)}),
                    ("size_none", {"low": 0, "high": 100}), ("size_scalar", {"low": 0, "high": 100, "size": ()}),
                    ("size_2x3", {"low": 0, "high": 100, "size": (2, 3)}), ("size_zero", {"low": 0, "high": 100, "size": (0,)}),
                    ("r_2e32", {"low": 0, "high": 2 ** 32, "size": (4,)}), ("r_2e40", {"low": 0, "high": 2 ** 40, "size": (4,)}),
                    ("r_max", {"low": 0, "high": 2 ** 63 - 1, "size": (4,)}), ("r_max_ep", {"low": 0, "high": 2 ** 63 - 1, "size": (4,), "endpoint": True}),
                    ("omit_endpoint", {"low": 0, "high": 5, "size": (3,), "endpoint": OMIT}),
                    ("omit_dtype", {"low": 0, "high": 5, "size": (3,), "dtype": OMIT}),
                    ("dt_null", {"low": 0, "high": 5, "size": (3,), "dtype": None}),
                    ("dt_float", {"low": 0, "high": 5, "size": (3,), "dtype": "float64"}),
                    ("omit_size", {"low": 0, "high": 5, "size": OMIT}), ("big", {"low": 0, "high": 1000, "size": (1000,)})]
        if not unsigned:
            variants += [("r_min", {"low": -2 ** 63, "high": 2 ** 63 - 1, "size": (4,)}),
                         ("r_min_ep", {"low": -2 ** 63, "high": 2 ** 63 - 1, "size": (4,), "endpoint": True}),
                         ("neg_one_arg", {"low": -3, "high": OMIT, "size": (3,)})]
        else:
            variants += [("u_2e63", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,), "dtype": "uint64"}),
                         ("u_max", {"low": 0, "high": 2 ** 64 - 1, "size": (4,), "dtype": "uint64"}),
                         ("u_max_ep", {"low": 0, "high": 2 ** 64 - 1, "size": (4,), "dtype": "uint64", "endpoint": True}),
                         ("u_2e63_default", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,)})]
        for dt in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
            lo = 0 if unsigned or dt.startswith("u") or dt == "bool" else -5
            variants.append((f"dt_{dt}", {"low": lo, "high": 2 if dt == "bool" else 17, "size": (6,), "dtype": dt}))
            vmin = 0 if dt == "bool" else int(np.iinfo(dt).min)
            vmax = 1 if dt == "bool" else int(np.iinfo(dt).max)
            flo = 0 if unsigned else vmin
            variants += [(f"dt_{dt}_full", {"low": flo, "high": vmax + 1, "size": (6,), "dtype": dt}),
                         (f"dt_{dt}_full_ep", {"low": flo, "high": vmax, "size": (6,), "dtype": dt, "endpoint": True}),
                         (f"dt_{dt}_over", {"low": 0, "high": vmax + 2, "size": (3,), "dtype": dt}),
                         (f"dt_{dt}_over_ep", {"low": 0, "high": vmax + 1, "size": (3,), "dtype": dt, "endpoint": True}),
                         (f"dt_{dt}_scalar", {"low": 1, "high": 2, "dtype": dt}),
                         (f"dt_{dt}_single", {"low": vmax, "high": vmax, "size": (3,), "dtype": dt, "endpoint": True})]
            if not unsigned:
                variants.append((f"dt_{dt}_under", {"low": vmin - 1, "high": 0, "size": (3,), "dtype": dt}))
        if first == "BigInteger":
            # Every value either integer overload takes, plus the Python ints past 64 bits only this one can spell.
            variants += [("u_2e63", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,), "dtype": "uint64"}),
                         ("u_max", {"low": 0, "high": 2 ** 64 - 1, "size": (4,), "dtype": "uint64"}),
                         ("u_max_ep", {"low": 0, "high": 2 ** 64 - 1, "size": (4,), "dtype": "uint64", "endpoint": True}),
                         ("u_2e63_default", {"low": 2 ** 63, "high": 2 ** 63 + 10, "size": (4,)})]
            variants += integer_big_extra()
        gen_sweep(out, PORTABLE, "Generator.integers", m, seeds, base, variants, np_fn)


@family("Generator", "choice")
def fam_gen_choice(out, surface, name, seeds):
    """Generator.choice(a, size, replace, p, axis, shuffle): integer and array populations (N-D along an axis),
    with/without replacement (both algorithms), shuffle off, p valid and invalid, NumPy's rejections."""
    p5 = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

    def np_fn(r, py):
        return r.choice(py["a"], **gen_kw(py, ("size", "replace", "p", "axis", "shuffle")))
    common = [("size_none", {"size": None}), ("size_scalar", {"size": ()}), ("size_2x3", {"size": (2, 3)}),
              ("size_zero", {"size": (0,)}), ("omit_size", {"size": OMIT}),
              ("noreplace", {"size": (3,), "replace": False}), ("noreplace_all", {"size": (5,), "replace": False}),
              ("noreplace_noshuffle", {"size": (3,), "replace": False, "shuffle": False}),
              ("noreplace_too_many", {"size": (6,), "replace": False}), ("omit_replace", {"size": (3,), "replace": OMIT}),
              ("omit_shuffle", {"size": (3,), "replace": False, "shuffle": OMIT}),
              ("p", {"size": (4,), "p": p5}), ("p_null", {"size": (4,), "p": None}), ("omit_p", {"size": (4,), "p": OMIT}),
              ("p_scalar_draw", {"size": None, "p": p5}), ("p_noreplace", {"size": (3,), "replace": False, "p": p5}),
              ("p_noreplace_noshuffle", {"size": (3,), "replace": False, "p": p5, "shuffle": False}),
              ("p_noreplace_zeros", {"size": (3,), "replace": False, "p": np.array([0.5, 0.5, 0.0, 0.0, 0.0])}),
              ("p_wrong_len", {"size": (3,), "p": np.array([0.5, 0.5])}), ("p_neg", {"size": (3,), "p": np.array([-0.1, 0.3, 0.3, 0.3, 0.2])}),
              ("p_nan", {"size": (3,), "p": np.array([NAN, 0.3, 0.3, 0.2, 0.2])}), ("p_sum", {"size": (3,), "p": np.array([0.1] * 5)}),
              ("p_0d", {"size": (3,), "p": np.array(0.5)}), ("p_2d", {"size": (3,), "p": np.array([[0.2] * 5])}),
              ("p_strided", {"size": (3,), "p": derived(np.array([0.1, 9, 0.2, 9, 0.3, 9, 0.25, 9, 0.15, 9]), lambda b: b[::2])}),
              ("axis1", {"size": (3,), "axis": 1}), ("omit_axis", {"size": (3,), "axis": OMIT})]
    for m in overloads(surface, "Generator", name):
        if m["params"][0]["type"] == "long":
            base = {"a": 5, "size": (3,)}
            variants = [(t, dict({"a": 5}, **v)) for t, v in common]
            variants += [("a1", {"a": 1, "size": (3,)}), ("a0", {"a": 0, "size": (3,)}), ("a0_empty", {"a": 0, "size": (0,)}),
                         ("aneg", {"a": -3, "size": (3,)}), ("a_big", {"a": 10 ** 6, "size": (4,)}),
                         ("a_big_noreplace", {"a": 10 ** 6, "size": (5,), "replace": False}),
                         ("a_2e40", {"a": 2 ** 40, "size": (4,)}), ("a_2e40_noreplace", {"a": 2 ** 40, "size": (4,), "replace": False})]
            gen_sweep(out, PORTABLE, "Generator.choice", m, seeds, base, variants, np_fn)
            continue
        pop = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        base = {"a": pop, "size": (3,)}
        variants = [(t, dict({"a": pop}, **v)) for t, v in common if t != "axis1"]
        m2 = np.arange(12, dtype=np.int64).reshape(3, 4)
        variants += [("a_f64", {"a": np.array([0.5, 1.5, 2.5, 3.5, 4.5]), "size": (4,)}),
                     ("a_bool", {"a": np.array([True, False, True, False, True]), "size": (4,)}),
                     ("a_c128", {"a": np.array([1 + 2j, 3 - 1j, 0j, 5j, -1 + 0j]), "size": (4,)}),
                     ("a_strided", {"a": derived(np.arange(10, dtype=np.int64), lambda b: b[::2]), "size": (4,)}),
                     ("a_negstride", {"a": derived(np.arange(5, dtype=np.int64), lambda b: b[::-1]), "size": (4,)}),
                     ("a_2d_axis0", {"a": m2, "size": (2,)}), ("a_2d_axis1", {"a": m2, "size": (2,), "axis": 1}),
                     ("a_2d_axisneg", {"a": m2, "size": (2,), "axis": -1}), ("a_2d_axis2", {"a": m2, "size": (2,), "axis": 2}),
                     ("a_2d_noreplace", {"a": m2, "size": (2,), "replace": False, "axis": 1}),
                     ("a_2d_p", {"a": m2, "size": (2,), "p": np.array([0.2, 0.3, 0.5])}),
                     ("a_2d_scalar", {"a": m2, "size": None}),
                     ("a_2d_fortran", {"a": derived(np.arange(12, dtype=np.int64), lambda b: b.reshape(4, 3).T), "size": (2,), "axis": 1}),
                     ("a_3d", {"a": np.arange(24, dtype=np.float64).reshape(2, 3, 4), "size": (3,), "axis": 2}),
                     ("a_0d_int", {"a": np.array(7, dtype=np.int64), "size": (4,)}),
                     ("a_0d_float", {"a": np.array(7.0), "size": (4,)}), ("a_0d_zero", {"a": np.array(0, dtype=np.int64), "size": (3,)}),
                     ("a_empty", {"a": np.array([], dtype=np.int64), "size": (3,)}),
                     ("a_empty_nosample", {"a": np.array([], dtype=np.int64), "size": (0,)}),
                     ("a_null", {"a": None, "size": (3,)})]
        gen_sweep(out, PORTABLE, "Generator.choice", m, seeds, base, variants, np_fn)


@family("Generator", "permutation")
def fam_gen_permutation(out, surface, name, seeds):
    """Generator.permutation(x, axis): an arange (the axis is ignored for an integer) or a shuffled copy along the axis."""
    for m in overloads(surface, "Generator", name):
        if m["params"][0]["type"] == "long":
            variants = [("x0", {"x": 0}), ("x1", {"x": 1}), ("xneg", {"x": -3}), ("x100", {"x": 100}),
                        ("axis1", {"x": 10, "axis": 1}), ("omit_axis", {"x": 10, "axis": OMIT})]
            gen_sweep(out, PORTABLE, "Generator.permutation", m, seeds, {"x": 10}, variants,
                      lambda r, py: r.permutation(py["x"], **gen_kw(py, ("axis",))))
            continue
        m2 = np.arange(12, dtype=np.int64).reshape(4, 3)
        variants = [("f64", {"x": np.linspace(0.0, 1.0, 7)}), ("bool", {"x": np.array([True, False, True, True])}),
                    ("d2", {"x": m2}), ("d2_axis1", {"x": m2, "axis": 1}), ("d2_axisneg", {"x": m2, "axis": -1}),
                    ("d2_axis2", {"x": m2, "axis": 2}), ("d3_axis2", {"x": np.arange(24.0).reshape(2, 3, 4), "axis": 2}),
                    ("fortran", {"x": derived(np.arange(12, dtype=np.int64), lambda b: b.reshape(3, 4).T), "axis": 1}),
                    ("strided", {"x": derived(np.arange(20, dtype=np.int64), lambda b: b[::3])}),
                    ("negstride", {"x": derived(np.arange(8, dtype=np.int64), lambda b: b[::-1])}),
                    ("empty", {"x": np.array([], dtype=np.float64)}), ("one", {"x": np.array([42], dtype=np.int64)}),
                    ("zerod", {"x": np.array(5, dtype=np.int64)}), ("null", {"x": None}), ("omit_axis", {"x": m2, "axis": OMIT})]
        gen_sweep(out, PORTABLE, "Generator.permutation", m, seeds, {"x": np.arange(10, dtype=np.int64)}, variants,
                  lambda r, py: r.permutation(py["x"], **gen_kw(py, ("axis",))))


@family("Generator", "shuffle")
def fam_gen_shuffle(out, surface, name, seeds):
    """Generator.shuffle(x, axis) in place along the axis (the operand's contents after the call are observed)."""
    def fresh(make, **extra):
        return lambda: dict({"x": make()}, **extra)

    def view(make_base, fn, **extra):
        return lambda: dict({"x": derived(make_base(), fn)}, **extra)
    variants = [("f64", fresh(lambda: np.linspace(0.0, 1.0, 9))), ("bool", fresh(lambda: np.array([True, False, True, False]))),
                ("d2", fresh(lambda: np.arange(12, dtype=np.int64).reshape(4, 3))),
                ("d2_axis1", fresh(lambda: np.arange(12, dtype=np.int64).reshape(4, 3), axis=1)),
                ("d2_axisneg", fresh(lambda: np.arange(12, dtype=np.int64).reshape(4, 3), axis=-1)),
                ("d2_axis2", fresh(lambda: np.arange(12, dtype=np.int64).reshape(4, 3), axis=2)),
                ("d3_axis1", fresh(lambda: np.arange(24.0).reshape(2, 3, 4), axis=1)),
                ("fortran", view(lambda: np.arange(12, dtype=np.int64), lambda b: b.reshape(3, 4).T)),
                ("strided", view(lambda: np.arange(20, dtype=np.int64), lambda b: b[::3])),
                ("negstride", view(lambda: np.arange(8, dtype=np.int64), lambda b: b[::-1])),
                ("cols_sliced", view(lambda: np.arange(20, dtype=np.int64), lambda b: b.reshape(4, 5)[:, 1:4], axis=1)),
                ("empty", fresh(lambda: np.array([], dtype=np.float64))), ("one", fresh(lambda: np.array([42], dtype=np.int64))),
                ("zerod", fresh(lambda: np.array(5, dtype=np.int64))), ("null", lambda: {"x": None}),
                ("omit_axis", fresh(lambda: np.arange(6, dtype=np.int64), axis=OMIT))]
    for m in overloads(surface, "Generator", name):
        gen_sweep(out, PORTABLE, "Generator.shuffle", m, seeds, lambda: {"x": np.arange(10, dtype=np.int64)}, variants,
                  lambda r, py: r.shuffle(py["x"], **gen_kw(py, ("axis",))), watch=("x",))


@family("Generator", "permuted")
def fam_gen_permuted(out, surface, name, seeds):
    """Generator.permuted(x, axis, out): every 1-D slice along the axis shuffled independently (axis=None: flattened);
    out as a fresh array, as x itself (in place), wrong shape (rejected)."""
    m2 = lambda: np.arange(12, dtype=np.int64).reshape(3, 4)
    variants = [("axis0", lambda: {"x": m2(), "axis": 0}), ("axis1", lambda: {"x": m2(), "axis": 1}),
                ("axisneg", lambda: {"x": m2(), "axis": -1}), ("axis2", lambda: {"x": m2(), "axis": 2}),
                ("axis_null", lambda: {"x": m2(), "axis": None}), ("omit_axis", lambda: {"x": m2(), "axis": OMIT}),
                ("out", lambda: {"x": m2(), "axis": 1, "out": np.zeros((3, 4), dtype=np.int64)}),
                ("out_self", lambda: {"x": m2(), "axis": 1, "out": Alias("x")}),
                ("out_wrong_shape", lambda: {"x": m2(), "axis": 1, "out": np.zeros((4, 3), dtype=np.int64)}),
                ("out_f64", lambda: {"x": m2(), "axis": 0, "out": np.zeros((3, 4))}),
                ("out_strided", lambda: {"x": m2(), "axis": 1, "out": derived(np.zeros(24, dtype=np.int64), lambda b: b.reshape(3, 8)[:, ::2])}),
                ("out_null", lambda: {"x": m2(), "axis": 1, "out": None}),
                ("f64_1d", lambda: {"x": np.linspace(0.0, 1.0, 9)}),
                ("fortran", lambda: {"x": derived(np.arange(12, dtype=np.int64), lambda b: b.reshape(4, 3).T), "axis": 0}),
                ("strided", lambda: {"x": derived(np.arange(20, dtype=np.int64), lambda b: b[::3])}),
                ("d3", lambda: {"x": np.arange(24.0).reshape(2, 3, 4), "axis": 2}),
                ("empty", lambda: {"x": np.zeros((0, 3))}), ("zerod", lambda: {"x": np.array(5, dtype=np.int64)}),
                ("zerod_axis0", lambda: {"x": np.array(5, dtype=np.int64), "axis": 0}), ("null", lambda: {"x": None}),
                ("null_axis", lambda: {"x": None, "axis": 1}), ("null_out", lambda: {"x": None, "out": np.zeros(3)}),
                ("null_out0d", lambda: {"x": None, "out": np.zeros(())}),
                ("zerod_out_i32", lambda: {"x": np.array(5, dtype=np.int64), "out": np.zeros((), dtype=np.int32)}),
                ("out_unsafe", lambda: {"x": m2(), "axis": 1, "out": np.zeros((3, 4), dtype=np.int32)})]
    for m in overloads(surface, "Generator", name):
        gen_sweep(out, PORTABLE, "Generator.permuted", m, seeds, lambda: {"x": np.arange(10, dtype=np.int64)}, variants,
                  lambda r, py: r.permuted(py["x"], **gen_kw(py, ("axis", "out"))), watch=("x", "out"))


@family("Generator", "bytes")
def fam_gen_bytes(out, surface, name, seeds):
    for m in overloads(surface, "Generator", name):
        variants = [(f"len{n}", {"length": n}) for n in (0, 1, 3, 4, 5, 7, 8, 13, 16, 100, -1, -3, -4, -7, -8, -9)]
        gen_sweep(out, PORTABLE, "Generator.bytes", m, seeds, {"length": 10}, variants, lambda r, py: r.bytes(py["length"]))


@family("Generator", "dirichlet")
def fam_gen_dirichlet(out, surface, name, seeds):
    """Generator.dirichlet: the gamma algorithm, and the beta-based one NumPy switches to when max(alpha) < 0.1."""
    for m in overloads(surface, "Generator", name):
        arr = m["params"][0]["type"].startswith("NDArray")

        def a(v):
            return np.array(v, dtype=np.float64) if arr else list(v)
        base = {"alpha": a([1.0, 2.0, 3.0]), "size": (4,)}
        variants = [(t, dict(base, **v)) for t, v in dims_variants("size", "Shape")[1:]] + [("omit_size", dict(base, size=OMIT))]
        variants += [("small", dict(base, alpha=a([0.05, 0.02, 0.08]))), ("tiny", dict(base, alpha=a([1e-3, 1e-3]))),
                     ("mixed", dict(base, alpha=a([0.05, 2.0]))), ("one", dict(base, alpha=a([2.5]))),
                     ("zero", dict(base, alpha=a([1.0, 0.0]))), ("neg", dict(base, alpha=a([1.0, -1.0]))),
                     ("nan", dict(base, alpha=a([1.0, NAN]))), ("empty", dict(base, alpha=a([]))), ("null", dict(base, alpha=None)),
                     ("large", dict(base, alpha=a([1e3, 2e3])))]
        if arr:
            variants += [("a2d", dict(base, alpha=np.array([[1.0, 2.0]]))), ("a0d", dict(base, alpha=np.array(1.5))),
                         ("a_i64", dict(base, alpha=np.array([1, 2, 3], dtype=np.int64))),
                         ("a_strided", dict(base, alpha=derived(np.array([1.0, 9, 2.0, 9, 3.0, 9]), lambda b: b[::2])))]
        gen_sweep(out, HOST, "Generator.dirichlet", m, seeds, base, variants,
                  lambda r, py: r.dirichlet(py["alpha"], **gen_kw(py, ("size",))))


@family("Generator", "multinomial")
def fam_gen_multinomial(out, surface, name, seeds):
    """Generator.multinomial(n, pvals, size): scalar and array n, 1-D and 2-D pvals broadcast, NumPy's rejections."""
    for m in overloads(surface, "Generator", name):
        nt, pt = m["params"][0]["type"], m["params"][1]["type"]

        def pv(v):
            return np.array(v, dtype=np.float64) if pt.startswith("NDArray") else list(v)

        def nv(v):
            return np.array(v, dtype=np.int64) if nt.startswith("NDArray") else v
        base = {"n": nv(10), "pvals": pv([0.2, 0.3, 0.5]), "size": (3,)}
        variants = [(t, dict(base, **v)) for t, v in dims_variants("size", "Shape")[1:]] + [("omit_size", dict(base, size=OMIT))]
        variants += [("n0", dict(base, n=nv(0))), ("nneg", dict(base, n=nv(-1))), ("n_big", dict(base, n=nv(10 ** 6))),
                     ("p_one", dict(base, pvals=pv([1.0]))), ("p_nan", dict(base, pvals=pv([NAN, 0.5, 0.5]))),
                     ("p_neg", dict(base, pvals=pv([-0.1, 0.6, 0.5]))), ("p_sum", dict(base, pvals=pv([0.6, 0.6, 0.1]))),
                     ("p_last_slack", dict(base, pvals=pv([0.2, 0.2, 0.9]))), ("p_empty", dict(base, pvals=pv([]))),
                     ("p_null", dict(base, pvals=None))]
        if pt.startswith("NDArray"):
            variants += [("p_2d", dict(base, pvals=np.array([[0.2, 0.8], [0.5, 0.5]]), size=OMIT)),
                         ("p_2d_size", dict(base, pvals=np.array([[0.2, 0.8], [0.5, 0.5]]), size=(3, 2))),
                         ("p_0d", dict(base, pvals=np.array(0.5))),
                         ("p_strided", dict(base, pvals=derived(np.array([0.2, 9, 0.3, 9, 0.5, 9]), lambda b: b[::2])))]
        if nt.startswith("NDArray"):
            variants += [("n_1d", dict(base, n=np.array([5, 10, 20], dtype=np.int64), size=OMIT)),
                         ("n_col", dict(base, n=np.array([[5], [10]], dtype=np.int64), pvals=np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]), size=OMIT)),
                         ("n_f64", dict(base, n=np.array([5.0, 10.0]), size=OMIT)), ("n_null", dict(base, n=None))]
        gen_sweep(out, HOST, "Generator.multinomial", m, seeds, base, variants,
                  lambda r, py: r.multinomial(py["n"], py["pvals"], **gen_kw(py, ("size",))))


@family("Generator", "multivariate_hypergeometric")
def fam_gen_mvhg(out, surface, name, seeds):
    """Generator.multivariate_hypergeometric(colors, nsample, size, method): 'marginals' (hypergeometric chain: libm)
    and 'count' (random_interval: portable), NumPy's rejections."""
    for m in overloads(surface, "Generator", name):
        arr = m["params"][0]["type"].startswith("NDArray")

        def col(v):
            return np.array(v, dtype=np.int64) if arr else list(v)
        np_fn = (lambda r, py: r.multivariate_hypergeometric(py["colors"], py["nsample"], **gen_kw(py, ("size", "method"))))
        for method, tier in (("marginals", HOST), ("count", PORTABLE)):
            base = {"colors": col([16, 8, 4]), "nsample": 6, "size": (3,), "method": method}
            variants = [(t, dict(base, **v)) for t, v in dims_variants("size", "Shape")[1:]] + [("omit_size", dict(base, size=OMIT))]
            variants += [("all", dict(base, nsample=28)), ("zero", dict(base, nsample=0)), ("too_many", dict(base, nsample=29)),
                         ("neg_n", dict(base, nsample=-1)), ("neg_color", dict(base, colors=col([4, -1, 3]))),
                         ("one_color", dict(base, colors=col([7]), nsample=3)), ("zero_colors", dict(base, colors=col([0, 0, 5]), nsample=2)),
                         ("empty", dict(base, colors=col([]), nsample=0)), ("big", dict(base, colors=col([10 ** 5, 2 * 10 ** 5]), nsample=1000)),
                         ("null", dict(base, colors=None))]
            if method == "count":
                # 'count' fills a sum(colors)-entry index array in BOTH libraries (8 GB at 10**9), so its large case stays
                # at a million entries; the int64 sum limits are checked before any allocation.
                variants += [("count_big", dict(base, colors=col([10 ** 6, 5]), nsample=7)),
                             ("count_overflow", dict(base, colors=col([2 ** 62, 2 ** 62]), nsample=2)),
                             ("count_intmax", dict(base, colors=col([2 ** 62, 2 ** 62 - 1]), nsample=2))]
            else:
                variants += [("omit_method", dict(base, method=OMIT)), ("bogus", dict(base, method="bogus")),
                             ("method_null", dict(base, method=None))]
            if arr:
                variants += [("c_2d", dict(base, colors=np.array([[1, 2]], dtype=np.int64))),
                             ("c_f64", dict(base, colors=np.array([4.0, 5.0]))),
                             ("c_strided", dict(base, colors=derived(np.array([16, 9, 8, 9, 4, 9], dtype=np.int64), lambda b: b[::2])))]
            gen_sweep(out, tier, "Generator.multivariate_hypergeometric", m, seeds, base,
                      [(f"{method}/{t}", v) for t, v in variants], np_fn, prefix=f"{method}/")


@family("Generator", "spawn")
def fam_gen_spawn(out, surface, name, seeds):
    """Generator.spawn(n): children over the bit generator's spawned seed sequences (the parent's counter advances)."""
    for m in overloads(surface, "Generator", name):
        gen_sweep(out, PORTABLE, "Generator.spawn", m, seeds, {"n_children": 3},
                  [("n0", {"n_children": 0}), ("n1", {"n_children": 1}), ("nneg", {"n_children": -1})],
                  lambda r, py: r.spawn(py["n_children"]))


@family("Generator", "bit_generator", "_bit_generator", "_poisson_lam_max", "ToString")
def fam_gen_attrs(out, surface, name, seeds):
    fn = {"bit_generator": lambda r, py: r.bit_generator, "_bit_generator": lambda r, py: r._bit_generator,
          "_poisson_lam_max": lambda r, py: r._poisson_lam_max, "ToString": lambda r, py: str(r)}[name]
    for m in overloads(surface, "Generator", name):
        gen_sweep(out, PORTABLE, f"Generator.{name}", m, seeds, {}, [], fn, calls2=False,
                  primes=("raw3",) if name.endswith("bit_generator") else ())


@family("Generator", "ctor")
def fam_gen_ctor(out, surface, name, seeds):
    """Generator(bit_generator): wraps the engine as is (primed state carried over); None is NumPy's AttributeError."""
    for m in overloads(surface, "Generator", name):
        emit_gen_ctor_overload(out, m, seeds, Recv("none"), state=False)


def emit_gen_ctor_overload(out, m, seeds, none, state):
    """Every case of the Generator(bit_generator) overload — the constructor's (no receiver, no state) or the
    np.random.Generator factory's (its RandomState receiver, whose state is recorded)."""
    for e in ENGINES:
        for sd in seeds[:3]:
            emit_m(out, PORTABLE, "Generator", m, none, {"bit_generator": bitgen_obj(e, sd)}, f"{e}_s{sd}",
                   lambda r, py: np.random.Generator(py["bit_generator"]), state=state)
        emit_m(out, PORTABLE, "Generator", m, none, {"bit_generator": bitgen_obj(e, 5, "raw3")}, f"{e}_primed",
               lambda r, py: np.random.Generator(py["bit_generator"]), state=state)
    emit_m(out, PORTABLE, "Generator", m, none, {"bit_generator": None}, "null",
           lambda r, py: np.random.Generator(py["bit_generator"]), state=state)


def make_module_class_family(cls_name):
    """np.random.<Class>(...) — numpy.random's classes reached through the module, NumPyRandom factories mirroring each
    constructor overload: replayed exactly as the constructor (the member is the class name, as NumPy's np.random.PCG64
    IS the class), on a RandomState receiver whose state is recorded — a factory must build, never draw."""
    def fam(out, surface, name, seeds):
        recv = Recv("RandomState", None, 0)
        for m in overloads(surface, "NumPyRandom", name):
            if cls_name in ENGINE_CLASS:
                emit_engine_ctor_overload(out, m, cls_name, seeds, recv, state=True)
            elif cls_name == "SeedSequence":
                emit_seedseq_ctor_overload(out, m, recv, state=True)
            else:
                emit_gen_ctor_overload(out, m, seeds, recv, state=True)
    return fam


for _cls in ENGINES + ["SeedSequence", "Generator"]:
    FAMILIES[("NumPyRandom", _cls)] = make_module_class_family(_cls)



# ======================================================================================================================
# Families: bit generators, their states, seed sequences, default_rng (P4)
# ======================================================================================================================

# State fields an OS-entropy-seeded engine draws, masked on both sides together with the seed sequence's pool and repr.
ENGINE_ENTROPY_FIELDS = {"MT19937": ("key",), "PCG64": ("state", "inc"), "PCG64DXSM": ("state", "inc"),
                         "Philox": ("key",), "SFC64": ("state",)}
SEEDSEQ_MASK = ("pool", "repr")


def engine_mask(engine):
    return ENGINE_ENTROPY_FIELDS[engine] + SEEDSEQ_MASK


# Seeds by the C# parameter type of an engine / SeedSequence constructor (each also taken by NumPy as that Python value).
SEED_VALUES = {
    "long": [0, 1, 42, 2 ** 32, 2 ** 63 - 1, -1],
    "ulong": [0, 2 ** 63, 2 ** 64 - 1],
    "BigInteger": [0, 2 ** 64, 2 ** 100, 2 ** 128 + 5, -1],
    "int[]": [[1, 2, 3], [], [0], [2 ** 31 - 1], [-1]],
    "long[]": [[1, 2, 3], [2 ** 40, 5], [], [-5]],
    "uint[]": [[1, 2, 3], [2 ** 32 - 1], []],
}


def seed_py(ctype, v):
    """The Python value NumPy receives for a C# seed of the given type: a uint[] stands for NumPy's uint32 ndarray (so
    the SeedSequence reprs agree), every other array for a Python list."""
    if v is None:
        return None
    return np.array(v, dtype=np.uint32) if ctype == "uint[]" else v


def seed_seqs():
    """ISeedSequence arguments: SeedSequences over every keyword, and the seedless one (NumPy refuses it)."""
    return [("ss42", seedseq_obj(42)), ("ss_key", seedseq_obj(42, spawn_key=[1, 2])),
            ("ss_pool8", seedseq_obj(42, pool_size=8)), ("ss_children", seedseq_obj(42, n_children_spawned=3)),
            ("ss_list", seedseq_obj([1, 2, 3])), ("ss_big", seedseq_obj(2 ** 100)), ("seedless", seedless_obj())]


def emit_engine_ctor_overload(out, m, E, seeds, recv, state):
    """Every case of one ENGINE(seed) overload — the constructor's own (`recv` = none, state=False) or the np.random
    factory that mirrors it (`recv` = the RandomState it is called on, whose state is recorded to prove the factory never
    draws from it). The member is the class name either way (NumPy's np.random.PCG64 IS the class)."""
    cls = ENGINE_CLASS[E]
    ps = m["params"]
    if not ps:
        emit_m(out, PORTABLE, E, m, recv, {}, "entropy", lambda r, py: cls(), state=state, state_mask=engine_mask(E))
        return
    if len(ps) == 3:
        fam_philox_ctor3(out, m, seeds, recv=recv, state=state)
        return
    pt = ps[0]["type"]
    if pt == "ISeedSequence":
        for tag, v in seed_seqs():
            emit_m(out, PORTABLE, E, m, recv, {"seed": v}, tag, lambda r, py: cls(py["seed"]), state=state)
        emit_m(out, PORTABLE, E, m, recv, {"seed": None}, "null", lambda r, py: cls(None), state=state,
               state_mask=engine_mask(E))
        return
    vals = (list(seeds) if pt == "long" else []) + SEED_VALUES[pt]
    for v in dict.fromkeys(map(lambda x: json.dumps(x), vals)):
        v = json.loads(v)
        emit_m(out, PORTABLE, E, m, recv, {"seed": v}, f"s{json.dumps(v, separators=(',', ':'))}",
               lambda r, py, pt=pt: cls(seed_py(pt, py["seed"])), state=state)
    if pt.endswith("[]"):
        emit_m(out, PORTABLE, E, m, recv, {"seed": None}, "null", lambda r, py: cls(None), state=state,
               state_mask=engine_mask(E))


def make_engine_ctor_family(E):
    def fam(out, surface, name, seeds):
        """ENGINE(seed) for every seed form: integers (fixed seeds and the edges), arrays (null is NumPy's None — OS
        entropy), seed sequences (every keyword, seedless refused), Philox's (seed, counter, key)."""
        for m in overloads(surface, E, "ctor"):
            emit_engine_ctor_overload(out, m, E, seeds, Recv("none"), state=False)
    return fam


def fam_philox_ctor3(out, m, seeds, recv=None, state=False):
    """Philox(seed, counter, key): seed with and without a counter, a key instead of a seed, both (refused), counters and
    keys past their word counts and negative (refused), every argument omitted (OS entropy). `recv`/`state` as for
    emit_engine_ctor_overload (the np.random.Philox factory passes its RandomState receiver)."""
    none = recv if recv is not None else Recv("none")

    def np_fn(r, py):
        return np.random.Philox(**{k: py[k] for k in ("seed", "counter", "key") if k in py})
    variants = [("seed", {"seed": value_obj(42)}), ("seed_counter", {"seed": value_obj(42), "counter": value_obj(5)}),
                ("seed_counter_list", {"seed": value_obj(42), "counter": value_obj([1, 2, 3, 4])}),
                ("seed_counter_arr", {"seed": value_obj(42), "counter": np.array([7, 0, 0, 1], dtype=np.uint64)}),
                ("seed_counter_big", {"seed": value_obj(42), "counter": value_obj(2 ** 200)}),
                ("seed_counter_neg", {"seed": value_obj(42), "counter": value_obj(-1)}),
                ("key", {"key": value_obj(7)}), ("key_list", {"key": value_obj([1, 2])}),
                ("key_arr", {"key": np.array([3, 4], dtype=np.uint64)}), ("key_big", {"key": value_obj(2 ** 128 - 1)}),
                ("key_too_big", {"key": value_obj(2 ** 128)}), ("key_neg", {"key": value_obj(-1)}),
                ("key_counter", {"key": value_obj(7), "counter": value_obj(3)}),
                ("seed_and_key", {"seed": value_obj(42), "key": value_obj(7)}),
                ("seed_seqseq", {"seed": seedseq_obj(9, spawn_key=[4])}), ("seed_list", {"seed": value_obj([1, 2, 3])}),
                ("seed_str", {"seed": value_obj("123")}), ("seed_float", {"seed": value_obj(1.5)}),
                ("seed_neg", {"seed": value_obj(-3)})]
    for tag, vals in variants:
        emit_m(out, PORTABLE, "Philox", m, none, vals, tag, np_fn, state=state)
    # Every argument None: OS entropy, like the omitted call below (the key and the seed sequence are masked).
    emit_m(out, PORTABLE, "Philox", m, none, {"seed": None, "counter": None, "key": None}, "nulls", np_fn, state=state,
           state_mask=engine_mask("Philox"))
    emit_m(out, PORTABLE, "Philox", m, none, {"seed": None, "counter": value_obj(9), "key": None}, "null_seed_counter",
           np_fn, state=state, state_mask=engine_mask("Philox"))
    emit_m(out, PORTABLE, "Philox", m, none, {}, "entropy", np_fn, state=state, state_mask=engine_mask("Philox"))


def bitgen_sweep(out, tier, member, m, E, seeds, base, variants, np_fn, primes=("raw3",), calls2=True, **kw):
    """The standard case set of one overload on a bare engine E: base on E(seed) for every seed, variants on two seeds,
    then priming (three raw words) and a repeated call."""
    recvs = [Recv("BitGenerator", E, sd) for sd in seeds]
    vrecv = recvs[:2]
    for recv in recvs:
        emit_m(out, tier, member, m, recv, base, "base", np_fn, **kw)
    for tag, vals in variants:
        for recv in vrecv:
            if not emit_m(out, tier, member, m, recv, vals, tag, np_fn, **kw):
                break
    for recv0 in vrecv:
        for pr in primes:
            emit_m(out, tier, member, m, Recv("BitGenerator", E, recv0.seed, prime=pr), base, f"prime_{pr}", np_fn, **kw)
        if calls2:
            emit_m(out, tier, member, m, recv0, base, "calls2", np_fn, calls=2, **kw)


def other_engine(E):
    return "SFC64" if E != "SFC64" else "PCG64"


def state_set_variants(E, typed):
    """Values for an engine's state setter: a state of the same engine from another seed and priming, another engine's
    (refused — expressible only through the base `BitGenerator.state` setter, whose parameter is the base State: the
    engine's own typed setter cannot even be handed another engine's State in C#), and states NumPy validates or not
    (bad positions, wrong word counts, unset arrays)."""
    v = [("same", {"value": bgstate_obj(E, 99, "raw3")}), ("same_fresh", {"value": bgstate_obj(E, 5)}),
         ("null", {"value": None})]
    if not typed:
        v.append(("other", {"value": bgstate_obj(other_engine(E), 5)}))
    if E == "MT19937":
        key = [(i * 69069 + 1) % 2 ** 32 for i in range(624)]
        v += [("pos0", {"value": bgstate_explicit(E, key=key, pos=0)}), ("pos624", {"value": bgstate_explicit(E, key=key, pos=624)}),
              ("pos625", {"value": bgstate_explicit(E, key=key, pos=625)}), ("posneg", {"value": bgstate_explicit(E, key=key, pos=-1)}),
              ("key10", {"value": bgstate_explicit(E, key=key[:10], pos=3)}), ("key_null", {"value": bgstate_explicit(E, pos=3)}),
              # A longer key: NumPy copies its first 624 words.
              ("key625", {"value": bgstate_explicit(E, key=key + [7], pos=3)})]
    elif E in ("PCG64", "PCG64DXSM"):
        v += [("explicit", {"value": bgstate_explicit(E, state=2 ** 100 + 7, inc=2 ** 64 + 1)}),
              ("even_inc", {"value": bgstate_explicit(E, state=5, inc=4)}),
              ("buffered", {"value": bgstate_explicit(E, state=5, inc=3, has_uint32=1, uinteger=123456)}),
              ("has_uint32_2", {"value": bgstate_explicit(E, state=5, inc=3, has_uint32=2, uinteger=9)})]
    elif E == "Philox":
        v += [("explicit", {"value": bgstate_explicit(E, counter=[1, 2, 3, 4], key=[5, 6], buffer=[7, 8, 9, 10], buffer_pos=2)}),
              ("buffer_pos4", {"value": bgstate_explicit(E, counter=[0, 0, 0, 0], key=[5, 6], buffer=[0, 0, 0, 0], buffer_pos=4)}),
              ("buffer_pos5", {"value": bgstate_explicit(E, counter=[0, 0, 0, 0], key=[5, 6], buffer=[0, 0, 0, 0], buffer_pos=5)}),
              ("buffer_pos_neg", {"value": bgstate_explicit(E, counter=[0, 0, 0, 0], key=[5, 6], buffer=[1, 2, 3, 4], buffer_pos=-1)}),
              ("counter_null_key3", {"value": bgstate_explicit(E, key=[5, 6, 7], buffer=[0, 0, 0, 0], buffer_pos=4)}),
              ("counter3_key_null", {"value": bgstate_explicit(E, counter=[1, 2, 3], buffer=[0, 0, 0, 0], buffer_pos=4)}),
              ("buffer_null", {"value": bgstate_explicit(E, counter=[1, 2, 3, 4], key=[5, 6], buffer_pos=4)}),
              ("buffer3", {"value": bgstate_explicit(E, counter=[1, 2, 3, 4], key=[5, 6], buffer=[1, 2, 3], buffer_pos=4)}),
              ("counter3", {"value": bgstate_explicit(E, counter=[1, 2, 3], key=[5, 6], buffer=[0, 0, 0, 0], buffer_pos=4)}),
              ("key_null", {"value": bgstate_explicit(E, counter=[1, 2, 3, 4], buffer=[0, 0, 0, 0], buffer_pos=4)})]
    else:
        v += [("explicit", {"value": bgstate_explicit(E, state=[1, 2, 3, 4], has_uint32=1, uinteger=77)}),
              ("state3", {"value": bgstate_explicit(E, state=[1, 2, 3])}), ("state_null", {"value": bgstate_explicit(E)}),
              # NumPy broadcasts the words into four (`state_vec[:] = …`): one word fills all four, none is an error.
              ("state1", {"value": bgstate_explicit(E, state=[9])}), ("state_empty", {"value": bgstate_explicit(E, state=[])})]
    return v


def make_engine_method_family(E):
    def fam(out, surface, name, seeds):
        """ENGINE.advance / jumped / state / _legacy_seeding on E(seed) receivers."""
        member = f"{E}.{name}"
        for m in overloads(surface, E, name):
            if name == "advance":
                variants = [(f"d{t}", {"delta": d}) for t, d in (("0", 0), ("2e64", 2 ** 64), ("max", 2 ** 128 - 1),
                                                                 ("2e128", 2 ** 128), ("neg", -1), ("2e200", 2 ** 200), ("7", 7))]
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {"delta": 1}, variants, lambda r, py: r.advance(py["delta"]))
            elif name == "jumped":
                # A new engine built as NumPy builds it (a fresh OS-entropy seed sequence, then the jumped state): its
                # seed sequence is masked, the state is exact; the receiver is untouched.
                # MT19937's jump is a polynomial evaluation per jump (NumPy loops `for i in range(jumps)`): no large
                # counts there; the counter-based engines take any count (one advance of jumps * step).
                counts = ((("0", 0), ("2", 2), ("neg", -1)) if E == "MT19937"
                          else (("0", 0), ("2", 2), ("7", 7), ("neg", -1), ("big", 2 ** 40)))
                variants = [(f"j{t}", {"jumps": j}) for t, j in counts]
                if m["params"][0]["type"] == "BigInteger":
                    variants.append(("j2e70", {"jumps": 2 ** 70}))
                if m["params"][0].get("optional"):
                    variants.append(("omit", {"jumps": OMIT}))
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {"jumps": 1}, variants,
                             lambda r, py: r.jumped(**({"jumps": py["jumps"]} if "jumps" in py else {})),
                             state_mask=SEEDSEQ_MASK)
            elif name == "state" and m["kind"] == "get":
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {}, [], lambda r, py: r.state, calls2=False)
            elif name == "state":
                def np_set(r, py):
                    r.state = py["value"]
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {"value": bgstate_obj(E, 31, "raw3")},
                             state_set_variants(E, typed=True), np_set, calls2=False)
            elif name == "_legacy_seeding":
                fam_legacy_seeding_overload(out, m, seeds)
            else:
                raise ValueError(f"no family for {m['sig']}")
    return fam


def fam_legacy_seeding_overload(out, m, seeds):
    """MT19937._legacy_seeding(seed): RandomState's seeding on a bare MT19937 — an int, init_by_array, or None (OS
    entropy, key masked)."""
    member = "MT19937._legacy_seeding"
    if not m["params"]:
        bitgen_sweep(out, PORTABLE, member, m, "MT19937", seeds, {}, [], lambda r, py: r._legacy_seeding(None),
                     calls2=False, state_mask=("key",))
        return
    pt = m["params"][0]["type"]
    if pt == "long":
        variants = [(f"s{v}", {"seed": v}) for v in (0, 42, 2 ** 32 - 1, 2 ** 32, -1)]
        base = {"seed": 1234}
    else:
        arrs = {"uint[]": [[1, 2, 3], [2 ** 32 - 1], []], "long[]": [[1, 2, 3], [2 ** 32], [-1], []],
                "int[]": [[1, 2, 3], [-1], [], list(range(700))]}[pt]
        variants = [(f"a{i}", {"seed": a}) for i, a in enumerate(arrs)]
        base = {"seed": [5, 6, 7]}
    # Every C# array is a Python LIST here, as in RandomState.seed and RandomState(int[]): a list of any length takes
    # init_by_array, where a one-element ndarray would be squeezed to the scalar seed (NumPy's `seed.squeeze()` branch).
    # (A SeedSequence entropy uint[] stands for the uint32 ndarray instead — its repr — see seed_py.)
    bitgen_sweep(out, PORTABLE, member, m, "MT19937", seeds, base, variants,
                 lambda r, py: r._legacy_seeding(py["seed"]), calls2=False)
    if pt.endswith("[]"):
        for sd in seeds[:2]:
            emit_m(out, PORTABLE, member, m, Recv("BitGenerator", "MT19937", sd), {"seed": None}, "null",
                   lambda r, py: r._legacy_seeding(None), state_mask=("key",))


for _E in ENGINES:
    FAMILIES[(_E, "ctor")] = make_engine_ctor_family(_E)
    for _n in ("advance", "jumped", "state", "_legacy_seeding"):
        FAMILIES[(_E, _n)] = make_engine_method_family(_E)


@family("BitGenerator", "random_raw", "seed_seq", "spawn", "state", "lock")
def fam_bitgen_base(out, surface, name, seeds):
    """BitGenerator's own members, invoked on every engine (and, for seed_seq/spawn, on a legacy-seeded MT19937, which
    has no seed sequence)."""
    for m in overloads(surface, "BitGenerator", name):
        if name == "lock":
            exempt(m["sig"], "a threading primitive; nothing NumPy-observable (plan §7)")
            continue
        for E in ENGINES:
            member = f"{E}.{name}"
            if name == "random_raw":
                variants = [("size_none", {"size": SHAPE_NONE}), ("size_scalar", {"size": ()}), ("size_2x3", {"size": (2, 3)}),
                            ("size_zero", {"size": (0,)}), ("omit_size", {"size": OMIT}),
                            ("no_output", {"size": (5,), "output": False}), ("no_output_scalar", {"size": SHAPE_NONE, "output": False}),
                            ("no_output_2x3", {"size": (2, 3), "output": False}),
                            ("omit_output", {"size": (4,), "output": OMIT}), ("big", {"size": (1000,)})]
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {"size": (6,)}, variants,
                             lambda r, py: r.random_raw(**gen_kw(py, ("size", "output"))))
            elif name == "seed_seq":
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {}, [], lambda r, py: r.seed_seq, calls2=False)
            elif name == "spawn":
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {"n_children": 3},
                             [("n0", {"n_children": 0}), ("n1", {"n_children": 1}), ("nneg", {"n_children": -2})],
                             lambda r, py: r.spawn(py["n_children"]))
            elif name == "state" and m["kind"] == "get":
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {}, [], lambda r, py: r.state, calls2=False)
            else:
                def np_set(r, py):
                    r.state = py["value"]
                bitgen_sweep(out, PORTABLE, member, m, E, seeds, {"value": bgstate_obj(E, 77)}, state_set_variants(E, typed=False),
                             np_set, calls2=False)
        if name in ("seed_seq", "spawn"):
            # A legacy-seeded MT19937 (RandomState(seed)._bit_generator): no seed sequence, and spawning is refused.
            for sd in seeds[:2]:
                recv = Recv("rs_bitgen", inner=Recv("RandomState", None, sd))
                vals = {} if name == "seed_seq" else {"n_children": 2}
                fn = (lambda r, py: r.seed_seq) if name == "seed_seq" else (lambda r, py: r.spawn(py["n_children"]))
                emit_m(out, PORTABLE, f"MT19937.{name}", m, recv, vals, "legacy_mt", fn)


# ---- the engines' state classes ----------------------------------------------------------------------------------------

# Where each State member lives in NumPy's state dict: the path of keys, and the element dtype of an array field.
STATE_FIELDS = {
    "MT19937": {"key": (("state", "key"), np.uint32), "pos": (("state", "pos"), None)},
    "PCG64": {"state": (("state", "state"), None), "inc": (("state", "inc"), None),
              "has_uint32": (("has_uint32",), None), "uinteger": (("uinteger",), None)},
    "Philox": {"counter": (("state", "counter"), np.uint64), "key": (("state", "key"), np.uint64),
               "buffer": (("buffer",), np.uint64), "buffer_pos": (("buffer_pos",), None),
               "has_uint32": (("has_uint32",), None), "uinteger": (("uinteger",), None)},
    "SFC64": {"state": (("state", "state"), np.uint64), "has_uint32": (("has_uint32",), None),
              "uinteger": (("uinteger",), None)},
}
STATE_FIELDS["PCG64DXSM"] = STATE_FIELDS["PCG64"]

# Setter values per field (valid and edge values; a state object holds whatever it is given — the engine validates).
STATE_SET_VALUES = {
    "key": {"MT19937": [[(i * 1812433253 + 7) % 2 ** 32 for i in range(624)], [1, 2, 3], None],
            "Philox": [[11, 12], [1, 2, 3], None]},
    "pos": [5, 0, 624, 625, -1],
    "state": {"PCG64": [2 ** 100 + 1, 0, 2 ** 128 - 1], "PCG64DXSM": [2 ** 100 + 1, 0, 2 ** 128 - 1],
              "SFC64": [[1, 2, 3, 4], [9], None]},
    "inc": [2 ** 64 + 1, 2, 2 ** 128 - 1],
    "has_uint32": [1, 0, 2, -1],
    "uinteger": [123456, 0, 2 ** 32 - 1],
    "counter": [[1, 2, 3, 4], [5], None],
    "buffer": [[9, 8, 7, 6], [1, 2], None],
    "buffer_pos": [0, 3, 4, 5],
}


def state_field_get(d, path):
    for k in path:
        d = d[k]
    return d


def state_field_set(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def bgstate_receivers(E, seeds):
    return [Recv("bgstate", inner=Recv("BitGenerator", E, sd, prime=pr)) for sd in seeds[:2] for pr in ("none", "raw3")]


def make_state_class_family(E):
    def fam(out, surface, name, seeds):
        """ENGINE.State: getters read NumPy's state dict, setters write it (the object's canonical text after the call is
        the receiver state), constructors build one from parts (the parameterless one: unset arrays, zeros)."""
        member = f"{E}.state"
        fields = STATE_FIELDS[E]
        for m in overloads(surface, f"{E}.State", name):
            if name == "ctor":
                none = Recv("none")
                ptypes = [p["name"] for p in m["params"]]

                def build(r, py, E=E, ptypes=tuple(ptypes)):
                    d = bgstate_explicit(E, **{k: v for k, v in py.items() if v is not None}).py if ptypes else \
                        bgstate_explicit(E).py
                    return d
                if not ptypes:
                    emit_m(out, PORTABLE, member, m, none, {}, "empty", build, state=False)
                    continue
                full = {"MT19937": {"key": STATE_SET_VALUES["key"]["MT19937"][0], "pos": 5},
                        "PCG64": {"state": 2 ** 100 + 1, "inc": 2 ** 64 + 1, "has_uint32": 1, "uinteger": 99},
                        "PCG64DXSM": {"state": 2 ** 100 + 1, "inc": 2 ** 64 + 1, "has_uint32": 1, "uinteger": 99},
                        "Philox": {"counter": [1, 2, 3, 4], "key": [5, 6], "buffer": [7, 8, 9, 10], "buffer_pos": 1,
                                   "has_uint32": 1, "uinteger": 42},
                        "SFC64": {"state": [1, 2, 3, 4], "has_uint32": 1, "uinteger": 42}}[E]
                emit_m(out, PORTABLE, member, m, none, full, "full", build, state=False)
                full2 = {"MT19937": {"key": [(i * 7 + 3) % 2 ** 32 for i in range(624)], "pos": 624},
                         "PCG64": {"state": 7, "inc": 9, "has_uint32": 0, "uinteger": 0},
                         "PCG64DXSM": {"state": 7, "inc": 9, "has_uint32": 0, "uinteger": 0},
                         "Philox": {"counter": [9, 8, 7, 6], "key": [3, 4], "buffer": [1, 1, 1, 1], "buffer_pos": 4,
                                    "has_uint32": 0, "uinteger": 0},
                         "SFC64": {"state": [5, 6, 7, 8], "has_uint32": 0, "uinteger": 7}}[E]
                emit_m(out, PORTABLE, member, m, none, full2, "full2", build, state=False)
                optional = [p["name"] for p in m["params"] if p.get("optional")]
                if optional:
                    emit_m(out, PORTABLE, member, m, none, dict(full, **{k: OMIT for k in optional}), "defaults", build, state=False)
                arrays = [p["name"] for p in m["params"] if p["type"].endswith("[]")]
                for k, a in enumerate(arrays):
                    emit_m(out, PORTABLE, member, m, none, dict(full, **{a: None}), "null_array" if k == 0 else f"null_{a}",
                           build, state=False)
                continue
            if name == "bit_generator":
                for recv in bgstate_receivers(E, seeds):
                    emit_m(out, PORTABLE, member, m, recv, {}, "get", lambda r, py: r["bit_generator"])
                continue
            path, adt = fields[name]
            if m["kind"] == "get":
                for recv in bgstate_receivers(E, seeds):
                    emit_m(out, PORTABLE, member, m, recv, {}, "get",
                           lambda r, py, path=path, adt=adt: (np.asarray(state_field_get(r, path), dtype=adt)
                                                            if adt is not None else state_field_get(r, path)))
                continue
            vals = STATE_SET_VALUES[name]
            vals = vals[E] if isinstance(vals, dict) else vals

            def np_set(r, py, path=path, adt=adt):
                v = py["value"]
                state_field_set(r, path, np.array(v, dtype=adt) if (adt is not None and v is not None) else v)
            for i, v in enumerate(vals):
                for recv in bgstate_receivers(E, seeds)[:2]:
                    emit_m(out, PORTABLE, member, m, recv, {"value": v}, f"v{i}", np_set)
    return fam


for _E in ENGINES:
    for _n in ("ctor", "bit_generator", "key", "pos", "state", "inc", "has_uint32", "uinteger", "counter", "buffer",
               "buffer_pos"):
        FAMILIES[(f"{_E}.State", _n)] = make_state_class_family(_E)


@family("BitGeneratorState", "bit_generator")
def fam_bgstate_base(out, surface, name, seeds):
    """BitGeneratorState.bit_generator (the base property) on every engine's state."""
    for m in overloads(surface, "BitGeneratorState", name):
        for E in ENGINES:
            for recv in bgstate_receivers(E, seeds)[:2]:
                emit_m(out, PORTABLE, f"{E}.state", m, recv, {}, "get", lambda r, py: r["bit_generator"])


@family("MT19937.State", "op_Implicit")
def fam_mt_state_implicit(out, surface, name, seeds):
    """The legacy tuple converts to MT19937's state — NumPy's MT19937.state setter translating a tuple: key and pos kept,
    the Gaussian cache dropped; another algorithm's tuple is refused with that setter's text."""
    none = Recv("none")

    def conv(r, py):
        t = py["legacy"]
        if t[0] != "MT19937":
            raise ValueError("state is not a legacy MT19937 state")
        return {"bit_generator": "MT19937", "state": {"key": np.asarray(t[1], dtype=np.uint32), "pos": int(t[2])}}
    for m in overloads(surface, "MT19937.State", name):
        for sd in seeds[:3]:
            for pr in ("none", "gauss"):
                emit_m(out, PORTABLE, "MT19937.state", m, none, {"legacy": legacy_state_of(Recv("RandomState", None, sd, prime=pr))},
                       f"s{sd}_{pr}", conv, state=False)
        emit_m(out, PORTABLE, "MT19937.state", m, none, {"legacy": legacy_state_explicit(KEY624, 17, 1, 0.5)}, "explicit", conv, state=False)
        emit_m(out, PORTABLE, "MT19937.state", m, none, {"legacy": legacy_state_explicit(KEY624, 5, algorithm="PCG64")}, "algo", conv,
               state=False)


# ---- seed sequences ----------------------------------------------------------------------------------------------------

SS_RECEIVER_SPECS = [seedseq_spec(42), seedseq_spec([1, 2, 3], spawn_key=[5], pool_size=8),
                     seedseq_spec(2 ** 100, n_children_spawned=2), seedseq_spec(0, spawn_key=[2 ** 70, 1])]


def ss_receivers():
    return [Recv("seedseq", extra={"ss": spec}) for spec in SS_RECEIVER_SPECS]


@family("SeedSequence", "ctor")
def fam_seedseq_ctor(out, surface, name, seeds):
    """SeedSequence(entropy[, spawn_key, pool_size, n_children_spawned]) for every entropy form NumPy accepts or refuses:
    ints, lists, uint32 arrays, strings (NumPy's seed-string rule), floats and negatives (refused), None (OS entropy)."""
    for m in overloads(surface, "SeedSequence", name):
        emit_seedseq_ctor_overload(out, m, Recv("none"), state=False)


def emit_seedseq_ctor_overload(out, m, none, state):
    """Every case of one SeedSequence(...) overload — the constructor's (`none` = the no-receiver, no state) or the
    np.random.SeedSequence factory's (its RandomState receiver, whose state is recorded)."""
    mask = SEEDSEQ_MASK
    ps = m["params"]
    if not ps:
        emit_m(out, PORTABLE, "SeedSequence", m, none, {}, "entropy", lambda r, py: np.random.SeedSequence(), state=state,
               state_mask=mask)
        return
    pt = ps[0]["type"]
    if pt == "object":
        def np_fn(r, py):
            kw = {k: py[k] for k in ("spawn_key", "pool_size", "n_children_spawned") if k in py}
            return np.random.SeedSequence(py["entropy"], **kw)
        variants = [("int", {"entropy": value_obj(42)}), ("big", {"entropy": value_obj(2 ** 100)}),
                    ("list", {"entropy": value_obj([1, 2, 3])}), ("nested", {"entropy": value_obj([[1, 2], [3]])}),
                    ("uint32", {"entropy": value_obj(np.array([7, 8], dtype=np.uint32))}),
                    ("str_dec", {"entropy": value_obj("123")}), ("str_lead0", {"entropy": value_obj("012")}),
                    ("str_hex", {"entropy": value_obj("0x1f")}), ("str_bad", {"entropy": value_obj("abc")}),
                    ("float", {"entropy": value_obj(1.5)}), ("neg", {"entropy": value_obj(-1)}),
                    ("bool", {"entropy": value_obj(True)}), ("ndarray", {"entropy": np.array([4, 5], dtype=np.int64)}),
                    ("ndarray_0d", {"entropy": np.array(9, dtype=np.int64)}),
                    ("key", {"entropy": value_obj(42), "spawn_key": value_obj([1, 2])}),
                    ("key_big", {"entropy": value_obj(42), "spawn_key": value_obj([2 ** 70])}),
                    ("key_str", {"entropy": value_obj(42), "spawn_key": value_obj("12")}),
                    ("key_nested", {"entropy": value_obj(42), "spawn_key": value_obj([[1, 2]])}),
                    ("key_int", {"entropy": value_obj(42), "spawn_key": value_obj(5)}),
                    ("key_ndarray", {"entropy": value_obj(42), "spawn_key": np.array([3, 4], dtype=np.int64)}),
                    ("key_0d", {"entropy": value_obj(42), "spawn_key": np.array(3, dtype=np.int64)}),
                    ("key_neg", {"entropy": value_obj(42), "spawn_key": value_obj([-1])}),
                    ("pool8", {"entropy": value_obj(42), "pool_size": 8}), ("pool3", {"entropy": value_obj(42), "pool_size": 3}),
                    ("children", {"entropy": value_obj(42), "n_children_spawned": 5}),
                    ("children_max", {"entropy": value_obj(42), "n_children_spawned": 2 ** 32 - 1}),
                    ("all", {"entropy": value_obj([9, 9]), "spawn_key": value_obj([1]), "pool_size": 6, "n_children_spawned": 2}),
                    ("omit_all", {"entropy": value_obj(3), "spawn_key": OMIT, "pool_size": OMIT, "n_children_spawned": OMIT})]
        for tag, vals in variants:
            emit_m(out, PORTABLE, "SeedSequence", m, none, vals, tag, np_fn, state=state)
        emit_m(out, PORTABLE, "SeedSequence", m, none, {"entropy": None}, "none", np_fn, state=state, state_mask=mask)
        emit_m(out, PORTABLE, "SeedSequence", m, none, {"entropy": None, "spawn_key": value_obj([4])}, "none_key", np_fn,
               state=state, state_mask=mask)
        return
    vals = SEED_VALUES[pt]
    for v in vals:
        emit_m(out, PORTABLE, "SeedSequence", m, none, {"entropy": v}, f"e{json.dumps(v, separators=(',', ':'))}",
               lambda r, py, pt=pt: np.random.SeedSequence(seed_py(pt, py["entropy"])), state=state)
    if pt.endswith("[]"):
        emit_m(out, PORTABLE, "SeedSequence", m, none, {"entropy": None}, "null",
               lambda r, py: np.random.SeedSequence(None), state=state, state_mask=mask)


@family("SeedSequence", "entropy", "spawn_key", "pool_size", "n_children_spawned", "pool", "state", "ToString",
        "generate_state", "spawn")
def fam_seedseq_members(out, surface, name, seeds):
    """SeedSequence's getters (entropy/spawn_key/state by NumPy's repr), generate_state (both dtypes, NumPy's refusals),
    spawn (the counter advances: the receiver's text after the call), repr."""
    for m in overloads(surface, "SeedSequence", name):
        emit_ss_member(out, m, name, "SeedSequence", ss_receivers())


def emit_ss_member(out, m, name, owner, receivers):
    member = f"SeedSequence.{name}" if owner != "SeedlessSeedSequence" else f"SeedlessSeedSequence.{name}"
    for recv in receivers:
        if name == "generate_state":
            def np_fn(r, py):
                return r.generate_state(py["n_words"], **({"dtype": py["dtype"]} if py.get("dtype") is not None else {}))
            variants = [("n4", {"n_words": 4}), ("n0", {"n_words": 0}), ("n1", {"n_words": 1}), ("n1000", {"n_words": 1000}),
                        ("u64", {"n_words": 3, "dtype": "uint64"}), ("u32", {"n_words": 3, "dtype": "uint32"}),
                        ("dt_null", {"n_words": 3, "dtype": None}), ("omit_dtype", {"n_words": 3, "dtype": OMIT}),
                        ("dt_i32", {"n_words": 3, "dtype": "int32"}), ("dt_f64", {"n_words": 3, "dtype": "float64"}),
                        ("neg", {"n_words": -1})]
            for tag, vals in variants:
                emit_m(out, PORTABLE, member, m, recv, vals, tag, np_fn)
        elif name == "spawn":
            for tag, n in (("n3", 3), ("n0", 0), ("n1", 1), ("nneg", -1)):
                emit_m(out, PORTABLE, member, m, recv, {"n_children": n}, tag, lambda r, py: r.spawn(py["n_children"]))
            emit_m(out, PORTABLE, member, m, recv, {"n_children": 2}, "calls2", lambda r, py: r.spawn(py["n_children"]), calls=2)
        elif name == "ToString":
            emit_m(out, PORTABLE, member, m, recv, {}, "repr", lambda r, py: repr(r))
        else:
            emit_m(out, PORTABLE, member, m, recv, {}, "get", lambda r, py, name=name: getattr(r, name))


@family("SeedlessSeedSequence", "ctor", "generate_state", "spawn")
def fam_seedless(out, surface, name, seeds):
    """SeedlessSeedSequence: constructible, cannot generate state (NumPy's NotImplementedError), spawns itself."""
    for m in overloads(surface, "SeedlessSeedSequence", name):
        if name == "ctor":
            emit_m(out, PORTABLE, "SeedlessSeedSequence", m, Recv("none"), {}, "new", lambda r, py: SeedlessSeedSequence(), state=False)
            continue
        emit_ss_member(out, m, name, "SeedlessSeedSequence", [Recv("seedless")])


@family("ISeedSequence", "generate_state")
def fam_iseedseq(out, surface, name, seeds):
    """The interface member, dispatched to a SeedSequence and to a SeedlessSeedSequence."""
    for m in overloads(surface, "ISeedSequence", name):
        emit_ss_member(out, m, name, "SeedSequence", ss_receivers()[:2])
        emit_ss_member(out, m, name, "SeedlessSeedSequence", [Recv("seedless")])


@family("ISpawnableSeedSequence", "spawn")
def fam_ispawnable(out, surface, name, seeds):
    for m in overloads(surface, "ISpawnableSeedSequence", name):
        emit_ss_member(out, m, name, "SeedSequence", ss_receivers()[:2])
        emit_ss_member(out, m, name, "SeedlessSeedSequence", [Recv("seedless")])


# ---- default_rng -------------------------------------------------------------------------------------------------------

@family("NumPyRandom", "default_rng")
def fam_default_rng(out, surface, name, seeds):
    """np.random.default_rng(seed) for every seed form: Generator(PCG64(seed)), a bit generator wrapped as is, a Generator
    passed through, a RandomState's engine shared, seed sequences, arrays; None and null arrays draw OS entropy."""
    recv = Recv("RandomState", None, 0)
    mask = engine_mask("PCG64")
    for m in overloads(surface, "NumPyRandom", name):
        ps = m["params"]

        def one(tag, vals, fn=lambda r, py: np.random.default_rng(*py.values()), masked=False):
            # The receiver's state is recorded: default_rng must not draw from the RandomState it is called on (the
            # masked fields belong to the NEW generator's entropy; the receiver's own MT19937 key is not masked by them).
            emit_m(out, PORTABLE, "default_rng", m, recv, vals, tag, fn, state_mask=mask if masked else ())
        if not ps:
            one("entropy", {}, lambda r, py: np.random.default_rng(), masked=True)
            continue
        pn, pt = ps[0]["name"], ps[0]["type"]
        if pt in ("long", "ulong", "BigInteger", "int[]", "long[]", "uint[]"):
            vals = (list(seeds) if pt == "long" else []) + SEED_VALUES[pt]
            for v in dict.fromkeys(map(json.dumps, vals)):
                v = json.loads(v)
                one(f"s{json.dumps(v, separators=(',', ':'))}", {pn: v},
                    lambda r, py, pt=pt, pn=pn: np.random.default_rng(seed_py(pt, py[pn])))
            if pt.endswith("[]"):
                one("null", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        elif pt == "BitGenerator":
            for E in ENGINES:
                one(f"{E}", {pn: bitgen_obj(E, 7)})
                one(f"{E}_primed", {pn: bitgen_obj(E, 8, "raw3")})
            one("null", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        elif pt == "Generator":
            for E in ENGINES:
                one(f"{E}", {pn: generator_obj(E, 9, "u32")})
            one("null", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        elif pt == "ISeedSequence":
            for tag, v in seed_seqs():
                one(tag, {pn: v})
            one("null", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        elif pt == "NumPyRandom":
            one("legacy", {pn: randomstate_obj(Recv("RandomState", None, 11))})
            one("legacy_gauss", {pn: randomstate_obj(Recv("RandomState", None, 12, prime="gauss"))})
            for E in ENGINES:
                one(f"over_{E}", {pn: randomstate_obj(Recv("RandomState", E, 13))})
            one("null", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        elif pt == "NDArray":
            for tag, a in (("i64", np.array([1, 2, 3], dtype=np.int64)), ("zerod", np.array(5, dtype=np.int64)),
                           ("u32", np.array([7, 8], dtype=np.uint32)), ("empty", np.array([], dtype=np.int64)),
                           ("f64", np.array([1.5])), ("neg", np.array([-1], dtype=np.int64)),
                           ("strided", derived(np.arange(10, dtype=np.int64), lambda b: b[::3])),
                           ("d2", np.array([[1, 2], [3, 4]], dtype=np.int64))):
                one(tag, {pn: a})
            one("null", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        elif pt == "object":
            for tag, v in (("int", value_obj(42)), ("big", value_obj(2 ** 100)), ("list", value_obj([1, 2, 3])),
                           ("str", value_obj("123")), ("float", value_obj(1.5)), ("neg", value_obj(-1)),
                           ("seedseq", seedseq_obj(5, spawn_key=[1])), ("seedless", seedless_obj()),
                           ("ndarray", np.array([4, 5], dtype=np.int64))):
                one(tag, {pn: v})
            # The dynamic dispatch on every engine: a bit generator wrapped, a Generator passed through, a
            # RandomState's engine shared (the legacy-seeded one too).
            for E in ENGINES:
                one(f"bitgen_{E}", {pn: bitgen_obj(E, 3)})
                one(f"generator_{E}", {pn: generator_obj(E, 4)})
                one(f"randomstate_{E}", {pn: randomstate_obj(Recv("RandomState", E, 6))})
            one("randomstate_legacy", {pn: randomstate_obj(Recv("RandomState", None, 6))})
            one("none", {pn: None}, lambda r, py: np.random.default_rng(None), masked=True)
        else:
            raise ValueError(f"no default_rng family for {m['sig']}")


# ---- the state objects: NativeRandomState (the legacy tuple) and NumPyRandom.State (the dict) ------------------------

NATIVE_FIELDS = {"Algorithm": 0, "Key": 1, "Pos": 2, "HasGauss": 3, "CachedGaussian": 4}


def state_receivers(seeds):
    """Legacy tuples/dicts of primed and fresh receivers (every engine for the dict; MT19937 for the tuple)."""
    tuples = [Recv("legacy_tuple", inner=Recv("RandomState", None, s, prime=p)) for s in seeds[:3] for p in ("none", "gauss")]
    tuples.append(Recv("legacy_tuple", inner=Recv("RandomState", "MT19937", seeds[0], prime="raw3")))
    dicts = [Recv("rs_dict", inner=Recv("RandomState", e, s, prime=p)) for e in [None] + ENGINES for s in seeds[:2]
             for p in ("none", "gauss")]
    return tuples, dicts


@family("NativeRandomState", "Algorithm", "Key", "Pos", "HasGauss", "CachedGaussian", "ctor")
def fam_native_state(out, surface, name, seeds):
    tuples, _ = state_receivers(seeds)
    for m in overloads(surface, "NativeRandomState", name):
        if name == "ctor":
            if m["params"][0]["type"] == "byte[]":
                exempt(m["sig"], "obsolete NumSharp constructor that always throws; no NumPy counterpart (plan §7)")
                continue
            none = Recv("none")

            def build(r, py):
                return ("MT19937", None if py["key"] is None else np.array(py["key"], dtype=np.uint32), py["pos"],
                        py.get("hasGauss", 0), py.get("cachedGaussian", 0.0))
            for tag, vals in [("full", {"key": KEY624, "pos": 5, "hasGauss": 1, "cachedGaussian": -0.5}),
                              ("defaults", {"key": KEY624, "pos": 624, "hasGauss": OMIT, "cachedGaussian": OMIT}),
                              ("omit_gauss", {"key": KEY624, "pos": 0, "hasGauss": 1, "cachedGaussian": OMIT}),
                              ("short", {"key": [1, 2, 3], "pos": 1}), ("nullkey", {"key": None, "pos": 3})]:
                emit_m(out, PORTABLE, "RandomState.get_state", m, none, vals, tag, build, state=False)
            continue
        idx = NATIVE_FIELDS[name]
        for recv in tuples:
            emit_m(out, PORTABLE, "RandomState.get_state", m, recv, {}, "field", lambda r, py, idx=idx: r[idx])


@family("NumPyRandom.State", "bit_generator", "gauss", "has_gauss", "state", "ctor")
def fam_rs_state_obj(out, surface, name, seeds):
    """NumPyRandom.State mirrors the dict get_state(legacy=False) returns: getters read its keys, setters write them
    (the object's canonical text after the call is the receiver state), constructors build one from parts."""
    _, dicts = state_receivers(seeds)
    for m in overloads(surface, "NumPyRandom.State", name):
        if name == "ctor":
            none = Recv("none")
            if not m["params"]:
                emit_m(out, PORTABLE, "RandomState.get_state", m, none, {}, "empty",
                       lambda r, py: {"state": None, "has_gauss": 0, "gauss": 0.0}, state=False)
                continue

            def build(r, py):
                d = dict(py["state"]) if py["state"] is not None else {"state": None}
                d["has_gauss"], d["gauss"] = py.get("has_gauss", 0), py.get("gauss", 0.0)
                return d
            for tag, vals in ([(f"{e}", {"state": bgstate_obj(e, 3, "raw3"), "has_gauss": 1, "gauss": 0.125}) for e in ENGINES]
                              + [("defaults", {"state": bgstate_obj("MT19937", 8), "has_gauss": OMIT, "gauss": OMIT}),
                                 ("omit_gauss", {"state": bgstate_obj("PCG64", 8), "has_gauss": 1, "gauss": OMIT}),
                                 ("nullstate", {"state": None, "has_gauss": 0, "gauss": 0.0})]):
                emit_m(out, PORTABLE, "RandomState.get_state", m, none, vals, tag, build, state=False)
            continue
        kind = m["kind"]
        for recv in dicts:
            if kind == "get":
                if name == "state":
                    fn = (lambda r, py: {k: v for k, v in r.items() if k not in ("has_gauss", "gauss")})
                else:
                    fn = (lambda r, py, name=name: r[name])
                emit_m(out, PORTABLE, "RandomState.get_state", m, recv, {}, "get", fn)
            else:
                if name == "state":
                    vals_list = [("same_engine", {"value": bgstate_obj(recv.inner.engine or "MT19937", 55, "raw3")}),
                                 ("other_engine", {"value": bgstate_obj("SFC64" if recv.inner.engine != "SFC64" else "PCG64", 56)}),
                                 ("null", {"value": None})]

                    def fn(r, py):
                        for k in [k for k in r if k not in ("has_gauss", "gauss")]:
                            del r[k]
                        if py["value"] is not None:
                            r.update(py["value"])
                        else:
                            r["state"] = None
                        return None
                elif name == "gauss":
                    vals_list = [("v", {"value": 1.5}), ("neg0", {"value": -0.0}), ("nan", {"value": NAN})]

                    def fn(r, py):
                        r["gauss"] = py["value"]
                        return None
                else:
                    vals_list = [("v1", {"value": 1}), ("v0", {"value": 0}), ("v7", {"value": 7})]

                    def fn(r, py):
                        r["has_gauss"] = py["value"]
                        return None
                for tag, vals in vals_list:
                    emit_m(out, PORTABLE, "RandomState.get_state", m, recv, vals, tag, fn)


# ======================================================================================================================
# Driver
# ======================================================================================================================

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


# The legacy families whose Windows answers can depend on C long being 32-bit (plan §6.5): their cases are regenerated
# under Linux NumPy and merged by `merge_lp64`. Extended as families are added.
LP64_FAMILIES = {("NumPyRandom", name) for name in LEGACY_INT64_CAST}
INT32_MIN, INT32_MAX = -2 ** 31, 2 ** 31 - 1


def windows_to_wsl(path):
    """C:/x/y (or with backslashes) -> /mnt/c/x/y, the WSL view of a Windows path."""
    path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(path)
    return "/mnt/" + drive.rstrip(":").lower() + rest.replace("\\", "/")


def run_lp64(seeds):
    """Runs this script under Linux NumPy (WSL, `~/np242/bin/python`, or the NUMSHARP_LP64_PYTHON command) in
    --lp64-only mode and returns its rows. Raises SystemExit when no LP64 NumPy 2.4.2 is available: the committed corpus
    needs the LP64 answers, so a generation without them must not silently drop the cases."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="random_api_lp64_")
    try:
        script = windows_to_wsl(os.path.abspath(__file__))
        cmd_prefix = os.environ.get("NUMSHARP_LP64_PYTHON", "~/np242/bin/python")
        seeds_arg = ",".join(str(x) for x in seeds)
        cmd = ["wsl", "-e", "bash", "-lc",
               f"{cmd_prefix} '{script}' --lp64-only --seeds {seeds_arg} --out '{windows_to_wsl(tmp)}'"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit("the LP64 sub-run failed (needs WSL with Linux NumPy 2.4.2 at ~/np242/bin/python, or "
                             f"NUMSHARP_LP64_PYTHON):\n{proc.stdout}\n{proc.stderr}")
        rows = []
        with open(os.path.join(tmp, "lp64_rows.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def lp64_out_of_int32(row):
    """Whether an LP64 row's answer only a 64-bit C long can give: an integer result element outside the int32 range."""
    if row.get("expects_throw"):
        return False
    res = row["expected"]["result"]
    if res["k"] == "int":
        return not (INT32_MIN <= int(res["v"]) <= INT32_MAX)
    if res["k"] == "array" and res["dtype"] == "int64":
        a = np.frombuffer(bytes.fromhex(res["hex"]), dtype=np.int64)
        return bool(((a < INT32_MIN) | (a > INT32_MAX)).any())
    return False


def merge_lp64(out, lp64_rows):
    """Moves the cases whose answer depends on C long being 64-bit into the LP64 tier, with Linux NumPy's answer.

    A case is LP64-dependent when the LP64 answer holds an integer outside int32 (Windows truncates, overflows to INT32_MIN
    or rejects the candidate against a 32-bit LONG_MAX, shifting the stream), or when the platforms disagree on raising
    (one raises and the other draws, or both raise with different errors: a bound derived from LONG_MAX, such as
    poisson's lam maximum). Anything else keeps the Windows row: a remaining value difference would be the platforms'
    libm, and the host tier is win-amd64-authored by design (those are printed, so an LP64 dependence the rule misses is
    seen rather than silently kept). Rows only the LP64 run has (Windows stopped a variant at its first error, Linux did
    not) join the LP64 tier when they qualify the same way.

    :param out: the Windows run's cases (mutated: moved rows leave their Windows tier).
    :param lp64_rows: the Linux run's rows for the same families, with the same stable ids.
    :returns: the number of rows moved/added.
    :raises SystemExit: when an id names different cases on the two platforms (the ids stopped being stable, so the
        merge would pair unrelated answers).
    """
    def case_identity(row):
        # `widened` records that the result arrived as int32 (true only where C long is 32-bit), so it is the one
        # params field allowed to differ between the platforms' versions of the same case.
        return {k: v for k, v in row["params"].items() if k != "widened"}, row["operands"]

    by_id = {}
    for tier in TIERS:
        for row in out.rows[tier]:
            if row["id"] in out.lp64_ids:
                by_id[row["id"]] = (tier, row)
    moved = 0
    libm_only = []
    for lrow in lp64_rows:
        win = by_id.get(lrow["id"])
        if win is not None and case_identity(win[1]) != case_identity(lrow):
            raise SystemExit(f"LP64 merge: id {lrow['id']} names different cases on Windows and Linux")
        l_raised = lrow.get("expects_throw", False)
        w_raised = True if win is None else win[1].get("expects_throw", False)
        if lp64_out_of_int32(lrow) or l_raised != w_raised:
            needed = True
        elif l_raised:
            needed = lrow["error"] != win[1]["error"]
        else:
            needed = False
            if win is not None and win[1]["expected"] != lrow["expected"]:
                if win[0] == PORTABLE:
                    # A portable row involves no libm, so a platform difference there IS the C long: e.g. randint's
                    # array bounds take the 32-bit path on Windows and the 64-bit one on Linux, whose element-wise
                    # conversion of float bounds scrambles a non-C layout — values in range, no error, yet LP64-only.
                    needed = True
                else:
                    libm_only.append(lrow["id"])
        if not needed:
            continue
        if win is not None:
            out.rows[win[0]].remove(win[1])
        out.rows["random_api_lp64"].append(lrow)
        moved += 1
    if libm_only:
        print(f"LP64 merge: {len(libm_only)} cases differ between Windows and Linux NumPy without an LP64 signature "
              "(kept with the Windows answer; libm):")
        for ident in libm_only[:40]:
            print(f"  {ident}")
    # LP64-flagged Windows rows the Linux run does not have: the Linux sweep stopped that variant at an earlier receiver
    # because NumPy RAISED there (validation precedes the draws and depends on neither engine nor seed — the sweep's own
    # one-error rule), so on the platform NumSharp models these receivers raise too and the Windows answers are not
    # NumSharp's. E.g. randint(low=[nan, 1.0], high=5): Windows' 32-bit path draws, Linux's 64-bit path raises int(nan).
    linux_ids = {r["id"] for r in lp64_rows}
    linux_raised = {(r["id"].split("/")[1], r["id"].split("/")[-1]) for r in lp64_rows if r.get("expects_throw")}
    dropped = []
    for ident, (tier, row) in by_id.items():
        if ident in linux_ids:
            continue
        parts = ident.split("/")
        if (parts[1], parts[-1]) in linux_raised and row in out.rows[tier]:
            out.rows[tier].remove(row)
            dropped.append(ident)
    if dropped:
        print(f"LP64 merge: {len(dropped)} Windows cases dropped (Linux NumPy raised on an earlier receiver of the same "
              "variant, so their C-long answer is not NumSharp's):")
        for ident in dropped[:40]:
            print(f"  {ident}")
    return moved


# ---- the NumPy signature table (parameter-name gate) --------------------------------------------------------------------

SIGNATURES_PATH = os.path.join(HERE, "random_numpy_signatures.json")

# The NumPy object behind each C# declaring type, and the members whose counterpart is elsewhere (module functions,
# the RandomState constructor behind the np.random factory).
NP_OWNERS = {
    "NumPyRandom": ("numpy.random.RandomState", lambda: np.random.RandomState),
    "Generator": ("numpy.random.Generator", lambda: np.random.Generator),
    "BitGenerator": ("numpy.random.BitGenerator", lambda: np.random.BitGenerator),
    "MT19937": ("numpy.random.MT19937", lambda: np.random.MT19937),
    "PCG64": ("numpy.random.PCG64", lambda: np.random.PCG64),
    "PCG64DXSM": ("numpy.random.PCG64DXSM", lambda: np.random.PCG64DXSM),
    "Philox": ("numpy.random.Philox", lambda: np.random.Philox),
    "SFC64": ("numpy.random.SFC64", lambda: np.random.SFC64),
    "SeedSequence": ("numpy.random.SeedSequence", lambda: np.random.SeedSequence),
    "SeedlessSeedSequence": ("numpy.random.bit_generator.SeedlessSeedSequence", lambda: SeedlessSeedSequence),
    "ISeedSequence": ("numpy.random.bit_generator.ISeedSequence", lambda: np.random.bit_generator.ISeedSequence),
    "ISpawnableSeedSequence": ("numpy.random.bit_generator.ISpawnableSeedSequence",
                               lambda: np.random.bit_generator.ISpawnableSeedSequence),
}
NP_SPECIAL = {
    ("NumPyRandom", "default_rng"): ("numpy.random.default_rng", lambda: np.random.default_rng),
    ("NumPyRandom", "RandomState"): ("numpy.random.RandomState", lambda: np.random.RandomState),
    ("NumPyRandom", "ranf"): ("numpy.random.ranf", lambda: np.random.ranf),
    ("NumPyRandom", "sample"): ("numpy.random.sample", lambda: np.random.sample),
    ("NumPyRandom", "get_bit_generator"): ("numpy.random.get_bit_generator", lambda: np.random.get_bit_generator),
    ("NumPyRandom", "set_bit_generator"): ("numpy.random.set_bit_generator", lambda: np.random.set_bit_generator),
    # numpy.random's classes through the module (np.random.PCG64(42), np.random.Generator(...)): the class signatures.
    ("NumPyRandom", "MT19937"): ("numpy.random.MT19937", lambda: np.random.MT19937),
    ("NumPyRandom", "PCG64"): ("numpy.random.PCG64", lambda: np.random.PCG64),
    ("NumPyRandom", "PCG64DXSM"): ("numpy.random.PCG64DXSM", lambda: np.random.PCG64DXSM),
    ("NumPyRandom", "Philox"): ("numpy.random.Philox", lambda: np.random.Philox),
    ("NumPyRandom", "SFC64"): ("numpy.random.SFC64", lambda: np.random.SFC64),
    ("NumPyRandom", "SeedSequence"): ("numpy.random.SeedSequence", lambda: np.random.SeedSequence),
    ("NumPyRandom", "Generator"): ("numpy.random.Generator", lambda: np.random.Generator),
}


def numpy_parameter_names(obj):
    """NumPy's parameter names of a callable, in order (`self` dropped), from inspect.signature or — for Cython
    methods without one — the first docstring line (`name(a, b=None, *, c)`); None when neither is readable. A `*args`
    parameter is reported as the single name `*`, a `**kwargs` one as `**`."""
    import inspect
    try:
        params = list(inspect.signature(obj).parameters.values())
    except (ValueError, TypeError):
        doc = ((getattr(obj, "__doc__", "") or "").strip().splitlines() or [""])[0]
        mm = re.match(r"^\w+\((.*)\)", doc)
        if not mm:
            return None
        names = []
        for part in mm.group(1).split(","):
            part = part.strip().strip("[]").strip()
            if not part or part in ("*", "/"):
                continue
            names.append("**" if part.startswith("**") else "*" if part.startswith("*") else part.split("=")[0].strip())
        return names
    out = []
    for prm in params:
        if prm.name in ("self", "cls"):
            continue
        out.append("*" if prm.kind == prm.VAR_POSITIONAL else "**" if prm.kind == prm.VAR_KEYWORD else prm.name)
    return out


def write_numpy_signatures(surface):
    """Writes test/oracle/random_numpy_signatures.json: for every C# method/constructor name of the random world whose
    NumPy counterpart exists, NumPy's qualified name and parameter names in order. The C# parameter-name gate checks
    each overload's parameters against it (names NumPy has, in NumPy's order; `*` accepts any name)."""
    table = {}
    for m in surface["members"]:
        if m["kind"] not in ("method", "ctor"):
            continue
        key = f"{m['type']}.{m['name']}"
        if key in table:
            continue
        if (m["type"], m["name"]) in NP_SPECIAL:
            qual, get = NP_SPECIAL[(m["type"], m["name"])]
            obj = get()
        elif m["type"] in NP_OWNERS:
            qual, get = NP_OWNERS[m["type"]]
            owner = get()
            if m["kind"] == "ctor":
                obj = owner
            else:
                obj = getattr(owner, m["name"], None)
                qual = f"{qual}.{m['name']}"
        else:
            continue
        if obj is None:
            continue
        names = numpy_parameter_names(obj)
        if names is None:
            raise SystemExit(f"no readable NumPy signature for {qual} (C# {key})")
        table[key] = {"numpy": qual, "params": names}
    with open(SIGNATURES_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"schema": 1, "numpy": np.__version__,
                             "note": "NumPy's parameter names per random-API member (gen_random_oracle.py); read by "
                                     "RandomApiCoverageTests' parameter-name gate.",
                             "members": dict(sorted(table.items()))}, indent=1) + "\n")
    print(f"random_numpy_signatures.json: {len(table)} members")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", help="comma-separated seeds (soak mode); default: the 10 fixed seeds")
    ap.add_argument("--out", help="output directory (default: the committed corpus directory)")
    ap.add_argument("--only", help="comma-separated 'Type.member' filters (development)")
    ap.add_argument("--partial", action="store_true", help="do not require every inventory member to be claimed")
    ap.add_argument("--lp64-only", action="store_true", help="(Linux NumPy) write only the LP64-eligible rows")
    ap.add_argument("--no-lp64", action="store_true", help="skip the LP64 sub-run (development only; the corpus then "
                                                            "keeps Windows answers that may depend on a 32-bit long)")
    ap.add_argument("--lp64-rows", help="the rows of an --lp64-only run made elsewhere (a Linux CI job) with the SAME "
                                        "seeds, merged instead of running the WSL sub-run")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else FIXED_SEEDS
    # Every family runs the legacy RandomState(int) on these seeds, whose domain is [0, 2**32); a repeated seed would
    # generate the same cases twice (their ids would collide into `#k` duplicates).
    bad = [x for x in seeds if not 0 <= x < 2 ** 32]
    if bad or len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds must be distinct integers in [0, 2**32); got {seeds}")
    out_dir = args.out or CORPUS_DIR
    # A soak / scratch directory is created on demand (a CI job passes a fresh path).
    os.makedirs(out_dir, exist_ok=True)
    surface = load_surface()
    only = set(args.only.split(",")) if args.only else None
    if args.lp64_only and np.dtype(np.long).itemsize != 8:
        raise SystemExit("--lp64-only needs a NumPy whose C long is 64-bit (Linux/macOS)")

    claimed = set()
    out = Case()
    for key in sorted(FAMILIES):
        type_name, name = key
        if only and f"{type_name}.{name}" not in only:
            continue
        if args.lp64_only and key not in LP64_FAMILIES:
            continue
        FAMILIES[key](out, surface, name, seeds)
        claimed.update(m["sig"] for m in overloads(surface, type_name, name))
    claimed.update(EXEMPT)

    if args.lp64_only:
        rows = [row for tier in TIERS for row in out.rows[tier] if row["id"] in out.lp64_ids]
        write_jsonl(os.path.join(out_dir, "lp64_rows.jsonl"), rows)
        print(f"lp64_rows.jsonl: {len(rows)} rows")
        return

    lp64_source = None
    if out.lp64_ids and not args.no_lp64:
        if args.lp64_rows:
            # A Linux job's --lp64-only output for the same seeds (the ids are content-derived, so the merge pairs them;
            # a row naming a different case than the Windows one aborts the merge).
            with open(args.lp64_rows, encoding="utf-8") as fh:
                lp64_rows = [json.loads(line) for line in fh if line.strip()]
            lp64_source = os.path.abspath(args.lp64_rows)
        else:
            lp64_rows = run_lp64(seeds)
            lp64_source = "wsl"
        moved = merge_lp64(out, lp64_rows)
        print(f"LP64 merge: {moved} cases take Linux NumPy's answer (C long = 64 bits)")

    if not only and not args.out:
        write_numpy_signatures(surface)
    missing = [m["sig"] for m in surface["members"] if m["sig"] not in claimed]
    if missing and not (args.partial or only):
        raise SystemExit(f"{len(missing)} inventory members are neither generated nor exempt:\n  " + "\n  ".join(missing))
    for t in TIERS:
        path = os.path.join(out_dir, f"{t}.jsonl")
        if not out.rows[t]:
            # An empty tier is never written (a stale file from an earlier run is removed): the replay names every
            # tier it expects, and an empty committed file would claim coverage that does not exist.
            if os.path.exists(path):
                os.remove(path)
            print(f"{t}.jsonl: (empty, not written)")
            continue
        write_jsonl(path, out.rows[t])
        print(f"{t}.jsonl: {len(out.rows[t])} cases")
    if missing:
        print(f"(partial: {len(missing)} members unclaimed)")
    if args.out:
        # A soak (or scratch) corpus records what produced it, for the replay's report and the uploaded evidence.
        manifest = {"seeds": seeds, "numpy": np.__version__, "python": sys.version.split()[0],
                    "platform": sys.platform, "lp64": lp64_source,
                    "tiers": {f"{t}.jsonl": len(out.rows[t]) for t in TIERS}, "partial": bool(missing)}
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
