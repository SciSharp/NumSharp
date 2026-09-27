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
                          or base.endswith("[]") or base.startswith("NDArray")):
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
    NativeRandomState struct), `rs_dict` its `get_state(legacy=False)` dict (C#: NumPyRandom.State). `none` is for
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
        """The receiver's part of a case id (stable: kind, engine, seed, priming, and the inner receiver)."""
        t = f"{self.kind}:{self.engine or '-'}:s{self.seed}:{self.prime}"
        return f"{t}({self.inner.tag()})" if self.inner is not None else t

    def build(self):
        """The NumPy receiver, primed."""
        if self.kind == "none":
            return None
        if self.kind == "legacy_tuple":
            return self.inner.build().get_state()
        if self.kind == "rs_dict":
            return self.inner.build().get_state(legacy=False)
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


def bitgen_state_text(st):
    """Canonical text of a bit generator state dict (+ has_gauss/gauss when it is a RandomState's)."""
    name = st["bit_generator"]
    core = st["state"]
    if name == "MT19937":
        key = np.asarray(core["key"], dtype=np.uint32)
        text = f"MT19937|pos={int(core['pos'])}|key={hashlib.sha256(key.tobytes()).hexdigest()}"
    elif name in ("PCG64", "PCG64DXSM"):
        text = (f"{name}|state={int(core['state'])}|inc={int(core['inc'])}"
                f"|has_uint32={int(st['has_uint32'])}|uinteger={int(st['uinteger'])}")
    elif name == "Philox":
        text = ("Philox|counter=" + ",".join(str(int(x)) for x in core["counter"]) +
                "|key=" + ",".join(str(int(x)) for x in core["key"]) +
                "|buffer=" + ",".join(str(int(x)) for x in st["buffer"]) +
                f"|buffer_pos={int(st['buffer_pos'])}|has_uint32={int(st['has_uint32'])}|uinteger={int(st['uinteger'])}")
    elif name == "SFC64":
        text = ("SFC64|state=" + ",".join(str(int(x)) for x in core["state"]) +
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
        return rsdict_text(r)
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
        return {k: (mask_text(v, fields) if k in ("state", "v") and isinstance(v, str) else mask_obs(v, fields))
                for k, v in o.items()}
    if isinstance(o, list):
        return [mask_obs(x, fields) for x in o]
    return o


def obs(v, ctype):
    """The canonical observation of a NumPy result against the C# return type `ctype` of the overload under test."""
    if ctype == "void":
        return {"k": "none"}
    if ctype in ("BitGenerator",) + tuple(ENGINES):
        return {"k": "bitgen", "type": type(v).__name__, "state": bitgen_state_text(v.state)}
    if ctype == "NumPyRandom":
        return {"k": "rs", "str": str(v), "state": bitgen_state_text(v.get_state(legacy=False))}
    if ctype == "Generator":
        return {"k": "gen", "str": str(v), "state": bitgen_state_text(v.bit_generator.state)}
    if ctype in ("NativeRandomState", "NumPyRandom.State", "object") and isinstance(v, (tuple, dict)):
        # By the VALUE: get_state() answers a non-MT19937 RandomState with the dict (and a warning) although the C#
        # overload is typed as the tuple — recorded as NumPy's answer; the divergence is classified on the C# side.
        if isinstance(v, tuple):
            return {"k": "legacy_state", "v": legacy_tuple_text(v)}
        return {"k": "rsdict", "v": rsdict_text(v)}
    if ctype == "BitGeneratorState":
        return {"k": "bgstate", "v": bitgen_state_text(v)}
    if ctype.endswith("[]") and ctype[:-2] in ("Generator", "BitGenerator", "SeedSequence", "NumPyRandom") + tuple(ENGINES):
        return {"k": "seq", "items": [obs(x, ctype[:-2]) for x in v]}
    if ctype in ("uint[]", "ulong[]", "long[]", "int[]"):
        dt = {"uint[]": np.uint32, "ulong[]": np.uint64, "long[]": np.int64, "int[]": np.int32}[ctype]
        return arr_obs(np.asarray(v, dtype=dt))
    if ctype.startswith("NDArray") or ctype == "T":
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
        return [[3], [], [2, 3], [0]]
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
    "geometric": dict(params=["p"], defaults=[NODEFAULT], base=[0.35], tier=HOST,
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
    "logseries": dict(params=["p"], defaults=[NODEFAULT], base=[0.6], tier=HOST,
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
    step = 1 if first in ints else 1.0
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
        f32[first] = np.array([b0, b0 + 1.0, b0 + 2.0], dtype=np.float32)
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
        return [("base", {pn: [3]}), ("empty", {pn: []}), ("d2x3", {pn: [2, 3]}), ("zero", {pn: [0]}),
                ("d2x0x3", {pn: [2, 0, 3]}), ("d1", {pn: [1]}), ("neg", {pn: [-1]}), ("neg2", {pn: [3, -2]})]
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
                return getattr(r, name)(*py.get(pn, []))
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
        unsigned = m["params"][0]["type"] == "ulong"

        def np_fn(r, py):
            kw = {}
            if py.get("high") is not None:
                kw["high"] = py["high"]
            if "size" in py:
                kw["size"] = py["size"]
            if py.get("dtype") is not None:
                kw["dtype"] = py["dtype"]
            return r.randint(py["low"], **kw)
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
                emit_m(out, PORTABLE, "RandomState.set_state", m, recv,
                       {"state": rs_dict_of(Recv("RandomState", recv.engine, 23, prime="raw3"))}, tag + "_same",
                       lambda r, py: r.set_state(py["state"]))


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
            emit_m(out, PORTABLE, "RandomState.RandomState", m, recv, values, tag, fn, state=False, state_mask=mask)
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
                    one(f"{e}_s{s}", {"bit_generator": bitgen_obj(e, s)}, lambda r, py: np.random.RandomState(py["bit_generator"]))
            one("primed", {"bit_generator": bitgen_obj("SFC64", 3, "raw3")},
                lambda r, py: np.random.RandomState(py["bit_generator"]))
            one("null", {"bit_generator": None}, lambda r, py: np.random.RandomState(None), mask=("key",))
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
                                                     lambda b: b[::2].reshape(2, 2)))),
              ("null_mean", dict(base, mean=None)), ("null_cov", dict(base, cov=None))]
    if "check_valid" in pn:
        v += [("cv_raise_bad", dict(base, cov=bad, check_valid="raise")), ("cv_ignore_bad", dict(base, cov=bad, check_valid="ignore")),
              ("cv_warn_bad", dict(base, cov=bad, check_valid="warn")), ("cv_raise_ok", dict(base, check_valid="raise")),
              ("cv_bogus", dict(base, check_valid="bogus")), ("omit_cv", dict(base, cov=bad, check_valid=OMIT))]
    if "tol" in pn:
        v += [("tol_tiny", dict(base, cov=sing, check_valid="raise", tol=1e-300)), ("tol_big", dict(base, cov=bad, check_valid="raise", tol=10.0)),
              ("omit_tol", dict(base, tol=OMIT))]
    if "method" in pn:
        v += [("m_eigh", dict(base, method="eigh")), ("m_cholesky", dict(base, method="cholesky")),
              ("m_cholesky_singular", dict(base, cov=sing, method="cholesky")), ("m_eigh_bad", dict(base, cov=bad, method="eigh")),
              ("m_bogus", dict(base, method="bogus")), ("omit_method", dict(base, method=OMIT))]
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
                         ("omit_method", {"size": (5,), "method": OMIT})]
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
                     ("dt_null", dict(base, dtype=None))]
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


@family("Generator", "integers")
def fam_gen_integers(out, surface, name, seeds):
    """Generator.integers(low, high, size, dtype, endpoint): the one-argument form, endpoint on and off, every dtype
    at its full range and just past it, NumPy's rejections (default dtype int64 — no C long involved)."""
    for m in overloads(surface, "Generator", name):
        unsigned = m["params"][0]["type"] == "ulong"

        def np_fn(r, py):
            kw = {}
            if py.get("high") is not None:
                kw["high"] = py["high"]
            kw.update(gen_kw(py, ("size", "dtype", "endpoint")))
            return r.integers(py["low"], **kw)
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
                variants += [("omit_method", dict(base, method=OMIT)), ("bogus", dict(base, method="bogus"))]
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
    none = Recv("none")
    for m in overloads(surface, "Generator", name):
        for e in ENGINES:
            for sd in seeds[:3]:
                emit_m(out, PORTABLE, "Generator.Generator", m, none, {"bit_generator": bitgen_obj(e, sd)}, f"{e}_s{sd}",
                       lambda r, py: np.random.Generator(py["bit_generator"]), state=False)
            emit_m(out, PORTABLE, "Generator.Generator", m, none, {"bit_generator": bitgen_obj(e, 5, "raw3")}, f"{e}_primed",
                   lambda r, py: np.random.Generator(py["bit_generator"]), state=False)
        emit_m(out, PORTABLE, "Generator.Generator", m, none, {"bit_generator": None}, "null",
               lambda r, py: np.random.Generator(py["bit_generator"]), state=False)


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
    return moved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", help="comma-separated seeds (soak mode); default: the 10 fixed seeds")
    ap.add_argument("--out", help="output directory (default: the committed corpus directory)")
    ap.add_argument("--only", help="comma-separated 'Type.member' filters (development)")
    ap.add_argument("--partial", action="store_true", help="do not require every inventory member to be claimed")
    ap.add_argument("--lp64-only", action="store_true", help="(Linux NumPy) write only the LP64-eligible rows")
    ap.add_argument("--no-lp64", action="store_true", help="skip the LP64 sub-run (development only; the corpus then "
                                                            "keeps Windows answers that may depend on a 32-bit long)")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else FIXED_SEEDS
    out_dir = args.out or CORPUS_DIR
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

    if out.lp64_ids and not args.no_lp64:
        moved = merge_lp64(out, run_lp64(seeds))
        print(f"LP64 merge: {moved} cases take Linux NumPy's answer (C long = 64 bits)")

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


if __name__ == "__main__":
    main()
