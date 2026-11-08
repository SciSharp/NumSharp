"""Compare NumSharp's np.* / NDArray overloads (api_dump.json) with NumPy 2.4.2's signatures.

Rules (what a ported NumPy call needs to bind in C#):
  - positional-only parameters: only their SLOT counts (they cannot be passed by name);
  - positional-or-keyword parameters: their NAMES must exist, in NumPy's relative ORDER, in one overload
    (a positional port passes them in that order, a keyword port by name);
  - keyword-only parameters: their NAMES must exist in that same overload, in any position.
  - ndarray methods whose NumPy signature ends in **kwargs (max, sum, std, ...) take the matching np.*
    function's remaining keywords (keepdims, initial, where, mean, correction) through those kwargs.
Excluded everywhere: like (array-function protocol), subok (no ndarray subclasses). Excluded on ufuncs:
casting / order / signature, which NumSharp models nowhere in its ufunc surface (CLAUDE.md).

Output: one row per shared name with a gap: missing names, order problems, NumPy signature.
"""
import inspect
import json
import re
import sys

import numpy as np

dump = json.load(open(sys.argv[1], encoding="utf-8"))
ALWAYS_EXCLUDED = {"like", "subok"}
UFUNC_EXCLUDED = {"casting", "order", "signature"}
P = inspect.Parameter


def numpy_signature(obj, name):
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


def effective_params(owner_name, name, obj):
    sig = numpy_signature(obj, name)
    if sig is None:
        return None, None, None
    is_ufunc = isinstance(obj, np.ufunc)
    excluded = ALWAYS_EXCLUDED | (UFUNC_EXCLUDED if is_ufunc else set())
    params = [p for p in sig.parameters.values() if p.name not in excluded and p.name != "self"]
    if owner_name == "ndarray" and any(p.kind == P.VAR_KEYWORD for p in params):
        fn = getattr(np, name, None)
        fsig = numpy_signature(fn, name) if fn is not None else None
        if fsig is not None:
            have = {p.name for p in params}
            extra = [p for i, p in enumerate(fsig.parameters.values())
                     if i > 0 and p.name not in have and p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)
                     and p.name not in excluded]
            params = [p for p in params if p.kind != P.VAR_KEYWORD] + [p.replace(kind=P.KEYWORD_ONLY) for p in extra]
    params = [p for p in params if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)]
    return sig, params, is_ufunc


def check_overload(ov, params):
    """Return (missing names, order problem) for one NumSharp overload against NumPy's parameters."""
    names = [prm["name"] for prm in ov]
    posonly = [p for p in params if p.kind == P.POSITIONAL_ONLY]
    poskw = [p for p in params if p.kind == P.POSITIONAL_OR_KEYWORD]
    kwonly = [p for p in params if p.kind == P.KEYWORD_ONLY]
    missing = [p.name for p in poskw + kwonly if p.name not in names]
    # Positional-only slots: the first len(posonly) parameters of the overload (any names).
    if len(posonly) > len(names):
        missing.append(f"<{len(posonly) - len(names)} positional slot(s)>")
    rest = names[len(posonly):]
    order = [rest.index(p.name) for p in poskw if p.name in rest]
    order_bad = order != sorted(order)
    # A NumPy positional-or-keyword parameter must sit at its NumPy position among the positional ones,
    # or a positional port binds the wrong argument (e.g. np.max(a, 1, o) binding o to keepdims).
    if not order_bad:
        expected = [p.name for p in poskw if p.name in rest]
        if [n for n in rest if n in expected] != expected[:len([n for n in rest if n in expected])]:
            order_bad = True
        # Every NumSharp-only parameter that sits BEFORE the last NumPy positional one shifts a positional port.
        last = max((rest.index(n) for n in expected), default=-1)
        if any(n not in {p.name for p in params} for n in rest[:last + 1]):
            order_bad = True
    return missing, order_bad


rows = {"np": [], "ndarray": []}
for owner_name, numpy_owner in (("np", np), ("ndarray", np.ndarray)):
    for name, overloads in sorted(dump[owner_name].items()):
        obj = getattr(numpy_owner, name, None)
        if obj is None or not callable(obj):
            continue
        sig, params, is_ufunc = effective_params(owner_name, name, obj)
        if sig is None:
            rows[owner_name].append((name, "?", "no NumPy signature", "", ""))
            continue
        best = None
        for ov in overloads:
            missing, order_bad = check_overload(ov, params)
            score = (len(missing), order_bad)
            if best is None or score < best[0]:
                best = (score, missing, order_bad)
        _, missing, order_bad = best
        if missing or order_bad:
            rows[owner_name].append((name, "ufunc" if is_ufunc else type(obj).__name__, ", ".join(missing),
                                     "ORDER" if order_bad else "", str(sig)))

with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as fh:
    for owner_name in ("np", "ndarray"):
        fh.write(f"## {owner_name}: {len(rows[owner_name])} names with a gap\n\n")
        fh.write("| name | kind | missing from the closest overload | order | NumPy signature |\n|---|---|---|---|---|\n")
        for r in rows[owner_name]:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | `{r[4]}` |\n")
        fh.write("\n")
for owner_name in ("np", "ndarray"):
    print(owner_name, len(rows[owner_name]), "names with a gap")
