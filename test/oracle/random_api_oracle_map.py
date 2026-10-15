"""
random_api_oracle_map.py — rebuilds the `random_api` fuzz key of coverage/oracle_map.json from the committed random-API
corpus and the coverage catalog (docs/plans/random-oracle-coverage.md §6.8).

Why a script: the coverage join is STRICT (coverage/oracle_evidence.py) — every `params.member` value of the
random_api tiers must resolve to catalog rows, and an override that the template would produce anyway is "redundant"
and fails too. The template `numpy.random.{value}` maps `<Owner>.<member>` (`Generator.normal`, `PCG64.advance`,
`SeedSequence.spawn`) and a constructor's class name (`MT19937`, `SeedSequence`, `default_rng`) to its row; this
script writes an override exactly for the members where that is not the whole answer:

- a template id that is not a catalog row (private members, NumSharp-only ones, SeedlessSeedSequence): mapped to its
  real row when one exists (SPECIAL), otherwise to nothing (`[]`, "tested, but no catalog row");
- RandomState members ALSO credit the np.random module function of the same name (NumPy's module function IS the
  global RandomState's bound method) — Generator members do not (the same-named module functions are the legacy ones);
- BitGenerator's own members invoked on an engine (`MT19937.random_raw`, `PCG64.seed_seq`, …) ALSO credit the
  `numpy.random.BitGenerator.<member>` row (the implementation every engine inherits).

Usage (after a corpus regeneration that changes the member set, and after `python coverage/generate_coverage.py`
has produced the catalog it reads):

    python test/oracle/random_api_oracle_map.py [--catalog coverage/generated/coverage.json]

It rewrites only the `random_api` line of coverage/oracle_map.json (inserting it between `r_` and `ravel_f` when
absent), validates the result as JSON, and prints the member / override counts. Run the coverage generator again
afterwards: the join then checks the new entry.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
CORPUS_DIR = os.path.join(ROOT, "test", "NumSharp.Tests.Oracle", "Fuzz", "corpus")
MAP_PATH = os.path.join(ROOT, "coverage", "oracle_map.json")
TIERS = ["random_api.jsonl", "random_api_host.jsonl", "random_api_mvn.jsonl", "random_api_lp64.jsonl"]
ENGINES = {"MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"}
BASE_MEMBERS = {"random_raw", "seed_seq", "spawn", "state"}

# Members whose NumPy counterpart is a different row than the template names (reviewed by hand).
SPECIAL = {
    "Generator._bit_generator": ["numpy.random.Generator.bit_generator"],
    "RandomState.bernoulli": ["numsharp.random.bernoulli"],
    "RandomState.ranf": ["numpy.random.ranf"],
    "RandomState.sample": ["numpy.random.sample"],
    # NumPy has these as numpy.random attributes outside __all__, so the catalog files them as NumSharp extensions; the
    # oracle still compares them with NumPy's own functions / str().
    "RandomState.get_bit_generator": ["numsharp.random.get_bit_generator"],
    "RandomState.set_bit_generator": ["numsharp.random.set_bit_generator"],
    "RandomState.ToString": ["numsharp.random.ToString"],
}

NOTE = ("Random-API oracle (gen_random_oracle.py): every overload of the random world by exact C# signature; "
        "params.member is <Owner>.<member> (a constructor is the class name alone). RandomState members also "
        "credit the np.random module function (the global RandomState's bound method); Generator members do "
        "not (the same-named module functions are the legacy ones). BitGenerator's members invoked on an "
        "engine also credit the BitGenerator row. Members without a catalog row map to nothing. "
        "Rebuilt by test/oracle/random_api_oracle_map.py.")


def load_catalog(path):
    """The catalog's row ids (the list of dicts with an `id` in coverage.json)."""
    with open(path, encoding="utf-8") as fh:
        cov = json.load(fh)
    rows = next(v for v in cov.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0])
    return {r["id"] for r in rows}


def corpus_members():
    """Every `params.member` value of the committed random_api tiers (a missing tier contributes nothing)."""
    members = set()
    for t in TIERS:
        path = os.path.join(CORPUS_DIR, t)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    members.add(json.loads(line)["params"]["member"])
    return members


def build_entry(members, catalog):
    """The `random_api` key: the template plus the overrides (see the module docstring)."""
    values = {}
    for mem in sorted(members):
        template_id = "numpy.random." + mem
        if mem in SPECIAL:
            values[mem] = [i for i in SPECIAL[mem] if i in catalog]
            continue
        owner, _, name = mem.partition(".")
        ids = [template_id] if template_id in catalog else []
        if owner == "RandomState" and "numpy.random." + name in catalog:
            ids.append("numpy.random." + name)
        if owner in ENGINES and name in BASE_MEMBERS and "numpy.random.BitGenerator." + name in catalog:
            ids.append("numpy.random.BitGenerator." + name)
        if ids != [template_id]:
            values[mem] = ids   # includes [] for members without any catalog row
    return {"param": "member", "ids": ["numpy.random.{value}"], "values": values, "note": NOTE}


def write_entry(entry):
    """Replaces (or inserts) the `random_api` line of coverage/oracle_map.json; the file must stay valid JSON."""
    line = '      "random_api": ' + json.dumps(entry, separators=(", ", ": ")) + ","
    with open(MAP_PATH, encoding="utf-8") as fh:
        text = fh.read()
    start = text.find('      "random_api": ')
    if start >= 0:
        end = text.index("\n", start)
        text = text[:start] + line + text[end:]
    else:
        anchor = '      "ravel_f": {'   # the keys are near-sorted; random_api sorts between r_ and ravel_f
        i = text.index(anchor)
        text = text[:i] + line + "\n" + text[i:]
    json.loads(text)   # still valid JSON
    with open(MAP_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=os.path.join(ROOT, "coverage", "generated", "coverage.json"),
                    help="the coverage catalog written by coverage/generate_coverage.py")
    args = ap.parse_args(argv)
    if not os.path.exists(args.catalog):
        raise SystemExit(f"{args.catalog} is missing: run python coverage/generate_coverage.py first")
    entry = build_entry(corpus_members(), load_catalog(args.catalog))
    write_entry(entry)
    nothing = sum(1 for v in entry["values"].values() if not v)
    print(f"random_api: {len(corpus_members())} members, {len(entry['values'])} overrides ({nothing} map to nothing)")


if __name__ == "__main__":
    main()
