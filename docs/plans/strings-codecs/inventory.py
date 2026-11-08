"""Lists the codec modules of CPython's ``encodings`` package that NumSharp refuses, classified the way
``tools/gen_python_codecs.py`` classifies them, with what implements each one in CPython and the aliases that reach it.

The discovery ``docs/plans/strings-codecs.md`` (section 1) quotes this script's output. Run it with the CPython the
oracles run (3.12, the NumPy 2.4.2 oracle interpreter); on Windows it lists 32 modules, elsewhere 30 (``mbcs`` and
``oem`` import Windows-only functions, so ``codecs.lookup`` reports them unknown there):

    python -I docs/plans/strings-codecs/inventory.py
"""
import codecs
import collections
import importlib
import inspect
import pkgutil
import sys

import encodings
from encodings import aliases

# The modules NumSharp implements statement for statement (gen_python_codecs.py's _UNICODE_KINDS).
_UNICODE = {"utf_8", "utf_8_sig", "utf_16", "utf_16_le", "utf_16_be", "utf_32", "utf_32_le", "utf_32_be", "latin_1",
            "ascii", "undefined"}


def classify(module):
    """The kind ``tools/gen_python_codecs.py`` gives one module of the ``encodings`` package.

    :param module: the module name (``shift_jis``, ``cp1252``, ...).
    :returns: ``(kind, codec name)`` with kind one of ``Unicode`` / ``Charmap`` / ``NotText`` / ``Unsupported``, or None
        for a module that is no codec on this platform (``aliases``; ``mbcs`` / ``oem`` off Windows).
    """
    if module == "aliases":
        return None
    try:
        info = codecs.lookup(module)
    except LookupError:
        return None
    if not getattr(info, "_is_text_encoding", True):
        return "NotText", info.name
    if module in _UNICODE:
        return "Unicode", info.name
    mod = importlib.import_module("encodings." + module)
    if hasattr(mod, "decoding_table") or hasattr(mod, "decoding_map"):
        # A charmap module whose every byte decodes to at most one character is a single-byte code page.
        single = True
        for b in range(256):
            try:
                if len(bytes([b]).decode(module)) > 1:
                    single = False
                    break
            except UnicodeDecodeError:
                pass
        if single:
            return "Charmap", info.name
    return "Unsupported", info.name


def implementation(module):
    """What a refused module runs in CPython: the import lines of its source.

    :param module: the module name.
    :returns: the module's ``import`` / ``from`` lines (``_codecs_jp`` + ``_multibytecodec`` for a CJK codec,
        ``codecs`` alone for one whose functions live in ``_codecs``), and its source's line count.
    """
    src = inspect.getsource(importlib.import_module("encodings." + module))
    imports = [ln.strip() for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))]
    return imports, len(src.splitlines())


def main():
    """Prints the counts by kind, then one line per refused module."""
    sys.stdout.reconfigure(encoding="utf-8")
    modules = sorted(m.name for m in pkgutil.iter_modules(encodings.__path__))
    kinds = {m: k for m in modules if (k := classify(m)) is not None}
    print(f"CPython {sys.version.split()[0]} on {sys.platform}: {len(modules)} modules in encodings")
    print("by kind:", dict(collections.Counter(k for k, _ in kinds.values())))
    by_module = collections.defaultdict(list)
    for alias, module in aliases.aliases.items():
        by_module[module].append(alias)
    refused = [m for m, (k, _) in kinds.items() if k == "Unsupported"]
    n_alias = sum(len(by_module[m]) for m in refused)
    print(f"refused: {len(refused)} modules, {n_alias} aliases of the {len(aliases.aliases)} in the alias table")
    for m in refused:
        imports, lines = implementation(m)
        print(f"  {m:20s} lookup name {kinds[m][1]!r:22s} aliases {sorted(by_module[m])}")
        print(f"  {'':20s} {lines} source lines, imports {imports}")


if __name__ == "__main__":
    main()
