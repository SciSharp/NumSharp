"""The one NumPy release NumSharp's oracle is generated with, read from the repository's ``numpy-version.txt``.

Every committed corpus, the coverage inventory, the benchmark twins and the interop parity environment are measured
against a single NumPy release. The release is written down exactly once, in ``numpy-version.txt`` at the repository
root; this module is how the generators under ``test/oracle/`` read it, ``coverage/generate_coverage.py`` reads the same
file, the workflows install ``numpy==$(cat numpy-version.txt)``, and the C# tests receive it as the ``NumPyVersion``
assembly metadata (``test/NumSharp.Tests/NumSharp.Tests.csproj``). Bumping NumPy is therefore one edit of that file,
followed by regenerating every corpus with the new release (``.claude/skills/oracle/references/regenerate.md``).

Import it from a generator that lives next to this file (the script's own folder is ``sys.path[0]`` when it runs as
``python test/oracle/gen_x.py``)::

    import numpy as np
    import numpy_pin

    numpy_pin.require(np, "gen_x.py")          # SystemExit unless the imported NumPy is the pinned release

Footgun: a generator run under ``python -I`` (isolated mode) does not get its own folder on ``sys.path``, so the
import fails exactly like the existing ``import layout_catalog`` sibling imports do; run the generators without ``-I``.
"""
import os

#: The repository root: this file lives in ``<root>/test/oracle/``.
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

#: The pin file every consumer reads; it holds the version and nothing else (one line, no comment).
PIN_FILE = os.path.join(REPO_ROOT, "numpy-version.txt")


def _read_pin(path):
    """Read the pinned NumPy version from ``numpy-version.txt``.

    The file is read as UTF-8 and stripped, so a trailing newline (or a Windows ``\\r\\n``) is not part of the version.

    :param path: the absolute path of ``numpy-version.txt``.
    :returns: the version string, for example ``"2.5.4"``.
    :raises SystemExit: the file is missing or empty, so no generator could tell which NumPy the corpus must come from;
        exiting (rather than defaulting to the installed NumPy) keeps a broken checkout from silently re-pinning every
        corpus to whatever NumPy happens to be installed.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            version = fh.read().strip()
    except OSError as error:
        raise SystemExit(f"cannot read the NumPy pin {path}: {error}") from error
    if not version:
        raise SystemExit(f"the NumPy pin {path} is empty")
    return version


#: The pinned NumPy release, for example ``"2.5.4"``.
NUMPY_VERSION = _read_pin(PIN_FILE)

#: The version without dots (``"254"``), the suffix of names that carry the release (the WSL venv ``~/np254``).
NUMPY_TAG = NUMPY_VERSION.replace(".", "")

#: The Linux interpreter the LP64 tiers run under (WSL): a venv named after the pin, so every bump builds a fresh one
#: (``/usr/bin/python3 -m venv --without-pip ~/np<tag>``, then ``pip --python ~/np<tag>/bin/python install
#: numpy==<pin>``). ``NUMSHARP_LP64_PYTHON`` overrides it where a generator reads that variable.
LINUX_PYTHON = f"~/np{NUMPY_TAG}/bin/python"


def require(np_module, what):
    """Refuse to generate anything unless the imported NumPy is the pinned release.

    A corpus generated with another release would commit that release's answers as NumSharp's contract, and every
    replay would then hold NumSharp to the wrong NumPy, so the generators stop before writing a byte.

    :param np_module: the imported ``numpy`` module (its ``__version__`` is checked).
    :param what: the generator or tier named in the message, for example ``"gen_flags_oracle.py"``.
    :returns: None.
    :raises SystemExit: ``np_module.__version__`` is not :data:`NUMPY_VERSION`.
    """
    found = np_module.__version__
    if found != NUMPY_VERSION:
        raise SystemExit(f"{what} must be generated with numpy {NUMPY_VERSION} (numpy-version.txt), found {found}")
