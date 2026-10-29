"""Create or refresh one of NumSharp.Tests.Interop's isolated Python environments.

NumSharp.Tests.Interop embeds CPython in the test process and uses a LIVE NumPy as its byte-exact
oracle, so the Python it embeds is part of the test's definition. The suite needs two different
Pythons, and they must not share site-packages:

  parity     numpy ONLY, at the exact release whose bundled OpenBLAS NumSharp.Interop.OpenBLAS
             ships (src/NumSharp.Interop.OpenBLAS/tools/openblas-manifest.json -> numpy_version).
             Every byte-exact live-NumPy test runs here. Nothing else is installed, so no third-party
             package can move numpy to another release, swap its BLAS, or load its own native
             runtime (torch's OpenMP/MKL, opencv's) into the oracle's process.
  ecosystem  the same numpy plus the third-party libraries the interop bridges into (torch, pandas,
             scipy, pyarrow, pillow, polars, opencv). Only the tests tagged [PythonEcosystem] run
             here. Those libraries release on their own schedule and pin numpy ranges of their own;
             keeping them out of `parity` means a future release that demands another numpy breaks
             THIS environment's install loudly instead of silently moving the oracle.

Each environment is a stdlib `venv` built from the interpreter that RUNS this script, at
<repo>/.venvs/<name>-py<major><minor>/, so one machine holds the same environment for several
Pythons side by side: run the script once per interpreter (e.g. `py -3.11 make_env.py parity` and
`py -3.12 make_env.py parity` on Windows, `python3.11 ...` / `python3.12 ...` elsewhere). Progress
goes to stderr; stdout carries exactly one line, the environment's interpreter path, so a caller
can capture it. Point a test run at an environment with NUMSHARP_PYTHONNET_PYTHON (PythonSession
treats that path as binding and embeds the venv, not its base install):

    PY=$(python test/NumSharp.Tests.Interop/python-envs/make_env.py parity)
    NUMSHARP_PYTHONNET_PYTHON="$PY" dotnet test test/NumSharp.Tests.Interop -f net10.0 \\
        --filter "TestCategory!=PythonEcosystem"

    PY=$(python test/NumSharp.Tests.Interop/python-envs/make_env.py ecosystem)
    NUMSHARP_PYTHONNET_PYTHON="$PY" NUMSHARP_PYTHONNET_REQUIRE_PACKAGES=1 \\
        dotnet test test/NumSharp.Tests.Interop -f net10.0 --filter "TestCategory=PythonEcosystem"

Re-running is cheap (pip reports everything already satisfied); --recreate starts from an empty
venv. Stdlib only, so it runs on a bare interpreter before anything is installed.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The directory holding this script and the requirement files it installs.
HERE = Path(__file__).resolve().parent

#: The repository root (test/NumSharp.Tests.Interop/python-envs -> three levels up).
REPO = HERE.parents[2]

#: The OpenBLAS pin the byte-parity tests depend on: numpy_version is the NumPy release whose bundled
#: scipy-openblas NumSharp.Interop.OpenBLAS ships byte-identically, openblas_version its build string.
MANIFEST = REPO / "src" / "NumSharp.Interop.OpenBLAS" / "tools" / "openblas-manifest.json"

#: Environment name -> requirement files installed IN ORDER on top of the pinned numpy. numpy is not
#: listed in any file on purpose: every environment gets it first, from the manifest pin, and every
#: later install runs under a constraint that holds it there.
ENVIRONMENTS = {
    "parity": [],
    # torch comes from PyTorch's CPU-only index (its own file, because a requirements file's
    # --index-url applies to the whole file): PyPI's default Linux wheel pulls ~3 GB of CUDA
    # libraries these CPU-only tests never load.
    "ecosystem": ["torch-cpu.txt", "ecosystem.txt"],
}

#: The CPython range an environment may be built on. The floor is numpy's (the pinned release
#: publishes cp311+ wheels only); the ceiling is pythonnet 3.0.5's, which the interop suite pins
#: and which refuses to drive anything newer. Building outside it would only fail later, and less
#: legibly, when PythonSession rejects the interpreter.
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 13)

#: numpy's macOS wheels come in two BLAS flavours: macosx_14_0_* link Apple Accelerate, the older
#: tags bundle scipy-openblas. pip on macOS 14+ prefers the newer tag, which would hand the oracle
#: a BLAS no NumSharp build shares; these are the OpenBLAS tags per architecture.
MACOS_OPENBLAS_TAGS = {"arm64": "macosx_11_0_arm64", "x86_64": "macosx_10_13_x86_64"}


def log(message):
    """Write one progress line to stderr, keeping stdout free for the interpreter path.

    :param message: the line to print.
    """
    print(message, file=sys.stderr, flush=True)


def run(command):
    """Run a command, streaming its output to stderr, and fail the script if it fails.

    :param command: the argument list (no shell), e.g. ``[python, "-m", "pip", "install", ...]``.
    :raises subprocess.CalledProcessError: when the command exits non-zero, so a failed install
        stops the script instead of producing an environment that is silently missing packages.
    """
    log("+ " + " ".join(str(part) for part in command))
    # stdout is redirected to OUR stderr: pip's progress must never mix into the one stdout line a
    # caller captures as the interpreter path.
    subprocess.run([str(part) for part in command], check=True, stdout=sys.stderr)


def read_manifest():
    """Read the numpy and OpenBLAS versions the byte-parity tests are pinned to.

    :returns: ``(numpy_version, openblas_version)``, e.g. ``("2.4.2", "0.3.31.dev")``.
    :raises SystemExit: when the manifest is missing or lacks either field; every environment
        depends on that pin, so there is nothing sensible to fall back to.
    """
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return manifest["numpy_version"], manifest["openblas_version"]
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"cannot read the numpy pin from {MANIFEST}: {error}")


def venv_python(env_dir):
    """Return the interpreter path inside a venv directory (the layout differs on Windows).

    :param env_dir: the venv's root directory.
    :returns: ``<env>/Scripts/python.exe`` on Windows, ``<env>/bin/python`` elsewhere.
    """
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def create_venv(env_dir, recreate):
    """Create the venv from the running interpreter unless it already exists.

    :param env_dir: where the venv lives.
    :param recreate: delete an existing venv first, so the environment is rebuilt from nothing.
    :returns: the venv's interpreter path.
    :raises SystemExit: when the directory exists but is not a venv of THIS interpreter's version
        (reusing it would install packages for the wrong Python).
    """
    if recreate and env_dir.exists():
        log(f"removing {env_dir}")
        shutil.rmtree(env_dir)

    python = venv_python(env_dir)
    if not python.exists():
        log(f"creating {env_dir} from {sys.executable}")
        # --upgrade-deps is deliberately NOT used: it would fetch the newest pip from PyPI, making
        # the environment depend on whatever pip release is current. ensurepip's bundled pip
        # supports everything this script asks of it (requirement files, -c, --index-url,
        # `pip download --platform`).
        run([sys.executable, "-m", "venv", env_dir])
        return python

    # An existing venv must belong to the same minor Python: `.venvs/parity-py312` built by 3.11
    # would be a lie in its own name, and pip would install cp311 wheels into it.
    probe = subprocess.run([str(python), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                           capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.split() != [str(sys.version_info[0]), str(sys.version_info[1])]:
        raise SystemExit(f"{env_dir} exists but is not a working Python "
                         f"{sys.version_info[0]}.{sys.version_info[1]} venv; rerun with --recreate")
    return python


def install_numpy(python, numpy_version):
    """Install the pinned numpy, forcing the scipy-openblas wheel on macOS.

    :param python: the venv's interpreter.
    :param numpy_version: the exact release to install (the manifest pin).
    :raises subprocess.CalledProcessError: when pip cannot fetch or install it.
    """
    pip = [python, "-m", "pip", "--disable-pip-version-check", "--no-input"]
    tag = MACOS_OPENBLAS_TAGS.get(platform.machine()) if sys.platform == "darwin" else None
    if tag is None:
        # Windows and Linux numpy wheels bundle scipy-openblas unconditionally.
        run(pip + ["install", "--progress-bar", "off", f"numpy=={numpy_version}"])
        return

    # macOS: download exactly the OpenBLAS-flavoured wheel, then install that FILE, so pip never
    # gets the chance to prefer the Accelerate one. --only-binary + --platform + --abi pin the tag.
    abi = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    with tempfile.TemporaryDirectory(prefix="numsharp-numpy-") as wheels:
        run(pip + ["download", "--only-binary=:all:", "--no-deps", "--platform", tag,
                   "--python-version", f"{sys.version_info[0]}{sys.version_info[1]}",
                   "--implementation", "cp", "--abi", abi, "-d", wheels, f"numpy=={numpy_version}"])
        downloaded = sorted(Path(wheels).glob("numpy-*.whl"))
        if len(downloaded) != 1:
            raise SystemExit(f"expected one {tag} numpy wheel, pip downloaded {len(downloaded)}")
        run(pip + ["install", "--progress-bar", "off", downloaded[0]])


def install_requirements(python, files, numpy_version):
    """Install each requirement file on top of the pinned numpy, which must not move.

    :param python: the venv's interpreter.
    :param files: requirement file names under this script's directory, installed in order.
    :param numpy_version: the pin the constraint file holds numpy at.
    :raises subprocess.CalledProcessError: when a file cannot be resolved without moving numpy
        (pip's resolver fails loudly on the constraint) or a download fails.
    """
    if not files:
        return

    # A constraint, not a requirement: it installs nothing by itself, it only forbids every
    # later resolution from choosing a different numpy. A package that genuinely needs another
    # numpy therefore FAILS this install instead of quietly replacing the oracle's numpy.
    with tempfile.TemporaryDirectory(prefix="numsharp-constraints-") as scratch:
        constraints = Path(scratch) / "constraints.txt"
        constraints.write_text(f"numpy=={numpy_version}\n", encoding="utf-8")
        for name in files:
            run([python, "-m", "pip", "--disable-pip-version-check", "--no-input", "install",
                 "--progress-bar", "off", "-c", constraints, "-r", HERE / name])


def verify(python, numpy_version, openblas_version):
    """Prove the environment's numpy is the pinned release linked to the pinned OpenBLAS.

    The byte-parity tests are only meaningful when NumPy and NumSharp call the same BLAS binary, so
    this checks what numpy REPORTS about itself rather than trusting the wheel that was requested.

    :param python: the venv's interpreter.
    :param numpy_version: the release numpy must report.
    :param openblas_version: the scipy-openblas build numpy's BLAS must report.
    :raises SystemExit: when the environment's numpy, or its BLAS name/version, differ from the pin.
    """
    script = ("import json, numpy; "
              "b = numpy.show_config(mode='dicts')['Build Dependencies']['blas']; "
              "print(json.dumps([numpy.__version__, b.get('name'), b.get('version')]))")
    result = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"numpy does not import in {python}:\n{result.stderr}")
    version, blas, blas_version = json.loads(result.stdout)
    if version != numpy_version or blas != "scipy-openblas" or blas_version != openblas_version:
        raise SystemExit(f"{python}: numpy {version} with BLAS {blas} {blas_version}; the byte-parity "
                         f"tests need numpy {numpy_version} with scipy-openblas {openblas_version}")
    log(f"numpy {version} / {blas} {blas_version}: matches the OpenBLAS manifest")


def main():
    """Parse the command line, build the requested environment, print its interpreter path.

    :returns: the process exit code (0 on success; failures raise SystemExit with a message).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=sorted(ENVIRONMENTS), help="the environment to create or refresh")
    parser.add_argument("--root", type=Path, default=REPO / ".venvs",
                        help="directory holding the environments (default: <repo>/.venvs, gitignored)")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the environment")
    args = parser.parse_args()

    current = sys.version_info[:2]
    if not MIN_PYTHON <= current <= MAX_PYTHON:
        raise SystemExit(f"Python {current[0]}.{current[1]} cannot host these environments: the interop "
                         f"suite needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]} "
                         "(numpy's wheels set the floor, pythonnet 3.0.5 the ceiling). Run this script "
                         "with an interpreter in that range.")

    numpy_version, openblas_version = read_manifest()
    env_dir = args.root.resolve() / f"{args.name}-py{current[0]}{current[1]}"
    python = create_venv(env_dir, args.recreate)
    install_numpy(python, numpy_version)
    install_requirements(python, ENVIRONMENTS[args.name], numpy_version)
    verify(python, numpy_version, openblas_version)

    print(python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
