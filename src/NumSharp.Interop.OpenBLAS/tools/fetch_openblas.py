#!/usr/bin/env python3
"""
Fetch the OpenBLAS binaries that `NumSharp.Interop.OpenBLAS` bundles as NuGet runtime assets.

WHY THIS EXISTS
---------------
The package's whole claim is that `np.dot` / `np.matmul` are BIT-IDENTICAL to a NumPy that calls
the same BLAS.  OpenBLAS' GEMM kernels are arch- and thread-count-specific, so "the same answer"
is only meaningful with respect to ONE SPECIFIC BINARY.  NumPy 2.x does not build its own OpenBLAS:
its wheels bundle the prebuilt `scipy-openblas32` / `scipy-openblas64` artifacts published on PyPI
by the openblas-libs project (https://github.com/MacPython/openblas-libs).  So bundling *those*
artifacts, at the version NumPy pins, means shipping literally the binary NumPy calls -- verified:
the DLL inside `scipy_openblas64-0.3.31.22.0-py3-none-win_amd64.whl` is byte-for-byte the DLL in
numpy 2.4.2's `numpy.libs/` (sha256 74a408729250596b0973e69fdd954eea07a70ff527a1dbaccf9ae21247b8037
-- which is also where NumPy's mangled filename suffix comes from; delvewheel names it by its hash).

The pin comes from NumPy's own build requirements, `requirements/ci_requirements.txt`, consumed by
`tools/wheels/cibw_before_build.sh`.  win-arm64 is deliberately `scipy-openblas32`, because that is
what NumPy's own build script selects there (`RUNNER_ARCH == ARM64 && RUNNER_OS == Windows`).

USAGE
-----
    python fetch_openblas.py                  # download + verify + extract into ../runtimes/
    python fetch_openblas.py --check          # verify only; non-zero exit if anything is missing
    python fetch_openblas.py --clean          # remove the staged runtime assets
    python fetch_openblas.py --refresh-manifest --distribution-version 0.3.31.22.0
                                              # re-derive openblas-manifest.json from PyPI

Nothing here runs during an ordinary `dotnet build`.  The staged binaries are gitignored; the
MANIFEST is checked in, and it is the pin: every download is verified against it twice (the wheel's
sha256, then the extracted library's own sha256), so a re-pointed URL or a mutated artifact fails
loudly rather than silently changing the bits the package produces.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
MANIFEST_PATH = os.path.join(HERE, "openblas-manifest.json")
RUNTIMES_DIR = os.path.join(PKG_ROOT, "runtimes")
CACHE_DIR = os.path.join(HERE, ".wheel-cache")
NOTICES_PATH = os.path.join(PKG_ROOT, "THIRD-PARTY-NOTICES.txt")

# .NET RID -> (distribution, PyPI wheel platform tag).
#
# Mirrors NumPy's own selection in tools/wheels/cibw_before_build.sh: scipy-openblas64 everywhere
# except Windows on ARM64, which NumPy builds against scipy-openblas32.  Matching that matters --
# the two have different symbol schemes AND different BLAS integer widths (see OpenBlasNative.Bind).
RID_MAP = {
    "win-x64":          ("scipy-openblas64", "win_amd64"),
    "win-arm64":        ("scipy-openblas32", "win_arm64"),
    "linux-x64":        ("scipy-openblas64", "manylinux2014_x86_64"),
    "linux-arm64":      ("scipy-openblas64", "manylinux2014_aarch64"),
    "linux-musl-x64":   ("scipy-openblas64", "musllinux_1_2_x86_64"),
    "linux-musl-arm64": ("scipy-openblas64", "musllinux_1_2_aarch64"),
    "osx-x64":          ("scipy-openblas64", "macosx_10_9_x86_64"),
    "osx-arm64":        ("scipy-openblas64", "macosx_11_0_arm64"),
}

NATIVE_SUFFIXES = (".dll", ".so", ".dylib")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def is_native_member(name):
    base = os.path.basename(name)
    if not base or base.startswith("."):
        return False
    if base.endswith(NATIVE_SUFFIXES):
        return True
    return ".so." in base


def pick_native_member(zf):
    """The one shared library in a scipy-openblas wheel (never the import .lib / .a)."""
    candidates = [
        n for n in zf.namelist()
        if is_native_member(n) and "/lib/" in n.replace("\\", "/")
    ]
    if not candidates:
        candidates = [n for n in zf.namelist() if is_native_member(n)]
    if not candidates:
        raise SystemExit("no shared library found in wheel")
    # Prefer the largest -- the real library, not a stub or symlink shim.
    return max(candidates, key=lambda n: zf.getinfo(n).file_size)


def download(url, expect_sha=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, os.path.basename(url.split("?")[0]))
    if os.path.isfile(cached):
        data = open(cached, "rb").read()
        if expect_sha is None or sha256_bytes(data) == expect_sha:
            return data
        os.remove(cached)  # poisoned cache entry
    print("    downloading %s" % os.path.basename(cached))
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    if expect_sha is not None and sha256_bytes(data) != expect_sha:
        raise SystemExit(
            "CHECKSUM MISMATCH for %s\n  expected %s\n  actual   %s\n"
            "The pinned artifact changed. Parity is a claim about one specific binary, so this is "
            "a hard failure, not a warning." % (url, expect_sha, sha256_bytes(data)))
    with open(cached, "wb") as fh:
        fh.write(data)
    return data


def refresh_manifest(distribution_version, numpy_version):
    """Re-derive the pin from PyPI. Downloads every wheel to record the EXTRACTED library's hash."""
    print("refreshing manifest for %s ..." % distribution_version)
    index = {}
    for dist in sorted({d for d, _ in RID_MAP.values()}):
        url = "https://pypi.org/pypi/%s/%s/json" % (dist, distribution_version)
        with urllib.request.urlopen(url) as resp:
            index[dist] = json.load(resp)

    runtimes = {}
    for rid in RID_MAP:
        dist, tag = RID_MAP[rid]
        # Platform tags can be compound, e.g.
        # "...-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", so anchor on the
        # separator before the tag and the dot after it rather than on the end of the name.
        matches = [u for u in index[dist]["urls"] if ("-%s." % tag) in u["filename"]]
        if len(matches) != 1:
            raise SystemExit("expected exactly one %s wheel for %s, got %d"
                             % (dist, tag, len(matches)))
        u = matches[0]
        print("  %-18s %s" % (rid, u["filename"]))
        blob = download(u["url"], u["digests"]["sha256"])
        with zipfile.ZipFile(os.path.join(CACHE_DIR, u["filename"])) as zf:
            member = pick_native_member(zf)
            payload = zf.read(member)
        runtimes[rid] = {
            "distribution": dist,
            "wheel": u["filename"],
            "url": u["url"],
            "wheel_sha256": u["digests"]["sha256"],
            "member": member,
            "file": os.path.basename(member),
            "sha256": sha256_bytes(payload),
            "size": len(payload),
        }

    manifest = {
        "_comment": (
            "PIN. Do not edit by hand -- regenerate with "
            "`python fetch_openblas.py --refresh-manifest`. These are the exact prebuilt OpenBLAS "
            "artifacts NumPy %s bundles (see numpy requirements/ci_requirements.txt), which is what "
            "makes NumSharp's matrix products bit-identical to that NumPy's." % numpy_version),
        "numpy_version": numpy_version,
        "distribution_version": distribution_version,
        "openblas_version": "0.3.31.dev",
        # SPDX ids only from the OSI/FSF-approved list: nuget.org rejects a package whose license
        # expression (NumSharp.Interop.OpenBLAS.csproj mirrors these) carries anything else, and it
        # does so at push time. 'BSD-3-Clause-Attribution' is a DIFFERENT, unapproved license and
        # once sat here; OpenBLAS's LICENSE is the plain 3-clause BSD.
        "licenses": [
            {"name": "OpenBLAS", "spdx": "BSD-3-Clause",
             "url": "https://github.com/OpenMathLib/OpenBLAS/"},
            {"name": "LAPACK", "spdx": "BSD-3-Clause",
             "url": "https://github.com/OpenMathLib/OpenBLAS/"},
            {"name": "GCC runtime library (libgfortran)", "spdx": "GPL-3.0-or-later WITH GCC-exception-3.1",
             "url": "https://gcc.gnu.org/git/?p=gcc.git;a=tree;f=libgfortran"},
            {"name": "openblas-libs (packaging)", "spdx": "BSD-2-Clause",
             "url": "https://github.com/MacPython/openblas-libs"},
        ],
        "runtimes": runtimes,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("wrote %s" % MANIFEST_PATH)
    write_notices(manifest)
    return manifest


def write_notices(manifest):
    """Emit the third-party notices that must travel with the redistributed binaries."""
    lines = []
    lines.append("THIRD-PARTY NOTICES for NumSharp.Interop.OpenBLAS")
    lines.append("=" * 76)
    lines.append("")
    lines.append("NumSharp.Interop.OpenBLAS itself is licensed Apache-2.0, like the rest of NumSharp.")
    lines.append("")
    lines.append("This package additionally REDISTRIBUTES prebuilt OpenBLAS shared libraries as")
    lines.append("NuGet runtime assets (runtimes/<rid>/native/). They are the unmodified artifacts")
    lines.append("published on PyPI by the openblas-libs project as %s"
                 % manifest["distribution_version"])
    lines.append("-- the very same binaries NumPy %s bundles in its own wheels, which is what makes"
                 % manifest["numpy_version"])
    lines.append("NumSharp's matrix products bit-identical to that NumPy's.")
    lines.append("")
    lines.append("Bundled components and their licenses:")
    lines.append("")
    for lic in manifest["licenses"]:
        lines.append("  %-38s %s" % (lic["name"], lic["spdx"]))
        lines.append("  %-38s %s" % ("", lic["url"]))
        lines.append("")
    lines.append("The libgfortran runtime is covered by the GCC Runtime Library Exception, which")
    lines.append("permits redistribution as part of a larger work under other terms; this is the")
    lines.append("same basis on which NumPy and SciPy ship it in every binary wheel.")
    lines.append("")
    lines.append("Bundled artifacts (sha256 of the extracted shared library):")
    lines.append("")
    for rid in sorted(manifest["runtimes"]):
        e = manifest["runtimes"][rid]
        lines.append("  %-18s %-28s %s" % (rid, e["file"], e["sha256"]))
    lines.append("")
    lines.append("Full OpenBLAS / LAPACK license text:")
    lines.append("  https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE")
    lines.append("  https://github.com/OpenMathLib/OpenBLAS/blob/develop/lapack-netlib/LICENSE")
    lines.append("")
    with open(NOTICES_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("wrote %s" % NOTICES_PATH)


def staged_path(rid, entry):
    return os.path.join(RUNTIMES_DIR, rid, "native", entry["file"])


def vendored_members(zf, main_member):
    """The vendored dependency shared libraries the main scipy-openblas library needs at load time.

    A scipy-openblas wheel carries its Fortran runtime beside the main library (Linux/macOS ship
    libgfortran + libquadmath, macOS adds libgcc_s); the Windows DLL is self-contained. auditwheel /
    delocate rename these with a content hash and patch the main library's DT_NEEDED / LC_LOAD_DYLIB
    to the mangled name, so a *system* libgfortran can never satisfy them — the vendored file MUST be
    co-staged or the main dlopen()/LoadLibrary fails on a clean machine and the backend silently
    stays uninstalled. fetch_openblas historically staged the main library alone, which is why
    OpenBlasEngine.Enabled came up false on a fresh Linux/macOS CI runner."""
    return [n for n in zf.namelist() if is_native_member(n) and n != main_member]


def vendored_dest(rid, member):
    """Where a vendored dep is staged, mirroring the wheel's own layout relative to the main library
    so the main library's baked-in search path resolves it unchanged: Linux keeps the deps beside the
    main in native/ (its RUNPATH is $ORIGIN), macOS keeps them in a .dylibs/ sibling of native/ (its
    load commands are @loader_path/../.dylibs/…)."""
    norm = member.replace("\\", "/")
    subdir = ".dylibs" if "/.dylibs/" in norm else "native"
    return os.path.join(RUNTIMES_DIR, rid, subdir, os.path.basename(member))


def fetch(manifest, check_only=False):
    missing, ok = [], 0
    for rid in sorted(manifest["runtimes"]):
        entry = manifest["runtimes"][rid]
        dest = staged_path(rid, entry)
        if os.path.isfile(dest) and sha256_bytes(open(dest, "rb").read()) == entry["sha256"]:
            ok += 1
            continue
        if check_only:
            missing.append(rid)
            continue

        print("  %-18s %s" % (rid, entry["file"]))
        download(entry["url"], entry["wheel_sha256"])
        with zipfile.ZipFile(os.path.join(CACHE_DIR, entry["wheel"])) as zf:
            payload = zf.read(entry["member"])
            actual = sha256_bytes(payload)
            if actual != entry["sha256"]:
                raise SystemExit(
                    "CHECKSUM MISMATCH for the library extracted from %s\n"
                    "  member   %s\n  expected %s\n  actual   %s"
                    % (entry["wheel"], entry["member"], entry["sha256"], actual))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(payload)
            # Co-stage the vendored Fortran runtime (libgfortran/libquadmath/libgcc_s) from the SAME
            # already-verified wheel, mirroring its layout so the main library loads on a clean host.
            for dep in vendored_members(zf, entry["member"]):
                ddest = vendored_dest(rid, dep)
                os.makedirs(os.path.dirname(ddest), exist_ok=True)
                with open(ddest, "wb") as fh:
                    fh.write(zf.read(dep))
                print("  %-18s %s (vendored dep)" % ("", os.path.basename(dep)))
        ok += 1

    if check_only and missing:
        print("MISSING staged runtime assets for: %s" % ", ".join(missing))
        print("Run: python %s" % os.path.relpath(__file__))
        return 1
    print("%d/%d runtime assets present and verified in %s"
          % (ok, len(manifest["runtimes"]), RUNTIMES_DIR))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only, do not download")
    ap.add_argument("--clean", action="store_true", help="delete staged runtime assets")
    ap.add_argument("--refresh-manifest", action="store_true", help="re-derive the pin from PyPI")
    ap.add_argument("--distribution-version", default="0.3.31.22.0")
    ap.add_argument("--numpy-version", default="2.4.2")
    args = ap.parse_args()

    if args.clean:
        if os.path.isdir(RUNTIMES_DIR):
            shutil.rmtree(RUNTIMES_DIR)
            print("removed %s" % RUNTIMES_DIR)
        return 0

    if args.refresh_manifest:
        manifest = refresh_manifest(args.distribution_version, args.numpy_version)
    else:
        manifest = load_manifest()
        write_notices(manifest)

    return fetch(manifest, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
