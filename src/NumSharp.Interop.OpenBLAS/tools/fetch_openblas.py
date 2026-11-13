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
the DLL inside `scipy_openblas64-0.3.34.106.0-py3-none-win_amd64.whl` is byte-for-byte the DLL in
numpy 2.5.4's `numpy.libs/` (`libscipy_openblas64_-ed4f167a5330424524f45258e7ca2c8d.dll`: delvewheel
names the copy by a hash of its content).

The pin comes from NumPy's own build requirements, `requirements/ci_requirements.txt`, consumed by
`tools/wheels/cibw_before_build.sh`.  win-arm64 is deliberately `scipy-openblas32`, because that is
what NumPy's own build script selects there (`RUNNER_ARCH == ARM64 && RUNNER_OS == Windows`).

USAGE
-----
    python fetch_openblas.py                  # download + verify + extract into ../runtimes/
    python fetch_openblas.py --check          # verify only; non-zero exit if anything is missing
    python fetch_openblas.py --clean          # remove the staged runtime assets
    python fetch_openblas.py --refresh-manifest --distribution-version 0.3.34.106.0
                                              # re-derive openblas-manifest.json from PyPI (the NumPy
                                              # release defaults to numpy-version.txt, the version
                                              # numpy.show_config() reports to --distribution-version)

Nothing here runs during an ordinary `dotnet build`.  The staged binaries are gitignored; the
MANIFEST is checked in, and it is the pin: every download is verified against it twice (the wheel's
sha256, then the extracted library's own sha256), so a re-pointed URL or a mutated artifact fails
loudly rather than silently changing the bits the package produces.

STAGED LAYOUT
-------------
    runtimes/<rid>/native/<main library>            every RID
    runtimes/<rid>/native/libgfortran-*.so.5 ...    Linux: the vendored GCC runtime beside the main
                                                    (its RUNPATH is $ORIGIN)
    runtimes/<rid>/native/.dylibs/*.dylib           macOS: the vendored GCC runtime NESTED under native/

The macOS nesting is deliberate and load-bearing.  The wheel itself parks the deps in a .dylibs/
SIBLING of lib/, and the main dylib's load commands say @loader_path/../.dylibs/<name>; but NuGet
delivers native assets only from under runtimes/<rid>/native/ and silently drops a sibling folder,
so staging the wheel's own layout produced a package that could not load on any macOS consumer.
The nested folder IS delivered; the runtime loader (MacOsVendoredRuntime in the package) turns it
into the sibling the load commands expect at first load.  The vendored files are pinned in the
manifest ("vendored" per RID) so a partial staging fails `--check`.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
#: The repository's one NumPy pin (numpy-version.txt): the release whose bundled OpenBLAS this package must ship.
NUMPY_PIN_PATH = os.path.normpath(os.path.join(PKG_ROOT, "..", "..", "numpy-version.txt"))
MANIFEST_PATH = os.path.join(HERE, "openblas-manifest.json")
RUNTIMES_DIR = os.path.join(PKG_ROOT, "runtimes")
CACHE_DIR = os.path.join(HERE, ".wheel-cache")
NOTICES_PATH = os.path.join(PKG_ROOT, "THIRD-PARTY-NOTICES.txt")

# .NET RID -> (distribution, PyPI wheel platform tag).
#
# Mirrors NumPy's own selection in tools/wheels/cibw_before_build.sh: scipy-openblas64 everywhere
# except Windows on ARM64, which NumPy builds against scipy-openblas32.  Matching that matters --
# the two have different symbol schemes AND different BLAS integer widths (see OpenBlasNative.Bind).
# The glibc Linux tags name the PLATFORM, not one tag spelling: wheel_matches() accepts any manylinux
# tag of the same architecture (openblas-libs moved from manylinux2014 to manylinux_2_27/2_28 with
# 0.3.33.0.0, the release line NumPy 2.5 pins).
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


#: The glibc platform tags: PEP 513/571/599's aliases (manylinux1 = glibc 2.5, manylinux2010 = 2.12,
#: manylinux2014 = 2.17) and PEP 600's perennial manylinux_<major>_<minor>_<arch>.
_MANYLINUX = re.compile(r"^manylinux(?:1|2010|2014|_[0-9]+_[0-9]+)_(?P<arch>.+)$")


def wheel_platform_tags(filename):
    """The compound platform tag of a scipy-openblas wheel file name, split into its components.

    The wheels are named ``<dist>-<version>-py3-none-<platform>.whl``, and ``<platform>`` may join
    several tags with dots (``manylinux_2_27_x86_64.manylinux_2_28_x86_64``).

    :param filename: the wheel's file name.
    :returns: the platform tag components, in file-name order.
    :raises SystemExit: the name is not a ``py3-none`` wheel, the only shape openblas-libs publishes;
        guessing at another shape could stage a library built for a different platform.
    """
    sep = "-py3-none-"
    i = filename.find(sep)
    if i < 0 or not filename.endswith(".whl"):
        raise SystemExit("unexpected wheel name: %s" % filename)
    return filename[i + len(sep):-len(".whl")].split(".")


def glibc_arch(tag):
    """The CPU architecture a glibc (manylinux) platform tag names.

    :param tag: one platform tag component, e.g. ``manylinux2014_aarch64``.
    :returns: the architecture (``x86_64``, ``aarch64``, ...), or None for a tag that is not a glibc
        manylinux tag (Windows, macOS and musllinux tags are matched exactly, never by family).
    """
    m = _MANYLINUX.match(tag)
    return m.group("arch") if m else None


def wheel_matches(filename, anchor):
    """Whether a wheel serves the platform a RID's anchor tag names.

    A wheel matches when one of its platform tag components IS the anchor, or -- for glibc Linux only --
    when one of them is a manylinux tag of the anchor's architecture. The family rule exists because
    openblas-libs retagged its Linux wheels from ``manylinux2014_<arch>`` (glibc 2.17) to
    ``manylinux_2_27_<arch>.manylinux_2_28_<arch>`` with 0.3.33.0.0: the RID is the same glibc
    platform either way, and an exact match found no wheel at all for any release NumPy 2.5 pins.

    :param filename: the candidate wheel's file name.
    :param anchor: the RID's platform tag from RID_MAP (or the first component of a manifest wheel's tag).
    :returns: True when the wheel is built for that platform.
    """
    tags = wheel_platform_tags(filename)
    if anchor in tags:
        return True
    arch = glibc_arch(anchor)
    return arch is not None and any(glibc_arch(t) == arch for t in tags)


def select_wheel(files, anchor):
    """The wheels of a PyPI release that serve a RID (the caller requires exactly one).

    An exact tag match wins when there is exactly one (the behaviour before the family rule, unchanged
    for every platform that never retagged); otherwise the family match of wheel_matches() decides. A
    release that ships two glibc wheels for one architecture therefore returns both, and the caller
    fails loudly instead of staging an arbitrary one.

    :param files: the release's file records from the PyPI JSON API (``filename``, ``url``, ``digests``).
    :param anchor: the RID's platform tag.
    :returns: the matching file records.
    """
    wheels = [f for f in files if f["filename"].endswith(".whl")]
    exact = [f for f in wheels if anchor in wheel_platform_tags(f["filename"])]
    if len(exact) == 1:
        return exact
    return [f for f in wheels if wheel_matches(f["filename"], anchor)]


def read_numpy_pin():
    """The repository's NumPy pin (numpy-version.txt), the default --numpy-version of a refresh.

    :returns: the pinned version string, or None outside a repository checkout (a copied package folder).
    """
    try:
        with open(NUMPY_PIN_PATH, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


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


def refresh_manifest(distribution_version, numpy_version, openblas_version):
    """Re-derive the pin from PyPI. Downloads every wheel to record the EXTRACTED library's hash.

    :param distribution_version: the scipy-openblas32/64 release NumPy pins in requirements/ci_requirements.txt.
    :param numpy_version: the NumPy release that pins it (numpy-version.txt).
    :param openblas_version: what that NumPy's ``numpy.show_config()`` reports as its BLAS version, which
        make_env.py verifies (``0.3.31.dev`` for NumPy 2.4.2, ``0.3.34.106.0`` for NumPy 2.5.4).
    :returns: the manifest written to openblas-manifest.json.
    :raises SystemExit: a RID has no wheel (or more than one) in that release, or a download fails its hash.
    """
    print("refreshing manifest for %s ..." % distribution_version)
    index = {}
    for dist in sorted({d for d, _ in RID_MAP.values()}):
        url = "https://pypi.org/pypi/%s/%s/json" % (dist, distribution_version)
        with urllib.request.urlopen(url) as resp:
            index[dist] = json.load(resp)

    runtimes = {}
    for rid in RID_MAP:
        dist, tag = RID_MAP[rid]
        # Platform tags can be compound ("...-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
        # and glibc Linux is matched by family: see select_wheel / wheel_matches.
        matches = select_wheel(index[dist]["urls"], tag)
        if len(matches) != 1:
            raise SystemExit("expected exactly one %s wheel for %s, got %d"
                             % (dist, tag, len(matches)))
        u = matches[0]
        print("  %-18s %s" % (rid, u["filename"]))
        blob = download(u["url"], u["digests"]["sha256"])
        with zipfile.ZipFile(os.path.join(CACHE_DIR, u["filename"])) as zf:
            member = pick_native_member(zf)
            payload = zf.read(member)
            # The vendored Fortran runtime is pinned too: every byte the package redistributes is
            # named in the manifest, and a staging that lost a dep (or carries a wrong one) fails
            # `--check` instead of shipping a dylib that cannot load.
            vendored = []
            for dep in vendored_members(zf, member):
                dep_blob = zf.read(dep)
                vendored.append({
                    "member": dep,
                    "file": os.path.basename(dep),
                    "sha256": sha256_bytes(dep_blob),
                    "size": len(dep_blob),
                })
        runtimes[rid] = {
            "distribution": dist,
            "wheel": u["filename"],
            "url": u["url"],
            "wheel_sha256": u["digests"]["sha256"],
            "member": member,
            "file": os.path.basename(member),
            "sha256": sha256_bytes(payload),
            "size": len(payload),
            "vendored": vendored,
        }

    manifest = {
        "_comment": (
            "PIN. Do not edit by hand -- regenerate with "
            "`python fetch_openblas.py --refresh-manifest`. These are the exact prebuilt OpenBLAS "
            "artifacts NumPy %s bundles (see numpy requirements/ci_requirements.txt), which is what "
            "makes NumSharp's matrix products bit-identical to that NumPy's." % numpy_version),
        "numpy_version": numpy_version,
        "distribution_version": distribution_version,
        "openblas_version": openblas_version,
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
    lines.append("Bundled artifacts (sha256 of the extracted shared library; the vendored GCC runtime")
    lines.append("libraries each wheel carries beside it are listed indented under their RID):")
    lines.append("")
    for rid in sorted(manifest["runtimes"]):
        e = manifest["runtimes"][rid]
        lines.append("  %-18s %-28s %s" % (rid, e["file"], e["sha256"]))
        for dep in e.get("vendored", []):
            lines.append("  %-18s %-28s %s" % ("", dep["file"], dep["sha256"]))
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
    """Where a vendored dep is staged.

    Linux keeps the deps beside the main in native/ (the main's RUNPATH is $ORIGIN), exactly as the
    wheel lays them out. macOS is the exception: the wheel parks them in a .dylibs/ SIBLING of lib/
    and the main dylib's load commands say @loader_path/../.dylibs/<name> — but NuGet delivers
    native assets only from UNDER runtimes/<rid>/native/ (nested paths included; a sibling folder is
    silently dropped from every consumer), so mirroring the wheel's sibling layout produced a package
    that looked complete and loaded on no consumer machine. The deps are therefore staged NESTED, as
    runtimes/<rid>/native/.dylibs/<name>, which NuGet does deliver; the runtime loader
    (MacOsVendoredRuntime) materializes the ../.dylibs sibling the load commands expect at first
    load — a relative symlink next to the dylib, or a verified per-user cache copy when that folder
    is read-only or a RID-specific publish has flattened the layout."""
    norm = member.replace("\\", "/")
    subdir = os.path.join("native", ".dylibs") if "/.dylibs/" in norm else "native"
    return os.path.join(RUNTIMES_DIR, rid, subdir, os.path.basename(member))


def legacy_sibling_dir(rid):
    """The pre-fix staging location of the macOS deps (runtimes/<rid>/.dylibs/), which the recursive
    pack glob would still sweep into the package as dead weight; removed whenever a RID is (re)staged."""
    return os.path.join(RUNTIMES_DIR, rid, ".dylibs")


def staged_ok(rid, entry):
    """True when the main library AND every vendored dep recorded in the manifest are staged with the
    pinned bytes. A manifest entry without a 'vendored' list (older pin) checks the main alone."""
    dest = staged_path(rid, entry)
    if not (os.path.isfile(dest) and sha256_bytes(open(dest, "rb").read()) == entry["sha256"]):
        return False
    for dep in entry.get("vendored", []):
        ddest = vendored_dest(rid, dep["member"])
        if not (os.path.isfile(ddest) and sha256_bytes(open(ddest, "rb").read()) == dep["sha256"]):
            return False
    return True


def fetch(manifest, check_only=False):
    missing, ok = [], 0
    for rid in sorted(manifest["runtimes"]):
        entry = manifest["runtimes"][rid]
        dest = staged_path(rid, entry)
        # Main + every pinned vendored dep must be present and byte-exact; a tree staged by an older
        # script (main alone, or the deps in the pre-fix sibling folder) is re-staged, not accepted.
        if staged_ok(rid, entry) and not os.path.isdir(legacy_sibling_dir(rid)):
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
            # already-verified wheel — under native/ (see vendored_dest) so NuGet delivers it — and
            # hold each file to the manifest's own pin when the manifest records one.
            pinned = {d["member"]: d["sha256"] for d in entry.get("vendored", [])}
            for dep in vendored_members(zf, entry["member"]):
                blob = zf.read(dep)
                if dep in pinned and sha256_bytes(blob) != pinned[dep]:
                    raise SystemExit(
                        "CHECKSUM MISMATCH for the vendored dependency extracted from %s\n"
                        "  member   %s\n  expected %s\n  actual   %s"
                        % (entry["wheel"], dep, pinned[dep], sha256_bytes(blob)))
                ddest = vendored_dest(rid, dep)
                os.makedirs(os.path.dirname(ddest), exist_ok=True)
                with open(ddest, "wb") as fh:
                    fh.write(blob)
                print("  %-18s %s (vendored dep)" % ("", os.path.basename(dep)))
            for dep in pinned:
                if dep not in set(vendored_members(zf, entry["member"])):
                    raise SystemExit("the wheel %s no longer carries the pinned vendored member %s"
                                     % (entry["wheel"], dep))
        legacy = legacy_sibling_dir(rid)
        if os.path.isdir(legacy):
            shutil.rmtree(legacy)
            print("  %-18s removed the pre-fix sibling folder %s" % ("", os.path.relpath(legacy, PKG_ROOT)))
        ok += 1

    if check_only and missing:
        print("MISSING or incomplete staged runtime assets for: %s" % ", ".join(missing))
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
    ap.add_argument("--distribution-version", default=None,
                    help="the scipy-openblas release to pin (default: the manifest's current one)")
    ap.add_argument("--numpy-version", default=None,
                    help="the NumPy release that pins it (default: numpy-version.txt)")
    ap.add_argument("--openblas-version", default=None,
                    help="the BLAS version that NumPy's show_config() reports (default: --distribution-version)")
    args = ap.parse_args()

    if args.clean:
        if os.path.isdir(RUNTIMES_DIR):
            shutil.rmtree(RUNTIMES_DIR)
            print("removed %s" % RUNTIMES_DIR)
        return 0

    if args.refresh_manifest:
        # Defaults re-derive the CURRENT pin: the manifest's own distribution and the repository's NumPy pin.
        current = load_manifest() if os.path.isfile(MANIFEST_PATH) else {}
        distribution_version = args.distribution_version or current.get("distribution_version")
        numpy_version = args.numpy_version or read_numpy_pin() or current.get("numpy_version")
        if not distribution_version or not numpy_version:
            raise SystemExit("--refresh-manifest needs --distribution-version and --numpy-version "
                             "(no manifest or numpy-version.txt to default them from)")
        manifest = refresh_manifest(distribution_version, numpy_version,
                                    args.openblas_version or distribution_version)
    else:
        manifest = load_manifest()
        write_notices(manifest)

    return fetch(manifest, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
