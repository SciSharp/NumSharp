#!/usr/bin/env bash
# ==============================================================================================
# Level D — the analyzer, as a green assertion over a build that MUST fail.
#
# The CounterExamples project deliberately does not compile: each method in BadShapes.cs trips one
# NDW00x analyzer error. This script builds it, asserts the build FAILED and that every expected
# error code appears in the output, then prints ANALYZER-OK. It mirrors verify_build_package.sh's
# step 11 (the analyzer catches these at CoreCompile and preempts the weaver).
# ==============================================================================================
set -u
cd "$(dirname "$0")" || exit 2

# If a caller (the `dotnet run -- level D` driver) handed us the SDK directory, put it on PATH: a
# non-interactive shell does not load the profile that normally provides 'dotnet'. cygpath converts
# the Windows path on Git Bash; on Linux/macOS it is already POSIX (cygpath absent -> fall through).
if [ -n "${NDSCOPING_DOTNET_DIR:-}" ]; then
  export PATH="$(cygpath -u "$NDSCOPING_DOTNET_DIR" 2>/dev/null || echo "$NDSCOPING_DOTNET_DIR"):$PATH"
fi

codes=(NDW002 NDW003 NDW005 NDW006 NDW009 NDW010 NDW011)

echo "building CounterExamples — expected to FAIL with analyzer errors ..."
echo "----------------------------------------------------------------"
out="$(dotnet build NDScoping.CounterExamples.csproj -c Debug --nologo -v q 2>&1)"
rc=$?

# Show only the analyzer error lines (the full log is verbose); fall back to all output if none matched.
grep -E "error NDW0(0[2356]|09|1[01])" <<< "$out" || echo "$out"
echo "----------------------------------------------------------------"

if [ $rc -eq 0 ]; then
  echo "ANALYZER-FAIL: the project BUILT — expected a compile failure (is the analyzer applied?)"
  exit 1
fi

missing=()
for c in "${codes[@]}"; do
  grep -q "$c" <<< "$out" || missing+=("$c")
done

if [ ${#missing[@]} -ne 0 ]; then
  echo "ANALYZER-FAIL: expected error codes not found: ${missing[*]}"
  exit 1
fi

echo "all expected analyzer errors present: ${codes[*]}"
echo "ANALYZER-OK"
