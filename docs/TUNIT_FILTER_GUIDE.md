# TUnit `--treenode-filter` Guide

## Filter Path Structure

```
/<Assembly>/<Namespace>/<ClassName>/<TestMethodName>[Property]
```

Use `*` as wildcard. Properties filter by `[Category(...)]` attributes.

## 5 Concrete Examples

### 1. Exclude OpenBugs (CI-style run)

```bash
dotnet test --no-build -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"
```

Runs all tests EXCEPT those marked `[OpenBugs]`. This is what CI uses.

### 2. Run ONLY OpenBugs (verify bug fixes)

```bash
dotnet test --no-build -- --treenode-filter "/*/*/*/*[Category=OpenBugs]"
```

Runs only failing bug reproductions to check if your fix works.

### 3. Run single test class

```bash
dotnet test --no-build -- --treenode-filter "/*/*/CountNonzeroTests/*"
```

Runs all tests in `CountNonzeroTests` class regardless of namespace.

### 4. Run single test method

```bash
dotnet test --no-build -- --treenode-filter "/*/*/*/Add_TwoNumbers_ReturnsSum"
```

Runs only the test named `Add_TwoNumbers_ReturnsSum`.

### 5. Run tests by namespace pattern

```bash
dotnet test --no-build -- --treenode-filter "/*/NumSharp.UnitTest.Backends.Kernels/*/*"
```

Runs all tests in the `NumSharp.UnitTest.Backends.Kernels` namespace.

## Quick Reference

| Goal | Filter |
|------|--------|
| Exclude category | `/*/*/*/*[Category!=OpenBugs]` |
| Include category only | `/*/*/*/*[Category=OpenBugs]` |
| Single class | `/*/*/ClassName/*` |
| Single method | `/*/*/*/MethodName` |
| Namespace | `/*/Full.Namespace.Path/*/*` |
| Multiple categories (AND) | `/*/*/*/*[(Category!=OpenBugs)&(Category!=WindowsOnly)]` |
| Multiple categories (OR) | `/*/*/*/*[(Category=OpenBugs)\|(Category=Misaligned)]` |
| Multiple classes | Use wildcards: `/*/*/*Comprehensive*/*` |

## Operators

| Op | Meaning | Where | Example |
|----|---------|-------|---------|
| `*` | Wildcard | Path & Properties | `*Tests*`, `[Category=Open*]` |
| `=` | Equals | Properties | `[Category=Unit]` |
| `!=` | Not equals | Properties | `[Category!=Slow]` |
| `&` | AND | Properties | `[(Category=A)&(Priority=High)]` |
| `\|` | OR | **Properties ONLY** | `[(Category=A)\|(Category=B)]` |

**Important:**
- OR (`|`) only works for property filters, NOT for path segments
- AND (`&`) requires outer brackets: `[(A)&(B)]` not `[A]&[B]`

## NumSharp Categories

| Category | Purpose | CI Behavior |
|----------|---------|-------------|
| `OpenBugs` | Known-failing bug reproductions | **Excluded** |
| `Misaligned` | NumSharp vs NumPy differences (tests pass) | Runs |
| `WindowsOnly` | Requires GDI+/System.Drawing | Excluded on Linux/macOS |

## Useful Flags

```bash
# Stop on first failure
dotnet test --no-build -- --fail-fast --treenode-filter "..."

# Detailed output (see passed tests too)
dotnet test --no-build -- --output Detailed --treenode-filter "..."

# List tests without running
dotnet test --no-build -- --list-tests

# Combine: detailed + fail-fast + filter (exclude OpenBugs and WindowsOnly)
dotnet test --no-build -- --output Detailed --fail-fast \
  --treenode-filter "/*/*/*/*[(Category!=OpenBugs)&(Category!=WindowsOnly)]"
```

## Advanced Filter Examples (Tested & Working)

### Example A: All Path Parameters Specified

```bash
dotnet test --no-build -- --treenode-filter "/*/NumSharp.UnitTest.Backends.Kernels/VarStdComprehensiveTests/Var_2D_Axis0"
```

**Result:** 1 test ✅

| Part | Value |
|------|-------|
| Assembly | `*` (wildcard - assembly name varies) |
| Namespace | `NumSharp.UnitTest.Backends.Kernels` |
| Class | `VarStdComprehensiveTests` |
| Method | `Var_2D_Axis0` |

### Example B: All Path Parameters with Wildcards

```bash
dotnet test --no-build -- --treenode-filter "/*/*.Backends.*/*Comprehensive*/*_2D_*"
```

**Result:** 29 tests ✅

| Part | Pattern | Matches |
|------|---------|---------|
| Assembly | `*` | Any |
| Namespace | `*.Backends.*` | Contains `.Backends.` |
| Class | `*Comprehensive*` | Contains `Comprehensive` |
| Method | `*_2D_*` | Contains `_2D_` |

### Example C: Multiple Classes via Wildcard Pattern

```bash
dotnet test --no-build -- --treenode-filter "/*/*/VarStd*/*"
```

**Result:** 77 tests ✅

Matches `VarStdComprehensiveTests` and any other class starting with `VarStd`.

**Note:** OR (`|`) does NOT work for path segments. Use wildcards instead.

### Example D: OR for Multiple Categories (Properties)

```bash
dotnet test --no-build -- --treenode-filter "/*/*/*/*[(Category=OpenBugs)|(Category=Misaligned)]"
```

**Result:** 277 tests ✅

Runs tests that have EITHER `[OpenBugs]` OR `[Misaligned]` attribute.

### Example E: Combined - Wildcards + Property Filters (AND)

```bash
dotnet test --no-build -- --treenode-filter "/*/*.Backends*/*/*[(Category!=OpenBugs)&(Category!=WindowsOnly)]"
```

**Result:** 1877 tests ✅

| Part | Pattern | Effect |
|------|---------|--------|
| Assembly | `*` | Any |
| Namespace | `*.Backends*` | Backend tests only |
| Class | `*` | All classes |
| Method | `*` | All methods |
| Properties | `[(Category!=OpenBugs)&(Category!=WindowsOnly)]` | Exclude both categories |

**Note:** Wrap AND conditions in outer brackets: `[(A)&(B)]`

## What Works vs What Doesn't

| Pattern Type | Works? | Example |
|--------------|--------|---------|
| Path wildcards | ✅ | `/*/*/Var*/*` |
| Property equals | ✅ | `[Category=OpenBugs]` |
| Property not equals | ✅ | `[Category!=OpenBugs]` |
| Property AND (with brackets) | ✅ | `[(A=1)&(B=2)]` |
| Property OR | ✅ | `[(A=1)\|(B=2)]` |
| Property AND without brackets | ❌ | `[A=1]&[B=2]` → 0 matches |
| Path OR `(A)\|(B)` | ❌ | Does not match |
| `~=` contains | ❌ | Not supported |
