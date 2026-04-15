# MSTest `--filter` Guide

## Filter Syntax

MSTest uses a simple property-based filter syntax:

```
--filter "Property=Value"
--filter "Property!=Value"
--filter "Property~Value"     # Contains
--filter "Property!~Value"    # Does not contain
```

Combine with `&` (AND) or `|` (OR):
```
--filter "Property1=A&Property2=B"   # AND
--filter "Property1=A|Property2=B"   # OR
```

## Available Properties

| Property | Description | Example |
|----------|-------------|---------|
| `TestCategory` | Category attribute | `TestCategory=OpenBugs` |
| `ClassName` | Test class name | `ClassName~BinaryOpTests` |
| `Name` | Test method name | `Name~Add_Int32` |
| `FullyQualifiedName` | Full namespace.class.method | `FullyQualifiedName~Backends.Kernels` |

## 5 Concrete Examples

### 1. Exclude OpenBugs (CI-style run)

```bash
dotnet test --no-build --filter "TestCategory!=OpenBugs"
```

Runs all tests EXCEPT those marked `[OpenBugs]`. This is what CI uses.

### 2. Run ONLY OpenBugs (verify bug fixes)

```bash
dotnet test --no-build --filter "TestCategory=OpenBugs"
```

Runs only failing bug reproductions to check if your fix works.

### 3. Run single test class

```bash
dotnet test --no-build --filter "ClassName~CountNonzeroTests"
```

Runs all tests in classes containing `CountNonzeroTests`.

### 4. Run single test method

```bash
dotnet test --no-build --filter "Name=Add_TwoNumbers_ReturnsSum"
```

Runs only the test named exactly `Add_TwoNumbers_ReturnsSum`.

### 5. Run tests by namespace pattern

```bash
dotnet test --no-build --filter "FullyQualifiedName~Backends.Kernels"
```

Runs all tests in the `Backends.Kernels` namespace.

## Quick Reference

| Goal | Filter |
|------|--------|
| Exclude category | `TestCategory!=OpenBugs` |
| Include category only | `TestCategory=OpenBugs` |
| Single class (exact) | `ClassName=BinaryOpTests` |
| Class contains | `ClassName~BinaryOp` |
| Method contains | `Name~Add_` |
| Namespace contains | `FullyQualifiedName~Backends.Kernels` |
| Multiple categories (AND) | `TestCategory!=OpenBugs&TestCategory!=WindowsOnly` |
| Multiple categories (OR) | `TestCategory=OpenBugs\|TestCategory=Misaligned` |

## Operators

| Op | Meaning | Example |
|----|---------|---------|
| `=` | Equals | `TestCategory=Unit` |
| `!=` | Not equals | `TestCategory!=Slow` |
| `~` | Contains | `Name~Integration` |
| `!~` | Does not contain | `ClassName!~Legacy` |
| `&` | AND | `TestCategory!=A&TestCategory!=B` |
| `\|` | OR | `TestCategory=A\|TestCategory=B` |

**Important:**
- Use `\|` (escaped pipe) for OR in bash
- Parentheses are NOT needed for combining filters
- Filter values are case-sensitive

## NumSharp Categories

| Category | Purpose | CI Behavior |
|----------|---------|-------------|
| `OpenBugs` | Known-failing bug reproductions | **Excluded** |
| `HighMemory` | Requires 8GB+ RAM | **Excluded** |
| `Misaligned` | NumSharp vs NumPy differences (tests pass) | Runs |
| `WindowsOnly` | Requires GDI+/System.Drawing | Excluded on Linux/macOS |
| `LongIndexing` | Tests > int.MaxValue elements | Runs |

## Useful Commands

```bash
# Stop on first failure (MSTest v3)
dotnet test --no-build -- --fail-on-failure

# Verbose output (see passed tests too)
dotnet test --no-build -v normal

# List tests without running
dotnet test --no-build --list-tests

# CI-style: exclude OpenBugs and HighMemory
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"

# Windows CI: full exclusion list
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"

# Linux/macOS CI: also exclude WindowsOnly
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=WindowsOnly"
```

## Advanced Filter Examples

### Example A: Specific Test Method

```bash
dotnet test --no-build --filter "FullyQualifiedName=NumSharp.UnitTest.Backends.Kernels.VarStdComprehensiveTests.Var_2D_Axis0"
```

**Result:** 1 test

### Example B: Pattern Matching Multiple Classes

```bash
dotnet test --no-build --filter "ClassName~Comprehensive&Name~_2D_"
```

**Result:** Tests in `*Comprehensive*` classes with `_2D_` in method name

### Example C: Namespace + Category Filter

```bash
dotnet test --no-build --filter "FullyQualifiedName~Backends.Kernels&TestCategory!=OpenBugs"
```

**Result:** All Kernels tests except OpenBugs

### Example D: Multiple Categories (OR)

```bash
dotnet test --no-build --filter "TestCategory=OpenBugs|TestCategory=Misaligned"
```

**Result:** Tests that have EITHER `[OpenBugs]` OR `[Misaligned]` attribute

## Migration from TUnit

| TUnit Filter | MSTest Filter |
|--------------|---------------|
| `--treenode-filter "/*/*/*/*[Category!=X]"` | `--filter "TestCategory!=X"` |
| `--treenode-filter "/*/*/ClassName/*"` | `--filter "ClassName~ClassName"` |
| `--treenode-filter "/*/*/*/MethodName"` | `--filter "Name=MethodName"` |
| `--treenode-filter "/*/Namespace/*/*"` | `--filter "FullyQualifiedName~Namespace"` |
