using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// Tier-1 correctness bugs discovered in the nditer-branch audit (Group 2 — IL kernels + SIMD).
///
/// Source documents:
///   docs/plans/NDITER_BRANCH_QUALITY_AUDIT_V2.md
///   docs/plans/audit_v2/02_ilkernel_simd.md
///
/// Each test compares NumSharp behavior against NumPy 2.4.2 (the canonical reference).
/// Tests are marked [OpenBugs] so CI excludes them. Remove the attribute when the
/// underlying bug is fixed in source.
/// </summary>
[TestClass]
public class AuditV2_ILKernelSimd
{
    // ======================================================================
    // T1.6 — Scalar IL path emits NaN <= x and NaN >= x as True
    //
    // File: src/NumSharp.Core/Backends/Kernels/DirectILKernelGenerator.Comparison.cs
    // Function: EmitComparisonOperation (lines 1009–1036)
    //
    // Root cause: the scalar IL path emits
    //     a <= b   as   !(Cgt a b)
    //     a >= b   as   !(Clt a b)
    // For NaN both Cgt and Clt return false (ordered compares), so the
    // negation flips false -> true. NumPy spec: every ordered compare
    // involving NaN returns False (only != returns True).
    //
    // The SIMD path is correct because Vector256.LessThanOrEqual / GreaterThanOrEqual
    // propagate NaN as false. Only the scalar tail / small-array fallback is buggy.
    // Half goes through op_LessThanOrEqual (correct). Decimal/Complex have their
    // own paths and are also correct.
    //
    // Remediation: emit float/double LessEqual as `Clt OR Ceq` and GreaterEqual
    // as `Cgt OR Ceq` so NaN drops out as `false OR false = false`.
    // ======================================================================

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.6")]
    public void T1_6_NaN_LessEqual_ShouldReturnFalse_Float()
    {
        var nan = np.array(new float[] { float.NaN });
        var one = np.array(new float[] { 1.0f });
        var result = nan <= one;
        result.GetValue<bool>(0).Should().BeFalse(
            "NumPy: float32 NaN <= 1.0 is False (all NaN ordered comparisons return False per IEEE 754)");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.6")]
    public void T1_6_NaN_GreaterEqual_ShouldReturnFalse_Float()
    {
        var nan = np.array(new float[] { float.NaN });
        var one = np.array(new float[] { 1.0f });
        var result = nan >= one;
        result.GetValue<bool>(0).Should().BeFalse(
            "NumPy: float32 NaN >= 1.0 is False (all NaN ordered comparisons return False per IEEE 754)");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.6")]
    public void T1_6_NaN_LessEqual_ShouldReturnFalse_Double()
    {
        var nan = np.array(new double[] { double.NaN });
        var one = np.array(new double[] { 1.0 });
        var result = nan <= one;
        result.GetValue<bool>(0).Should().BeFalse(
            "NumPy: float64 NaN <= 1.0 is False per IEEE 754");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.6")]
    public void T1_6_NaN_GreaterEqual_ShouldReturnFalse_Double()
    {
        var nan = np.array(new double[] { double.NaN });
        var one = np.array(new double[] { 1.0 });
        var result = nan >= one;
        result.GetValue<bool>(0).Should().BeFalse(
            "NumPy: float64 NaN >= 1.0 is False per IEEE 754");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.6")]
    public void T1_6_LessEqual_NaN_Reverse_ShouldReturnFalse()
    {
        // Reverse direction: 1 <= NaN — symmetric bug.
        var one = np.array(new float[] { 1.0f });
        var nan = np.array(new float[] { float.NaN });
        var result = one <= nan;
        result.GetValue<bool>(0).Should().BeFalse(
            "NumPy: 1.0 <= NaN is False (NaN on either side returns False for ordered compares)");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.6")]
    public void T1_6_GreaterEqual_NaN_Reverse_ShouldReturnFalse()
    {
        var one = np.array(new float[] { 1.0f });
        var nan = np.array(new float[] { float.NaN });
        var result = one >= nan;
        result.GetValue<bool>(0).Should().BeFalse(
            "NumPy: 1.0 >= NaN is False (NaN on either side returns False for ordered compares)");
    }

    // ======================================================================
    // T1.6 SIBLING CHECKS — Less / Greater behave correctly
    //
    // These pass today. Documented here so the regression surface is explicit:
    // if a future change to EmitComparisonOperation reorders branches, breaking
    // < or > should be caught immediately.
    // ======================================================================

    [TestMethod]
    public void T1_6_Sibling_NaN_Less_ReturnsFalse_Already()
    {
        var nan = np.array(new float[] { float.NaN });
        var one = np.array(new float[] { 1.0f });
        (nan < one).GetValue<bool>(0).Should().BeFalse();
    }

    [TestMethod]
    public void T1_6_Sibling_NaN_Greater_ReturnsFalse_Already()
    {
        var nan = np.array(new float[] { float.NaN });
        var one = np.array(new float[] { 1.0f });
        (nan > one).GetValue<bool>(0).Should().BeFalse();
    }

    [TestMethod]
    public void T1_6_Sibling_NaN_Equal_ReturnsFalse_Already()
    {
        var nan = np.array(new float[] { float.NaN });
        var one = np.array(new float[] { 1.0f });
        (nan == one).GetValue<bool>(0).Should().BeFalse();
    }

    [TestMethod]
    public void T1_6_Sibling_NaN_NotEqual_ReturnsTrue_Already()
    {
        var nan = np.array(new float[] { float.NaN });
        var one = np.array(new float[] { 1.0f });
        (nan != one).GetValue<bool>(0).Should().BeTrue();
    }

    // ======================================================================
    // T1.11 — np.fmax / np.fmin propagate NaN instead of skipping
    //
    // File: src/NumSharp.Core/Math/np.maximum.cs (fmax)
    //       src/NumSharp.Core/Math/np.minimum.cs (fmin)
    //
    // Root cause: fmax and fmin are implemented as thin wrappers that delegate
    // to np.clip with a_min=x2 / a_max=x2 respectively. clip propagates NaN
    // (matching np.maximum/np.minimum semantics), but the NumPy contract for
    // fmax/fmin is the OPPOSITE — they must IGNORE NaN where possible and
    // return the non-NaN operand.
    //
    // NumPy 2.x semantics:
    //   np.maximum / np.minimum -> NaN propagates ("NaN wins")
    //   np.fmax    / np.fmin    -> NaN is skipped (returns the non-NaN side)
    // Implemented identically in NumSharp ⇒ fmax/fmin are wrong.
    //
    // Remediation: route fmax/fmin through a NaN-aware element kernel that
    // chooses the non-NaN operand. Equivalent to NumPy's `npy_fmax` /
    // `npy_fmin` core loops.
    // ======================================================================

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.11")]
    public void T1_11_Fmax_SkipsNaN_ReturnsOtherOperand()
    {
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var b = np.array(new double[] { double.NaN, 2.0, double.NaN });
        var result = np.fmax(a, b);
        // NumPy 2.4.2: np.fmax(a, b) == [1.0, 2.0, 3.0]
        result.GetDouble(0).Should().Be(1.0);
        result.GetDouble(1).Should().Be(2.0, "fmax should skip NaN and return the other operand");
        result.GetDouble(2).Should().Be(3.0);
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.11")]
    public void T1_11_Fmin_SkipsNaN_ReturnsOtherOperand()
    {
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var b = np.array(new double[] { double.NaN, 2.0, double.NaN });
        var result = np.fmin(a, b);
        // NumPy 2.4.2: np.fmin(a, b) == [1.0, 2.0, 3.0]
        result.GetDouble(0).Should().Be(1.0);
        result.GetDouble(1).Should().Be(2.0, "fmin should skip NaN and return the other operand");
        result.GetDouble(2).Should().Be(3.0);
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.11")]
    public void T1_11_Fmax_NaN_And_Number_ReturnsNumber()
    {
        var a = np.array(new double[] { double.NaN });
        var b = np.array(new double[] { 5.0 });
        var result = np.fmax(a, b);
        result.GetDouble(0).Should().Be(5.0, "fmax(NaN, 5) must return 5 per NumPy");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.11")]
    public void T1_11_Fmin_NaN_And_Number_ReturnsNumber()
    {
        var a = np.array(new double[] { double.NaN });
        var b = np.array(new double[] { 5.0 });
        var result = np.fmin(a, b);
        result.GetDouble(0).Should().Be(5.0, "fmin(NaN, 5) must return 5 per NumPy");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.11")]
    public void T1_11_Fmax_BothNaN_ReturnsNaN()
    {
        // When both sides are NaN, NumPy returns NaN (no other operand to fall back to).
        var a = np.array(new double[] { double.NaN });
        var b = np.array(new double[] { double.NaN });
        var result = np.fmax(a, b);
        double.IsNaN(result.GetDouble(0)).Should().BeTrue(
            "fmax(NaN, NaN) is NaN — only one NaN can be skipped");
    }

    // np.maximum / np.minimum are NaN-propagating per NumPy contract — those
    // tests pass today and would catch a regression if someone wired them to
    // the fmax/fmin code path while fixing T1.11.

    [TestMethod]
    public void T1_11_Sibling_Maximum_PropagatesNaN_Already()
    {
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var b = np.array(new double[] { double.NaN, 2.0, double.NaN });
        var result = np.maximum(a, b);
        double.IsNaN(result.GetDouble(0)).Should().BeTrue("np.maximum propagates NaN");
        double.IsNaN(result.GetDouble(1)).Should().BeTrue();
        double.IsNaN(result.GetDouble(2)).Should().BeTrue();
    }

    [TestMethod]
    public void T1_11_Sibling_Minimum_PropagatesNaN_Already()
    {
        var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
        var b = np.array(new double[] { double.NaN, 2.0, double.NaN });
        var result = np.minimum(a, b);
        double.IsNaN(result.GetDouble(0)).Should().BeTrue("np.minimum propagates NaN");
        double.IsNaN(result.GetDouble(1)).Should().BeTrue();
        double.IsNaN(result.GetDouble(2)).Should().BeTrue();
    }

    // ======================================================================
    // T1.36 — int ** -int returns silent integer-truncation values
    //
    // File: src/NumSharp.Core/Backends/Default/Math/Default.Power.cs
    // Functions: PowSByte/PowInt16/PowInt32/PowInt64 (handle b < 0 branches)
    //
    // Current behavior (deterministic but wrong-by-API):
    //     base=2  exp=-1 -> 0
    //     base=1  exp=-1 -> 1
    //     base=-1 exp=-1 -> -1
    //     base=10 exp=-1 -> 0
    //
    // NumPy 2.4.2 behavior:
    //     np.array([2], dtype=np.int32) ** np.array([-1], dtype=np.int32)
    //     -> ValueError: Integers to negative integer powers are not allowed.
    //
    // The current "floor of real reciprocal" implementation produces predictable
    // numbers but silently hides the user's intent. NumPy intentionally rejects
    // this because the integer dtype cannot represent the true fractional answer.
    //
    // Remediation: detect any negative element in the integer exponent before
    // entering the per-element loop in PowerInteger and throw a ValueError-
    // equivalent (e.g. ArgumentException) matching NumPy's diagnostic.
    // ======================================================================

    [TestMethod]
    public void T1_36_Int32_NegativeExponent_ShouldThrow()
    {
        var a = np.array(new int[] { 2 });
        var b = np.array(new int[] { -1 });
        Action act = () => { var r = np.power(a, b); };
        act.Should().Throw<Exception>(
            "NumPy raises 'Integers to negative integer powers are not allowed.' " +
            "Silently returning truncated values (e.g. 0) hides loss of precision.");
    }

    [TestMethod]
    public void T1_36_Int64_NegativeExponent_ShouldThrow()
    {
        var a = np.array(new long[] { 5L });
        var b = np.array(new long[] { -2L });
        Action act = () => { var r = np.power(a, b); };
        act.Should().Throw<Exception>(
            "NumPy raises ValueError for int**-int. Currently returns 0 silently.");
    }

    [TestMethod]
    public void T1_36_Int16_NegativeExponent_ShouldThrow()
    {
        var a = np.array(new short[] { (short)4 });
        var b = np.array(new short[] { (short)-3 });
        Action act = () => { var r = np.power(a, b); };
        act.Should().Throw<Exception>(
            "NumPy raises ValueError for int**-int.");
    }

    [TestMethod]
    public void T1_36_NumPy_Even_Throws_For_Base_One()
    {
        // Important parity nuance: NumPy throws even when the mathematical answer
        // would be representable (e.g. base=1, base=-1). The check is on the
        // exponent dtype/sign, not on the per-element feasibility. NumSharp
        // currently returns the "mathematically correct" answer (1 and -1) for
        // these special cases — also wrong w.r.t. NumPy contract.
        var a = np.array(new int[] { 1, -1 });
        var b = np.array(new int[] { -1, -1 });
        Action act = () => { var r = np.power(a, b); };
        act.Should().Throw<Exception>(
            "NumPy throws unconditionally for any negative integer exponent — " +
            "it does not special-case base=±1.");
    }

    // ======================================================================
    // EXTRA — Missing `sbyte` in IsSimdSupported<T>
    //
    // File: src/NumSharp.Core/Backends/Kernels/DirectILKernelGenerator.Binary.cs:451
    // Function: IsSimdSupported<T>()
    //
    // Issue: sbyte is excluded from the SIMD allow-list even though
    // Vector256<sbyte> is fully supported on .NET 8+. byte is in the list,
    // sbyte is not — asymmetric coverage. This is a performance gap rather
    // than a correctness defect: sbyte ops still produce correct results via
    // the scalar fallback, just slower than byte equivalents.
    //
    // Documented as a OpenBugs test so any future perf benchmark comparing
    // sbyte vs byte can use this as a regression marker once fixed.
    // ======================================================================

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-sbyte-simd")]
    public void Extra_Sbyte_Add_ProducesCorrectResults_ButNotSimd()
    {
        // This test passes today (correctness OK) but documents the perf gap.
        // Remove [OpenBugs] when sbyte is added to IsSimdSupported<T>().
        var a = np.array(new sbyte[] { 1, 2, 3, 4 });
        var b = np.array(new sbyte[] { 10, 20, 30, 40 });
        var r = a + b;
        r.GetValue<sbyte>(0).Should().Be((sbyte)11);
        r.GetValue<sbyte>(3).Should().Be((sbyte)44);

        // Marked OpenBugs so it appears in the audit-driven test list; reviewer
        // should verify SIMD path is reached after fix via micro-benchmark.
        Assert.Inconclusive(
            "Correctness OK. Perf gap: sbyte missing from IsSimdSupported<T> in " +
            "DirectILKernelGenerator.Binary.cs:451. Vector256<sbyte> is supported on .NET 8+.");
    }
}
