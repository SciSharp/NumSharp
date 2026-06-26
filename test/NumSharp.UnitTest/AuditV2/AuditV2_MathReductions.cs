using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// NDITER branch audit v2 — Tier 1 correctness bugs in DefaultEngine Math/Reductions/BLAS.
///
/// Source documents:
///   docs/plans/NDITER_BRANCH_QUALITY_AUDIT_V2.md
///   docs/plans/audit_v2/03_default_math_reductions.md
///
/// Each test asserts the NumPy 2.4.2-correct behavior. Tests are tagged [OpenBugs] so they
/// FAIL today (documenting the bug) and PASS once the underlying issue is fixed.
/// Remove [OpenBugs] when the bug is fixed.
/// </summary>
[TestClass]
public class AuditV2_MathReductions
{
    // ============================================================================
    // T1.3 — np.power on sliced/broadcast integer arrays CRASHES
    // ============================================================================
    // File: src/NumSharp.Core/Backends/Default/Math/Default.Power.cs (PowerInteger, lines 50-128)
    //
    // PowerInteger calls `lhs.Unsafe.Address` / `rhs.Unsafe.Address` directly, which
    // throws `InvalidOperationException` when either operand is sliced or broadcasted
    // (see NDArray.Unmanaged.cs guard). The fast-path is taken whenever both operands
    // share the same integer dtype and shape, regardless of contiguity. Even if the
    // address WERE available the loop `d[i] = Pow(a[i], b[i])` ignores strides.
    //
    // NumPy: walks strides natively, returns correct values.
    // NumSharp: crashes with InvalidOperationException("Can't return a memory address
    //           when NDArray is sliced or broadcasted.").

    /// <summary>
    /// T1.3a — np.power on sliced int32 must not crash and must produce NumPy-correct values.
    /// Fixed: PowerInteger fast-path with Unsafe.Address removed; ExecuteBinaryOp now uses
    /// integer-aware IL kernels that walk strides natively.
    /// </summary>
    [TestMethod]
    public void T1_3a_Power_SlicedInt32_ShouldNotCrash()
    {
        var arr = np.arange(20).astype(NPTypeCode.Int32);
        var sliced = arr["::2"];                        // [0,2,4,6,8,10,12,14,16,18], sliced view
        var b = np.arange(10).astype(NPTypeCode.Int32); // [0,1,2,3,4,5,6,7,8,9]

        Action act = () => np.power(sliced, b);
        act.Should().NotThrow(
            "NumPy walks strides natively and never crashes on power(sliced_int, int); " +
            "PowerInteger fast-path must guard for contiguity or stride-walk properly.");

        // Once it stops crashing, also check a few values vs NumPy ground truth:
        //   np.power([0,2,4,6,8,10,12,14,16,18], [0,1,2,3,4,5,6,7,8,9])
        //   = [1, 2, 16, 216, 4096, 100000, 2985984, 105413504, 0, 790794752]   (int32 wrap)
        var r = np.power(sliced, b);
        r.GetInt32(0).Should().Be(1, "0**0 = 1");
        r.GetInt32(1).Should().Be(2, "2**1 = 2");
        r.GetInt32(2).Should().Be(16, "4**2 = 16");
        r.GetInt32(3).Should().Be(216, "6**3 = 216");
    }

    /// <summary>
    /// T1.3b — np.power on broadcast int32 must not crash.
    /// Fixed alongside T1.3a.
    /// </summary>
    [TestMethod]
    public void T1_3b_Power_BroadcastInt32_ShouldNotCrash()
    {
        var a = np.array(new int[] { 2 });
        var b = np.array(new int[] { 3, 3, 3 });
        var bc = np.broadcast_to(a, b.Shape);   // shape (3,), stride 0

        Action act = () => np.power(bc, b);
        act.Should().NotThrow(
            "NumPy: np.power(broadcast_to([2],(3,)), [3,3,3]) = [8,8,8]; " +
            "NumSharp PowerInteger fast-path crashes on broadcasted operand.");

        var r = np.power(bc, b);
        r.GetInt32(0).Should().Be(8);
        r.GetInt32(1).Should().Be(8);
        r.GetInt32(2).Should().Be(8);
    }

    // ============================================================================
    // T1.4 — np.reciprocal on sliced integer arrays CRASHES
    // ============================================================================
    // File: src/NumSharp.Core/Backends/Default/Math/Default.Reciprocal.cs
    //       (ReciprocalInteger, lines 24-95)
    //
    // Same pattern as T1.3 — uses `nd.Unsafe.Address` without guarding contiguity.

    /// <summary>
    /// T1.4 — np.reciprocal on sliced int32 must not crash.
    /// </summary>
    [TestMethod]
    public void T1_4_Reciprocal_SlicedInt32_ShouldNotCrash()
    {
        var a = np.arange(1, 21).astype(NPTypeCode.Int32);  // [1..20]
        var sliced = a["::2"];                              // [1,3,5,7,9,11,13,15,17,19]

        Action act = () => np.reciprocal(sliced);
        act.Should().NotThrow(
            "NumPy: np.reciprocal([1,3,5,...,19]) = [1,0,0,0,0,0,0,0,0,0] (integer trunc); " +
            "NumSharp ReciprocalInteger fast-path crashes on sliced operand.");

        var r = np.reciprocal(sliced);
        r.GetInt32(0).Should().Be(1, "1/1 = 1");
        for (int i = 1; i < 10; i++)
            r.GetInt32(i).Should().Be(0, "1/x truncates to 0 for |x|>=2 in integer dtype");
    }

    // ============================================================================
    // T1.5 — np.dot(N-D, 1-D) for N>=3 returns wrong shape (axis hardcoded to 1)
    // ============================================================================
    // File: src/NumSharp.Core/Backends/Default/Math/BLAS/Default.Dot.cs:60
    //
    //   if (leftshape.NDim >= 2 && rightshape.NDim == 1)
    //   {
    //       return np.sum(left * right, axis: 1);   // <-- hardcoded axis 1!
    //   }
    //
    // The comment above is even tagged "//TODO! this doesn't seem right, read desc".
    // NumPy: "If a is an N-D array and b is a 1-D array, it is a sum product over the
    //         last axis of a and b." → axis = ndim-1.
    // For 3D@1D the hardcoded axis:1 sums the WRONG dimension (and also corrupts
    // storage layout — the resulting buffer is smaller than the wrong shape claims).

    /// <summary>
    /// T1.5 — np.dot(3D, 1D) should produce shape (lhs[0], lhs[1]) by summing the last axis.
    /// </summary>
    [TestMethod]
    public void T1_5_Dot_3D_1D_ShouldSumLastAxis()
    {
        // NumPy ground truth:
        //   a = np.arange(24).reshape(2,3,4); b = np.array([1,2,3,4])
        //   np.dot(a, b) -> shape (2,3), values [[20,60,100],[140,180,220]]
        var a = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Int32);
        var b = np.array(new int[] { 1, 2, 3, 4 });

        var r = np.dot(a, b);
        r.shape.Should().Equal(new long[] { 2, 3 },
            "NumPy: np.dot((2,3,4), (4,)) = shape (2,3); NumSharp uses hardcoded axis:1 in " +
            "Default.Dot.cs:60 -> returns (2,4) instead.");

        // Values:
        r.GetInt32(0, 0).Should().Be(20);   // 0*1+1*2+2*3+3*4 = 20
        r.GetInt32(0, 1).Should().Be(60);   // 4*1+5*2+6*3+7*4 = 60
        r.GetInt32(0, 2).Should().Be(100);  // 8*1+9*2+10*3+11*4 = 100
        r.GetInt32(1, 0).Should().Be(140);
        r.GetInt32(1, 1).Should().Be(180);
        r.GetInt32(1, 2).Should().Be(220);
    }

    // ============================================================================
    // T1.14 — np.convolve accumulates in double, loses int64 precision
    // ============================================================================
    // File: src/NumSharp.Core/Math/NdArray.Convolve.cs:138-188 (ConvolveFullTyped<T>)
    //
    // The inner loop accumulates `double sum = 0` even for int64 input. Values whose
    // magnitude exceeds 2^53 (double's mantissa precision) lose precision. NumPy uses
    // type-specific accumulators (int64 sum for int64 input).

    /// <summary>
    /// T1.14 — np.convolve with int64 values around 2^53 must preserve full int64 precision.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.14")]
    public void T1_14_Convolve_Int64_PrecisionPreserved()
    {
        // 2^53 + 1 is the smallest positive int64 not exactly representable as double.
        long big = (1L << 53) + 1L;
        var a = np.array(new long[] { big, big, big });
        var v = np.array(new long[] { 1L, 1L });

        var r = a.convolve(v, "full");

        // NumPy ground truth:
        //   [big, 2*big, 2*big, big]
        //   = [9007199254740993, 18014398509481986, 18014398509481986, 9007199254740993]
        r.GetInt64(0).Should().Be(big,
            $"NumPy: convolve preserves int64 precision; expected r[0] = {big}");
        r.GetInt64(1).Should().Be(2 * big,
            $"NumPy: r[1] = 2*{big} = {2 * big}; double accumulator rounds to {(long)(double)(2 * big)} losing precision.");
        r.GetInt64(2).Should().Be(2 * big);
        r.GetInt64(3).Should().Be(big);
    }

    // ============================================================================
    // T1.21 — arctan2 dtype promotion for small integer inputs
    // ============================================================================
    //
    // FALSE POSITIVE — VERIFIED AGAINST NUMPY 2.4.2.
    //
    // The audit (audit_v2/03_default_math_reductions.md:202) claims NumPy maps every
    // non-float input to float64. This is INCORRECT for NumPy 2.x. Empirical NumPy 2.4.2:
    //
    //   np.arctan2(int8,  int8 ) -> float16
    //   np.arctan2(uint8, uint8) -> float16
    //   np.arctan2(bool,  bool ) -> float16
    //   np.arctan2(int16, int16) -> float32
    //   np.arctan2(uint16,uint16) -> float32
    //   np.arctan2(int32, int32) -> float64
    //   np.arctan2(int64, int64) -> float64
    //
    // NumSharp's PromoteATan2Single in Default.ATan2.cs:110-120 matches this exactly.
    // No fix needed.

    /// <summary>
    /// T1.21 — confirm NumSharp matches NumPy 2.x arctan2 dtype rules (NOT a bug).
    /// This test PASSES today and documents the alignment.
    /// </summary>
    [TestMethod]
    public void T1_21_ATan2_Int8_PromotesToHalf_MatchesNumPy2x()
    {
        // NumPy 2.4.2: np.arctan2(int8([1]), int8([1])).dtype == float16
        var r = np.arctan2(np.array(new sbyte[] { 1 }), np.array(new sbyte[] { 1 }));
        r.typecode.Should().Be(NPTypeCode.Half,
            "NumPy 2.4.2: arctan2 over int8/int8 returns float16; NumSharp PromoteATan2Single agrees.");

        var r2 = np.arctan2(np.array(new short[] { 1 }), np.array(new short[] { 1 }));
        r2.typecode.Should().Be(NPTypeCode.Single, "NumPy 2.4.2: int16/int16 -> float32");

        var r3 = np.arctan2(np.array(new int[] { 1 }), np.array(new int[] { 1 }));
        r3.typecode.Should().Be(NPTypeCode.Double, "NumPy 2.4.2: int32/int32 -> float64");
    }

    // ============================================================================
    // T1.22 — np.ceil / np.floor / np.trunc on Boolean promotes to Double
    // ============================================================================
    // Files:
    //   src/NumSharp.Core/Backends/Default/Math/Default.Ceil.cs
    //   src/NumSharp.Core/Backends/Default/Math/Default.Floor.cs
    //   src/NumSharp.Core/Backends/Default/Math/Default.Truncate.cs
    //
    // Each checks `nd.GetTypeCode.IsInteger()` to short-circuit (preserve dtype),
    // but IsInteger() returns FALSE for Boolean (NPTypeCode.cs:756-766). So Boolean
    // falls through to ExecuteUnaryOp which promotes to Double.
    //
    // NumPy: np.ceil(bool) -> bool (no-op).

    /// <summary>
    /// T1.22a — np.ceil(bool) should preserve dtype.
    /// </summary>
    [TestMethod]
    public void T1_22a_Ceil_Bool_PreservesDtype()
    {
        var r = np.ceil(np.array(new bool[] { true, false }));
        r.typecode.Should().Be(NPTypeCode.Boolean,
            "NumPy: np.ceil(bool) returns bool unchanged; NumSharp promotes to Double.");
        r.GetBoolean(0).Should().BeTrue();
        r.GetBoolean(1).Should().BeFalse();
    }

    /// <summary>
    /// T1.22b — np.floor(bool) should preserve dtype.
    /// </summary>
    [TestMethod]
    public void T1_22b_Floor_Bool_PreservesDtype()
    {
        var r = np.floor(np.array(new bool[] { true, false }));
        r.typecode.Should().Be(NPTypeCode.Boolean,
            "NumPy: np.floor(bool) returns bool unchanged.");
    }

    /// <summary>
    /// T1.22c — np.trunc(bool) should preserve dtype.
    /// </summary>
    [TestMethod]
    public void T1_22c_Trunc_Bool_PreservesDtype()
    {
        var r = np.trunc(np.array(new bool[] { true, false }));
        r.typecode.Should().Be(NPTypeCode.Boolean,
            "NumPy: np.trunc(bool) returns bool unchanged.");
    }

    // ============================================================================
    // T1.28 — np.negative behavior diverges from NumPy and from operator -
    // ============================================================================
    // Already covered in AuditV2_MathSelectionSorting.cs:
    //   T1_28a_NpNegative_RejectsByteArray_OperatorWorks (uint8 wrap should match operator)
    //   T1_28b_NpNegative_AcceptsBool_NumPyRejects        (bool should raise like NumPy)
    //
    // Additional T1.28 coverage below extends the matrix to uint16/uint32/uint64/Char,
    // which also throw NotSupportedException but should wrap per NumPy two's-complement.

    /// <summary>
    /// T1.28c — np.negative(uint16) must wrap (NumPy: [65535, 65534, 65533]).
    /// </summary>
    [TestMethod]
    public void T1_28c_Negative_UInt16_ShouldWrap()
    {
        var arr = np.array(new ushort[] { 1, 2, 3 });
        Action act = () => np.negative(arr);
        act.Should().NotThrow(
            "NumPy: np.negative(uint16([1,2,3])) wraps. NumSharp throws NotSupportedException " +
            "(see NDArray.negative.cs case Char/UInt16/UInt32/UInt64/default).");
    }

    /// <summary>
    /// T1.28d — np.negative(uint32) must wrap.
    /// </summary>
    [TestMethod]
    public void T1_28d_Negative_UInt32_ShouldWrap()
    {
        var arr = np.array(new uint[] { 1, 2, 3 });
        Action act = () => np.negative(arr);
        act.Should().NotThrow(
            "NumPy: np.negative(uint32([1,2,3])) = uint32([4294967295,4294967294,4294967293]).");
    }

    /// <summary>
    /// T1.28e — np.negative(uint64) must wrap.
    /// </summary>
    [TestMethod]
    public void T1_28e_Negative_UInt64_ShouldWrap()
    {
        var arr = np.array(new ulong[] { 1, 2, 3 });
        Action act = () => np.negative(arr);
        act.Should().NotThrow(
            "NumPy: np.negative(uint64) wraps modulo 2^64.");
    }

    // ============================================================================
    // T1.35 — np.matmul(1D, 2D) rejected with NotSupportedException
    // ============================================================================
    // File: src/NumSharp.Core/Backends/Default/Math/BLAS/Default.MatMul.cs:19-21
    //
    //   if (lhs.ndim == 1 && rhs.ndim == 2)
    //       throw new NotSupportedException("Input operand 1 has a mismatch ...");
    //
    // The comment immediately above the throw correctly DESCRIBES NumPy's behavior:
    //   "If the first argument is 1-D, it is promoted to a matrix by prepending a 1
    //    to its dimensions. After matrix multiplication the prepended 1 is removed."
    // But the code raises instead of implementing it. np.dot(1D, 2D) already does the
    // right thing in Default.Dot.cs:64-72 — matmul could copy that approach.

    /// <summary>
    /// T1.35 — np.matmul(1D, 2D) should multiply by prepending a 1-axis, then squeezing.
    /// </summary>
    [TestMethod]
    public void T1_35_Matmul_1D_2D_ShouldSucceed()
    {
        // NumPy: np.matmul([1,2,3], [[1,2],[3,4],[5,6]]) = [22, 28]
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });

        Action act = () => np.matmul(a, b);
        act.Should().NotThrow(
            "NumPy gufunc signature (n?,k),(k,m?)->(n?,m?) accepts 1D@2D by prepending. " +
            "NumSharp throws NotSupportedException in Default.MatMul.cs:19-21.");

        var r = np.matmul(a, b);
        r.shape.Should().Equal(new long[] { 2 }, "result shape is (2,) after squeezing prepended axis");
        r.GetInt32(0).Should().Be(22, "1*1 + 2*3 + 3*5 = 22");
        r.GetInt32(1).Should().Be(28, "1*2 + 2*4 + 3*6 = 28");
    }

    // ============================================================================
    // T1.37 — var / std with ddof >= n returns NaN; NumPy returns +inf
    // ============================================================================
    // File: src/NumSharp.Core/Backends/Kernels/DirectILKernelGenerator.Masking.VarStd.cs:42-43
    //
    //   if (size <= ddof)
    //       return double.NaN; // Division by zero or negative
    //
    // NumPy uses raw IEEE divide: sumOfSquares / (n - ddof). When n-ddof = 0:
    //   - non-zero numerator -> +inf  (with RuntimeWarning)
    //   - zero numerator     -> NaN  (constant array, 0/0)
    //
    // NumSharp's element-wise VarSimdHelper unconditionally returns NaN, so non-constant
    // arrays with ddof >= n return NaN instead of +inf.
    //
    // The AXIS path (Default.Reduction.Var.cs:355-366) uses Math.Max(axisSize-ddof, 0)
    // and `*= adjustment`, which yields +inf correctly for non-zero variance but NaN for
    // constant slices (0*inf=NaN). Axis path appears to match NumPy. Bug is element-wise only.

    /// <summary>
    /// T1.37a — np.var on non-constant array with ddof >= n must return +inf (NumPy).
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.37")]
    public void T1_37a_Var_DdofGreaterThanN_NonConstant_ReturnsPositiveInfinity()
    {
        var r = np.var(np.array(new double[] { 1, 2, 3, 4, 5 }), ddof: 10);
        r.GetDouble().Should().Be(double.PositiveInfinity,
            "NumPy: var([1,2,3,4,5], ddof=10) = +inf (sumSqDiff > 0, divisor = -5 in raw / 0 clamped). " +
            "NumSharp VarSimdHelper hardcodes return NaN when size <= ddof.");
    }

    /// <summary>
    /// T1.37b — np.std on non-constant array with ddof >= n must return +inf.
    /// </summary>
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.37")]
    public void T1_37b_Std_DdofGreaterThanN_NonConstant_ReturnsPositiveInfinity()
    {
        var r = np.std(np.array(new double[] { 1, 2, 3, 4, 5 }), ddof: 10);
        r.GetDouble().Should().Be(double.PositiveInfinity,
            "NumPy: std = sqrt(var) = sqrt(+inf) = +inf. NumSharp returns NaN.");
    }

    /// <summary>
    /// T1.37c — np.var on constant array with ddof >= n must return NaN (0/0).
    /// This case is correct in NumSharp but pinned to document expected behavior.
    /// </summary>
    [TestMethod]
    public void T1_37c_Var_DdofGreaterThanN_ConstantArray_ReturnsNaN()
    {
        var r = np.var(np.array(new double[] { 1, 1, 1, 1 }), ddof: 10);
        double v = r.GetDouble();
        double.IsNaN(v).Should().BeTrue("NumPy: var(const, ddof>=n) = NaN (0/0 IEEE). Already correct in NumSharp.");
    }

    // ============================================================================
    // T1.55 — np.copyto(dst_sbyte, src_float) throws NotSupportedException
    // ============================================================================
    // File: src/NumSharp.Core/Manipulation/np.copyto.cs (delegates to NpyIter.Copy)
    //
    // NumPy: raises TypeError("Cannot cast array data from dtype('float32') to
    //                          dtype('int8') according to the rule 'same_kind'").
    // NumSharp: throws NotSupportedException("Unsupported type: SByte") from a deep
    //           internal NpyIter copy path — indicating SByte path is unimplemented,
    //           not a casting-rule violation. Same call on Byte succeeds (mis-truncates).
    //
    // Both raise SOMETHING, but the error message is misleading and the underlying
    // cause (SByte casting path missing) is a real implementation gap.

    /// <summary>
    /// T1.55 — np.copyto(sbyte_dst, float_src) should raise a TypeError-style message,
    /// not NotSupportedException("Unsupported type: SByte"). The current message implies
    /// SByte isn't supported at all (it is for many other ops).
    /// </summary>
    [TestMethod]
    public void T1_55_CopyTo_SByte_From_Float_RaisesProperError()
    {
        var dst = np.zeros(new Shape(3), NPTypeCode.SByte);
        var src = np.array(new float[] { 1.5f, 2.5f, 3.5f });

        Action act = () => np.copyto(dst, src);

        // NumPy raises TypeError. NumSharp throws NotSupportedException with the
        // misleading message "Unsupported type: SByte". A correct implementation either
        //   (a) succeeds (matching NumPy `casting='unsafe'` would truncate 1.5 -> 1)
        //   (b) raises a casting-rule error consistent with the dtype combination.
        // Today's behavior is neither: it claims SByte is wholly unsupported.
        // Test asserts that calling NotSupported(SByte) is NOT the expected outcome —
        // when the SByte cast path is implemented, this should EITHER NotThrow OR
        // throw an InvalidCastException-like exception with a casting-rule message.
        try
        {
            act();
            // If it succeeds (unsafe cast), make sure values are int8-truncated:
            dst.GetSByte(0).Should().Be(1);
            dst.GetSByte(1).Should().Be(2);
            dst.GetSByte(2).Should().Be(3);
        }
        catch (NotSupportedException nse) when (nse.Message.Contains("SByte"))
        {
            Assert.Fail(
                "NumSharp throws NotSupportedException(\"Unsupported type: SByte\") which falsely " +
                "implies SByte isn't supported at all. NumPy would raise TypeError about casting rules. " +
                "Expected either successful unsafe cast or a casting-rule error. Actual: " + nse.Message);
        }
        catch (Exception)
        {
            // Any other exception (e.g. proper casting error) is acceptable.
        }
    }
}
