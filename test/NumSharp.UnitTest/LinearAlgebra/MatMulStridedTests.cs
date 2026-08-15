using System;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra;

/// <summary>
/// Tests for the stride-aware GEMM path in np.dot / np.matmul.
/// Every dtype (all 12 supported by NumSharp) must produce bit-identical
/// results on transposed and sliced views as it does on contiguous copies —
/// without materializing copies anywhere along the call chain.
///
/// Reference for each case is the same operation with both operands
/// materialized contiguously via .copy(). The stride-native kernels are
/// required to match that reference exactly (bit-exact for same-type paths,
/// which preserve FMA order; mixed-type paths use a double accumulator).
/// </summary>
[TestClass]
public class MatMulStridedTests
{
    // =====================================================================
    // Float — SIMD stride-aware GEMM (BLIS packers)
    // =====================================================================

    [TestMethod]
    public void Dot_Float_TransposedA_Small_SimplePath()
    {
        // At shape (4,3) strides (1,4) — aStride0==1 → PackA SIMD load path.
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Single);
        var at = a.transpose();

        var result = np.dot(at, a);
        var reference = np.dot(at.copy(), a);

        at.Shape.IsContiguous.Should().BeFalse();
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_TransposedA_Large_BlockedPath()
    {
        // Dims > BLOCKING_THRESHOLD (128) → blocked GEBP with packer.
        np.random.seed(42);
        var l = np.random.randn(200L, 150L).astype(NPTypeCode.Single);
        var lt = l.transpose();

        var result = np.dot(lt, l);
        var reference = np.dot(lt.copy(), l);

        lt.Shape.IsContiguous.Should().BeFalse();
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_TransposedB_Small_SimplePath()
    {
        var b = np.arange(8).reshape(4, 2).astype(NPTypeCode.Single);
        var bt = b.transpose();

        var result = np.dot(bt, b);
        var reference = np.dot(bt.copy(), b);

        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_ContigByTransposedB_Large()
    {
        // L @ Lt — B is transposed-contiguous, exercises PackB bStride0==1.
        np.random.seed(7);
        var l = np.random.randn(500L, 400L).astype(NPTypeCode.Single);
        var lt = l.transpose();

        var result = np.dot(l, lt);
        var reference = np.dot(l, lt.copy());

        lt.Shape.IsContiguous.Should().BeFalse();
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_BothTransposed_Small()
    {
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Single);
        var b = np.arange(12).reshape(4, 3).astype(NPTypeCode.Single);
        var at = a.transpose();
        var bt = b.transpose();

        var result = np.dot(at, bt);
        var reference = np.dot(at.copy(), bt.copy());

        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_BothTransposed_Large_BlockedPath()
    {
        np.random.seed(11);
        var a = np.random.randn(200L, 300L).astype(NPTypeCode.Single);
        var b = np.random.randn(200L, 150L).astype(NPTypeCode.Single);
        var bt = b.transpose();
        var result = np.dot(bt, a);
        var reference = np.dot(bt.copy(), a);

        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_SlicedRows_BlockedPath()
    {
        // Every other row — strides (2*cols, 1), non-contiguous, offset 0.
        np.random.seed(23);
        var big = np.random.randn(400L, 200L).astype(NPTypeCode.Single);
        var sliced = big["::2, :"];
        var b = np.random.randn(200L, 100L).astype(NPTypeCode.Single);

        sliced.Shape.IsContiguous.Should().BeFalse();
        var result = np.dot(sliced, b);
        var reference = np.dot(sliced.copy(), b);

        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_SlicedWithOffset_AppliesOffsetCorrectly()
    {
        // 2D slice — non-contiguous with Shape.offset > 0. Dispatcher must
        // add offset to the base pointer before passing to the kernel.
        var big = np.arange(48).reshape(6, 8).astype(NPTypeCode.Single);
        var sliced = big["1:, 2:"];
        var b = np.arange(12).reshape(6, 2).astype(NPTypeCode.Single);

        sliced.Shape.offset.Should().BeGreaterThan(0);
        sliced.Shape.IsContiguous.Should().BeFalse();

        var result = np.dot(sliced, b);
        var reference = np.dot(sliced.copy(), b);

        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_Contiguous_UnchangedBehavior()
    {
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Single);
        var b = np.arange(8).reshape(4, 2).astype(NPTypeCode.Single);
        var result = np.dot(a, b);

        result.GetSingle(0, 0).Should().Be(28f);
        result.GetSingle(0, 1).Should().Be(34f);
        result.GetSingle(1, 0).Should().Be(76f);
        result.GetSingle(1, 1).Should().Be(98f);
        result.GetSingle(2, 0).Should().Be(124f);
        result.GetSingle(2, 1).Should().Be(162f);
    }

    // =====================================================================
    // Double — SIMD stride-aware GEMM (simple path + blocked GEBP)
    // =====================================================================

    [TestMethod]
    public void Dot_Double_TransposedA_Small()
    {
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Double);
        var at = a.transpose();
        var result = np.dot(at, a);
        var reference = np.dot(at.copy(), a);
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Double_ContigByTransposedB_Simple()
    {
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Double);
        var b = np.arange(8).reshape(2, 4).astype(NPTypeCode.Double);
        var bt = b.transpose();
        var result = np.dot(a, bt);
        var reference = np.dot(a, bt.copy());
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Double_Contiguous_UnchangedBehavior()
    {
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Double);
        var b = np.arange(8).reshape(4, 2).astype(NPTypeCode.Double);
        var result = np.dot(a, b);
        result.GetDouble(0, 0).Should().Be(28.0);
        result.GetDouble(1, 1).Should().Be(98.0);
        result.GetDouble(2, 0).Should().Be(124.0);
    }

    [TestMethod]
    public void Dot_Double_TransposedA_Large_BlockedPath()
    {
        // Dims > BLOCKING_THRESHOLD (128) → blocked GEBP; aStride0==1 →
        // PackADoublePanelsStrided's two-vector-load fast path. Bit-exact vs
        // the contiguous copy because packing normalizes both operands into
        // identical panels, so the micro-kernel FMA order is the same.
        np.random.seed(42);
        var l = np.random.randn(200L, 150L).astype(NPTypeCode.Double);
        var lt = l.transpose();

        var result = np.dot(lt, l);
        var reference = np.dot(lt.copy(), l);

        lt.Shape.IsContiguous.Should().BeFalse();
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Double_ContigByTransposedB_Large_BlockedPath()
    {
        // L @ Lt — B transposed-contiguous, exercises PackB bStride0==1.
        // This is the np.cov pattern (dot(Xc, Xc.T)) that motivated the
        // blocked double kernel.
        np.random.seed(7);
        var l = np.random.randn(500L, 400L).astype(NPTypeCode.Double);
        var lt = l.transpose();

        var result = np.dot(l, lt);
        var reference = np.dot(l, lt.copy());

        lt.Shape.IsContiguous.Should().BeFalse();
        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Double_SlicedWithOffset_Large_BlockedPath()
    {
        // 2D slice — non-contiguous with Shape.offset > 0 at blocked size,
        // exercises PackA's general (scalar) branch plus the offset path.
        np.random.seed(23);
        var big = np.random.randn(300L, 260L).astype(NPTypeCode.Double);
        var sliced = big["1:, 2:"];                      // (299, 258)
        var b = np.random.randn(258L, 130L).astype(NPTypeCode.Double);

        sliced.Shape.offset.Should().BeGreaterThan(0);
        sliced.Shape.IsContiguous.Should().BeFalse();

        var result = np.dot(sliced, b);
        var reference = np.dot(sliced.copy(), b);

        np.array_equal(result, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Double_MidSizeTransposedB_BlockedReroute_ExactVsInt64Oracle()
    {
        // All dims <= BLOCKING_THRESHOLD but bStride1 != 1 with M*N*K >= 4096:
        // the strided-B mid-size reroute sends this to the blocked kernel
        // (the simple path's inner loop is scalar here). Small integer values
        // keep every sum exact, so the result must agree bit-for-bit with the
        // independent INumber<long> kernel whatever the accumulation order.
        var ai = (np.arange(100 * 100) % 13).reshape(100, 100);
        var bi = (np.arange(100 * 100) % 11).reshape(100, 100);
        var bt = bi.astype(NPTypeCode.Double).transpose();

        bt.Shape.IsContiguous.Should().BeFalse();
        var result = np.dot(ai.astype(NPTypeCode.Double), bt);
        var oracle = np.dot(ai.astype(NPTypeCode.Int64), bi.transpose().astype(NPTypeCode.Int64)).astype(NPTypeCode.Double);

        np.array_equal(result, oracle).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Double_Contiguous_Large_ExactVsInt64Oracle()
    {
        // Independent oracle for the blocked path on contiguous inputs: with
        // small integer values every product and partial sum is exact in both
        // int64 and double (max ≈ 16800 « 2^53), so the blocked double GEMM
        // must agree bit-for-bit with the INumber<long> kernel — a completely
        // separate code path — regardless of accumulation order.
        var ai = (np.arange(150 * 140) % 13).reshape(150, 140);
        var bi = (np.arange(140 * 130) % 11).reshape(140, 130);
        var ad = ai.astype(NPTypeCode.Double);
        var bd = bi.astype(NPTypeCode.Double);

        var result = np.dot(ad, bd);
        var oracle = np.dot(ai.astype(NPTypeCode.Int64), bi.astype(NPTypeCode.Int64)).astype(NPTypeCode.Double);

        np.array_equal(result, oracle).Should().BeTrue();
    }

    // =====================================================================
    // Integer & other non-SIMD dtypes — stride-native INumber<T> kernel.
    // Each covers TN, NT, and sliced-row patterns to exercise both branches
    // of the generic kernel (bStride1==1 vs fully scalar).
    // =====================================================================

    [TestMethod]
    public void Dot_Byte_StrideNative()
    {
        // Values kept small so byte arithmetic doesn't overflow meaningfully
        // for correctness comparison (both paths wrap identically).
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Byte);
        var at = a.transpose();

        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();   // TN
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();   // NT
    }

    [TestMethod]
    public void Dot_Int16_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int16);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_UInt16_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.UInt16);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Int32_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int32);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_UInt32_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.UInt32);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Int64_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int64);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_UInt64_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.UInt64);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Char_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Char);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Decimal_StrideNative()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Decimal);
        var at = a.transpose();
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
        np.array_equal(np.dot(a, at), np.dot(a, at.copy())).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Bool_StrideNative()
    {
        // NumPy bool dot: C[i,j] = OR over k of (A[i,k] AND B[k,j]).
        var ap = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
        var bp = np.arange(6).reshape(3, 2).astype(NPTypeCode.Int32);
        var a = (ap > 2).astype(NPTypeCode.Boolean);
        var b = (bp > 2).astype(NPTypeCode.Boolean);
        var bt = b.transpose();  // (2,3) non-contig

        // a @ b and a @ bt.T should give the same result; testing stride path.
        var contig = np.dot(a, b);
        var strided = np.dot(bt, a.transpose());  // bt (2,3) @ a.T (3,2) -> (2,2)
        var strided_ref = np.dot(bt.copy(), a.transpose().copy());
        np.array_equal(strided, strided_ref).Should().BeTrue();
    }

    // =====================================================================
    // Sliced-row patterns (non-transpose, non-contiguous) per dtype —
    // exercise the bStride1 == 1 fast branch of the generic kernel.
    // =====================================================================

    [TestMethod]
    public void Dot_Int32_SlicedRows()
    {
        var big = np.arange(40).reshape(8, 5).astype(NPTypeCode.Int32);
        var sliced = big["::2, :"];            // (4,5) non-contig, offset 0
        var b = np.arange(10).reshape(5, 2).astype(NPTypeCode.Int32);

        sliced.Shape.IsContiguous.Should().BeFalse();
        np.array_equal(np.dot(sliced, b), np.dot(sliced.copy(), b)).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Int64_SlicedWithOffset()
    {
        // 2D slice → non-zero offset, exercises the offset path per-dtype.
        var big = np.arange(48).reshape(6, 8).astype(NPTypeCode.Int64);
        var sliced = big["1:, 2:"];
        var b = np.arange(12).reshape(6, 2).astype(NPTypeCode.Int64);

        sliced.Shape.offset.Should().BeGreaterThan(0);
        sliced.Shape.IsContiguous.Should().BeFalse();

        np.array_equal(np.dot(sliced, b), np.dot(sliced.copy(), b)).Should().BeTrue();
    }

    // =====================================================================
    // Mixed-type — stride-native path with double accumulator.
    // =====================================================================

    [TestMethod]
    public void Dot_Int32ByFloat32_Transposed_MixedType()
    {
        var ai = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int32);
        var af = np.arange(10).reshape(2, 5).astype(NPTypeCode.Single);
        var aft = af.transpose();  // (5,2) non-contig float

        // int32 @ float32 -> float64 per NumPy promotion
        var mixed = np.dot(ai, aft);
        var mixed_ref = np.dot(ai, aft.copy());
        np.array_equal(mixed, mixed_ref).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Int32_TransposedA_SameTypePath()
    {
        var a = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int32);
        var at = a.transpose();

        // Same-type INumber<int> kernel path — not mixed-type.
        np.array_equal(np.dot(at, a), np.dot(at.copy(), a)).Should().BeTrue();
    }

    // =====================================================================
    // MLP-shape regression — the original fix target from FullyConnectedFused.
    // =====================================================================

    [TestMethod]
    public void Dot_Float_MlpGradW_InputTransposed()
    {
        np.random.seed(1337);
        var input = np.random.randn(64L, 784L).astype(NPTypeCode.Single);
        var gradPreact = np.random.randn(64L, 128L).astype(NPTypeCode.Single);
        var inputT = input.transpose();

        var gradW = np.dot(inputT, gradPreact);
        var reference = np.dot(inputT.copy(), gradPreact);

        gradW.shape[0].Should().Be(784);
        gradW.shape[1].Should().Be(128);
        np.array_equal(gradW, reference).Should().BeTrue();
    }

    [TestMethod]
    public void Dot_Float_MlpInputGrad_WeightTransposed()
    {
        np.random.seed(1337);
        var w = np.random.randn(784L, 128L).astype(NPTypeCode.Single);
        var gradPreact = np.random.randn(64L, 128L).astype(NPTypeCode.Single);
        var wT = w.transpose();

        var inputGrad = np.dot(gradPreact, wT);
        var reference = np.dot(gradPreact, wT.copy());

        inputGrad.shape[0].Should().Be(64);
        inputGrad.shape[1].Should().Be(784);
        np.array_equal(inputGrad, reference).Should().BeTrue();
    }
}
