using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for <see cref="np.take_along_axis(NDArray,NDArray,int?)"/> — the per-slice gather
/// (NumPy <c>numpy.take_along_axis</c>, an advanced-indexing composition). All expected values
/// come from running NumPy 2.4.2. The op is IL-kernel-backed (a whole-array strided odometer,
/// <c>DirectILKernelGenerator.TakeAlongAxis.cs</c>) and dtype-agnostic via a byte-width-keyed
/// element copy. Buckets:
/// <list type="bullet">
///   <item>argsort / argmax(keepdims) reconstruction along an axis; default axis == -1.</item>
///   <item><c>axis=None</c> C-order flatten.</item>
///   <item>J != M along the axis; negative indices (single wrap).</item>
///   <item>Index broadcasting on non-axis dims; source broadcasting on non-axis dims.</item>
///   <item>Memory layouts: F-contiguous, negative-stride, sliced, transposed, broadcast view.</item>
///   <item>Per-dtype coverage on all 15 supported types.</item>
///   <item>Result independence (a fresh writeable C-contiguous copy).</item>
///   <item>Error parity: axis / dtype / ndim / broadcast / out-of-bounds, verbatim messages.</item>
/// </list>
/// </summary>
[TestClass]
public class TakeAlongAxisTests
{
    private static NDArray A() => np.array(new int[,] { { 10, 30, 20 }, { 60, 40, 50 } });

    // ===================================================================
    // Core semantics
    // ===================================================================

    [TestMethod]
    public void ArgsortReconstruct_Axis1()
    {
        var a = A();
        var r = np.take_along_axis(a, np.argsort(a, 1), 1);
        r.shape.Should().Equal(2, 3);
        r.ToArray<int>().Should().Equal(10, 20, 30, 40, 50, 60);
    }

    [TestMethod]
    public void ArgsortReconstruct_Axis0()
    {
        var a = A();
        var r = np.take_along_axis(a, np.argsort(a, 0), 0);
        r.ToArray<int>().Should().Equal(10, 30, 20, 60, 40, 50);
    }

    [TestMethod]
    public void DefaultAxisIsMinusOne()
    {
        var a = A();
        // NumPy 2.3+ default axis == -1 (last).
        var r = np.take_along_axis(a, np.argsort(a, 1));
        r.ToArray<int>().Should().Equal(10, 20, 30, 40, 50, 60);
    }

    [TestMethod]
    public void NegativeAxis()
    {
        var a = A();
        var r = np.take_along_axis(a, np.argsort(a, 0), -2);
        r.ToArray<int>().Should().Equal(10, 30, 20, 60, 40, 50);
    }

    [TestMethod]
    public void KeepdimsArgmax_JEquals1()
    {
        var a = A();
        var r = np.take_along_axis(a, np.argmax(a, 1, keepdims: true), 1);
        r.shape.Should().Equal(2, 1);
        r.ToArray<int>().Should().Equal(30, 60);
    }

    [TestMethod]
    public void J_GreaterThan_M()
    {
        var a = A();
        var idx = np.array(new int[,] { { 0, 1, 2, 0, 1 }, { 2, 1, 0, 2, 1 } });
        var r = np.take_along_axis(a, idx, 1);
        r.shape.Should().Equal(2, 5);
        r.ToArray<int>().Should().Equal(10, 30, 20, 10, 30, 50, 40, 60, 50, 40);
    }

    [TestMethod]
    public void NegativeIndices_Wrap()
    {
        var a = A();
        var idx = np.array(new int[,] { { -1, -2, -3 }, { -1, -1, -1 } });
        var r = np.take_along_axis(a, idx, 1);
        r.ToArray<int>().Should().Equal(20, 30, 10, 50, 50, 50);
    }

    [TestMethod]
    public void AxisNone_FlattensCOrder()
    {
        var a = A();
        var r = np.take_along_axis(a, np.array(new int[] { 0, 5, 3, 1, 2, 4 }), null);
        r.shape.Should().Equal(6);
        r.ToArray<int>().Should().Equal(10, 50, 60, 30, 20, 40);
    }

    [TestMethod]
    public void AxisNone_FlattensNonContiguousSourceInLogicalOrder()
    {
        // F-contiguous source must flatten in C (logical) order, not memory order.
        var af = np.asfortranarray(A());
        var r = np.take_along_axis(af, np.array(new int[] { 0, 1, 2, 3, 4, 5 }), null);
        r.ToArray<int>().Should().Equal(10, 30, 20, 60, 40, 50);
    }

    [TestMethod]
    public void Index3D_LastAxis()
    {
        var a = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Int32);
        var r = np.take_along_axis(a, np.argsort(a, 2), 2);
        r.shape.Should().Equal(2, 3, 4);
        r.ToArray<int>().Should().Equal(Enumerable_Range(0, 24));
    }

    [TestMethod]
    public void Index3D_Axis0()
    {
        var a = np.arange(24).reshape(2, 3, 4).astype(NPTypeCode.Int32);
        var r = np.take_along_axis(a, np.argsort(a, 0), 0);
        r.ToArray<int>().Should().Equal(Enumerable_Range(0, 24));
    }

    // ===================================================================
    // Broadcasting
    // ===================================================================

    [TestMethod]
    public void IndexBroadcast_NonAxisDim_Row()
    {
        var a = A();
        // idx (1,3) broadcasts over arr (2,3) axis 1 -> (2,3).
        var r = np.take_along_axis(a, np.array(new int[,] { { 0, 2, 1 } }), 1);
        r.shape.Should().Equal(2, 3);
        r.ToArray<int>().Should().Equal(10, 20, 30, 60, 50, 40);
    }

    [TestMethod]
    public void IndexBroadcast_NonAxisDim_Col()
    {
        var a = A();
        var r = np.take_along_axis(a, np.array(new int[,] { { 0 }, { 2 } }), 1);
        r.shape.Should().Equal(2, 1);
        r.ToArray<int>().Should().Equal(10, 50);
    }

    [TestMethod]
    public void SourceBroadcast_NonAxisDim()
    {
        // arr (1,3) broadcasts on dim 0 up to idx's 4; axis dim J=5 (> M=3).
        var a = np.arange(3).reshape(1, 3).astype(NPTypeCode.Int32);
        var idx = np.array(new int[,]
        {
            { 0, 1, 2, 2, 1 }, { 0, 0, 0, 1, 2 }, { 2, 2, 2, 2, 2 }, { 1, 1, 1, 1, 1 },
        });
        var r = np.take_along_axis(a, idx, 1);
        r.shape.Should().Equal(4, 5);
        r.ToArray<int>().Should().Equal(0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1);
    }

    [TestMethod]
    public void SourceBroadcastView_ReadOnly_AlongNonAxis()
    {
        // A genuine stride-0 broadcast view (read-only) as the source.
        var b = np.broadcast_to(np.array(new int[,] { { 10, 20, 30 } }), new Shape(4, 3));
        var idx = np.array(new int[,] { { 2, 0, 1 }, { 0, 0, 0 }, { 1, 1, 1 }, { 2, 2, 2 } });
        var r = np.take_along_axis(b, idx, 1);
        r.ToArray<int>().Should().Equal(30, 10, 20, 10, 10, 10, 20, 20, 20, 30, 30, 30);
    }

    [TestMethod]
    public void SourceBroadcastView_AlongAxis()
    {
        // Broadcast on the AXIS itself (stride 0): every in-range index gathers the same element.
        var c = np.broadcast_to(np.array(new int[,] { { 7 }, { 8 } }), new Shape(2, 5));
        var idx = np.array(new int[,] { { 0, 4, 2, 1, 3 }, { 3, 3, 0, 4, 1 } });
        var r = np.take_along_axis(c, idx, 1);
        r.ToArray<int>().Should().Equal(7, 7, 7, 7, 7, 8, 8, 8, 8, 8);
    }

    // ===================================================================
    // Memory layouts of the source
    // ===================================================================

    [TestMethod]
    public void Source_FContiguous()
    {
        var af = np.asfortranarray(A());
        var r = np.take_along_axis(af, np.argsort(af, 1), 1);
        r.ToArray<int>().Should().Equal(10, 20, 30, 40, 50, 60);
    }

    [TestMethod]
    public void Source_NegativeStride()
    {
        var arev = A()[":, ::-1"];   // reversed columns
        var r = np.take_along_axis(arev, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }), 1);
        r.ToArray<int>().Should().Equal(20, 30, 10, 60, 40, 50);
    }

    [TestMethod]
    public void Source_SlicedStrided()
    {
        var asl = np.arange(20).reshape(4, 5).astype(NPTypeCode.Int32)["1:3, ::2"];   // (2,3) strided
        var r = np.take_along_axis(asl, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }), 1);
        r.ToArray<int>().Should().Equal(5, 7, 9, 14, 12, 10);
    }

    [TestMethod]
    public void Source_Transposed()
    {
        var atr = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32).T;   // (3,2) non-contig
        var r = np.take_along_axis(atr, np.array(new int[,] { { 0, 1 }, { 1, 0 }, { 0, 1 } }), 0);
        r.shape.Should().Equal(3, 2);
        r.ToArray<int>().Should().Equal(0, 4, 1, 3, 0, 4);
    }

    // ===================================================================
    // Dtype coverage (all 15)
    // ===================================================================

    [TestMethod]
    public void Dtype_AllFifteen_PreserveAndGather()
    {
        // idx (2,3) reorders each row; result dtype == source dtype.
        var idx = np.array(new int[,] { { 2, 0, 1 }, { 1, 2, 0 } });

        // base {{1,3,2},{6,4,5}} with idx {{2,0,1},{1,2,0}} -> {2,1,3,4,5,6}.
        Check<bool>(np.array(new bool[,] { { true, false, true }, { false, true, false } }), idx,
                    new[] { true, true, false, true, false, false });
        Check<byte>(Base(NPTypeCode.Byte), idx, new byte[] { 2, 1, 3, 4, 5, 6 });
        Check<sbyte>(Base(NPTypeCode.SByte), idx, new sbyte[] { 2, 1, 3, 4, 5, 6 });
        Check<short>(Base(NPTypeCode.Int16), idx, new short[] { 2, 1, 3, 4, 5, 6 });
        Check<ushort>(Base(NPTypeCode.UInt16), idx, new ushort[] { 2, 1, 3, 4, 5, 6 });
        Check<int>(Base(NPTypeCode.Int32), idx, new[] { 2, 1, 3, 4, 5, 6 });
        Check<uint>(Base(NPTypeCode.UInt32), idx, new uint[] { 2, 1, 3, 4, 5, 6 });
        Check<long>(Base(NPTypeCode.Int64), idx, new long[] { 2, 1, 3, 4, 5, 6 });
        Check<ulong>(Base(NPTypeCode.UInt64), idx, new ulong[] { 2, 1, 3, 4, 5, 6 });
        Check<char>(np.array(new char[,] { { 'a', 'b', 'c' }, { 'd', 'e', 'f' } }), idx,
                    new[] { 'c', 'a', 'b', 'e', 'f', 'd' });
        Check<Half>(Base(NPTypeCode.Half), idx, new[] { (Half)2, (Half)1, (Half)3, (Half)4, (Half)5, (Half)6 });
        Check<float>(Base(NPTypeCode.Single), idx, new float[] { 2, 1, 3, 4, 5, 6 });
        Check<double>(Base(NPTypeCode.Double), idx, new double[] { 2, 1, 3, 4, 5, 6 });
        Check<decimal>(Base(NPTypeCode.Decimal), idx, new decimal[] { 2, 1, 3, 4, 5, 6 });
        Check<Complex>(Base(NPTypeCode.Complex), idx,
                       new[] { new Complex(2, 0), new Complex(1, 0), new Complex(3, 0),
                               new Complex(4, 0), new Complex(5, 0), new Complex(6, 0) });
    }

    // ===================================================================
    // Result independence
    // ===================================================================

    [TestMethod]
    public void Result_IsIndependentWriteableContiguousCopy()
    {
        var a = A();
        var r = np.take_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 0, 1, 2 } }), 1);
        r.Shape.IsWriteable.Should().BeTrue();
        r.Shape.IsContiguous.Should().BeTrue();
        r.SetValue(999, 0, 0);
        ((int)a.GetValue(0, 0)).Should().Be(10);   // source untouched
    }

    // ===================================================================
    // Empty results
    // ===================================================================

    [TestMethod]
    public void Empty_ZeroJ()
    {
        var a = A();
        var r = np.take_along_axis(a, np.zeros(new Shape(2, 0)).astype(NPTypeCode.Int64), 1);
        r.shape.Should().Equal(2, 0);
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Empty_ZeroSourceRow()
    {
        var a = np.zeros(new Shape(0, 3));
        var r = np.take_along_axis(a, np.zeros(new Shape(0, 3)).astype(NPTypeCode.Int64), 1);
        r.shape.Should().Equal(0, 3);
    }

    // ===================================================================
    // Error parity (verbatim NumPy messages)
    // ===================================================================

    [TestMethod]
    public void Err_AxisNone_IndicesNot1D()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.array(new int[,] { { 0, 1 }, { 1, 0 } }), null);
        act.Should().Throw<ValueError>()
           .WithMessage("when axis=None, `indices` must have a single dimension.");
    }

    [TestMethod]
    public void Err_FloatIndices()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.array(new double[,] { { 0.0, 1, 2 }, { 0, 1, 2 } }), 1);
        act.Should().Throw<IndexError>().WithMessage("`indices` must be an integer array");
    }

    [TestMethod]
    public void Err_BoolIndices()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.array(new bool[,] { { true, false, true }, { false, true, false } }), 1);
        act.Should().Throw<IndexError>().WithMessage("`indices` must be an integer array");
    }

    [TestMethod]
    public void Err_NdimMismatch()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.array(new int[] { 0, 1, 2 }), 1);
        act.Should().Throw<ValueError>()
           .WithMessage("`indices` and `arr` must have the same number of dimensions");
    }

    [TestMethod]
    public void Err_AxisOutOfBounds()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.zeros(new Shape(2, 3)).astype(NPTypeCode.Int64), 2);
        act.Should().Throw<AxisError>().WithMessage("*axis 2 is out of bounds for array of dimension 2*");
    }

    [TestMethod]
    public void Err_AxisOutOfBounds_Negative()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.zeros(new Shape(2, 3)).astype(NPTypeCode.Int64), -3);
        act.Should().Throw<AxisError>().WithMessage("*axis -3 is out of bounds for array of dimension 2*");
    }

    [TestMethod]
    public void Err_IndexOutOfBounds_Positive()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.array(new int[,] { { 3, 0, 0 }, { 0, 0, 0 } }), 1);
        act.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 1 with size 3");
    }

    [TestMethod]
    public void Err_IndexOutOfBounds_NegativeAfterWrap()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.array(new int[,] { { -4, 0, 0 }, { 0, 0, 0 } }), 1);
        // The ORIGINAL (pre-wrap) index is reported.
        act.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 1 with size 3");
    }

    [TestMethod]
    public void Err_BroadcastMismatch()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.zeros(new Shape(9, 5)).astype(NPTypeCode.Int64), 1);
        act.Should().Throw<IndexError>()
           .WithMessage("shape mismatch: indexing arrays could not be broadcast together with shapes (2,1) (9,5) ");
    }

    [TestMethod]
    public void Err_BroadcastMismatch_Axis0()
    {
        var a = A();
        Action act = () => np.take_along_axis(a, np.zeros(new Shape(9, 5)).astype(NPTypeCode.Int64), 0);
        act.Should().Throw<IndexError>()
           .WithMessage("shape mismatch: indexing arrays could not be broadcast together with shapes (9,5) (1,3) ");
    }

    [TestMethod]
    public void Err_BroadcastMismatch_3D()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        Action act = () => np.take_along_axis(a, np.zeros(new Shape(2, 5, 4)).astype(NPTypeCode.Int64), 2);
        act.Should().Throw<IndexError>()
           .WithMessage("shape mismatch: indexing arrays could not be broadcast together with shapes (2,1,1) (1,3,1) (2,5,4) ");
    }

    [TestMethod]
    public void Err_IndexOutOfBounds_EmptyAxis()
    {
        var a = np.zeros(new Shape(2, 0));
        Action act = () => np.take_along_axis(a, np.zeros(new Shape(2, 3)).astype(NPTypeCode.Int64), 1);
        act.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 1 with size 0");
    }

    // ===================================================================
    // Helpers
    // ===================================================================

    private static NDArray Base(NPTypeCode tc) =>
        np.array(new int[,] { { 1, 3, 2 }, { 6, 4, 5 } }).astype(tc);

    private static void Check<T>(NDArray src, NDArray idx, T[] expected) where T : unmanaged
    {
        var r = np.take_along_axis(src, idx, 1);
        r.typecode.Should().Be(src.typecode);
        r.shape.Should().Equal(2, 3);
        r.ToArray<T>().Should().Equal(expected);
    }

    private static int[] Enumerable_Range(int start, int count)
    {
        var a = new int[count];
        for (int i = 0; i < count; i++) a[i] = start + i;
        return a;
    }
}
