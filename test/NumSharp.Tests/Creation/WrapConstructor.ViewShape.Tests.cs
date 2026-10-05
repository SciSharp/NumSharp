using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     The wrap constructors <c>new NDArray(Array, Shape, char)</c> and <c>new NDArray(IArraySlice, Shape, char)</c> (and
    ///     their <see cref="NDArray{TDType}"/> twins) are the C# port of NumPy's <c>np.ndarray(shape, buffer=values,
    ///     order=order)</c>, which reads <c>shape</c> as DIMENSIONS ONLY: C strides by default, F strides for
    ///     <c>order='F'</c>, offset 0, writeable. Every expected value here is NumPy 2.4.2's answer for the same call, probed
    ///     on 2026-10-05.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Before the fix (plan <c>docs/plans/view-shape-open-bugs.md</c>, bug 1) the constructors installed the shape's
    ///     strides, offset and flags over the caller's buffer as given. For a stepped view's shape that read past the end of
    ///     the buffer; a reversed view's shape read the values backwards, a transposed view's in F order, and a broadcast
    ///     view's repeated row 0 and made the result read-only. The first four tests were <c>[OpenBugs]</c> pins
    ///     (<c>2f92d91d</c>).
    ///     </para>
    ///     <para>
    ///     Every test checks the layout (strides, offset) BEFORE it reads an element, so a regression fails without reading
    ///     past a buffer.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class WrapConstructorViewShapeTests
    {
        /// <summary>The shape of <c>arange(20)[::4]</c>: dimensions <c>(5,)</c>, stride 4.</summary>
        /// <returns>The view's shape; the view itself is disposed (a <see cref="Shape"/> holds no storage).</returns>
        private static Shape SteppedShape()
        {
            using var x = np.arange(20)["::4"];
            return x.Shape;
        }

        /// <summary>
        ///     Checks the dimensions-only layout every wrap must produce: the given dimensions, the given strides (C or F of
        ///     those dimensions), offset 0, writeable and not broadcast.
        /// </summary>
        /// <param name="a">The wrapped array.</param>
        /// <param name="dims">The expected dimensions.</param>
        /// <param name="strides">The expected element strides.</param>
        private static void ShouldHaveDenseLayout(NDArray a, long[] dims, long[] strides)
        {
            a.shape.Should().Equal(dims);
            a.Shape.strides.Should().Equal(strides, "a wrap lays the buffer out from the dimensions alone");
            a.Shape.offset.Should().Be(0);
            a.Shape.IsWriteable.Should().BeTrue("the caller's buffer is writable, whatever the shape's source was");
            a.Shape.IsBroadcasted.Should().BeFalse();
        }

        /// <summary>
        ///     <c>new NDArray(Array, x.Shape)</c> for a stepped view <c>x</c> lays the values out from the shape's dimensions in
        ///     C order, as NumPy's <c>np.ndarray(x.shape, buffer=values)</c> does, not under the view's stride 4 (which read
        ///     <c>[1, 5, heap, heap, heap]</c>). Was an <c>[OpenBugs]</c> pin.
        /// </summary>
        [TestMethod]
        public void ArrayValues_SteppedViewShape_LaysOutInCOrder()
        {
            var values = new double[] { 1, 2, 3, 4, 5 };
            using var a = new NDArray(values, SteppedShape());

            ShouldHaveDenseLayout(a, new long[] { 5 }, new long[] { 1 });
            a.ToArray<double>().Should().Equal(values);
        }

        /// <summary>
        ///     <c>new NDArray(IArraySlice, x.Shape)</c> follows the same rule through the slice overload. Was an
        ///     <c>[OpenBugs]</c> pin.
        /// </summary>
        [TestMethod]
        public void ArraySliceValues_SteppedViewShape_LaysOutInCOrder()
        {
            using var a = new NDArray(ArraySlice.FromArray(new double[] { 1, 2, 3, 4, 5 }), SteppedShape());

            ShouldHaveDenseLayout(a, new long[] { 5 }, new long[] { 1 });
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>
        ///     A 2-D column-stepped view (<c>arange(24).reshape(4, 6)[:, ::2]</c>, strides <c>(6, 2)</c>, which reached element
        ///     22 of a 12-element buffer): NumPy's <c>np.ndarray((4, 3), buffer=arange(12))</c> is
        ///     <c>[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]</c>. Was an <c>[OpenBugs]</c> pin.
        /// </summary>
        [TestMethod]
        public void ArrayValues_ColumnSteppedView2DShape_LaysOutInCOrder()
        {
            using var v = np.arange(24).reshape(4, 6)[":, ::2"];
            var values = Enumerable.Range(0, 12).Select(i => (long)i).ToArray();
            using var a = new NDArray(values, v.Shape);

            ShouldHaveDenseLayout(a, new long[] { 4, 3 }, new long[] { 3, 1 });
            a.ToArray<long>().Should().Equal(values, "values fill the array in C order");
        }

        /// <summary>
        ///     <c>order: 'F'</c> on a 1-D stepped view's shape gives the dense stride 1 (a 1-D array is both C- and F-ordered).
        ///     The old constructor rebuilt the shape for <c>'F'</c> only when <c>NDim &gt; 1</c>, so the stride 4 survived.
        ///     Was an <c>[OpenBugs]</c> pin.
        /// </summary>
        [TestMethod]
        public void ArrayValues_SteppedViewShape_FOrder1D_LaysOutDensely()
        {
            using var a = new NDArray(new double[] { 1, 2, 3, 4, 5 }, SteppedShape(), 'F');

            ShouldHaveDenseLayout(a, new long[] { 5 }, new long[] { 1 });
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>
        ///     A reversed view's shape (stride -1, offset 4) read the values backwards. NumPy's
        ///     <c>np.ndarray(arange(5)[::-1].shape, buffer=[1, 2, 3, 4, 5])</c> is <c>[1, 2, 3, 4, 5]</c>.
        /// </summary>
        [TestMethod]
        public void ArrayValues_ReversedViewShape_ReadsForwards()
        {
            using var r = np.arange(5)["::-1"];
            using var a = new NDArray(new double[] { 1, 2, 3, 4, 5 }, r.Shape);

            ShouldHaveDenseLayout(a, new long[] { 5 }, new long[] { 1 });
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>
        ///     A transposed view's shape (dense, strides <c>(1, 3)</c>) is still only a shape: NumPy's
        ///     <c>np.ndarray(arange(6).reshape(2, 3).T.shape, buffer=arange(6.))</c> is <c>[[0, 1], [2, 3], [4, 5]]</c>, strides
        ///     <c>(2, 1)</c>. NumSharp read it in F order (<c>[[0, 3], [1, 4], [2, 5]]</c>).
        /// </summary>
        [TestMethod]
        public void ArrayValues_TransposedViewShape_COrder_IsCLayout()
        {
            using var t = np.arange(6).reshape(2, 3).T;
            using var a = new NDArray(new double[] { 0, 1, 2, 3, 4, 5 }, t.Shape);

            ShouldHaveDenseLayout(a, new long[] { 3, 2 }, new long[] { 2, 1 });
            a.flatten().ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            a.GetDouble(0, 1).Should().Be(1.0);
            a.GetDouble(1, 0).Should().Be(2.0);
        }

        /// <summary>
        ///     The same transposed shape with <c>order: 'F'</c> is the F layout of its dimensions: NumPy's
        ///     <c>np.ndarray((3, 2), buffer=arange(6.), order='F')</c> is <c>[[0, 3], [1, 4], [2, 5]]</c>, strides <c>(1, 3)</c>.
        ///     This case was already right; it guards the F branch of the fix.
        /// </summary>
        [TestMethod]
        public void ArrayValues_TransposedViewShape_FOrder_IsFLayout()
        {
            using var t = np.arange(6).reshape(2, 3).T;
            using var a = new NDArray(new double[] { 0, 1, 2, 3, 4, 5 }, t.Shape, 'F');

            ShouldHaveDenseLayout(a, new long[] { 3, 2 }, new long[] { 1, 3 });
            a.GetDouble(0, 1).Should().Be(3.0);
            a.GetDouble(1, 0).Should().Be(1.0);
            a.GetDouble(2, 1).Should().Be(5.0);
        }

        /// <summary>
        ///     A broadcast view's shape (strides <c>(0, 1)</c>, read-only) repeated row 0 and made the result read-only. NumPy's
        ///     <c>np.ndarray(broadcast_to(arange(3.), (2, 3)).shape, buffer=arange(6.))</c> is <c>[[0, 1, 2], [3, 4, 5]]</c>
        ///     and writeable.
        /// </summary>
        [TestMethod]
        public void ArrayValues_BroadcastViewShape_IsWriteableCLayout()
        {
            using var b = np.broadcast_to(np.arange(3.0), new Shape(2, 3));
            using var a = new NDArray(new double[] { 0, 1, 2, 3, 4, 5 }, b.Shape);

            ShouldHaveDenseLayout(a, new long[] { 2, 3 }, new long[] { 3, 1 });
            a.flatten().ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            a.SetDouble(9.0, 1, 2);
            a.GetDouble(1, 2).Should().Be(9.0);
        }

        /// <summary>
        ///     A contiguous offset slice's shape (<c>arange(20)[2:7]</c>) already gave NumPy's answer
        ///     (<c>[1, 2, 3, 4, 5]</c>); this guards it.
        /// </summary>
        [TestMethod]
        public void ArrayValues_ContiguousSliceShape_IsUnchanged()
        {
            using var s = np.arange(20)["2:7"];
            using var a = new NDArray(new double[] { 1, 2, 3, 4, 5 }, s.Shape);

            ShouldHaveDenseLayout(a, new long[] { 5 }, new long[] { 1 });
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>
        ///     A 0-d shape over one value is NumPy's <c>np.ndarray((), buffer=[7.])</c>: the 0-d array <c>7.0</c>.
        /// </summary>
        [TestMethod]
        public void ArrayValues_ScalarShape_IsZeroDimensional()
        {
            using var a = new NDArray(new double[] { 7 }, Shape.Scalar);

            a.ndim.Should().Be(0);
            a.Shape.IsWriteable.Should().BeTrue();
            a.GetDouble().Should().Be(7.0);
        }

        /// <summary>
        ///     An EMPTY stepped view's shape (<c>zeros((0, 3))[::2]</c>, strides <c>(0, 1)</c>) gets the C strides of its
        ///     dimensions, <c>(3, 1)</c>, as NumPy's <c>np.ndarray((0, 3), buffer=zeros(0))</c> does (byte strides
        ///     <c>(24, 8)</c>). The C flag alone would have kept <c>(0, 1)</c>: it is set on every empty shape.
        /// </summary>
        [TestMethod]
        public void ArrayValues_EmptySteppedViewShape_GetsCStrides()
        {
            using var e = np.zeros(new Shape(0, 3))["::2"];
            using var a = new NDArray(new double[0], e.Shape);

            ShouldHaveDenseLayout(a, new long[] { 0, 3 }, new long[] { 3, 1 });
            a.size.Should().Be(0);
        }

        /// <summary>
        ///     <c>'F'</c> lays a 3-D shape out column-major for any source layout, here a stepped view's shape: NumPy's
        ///     <c>np.ndarray((2, 3, 2), buffer=arange(12.), order='F')</c> has element <c>[i, j, k]</c> at
        ///     <c>i + 2 * j + 6 * k</c>.
        /// </summary>
        [TestMethod]
        public void ArrayValues_FOrder3D_IsFLayout()
        {
            using var v = np.zeros(new Shape(4, 3, 4))["::2, :, ::2"];
            var values = Enumerable.Range(0, 12).Select(i => (double)i).ToArray();
            using var a = new NDArray(values, v.Shape, 'F');

            ShouldHaveDenseLayout(a, new long[] { 2, 3, 2 }, new long[] { 1, 2, 6 });
            a.GetDouble(1, 2, 1).Should().Be(1 + 2 * 2 + 6 * 1);
            a.GetDouble(0, 1, 1).Should().Be(2 + 6);
        }

        /// <summary>
        ///     NumPy's <c>ndarray</c> accepts every order code: <c>'F'</c> and <c>'f'</c> are F, and <c>'C'</c>, <c>'A'</c>,
        ///     <c>'K'</c> in either case are C (probed: all give the C layout for <c>(2, 3)</c>).
        /// </summary>
        [TestMethod]
        public void OrderCodes_FIsColumnMajor_EveryOtherValidCodeIsRowMajor()
        {
            var values = new double[] { 0, 1, 2, 3, 4, 5 };
            foreach (char o in new[] { 'C', 'c', 'A', 'a', 'K', 'k' })
            {
                using var a = new NDArray(values, new Shape(2, 3), o);
                a.Shape.strides.Should().Equal(new long[] { 3, 1 }, $"order '{o}' is the C layout");
            }

            foreach (char o in new[] { 'F', 'f' })
            {
                using var a = new NDArray(values, new Shape(2, 3), o);
                a.Shape.strides.Should().Equal(new long[] { 1, 2 }, $"order '{o}' is the F layout");
            }
        }

        /// <summary>
        ///     An invalid order code raises NumPy's <c>ValueError</c> text verbatim, from both overloads (it used to be ignored).
        /// </summary>
        [TestMethod]
        public void OrderCodes_Invalid_RaisesNumPysValueError()
        {
            Action arrayOverload = () => new NDArray(new double[] { 1, 2 }, new Shape(2), 'X');
            arrayOverload.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'X')");

            Action sliceOverload = () => new NDArray(ArraySlice.FromArray(new double[] { 1, 2 }), new Shape(2), 'Q');
            sliceOverload.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
        }

        /// <summary>
        ///     Every dtype takes the dimensions-only layout through both overloads: the stepped view's shape over five values
        ///     reads them in order.
        /// </summary>
        [TestMethod]
        public void AllDtypes_SteppedViewShape_ReadInOrder()
        {
            var shape = SteppedShape();
            Array[] arrays =
            {
                new[] { true, false, true, true, false },
                new byte[] { 1, 2, 3, 4, 5 },
                new sbyte[] { 1, 2, 3, 4, 5 },
                new short[] { 1, 2, 3, 4, 5 },
                new ushort[] { 1, 2, 3, 4, 5 },
                new[] { 1, 2, 3, 4, 5 },
                new uint[] { 1, 2, 3, 4, 5 },
                new long[] { 1, 2, 3, 4, 5 },
                new ulong[] { 1, 2, 3, 4, 5 },
                new[] { 'a', 'b', 'c', 'd', 'e' },
                new[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5 },
                new[] { 1f, 2f, 3f, 4f, 5f },
                new[] { 1.0, 2.0, 3.0, 4.0, 5.0 },
                new[] { 1m, 2m, 3m, 4m, 5m },
                new[] { new Complex(1, -1), new Complex(2, -2), new Complex(3, -3), new Complex(4, -4), new Complex(5, -5) },
            };

            foreach (var values in arrays)
            {
                string what = values.GetType().GetElementType()!.Name;
                using (var a = new NDArray(values, shape))
                {
                    a.Shape.strides.Should().Equal(new long[] { 1 }, what);
                    for (int i = 0; i < values.Length; i++)
                        a.GetAtIndex(i).Should().Be(values.GetValue(i), $"{what} element {i} (Array overload)");
                }

                using (var b = new NDArray(ArraySlice.FromArray(values), shape))
                {
                    b.Shape.strides.Should().Equal(new long[] { 1 }, what);
                    for (int i = 0; i < values.Length; i++)
                        b.GetAtIndex(i).Should().Be(values.GetValue(i), $"{what} element {i} (IArraySlice overload)");
                }
            }
        }

        /// <summary>
        ///     The generic twins <c>new NDArray&lt;T&gt;(Array, Shape)</c> and <c>new NDArray&lt;T&gt;(IArraySlice, Shape)</c>
        ///     forward to the fixed constructors.
        /// </summary>
        [TestMethod]
        public void GenericTwins_SteppedViewShape_ReadInOrder()
        {
            using var a = new NDArray<double>(new double[] { 1, 2, 3, 4, 5 }, SteppedShape());
            a.Shape.strides.Should().Equal(new long[] { 1 });
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);

            using var b = new NDArray<int>(ArraySlice.FromArray(new[] { 1, 2, 3, 4, 5 }), SteppedShape());
            b.Shape.strides.Should().Equal(new long[] { 1 });
            b.ToArray<int>().Should().Equal(1, 2, 3, 4, 5);
        }

        /// <summary>
        ///     A one-dimensional array is wrapped in place, so a write through the result reaches the caller's array, at the
        ///     element the dense layout names (with the old stride 4, index 1 was element 4).
        /// </summary>
        [TestMethod]
        public void OneDimensionalValues_AreWrappedInPlace()
        {
            var values = new double[] { 1, 2, 3, 4, 5 };
            using var a = new NDArray(values, SteppedShape());

            a.SetDouble(99.0, 1);
            values[1].Should().Be(99.0);
            values[4].Should().Be(5.0);
        }

        /// <summary>
        ///     A multi-dimensional C# array is flattened row by row and laid out by the shape's dimensions: a
        ///     <c>double[2, 3]</c> with the shape of a stepped <c>(3, 2)</c> view is <c>[[1, 2], [3, 4], [5, 6]]</c>.
        /// </summary>
        [TestMethod]
        public void MultiDimensionalValues_SteppedViewShape_FlattenThenLayOut()
        {
            using var v = np.zeros(new Shape(6, 4))["::2, ::2"];
            using var a = new NDArray(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, v.Shape);

            ShouldHaveDenseLayout(a, new long[] { 3, 2 }, new long[] { 2, 1 });
            a.flatten().ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        }

        /// <summary>
        ///     The exact-size rule is unchanged: a buffer whose element count differs from the shape's size raises
        ///     <see cref="ArgumentException"/>. (NumPy raises <c>TypeError: buffer is too small for requested array</c> for a
        ///     smaller buffer and reads a prefix of a larger one; the plan records that as a separate follow-up.)
        /// </summary>
        [TestMethod]
        public void SizeMismatch_StillRaises()
        {
            Action smaller = () => new NDArray(new double[] { 1, 2, 3, 4, 5 }, new Shape(6));
            smaller.Should().Throw<ArgumentException>();

            Action larger = () => new NDArray(new double[] { 1, 2, 3, 4, 5 }, new Shape(3));
            larger.Should().Throw<ArgumentException>();
        }

        /// <summary>
        ///     <c>np.ones</c> moved from the wrap constructor to the storage constructor so that it keeps a KEEPORDER layout:
        ///     NumPy's <c>np.ones_like(f_array)</c> is F-contiguous, and <c>np.ones</c> of a dense permuted shape keeps the
        ///     permutation (<see cref="Shape.ForFreshAllocation"/>).
        /// </summary>
        [TestMethod]
        public void Ones_KeepsKeepOrderLayouts()
        {
            using var f = np.asfortranarray(np.arange(20.0).reshape(4, 5));
            using var like = np.ones_like(f);
            like.Shape.IsFContiguous.Should().BeTrue("ones_like keeps an F input's order (order='K')");
            like.Shape.strides.Should().Equal(new long[] { 1, 4 });
            like.ToArray<double>().Should().OnlyContain(v => v == 1.0);

            using var p = np.transpose(np.zeros(new Shape(3, 2, 4)), new[] { 1, 0, 2 });
            using var ones = np.ones(p.Shape);
            ones.Shape.strides.Should().Equal(p.Shape.strides, "a dense permutation is a fresh layout and is kept");
            ones.ToArray<double>().Should().OnlyContain(v => v == 1.0);
        }
    }
}
