using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the array library below numpy.matrix
    // =====================================================================
    //
    // Found 2026-10-09 by the matrix tier (test/oracle/gen_matrix_oracle.py, replayed by
    // NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests.Matrix.cs): numpy.matrix as NDMatrix, every member over 14 source
    // layouts and the dtypes, with the result's class and memory facet compared besides its values. The matrix class
    // itself is held to NumPy without exception. The 172 cases that still differ are carved into
    // NumSharp.Tests.Oracle/Fuzz/corpus/gaps/matrix_gaps.jsonl under the gap ids MX01-MX11 (the GAPS table of the
    // generator) and replayed by Matrix_KnownGaps: each one is an NDArray member or an np.* function that NDMatrix
    // delegates to, differing from NumPy for the layout or dtype of the case.
    //
    // This file reproduces the gaps through PUBLIC calls on plain arrays, so each can be fixed where it lives:
    //
    //   * Silent wrong values: np.dot of an N-D operand with an M-D one computes complex128 in float64 (MX10), the
    //     flat np.prod in a cast loop returns NaN / garbage or throws (MX08).
    //   * A crash: fancy assignment with two index entries into a C-contiguous view at an offset (MX07).
    //   * Result LAYOUT (values equal): np.cumsum / np.cumprod (MX02), ndarray.imag of a real F array (MX03),
    //     ndarray.round of a broadcast operand (MX04), np.conjugate of a transposed strided view (MX05).
    //
    // Not repeated here, because an older file already pins the same root cause: the fancy getter's result layout
    // (MX01: OpenBugs.FancyIndex.cs, group 2), reshape(-1) of a broadcast view (MX06: OpenBugs.ViewFlags.cs, VF24),
    // complex128 np.prod's zero sign (MX09: OpenBugs.Reductions.cs), np.repeat of an offset view (MX11:
    // OpenBugs.ViewFlags.cs, VF39).
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call. EXPECTED (when fixed): each test passes;
    // remove its [OpenBugs], move it next to its API's tests, and drop the gap's rule from the generator's _gap_of so
    // the cases return to the gate.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     <c>np.dot(a, b)</c> with a (1, 4, 3) complex128 <c>a</c> and a (3, 4) complex128 <c>b</c>: NumPy returns
        ///     <c>[32-8j, 38-8j, 44-8j, 50-8j]</c> in its first row. NumSharp's generic N-D route converts every
        ///     element to float64, so the imaginary parts are lost (gap MX10; int64 past 2^53 and decimal lose
        ///     precision on the same route).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX10")]
        public void Matrix_DotNDWithMD_Complex_ShouldKeepTheImaginaryParts()
        {
            using var real = np.arange(12).reshape(1, 4, 3).astype(np.complex128);
            using var a = real + new Complex(0, 1);
            using var b = np.arange(12).reshape(3, 4).astype(np.complex128) * new Complex(1, -1);
            using var r = np.dot(a, b);

            r.shape.Should().Equal(1L, 4L, 4L);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(32, -8));
            r.GetAtIndex<Complex>(3).Should().Be(new Complex(50, -8));
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of an int8 array in a cast loop. NumPy: <c>dtype=bool</c> gives False,
        ///     <c>dtype=float16</c> gives -0.0, <c>dtype=complex128</c> gives (-0+0j). NumSharp throws
        ///     NotSupportedException for bool and returns NaN for the two others (gap MX08; the axis forms are
        ///     right).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX08")]
        public void Matrix_FlatProd_InACastLoop_ShouldMatchNumPy()
        {
            using var a = np.array(new sbyte[,] { { -1, 0, 1, 2 }, { 3, -1, 0, 1 }, { 2, 3, -1, 0 } });

            using var asBool = np.prod(a, dtype: np.@bool);
            ((bool)asBool).Should().BeFalse();

            using var asHalf = np.prod(a, dtype: np.float16);
            Half half = asHalf.GetAtIndex<Half>(0);
            Half.IsNaN(half).Should().BeFalse();
            ((double)half).Should().Be(0.0);

            using var asComplex = np.prod(a, dtype: np.complex128);
            asComplex.GetAtIndex<Complex>(0).Should().Be(Complex.Zero);
        }

        /// <summary>
        ///     <c>view[[0, 2], [1, 3]] = -1</c> where <c>view</c> is the middle <c>np.unstack</c> child of a
        ///     (3, 4, 4) array (a C-contiguous (4, 4) view at offset 16): NumPy stores the two elements. NumSharp
        ///     throws IndexOutOfRangeException: the computed offsets include the view's offset and the bound they
        ///     are tested against is the view's own size (gap MX07). A view near the start of its buffer hides the
        ///     bug, because its offsets stay below that bound.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX07")]
        public void Matrix_FancySetOnAnOffsetView_ShouldStore()
        {
            using var owner = np.arange(48).reshape(3, 4, 4);
            NDArray[] children = np.unstack(owner);
            NDArray view = children[1];

            view[new object[] { new long[] { 0, 2 }, new long[] { 1, 3 } }] = -1L;

            view.ToArray<long>().Should().Equal(16L, -1L, 18L, 19L, 20L, 21L, 22L, 23L, 24L, 25L, 26L, -1L, 28L, 29L, 30L, 31L);
        }

        /// <summary>
        ///     <c>np.cumprod(a, 0)</c> of an F-contiguous (2, 3) int64 array: NumPy's accumulate allocates the
        ///     result in the operand's order (F-contiguous, strides (8, 16)). NumSharp returns a C-ordered array
        ///     (gap MX02). The values are equal.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX02")]
        public void Matrix_Cumprod_OfAnFOrderedOperand_ShouldKeepTheOrder()
        {
            using var c = np.arange(1, 7).reshape(2, 3);
            using var f = np.asfortranarray(c);
            using var r = np.cumprod(f, 0);

            r.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L, 10L, 18L);
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
        }

        /// <summary>
        ///     <c>a.imag</c> of a real F-contiguous array: NumPy allocates the zeros in Fortran order
        ///     (<c>PyArray_ISFORTRAN(self)</c>). NumSharp allocates C order (gap MX03).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX03")]
        public void Matrix_ImagOfARealFArray_ShouldBeFOrdered()
        {
            using var c = np.arange(1, 7).reshape(2, 3);
            using var f = np.asfortranarray(c);
            using var r = f.imag;

            r.flags.writeable.Should().BeFalse();
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
        }

        /// <summary>
        ///     <c>a.round()</c> of a complex128 array broadcast from one row to (3, 2): NumPy rounds a KEEPORDER
        ///     copy, which is F-contiguous for a row broadcast (the stride-0 axis is innermost). NumSharp returns a
        ///     C-ordered array (gap MX04). The values are equal.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX04")]
        public void Matrix_RoundOfABroadcastComplexOperand_ShouldKeepTheOrder()
        {
            using var row = np.array(new[] { new Complex(1.5, 2.5), new Complex(2.5, -0.5) });
            using var b = np.broadcast_to(row, new Shape(3, 2));
            using var r = b.round();

            r.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 2));
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
        }

        /// <summary>
        ///     <c>np.conjugate(v)</c> where <c>v</c> is the transpose of a stepped (2, 3) view of a (4, 6)
        ///     complex128 array: NumPy's ufunc output keeps the operand's memory order (F-contiguous, strides
        ///     (16, 48)). NumSharp writes C order (gap MX05, reached through <c>matrix.H</c>). The values are equal.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "matrix-MX05")]
        public void Matrix_ConjugateOfATransposedSteppedView_ShouldKeepTheOrder()
        {
            using var owner = np.arange(24).reshape(4, 6).astype(np.complex128) * new Complex(1, 1);
            using var stepped = owner["::2, ::2"];
            using var v = stepped.T;
            using var r = np.conjugate(v);

            r.shape.Should().Equal(3L, 2L);
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(12, -12));
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
        }
    }
}
