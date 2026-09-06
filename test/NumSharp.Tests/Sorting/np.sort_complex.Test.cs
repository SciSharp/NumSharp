using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    /// np.sort_complex parity with NumPy 2.4.2 (every expected value produced by running NumPy).
    /// Implementation: port of numpy/lib/_function_base_impl.py — copy + sort along the LAST axis
    /// in the input's own dtype, then up-cast to Complex (NumSharp's single complex width).
    /// </summary>
    [TestClass]
    public class NpSortComplexTests
    {
        private static Complex C(double re, double im = 0) => new Complex(re, im);

        private static void AssertComplex(NDArray r, params Complex[] expected)
        {
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.size.Should().Be(expected.Length);
            var flat = r.ravel();
            for (int i = 0; i < expected.Length; i++)
            {
                var v = flat.GetValue<Complex>(i);
                if (double.IsNaN(expected[i].Real)) double.IsNaN(v.Real).Should().BeTrue($"[{i}].Real should be NaN");
                else v.Real.Should().Be(expected[i].Real, $"[{i}].Real");
                if (double.IsNaN(expected[i].Imaginary)) double.IsNaN(v.Imaginary).Should().BeTrue($"[{i}].Imag should be NaN");
                else v.Imaginary.Should().Be(expected[i].Imaginary, $"[{i}].Imag");
            }
        }

        [TestMethod]
        public void SortComplex_IntInput()
        {
            // np.sort_complex([5, 3, 6, 2, 1]) == [1.+0.j, 2.+0.j, 3.+0.j, 5.+0.j, 6.+0.j]
            var r = np.sort_complex(np.array(new[] { 5, 3, 6, 2, 1 }));
            AssertComplex(r, C(1), C(2), C(3), C(5), C(6));
        }

        [TestMethod]
        public void SortComplex_ComplexInput_RealThenImag()
        {
            // np.sort_complex([1+2j, 2-1j, 3-2j, 3-3j, 3+5j]) == [1+2j, 2-1j, 3-3j, 3-2j, 3+5j]
            var r = np.sort_complex(np.array(new[] { C(1, 2), C(2, -1), C(3, -2), C(3, -3), C(3, 5) }));
            AssertComplex(r, C(1, 2), C(2, -1), C(3, -3), C(3, -2), C(3, 5));
        }

        [TestMethod]
        public void SortComplex_2D_SortsLastAxis()
        {
            // np.sort_complex([[3+1j, 1+2j], [2-1j, 0+5j]]) == [[1+2j, 3+1j], [0+5j, 2-1j]]
            var r = np.sort_complex(np.array(new[,] { { C(3, 1), C(1, 2) }, { C(2, -1), C(0, 5) } }));
            r.Shape.Should().Be(new Shape(2, 2));
            AssertComplex(r, C(1, 2), C(3, 1), C(0, 5), C(2, -1));
        }

        [TestMethod]
        public void SortComplex_2D_IntRows()
        {
            // np.sort_complex([[3,1,2],[9,7,8]]) == [[1,2,3],[7,8,9]] + 0j (per-row, NOT flattened)
            var r = np.sort_complex(np.array(new[,] { { 3, 1, 2 }, { 9, 7, 8 } }));
            r.Shape.Should().Be(new Shape(2, 3));
            AssertComplex(r, C(1), C(2), C(3), C(7), C(8), C(9));
        }

        [TestMethod]
        public void SortComplex_FloatNaN_SortsLast()
        {
            // np.sort_complex([nan, 2.0, 1.0]) == [1.+0.j, 2.+0.j, nan+0.j]
            var r = np.sort_complex(np.array(new[] { double.NaN, 2.0, 1.0 }));
            AssertComplex(r, C(1), C(2), C(double.NaN, 0));
        }

        [TestMethod]
        public void SortComplex_ComplexNaN_NumPyOrdering()
        {
            // np.sort_complex([nan+1j, 2+3j, 1+nanj, 0+1j]) == [0+1j, 2+3j, 1+nanj, nan+1j]
            var r = np.sort_complex(np.array(new[] { C(double.NaN, 1), C(2, 3), C(1, double.NaN), C(0, 1) }));
            AssertComplex(r, C(0, 1), C(2, 3), C(1, double.NaN), C(double.NaN, 1));
        }

        [TestMethod]
        public void SortComplex_AllDtypes_LandOnComplex()
        {
            // NumPy: i1/u1/i2/u2 → complex64, else → complex128; NumSharp's single width = Complex.
            // Values are identical in every cell (probed).
            AssertComplex(np.sort_complex(np.array(new[] { true, false, true })), C(0), C(1), C(1));
            AssertComplex(np.sort_complex(np.array(new sbyte[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new byte[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new short[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new ushort[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new uint[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new long[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new ulong[] { 3, 1, 2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new Half[] { (Half)3, (Half)1, (Half)2 })), C(1), C(2), C(3));
            AssertComplex(np.sort_complex(np.array(new float[] { 3, 1, 2 })), C(1), C(2), C(3));
            // Char/Decimal: no NumPy analog — sorted in their own dtype then cast (house decision)
            AssertComplex(np.sort_complex(np.array(new[] { 'c', 'a', 'b' })), C(97), C(98), C(99));
            AssertComplex(np.sort_complex(np.array(new decimal[] { 3, 1, 2 })), C(1), C(2), C(3));
        }

        [TestMethod]
        public void SortComplex_SortsInOriginalDtype_Int64Exact()
        {
            // The sort runs BEFORE the complex cast: 2^53 and 2^53+1 collapse to the same double,
            // so sorting after the cast could not order them — NumPy sorts the exact int64s first.
            long big = (1L << 53);
            var r = np.sort_complex(np.array(new[] { big + 1, big, 1L }));
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetValue<Complex>(0).Real.Should().Be(1.0);
            // both big values map to the same double after the cast — order among them came from int64
            r.GetValue<Complex>(1).Real.Should().Be((double)big);
            r.GetValue<Complex>(2).Real.Should().Be((double)(big + 1));
        }

        [TestMethod]
        public void SortComplex_Empty()
        {
            // np.sort_complex(np.array([], dtype=i1)) == array([], dtype=complex64) — NumSharp: Complex
            var r = np.sort_complex(np.array(new sbyte[0]));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void SortComplex_0d_Throws()
        {
            // np.sort_complex(np.array(5.0)) → AxisError: axis -1 is out of bounds for array of
            // dimension 0 (leaked from b.sort(), reproduced by delegating to np.sort)
            var act = () => np.sort_complex(NDArray.Scalar(5.0));
            act.Should().Throw<ArgumentException>().WithMessage("axis -1 is out of bounds for array of dimension 0*");
        }

        [TestMethod]
        public void SortComplex_InputNotMutated()
        {
            var src = np.array(new[] { 3.0, 1.0, 2.0 });
            np.sort_complex(src);
            src.GetValue<double>(0).Should().Be(3.0);
            src.GetValue<double>(1).Should().Be(1.0);
            src.GetValue<double>(2).Should().Be(2.0);
        }

        [TestMethod]
        public void SortComplex_StridedAndReversedViews()
        {
            // np.sort_complex(a[::2]) == [7,8,9]+0j for a=[9,1,8,2,7,3]
            var a = np.array(new[] { 9.0, 1.0, 8.0, 2.0, 7.0, 3.0 });
            AssertComplex(np.sort_complex(a["::2"]), C(7), C(8), C(9));
            // np.sort_complex(a[::-1]) == [1,2,3,7,8,9]+0j
            AssertComplex(np.sort_complex(a["::-1"]), C(1), C(2), C(3), C(7), C(8), C(9));
        }

        [TestMethod]
        public void SortComplex_Inf()
        {
            // np.sort_complex([inf, -inf, 1.0]) == [-inf+0j, 1+0j, inf+0j]
            var r = np.sort_complex(np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, 1.0 }));
            AssertComplex(r, C(double.NegativeInfinity), C(1), C(double.PositiveInfinity));
        }

        [TestMethod]
        public void SortComplex_ComplexInput_ReturnsFreshCopy()
        {
            // complex input skips the cast but must still be a copy (NumPy: array(a, copy=True))
            var src = np.array(new[] { C(3, 1), C(1, 2) });
            var r = np.sort_complex(src);
            r.GetValue<Complex>(0).Should().Be(C(1, 2));
            src.GetValue<Complex>(0).Should().Be(C(3, 1), "input must not be mutated");
        }
    }
}
