using System;
using System.Linq;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The memory layout of FRESH results (parity QA-36): a NumPy ufunc allocates its output through NpyIter in the K
    ///     order of its operands (<c>npyiter_find_best_axis_ordering</c> + <c>npyiter_new_temp_array</c>: the axes sorted by
    ///     the operands' strides, negative strides flipped, a broadcast axis abstaining), and the functions built on ufuncs
    ///     inherit it, so a permuted, F or flipped-F operand keeps its memory order. Every expectation is NumPy 2.4.2's
    ///     <c>(flags.c_contiguous, flags.f_contiguous, strides)</c> for the same call.
    /// </summary>
    /// <remarks>
    ///     The operands: <c>P = np.arange(24.).reshape(3, 2, 4).transpose(1, 0, 2)</c> (strides (32, 64, 8)),
    ///     <c>F = np.asfortranarray(np.arange(1., 25.).reshape(2, 3, 4))</c> (strides (8, 16, 48)), <c>FF = np.flip(F, 2)</c>
    ///     (strides (8, 16, -48)), <c>B = np.broadcast_to(np.arange(1., 5.), (2, 3, 4))</c> (strides (0, 0, 8)), and the
    ///     int64 <c>Pi</c> laid out as P.
    /// </remarks>
    [TestClass]
    public class UfuncResultLayoutTests
    {
        /// <summary>A permuted float64 operand, strides (32, 64, 8): neither C nor F, dense.</summary>
        /// <returns>A fresh view (its own buffer kept alive by it).</returns>
        private static NDArray P() => np.arange(24.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });

        /// <summary>An F-contiguous float64 operand, strides (8, 16, 48).</summary>
        /// <returns>A fresh array.</returns>
        private static NDArray F() => np.asfortranarray(np.arange(1.0, 25.0).reshape(2, 3, 4));

        /// <summary>An F operand flipped on its last axis, strides (8, 16, -48): NpyIter flips the negative stride back.</summary>
        /// <returns>A fresh view.</returns>
        private static NDArray FF() => np.flip(F(), 2);

        /// <summary>A row broadcast to (2, 3, 4), strides (0, 0, 8): its stride-0 axes abstain in NpyIter's vote.</summary>
        /// <returns>A fresh read-only view.</returns>
        private static NDArray B() => np.broadcast_to(np.arange(1.0, 5.0), new Shape(2, 3, 4));

        /// <summary>The int64 twin of <see cref="P"/>, strides (32, 64, 8).</summary>
        /// <returns>A fresh view.</returns>
        private static NDArray Pi() => np.arange(1L, 25L).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });

        /// <summary>Asserts NumPy's flags and byte strides for a result.</summary>
        /// <param name="r">The result.</param>
        /// <param name="c">NumPy's <c>flags.c_contiguous</c>.</param>
        /// <param name="f">NumPy's <c>flags.f_contiguous</c>.</param>
        /// <param name="strides">NumPy's byte strides.</param>
        private static void Layout(NDArray r, bool c, bool f, params long[] strides)
        {
            r.strides.Should().Equal(strides);
            r.flags.c_contiguous.Should().Be(c);
            r.flags.f_contiguous.Should().Be(f);
        }

        /// <summary>
        ///     Unary ufuncs keep a permuted input's order and lay a flipped F input out as F; a K-order ravel then reads the
        ///     result in memory order (<c>np.ravel(np.negative(P), order='K')</c> is <c>-0, -1, -2, …</c>).
        /// </summary>
        [TestMethod]
        public void Unary_KeepsTheInputsMemoryOrder()
        {
            using var p = P();
            using var ff = FF();
            using var negative = np.negative(p);
            using var positive = np.positive(p);
            using var sqrt = np.sqrt(ff);
            using var positiveFF = np.positive(ff);
            using var memoryOrder = np.ravel(negative, 'K');

            Layout(negative, false, false, 32, 64, 8);
            Layout(positive, false, false, 32, 64, 8);
            Layout(sqrt, false, true, 8, 16, 48);
            Layout(positiveFF, false, true, 8, 16, 48);
            memoryOrder.ToArray<double>().Take(6).Should().Equal(-0.0, -1.0, -2.0, -3.0, -4.0, -5.0);
        }

        /// <summary>
        ///     Binary ufuncs — arithmetic, the float-tier family, ldexp and the shifts — keep the operands' K order, with a
        ///     0-d operand that never votes.
        /// </summary>
        [TestMethod]
        public void Binary_KeepsTheOperandsMemoryOrder()
        {
            using var p = P();
            using var f = F();
            using var pi = Pi();
            using var two = np.array(2.0);
            using var minusP = -p;
            using var sum = p + p;
            using var withScalar = p + two;
            using var copysign = np.copysign(p, minusP);
            using var hypot = np.hypot(f, f);
            using var arctan2 = np.arctan2(f, two);
            using var piAsDouble = pi.astype(np.float64);
            using var ldexp = np.ldexp(piAsDouble, np.array(2L));
            using var shifted = np.left_shift(pi, np.array(1L));

            Layout(sum, false, false, 32, 64, 8);
            Layout(withScalar, false, false, 32, 64, 8);
            Layout(copysign, false, false, 32, 64, 8);
            Layout(hypot, false, true, 8, 16, 48);
            Layout(arctan2, false, true, 8, 16, 48);
            Layout(ldexp, false, false, 32, 64, 8);
            Layout(shifted, false, false, 32, 64, 8);
        }

        /// <summary>Comparison and logical ufuncs and np.isclose: a bool result in the operands' K order.</summary>
        [TestMethod]
        public void ComparisonAndLogical_KeepTheOperandsMemoryOrder()
        {
            using var p = P();
            using var greater = p > 3.0;
            using var low = p > 2.0;
            using var high = p < 20.0;
            using var logicalAnd = np.logical_and(low, high);
            using var isclose = np.isclose(p, p);

            Layout(greater, false, false, 4, 8, 1);
            Layout(logicalAnd, false, false, 4, 8, 1);
            Layout(isclose, false, false, 4, 8, 1);
        }

        /// <summary>np.where allocates through a KEEPORDER iterator over (condition, x, y).</summary>
        [TestMethod]
        public void Where_KeepsTheOperandsMemoryOrder()
        {
            using var p = P();
            using var f = F();
            using var pCond = p > 3.0;
            using var fCond = f > 3.0;
            using var minusF = -f;
            using var fromP = np.where(pCond, p, 0.0);
            using var fromF = np.where(fCond, f, minusF);

            Layout(fromP, false, false, 32, 64, 8);
            Layout(fromF, false, true, 8, 16, 48);
            fromP.ToArray<double>().Should().Equal(p.ToArray<double>().Select(v => v > 3.0 ? v : 0.0));
        }

        /// <summary>
        ///     The scans along an axis: PyUFunc_Accumulate's outer iterator allocates the output in the input's K order (the
        ///     NaN scans inherit it from NumPy's <c>np.array(a, copy=True)</c> of the input, which lays a broadcast input out
        ///     by its stride ranking).
        /// </summary>
        [TestMethod]
        public void Scans_KeepTheInputsMemoryOrder()
        {
            using var p = P();
            using var f = F();
            using var ff = FF();
            using var b = B();
            using var cumsum2 = np.cumsum(p, axis: 2);
            using var cumsum0 = np.cumsum(p, axis: 0);
            using var cumprodF = np.cumprod(f, axis: 0);
            using var cumprodFF = np.cumprod(ff, axis: 2);
            using var nancumsumB = np.nancumsum(b, axis: 1);
            using var nancumprodP = np.nancumprod(p, axis: 1);

            Layout(cumsum2, false, false, 32, 64, 8);
            Layout(cumsum0, false, false, 32, 64, 8);
            Layout(cumprodF, false, true, 8, 16, 48);
            Layout(cumprodFF, false, true, 8, 16, 48);
            Layout(nancumsumB, false, false, 24, 8, 48);
            Layout(nancumprodP, false, false, 32, 64, 8);
            // NumPy: np.cumprod(np.flip(F, 2), axis=2)[1, 2] is [24, 552, 12144, 255024] (the row 21..24 reversed, then
            // multiplied up): the relabelled result reads in logical order.
            cumprodFF["1, 2"].ToArray<double>().Should().Equal(24.0, 552.0, 12144.0, 255024.0);
        }

        /// <summary>np.diff (a subtract of two slices) and np.nan_to_num (np.array's K copy) keep the input's order.</summary>
        [TestMethod]
        public void DiffAndNanToNum_KeepTheInputsMemoryOrder()
        {
            using var p = P();
            using var f = F();
            using var ff = FF();
            using var diffP = np.diff(p, axis: 1);
            using var diffF = np.diff(f, axis: 1);
            using var nanToNumP = np.nan_to_num(p);
            using var nanToNumFF = np.nan_to_num(ff);

            Layout(diffP, false, false, 32, 64, 8);
            Layout(diffF, false, true, 8, 16, 32);
            Layout(nanToNumP, false, false, 32, 64, 8);
            Layout(nanToNumFF, false, true, 8, 16, 48);
        }

        /// <summary>The two-output ufuncs lay both fresh outputs out in the operands' K order.</summary>
        [TestMethod]
        public void TwoOutputUfuncs_KeepTheOperandsMemoryOrder()
        {
            using var p = P();
            using var f = F();
            var (mantissa, exponent) = np.frexp(p);
            var (fractional, integral) = np.modf(f);
            var (quotient, remainder) = np.divmod(p, 3.0);

            Layout(mantissa, false, false, 32, 64, 8);
            Layout(exponent, false, false, 16, 32, 4);
            Layout(integral, false, true, 8, 16, 48);
            Layout(fractional, false, true, 8, 16, 48);
            Layout(quotient, false, false, 32, 64, 8);
            Layout(remainder, false, false, 32, 64, 8);
            foreach (var r in new[] { mantissa, exponent, fractional, integral, quotient, remainder })
                r.Dispose();
        }

        /// <summary>
        ///     The other direction: np.around with decimals != 0 allocates <c>PyArray_Empty(..., PyArray_ISFORTRAN(a))</c>,
        ///     C unless the input is F-only, never the input's K order (an integer input's cast back keeps that layout).
        /// </summary>
        [TestMethod]
        public void AroundWithDecimals_IsCUnlessTheInputIsFOnly()
        {
            using var p = P();
            using var f = F();
            using var b = B();
            using var pi = Pi();
            using var aroundP = np.around(p, 2);
            using var aroundF = np.around(f, 2);
            using var aroundB = np.around(b, 2);
            using var aroundInt = np.around(pi, -1);

            Layout(aroundP, true, false, 96, 32, 8);
            Layout(aroundF, false, true, 8, 16, 48);
            Layout(aroundB, true, false, 96, 32, 8);
            Layout(aroundInt, true, false, 96, 32, 8);
        }

        /// <summary>
        ///     np.asarray with a dtype change under <c>order='A'</c> copies in K order (PyArray_FromArray without a contiguity
        ///     flag), while the same dtype copies C (<c>PyArray_NewCopy(order='A')</c>).
        /// </summary>
        [TestMethod]
        public void AsarrayOrderA_CastCopiesInKOrder()
        {
            using var p = P();
            using var b = B();
            using var castP = np.asarray(p, np.float32, 'A');
            using var castB = np.asarray(b, np.float32, 'A');
            using var anyP = np.asanyarray(p, np.float32, 'A');
            using var sameP = np.asarray(p, np.float64, 'A', copy: true);

            Layout(castP, false, false, 16, 32, 4);
            Layout(castB, false, false, 12, 4, 24);
            Layout(anyP, false, false, 16, 32, 4);
            Layout(sameP, true, false, 96, 32, 8);
        }

        /// <summary>
        ///     Compositions: np.angle of a real input is <c>arctan2(0, z)</c> (an F input stays F; a broadcast one is C, as
        ///     the cast to float64 does not vote), and einsum's allocated output keeps a permuted operand's order.
        /// </summary>
        [TestMethod]
        public void Compositions_InheritTheUfuncLayout()
        {
            using var p = P();
            using var f = F();
            using var b = B();
            using var angleF = np.angle(f);
            using var angleB = np.angle(b);
            using var einsum = np.einsum("ijk,ijk->ijk", p, p);

            Layout(angleF, false, true, 8, 16, 48);
            Layout(angleB, true, false, 96, 32, 8);
            Layout(einsum, false, false, 32, 64, 8);
        }

        /// <summary>
        ///     Past NumPy's 64 dimensions the same rule holds through the long-axes sort: a 70-dimensional array whose three
        ///     long axes are swapped out of C order keeps their memory order through unary and binary ufuncs and a scan,
        ///     and a K-order ravel of the result reads the input's memory order.
        /// </summary>
        [TestMethod, Misaligned]
        public void SeventyDimensions_KeepTheMemoryOrderOfTheLongAxes()
        {
            var dims = Enumerable.Repeat(1L, 70).ToArray();
            dims[10] = 3;
            dims[40] = 2;
            dims[69] = 4;
            using var c = np.arange(24.0).reshape(new Shape(dims));
            using var x = np.swapaxes(c, 10, 40);   // axis 40 (extent 3) is now outermost in memory, axis 10 (extent 2) next

            using var negative = np.negative(x);
            using var sum = x + x;
            using var scan = np.cumsum(x, axis: 40);

            foreach (var r in new[] { negative, sum, scan })
            {
                r.strides[40].Should().Be(64);
                r.strides[10].Should().Be(32);
                r.strides[69].Should().Be(8);
            }
            using var memoryOrder = np.ravel(negative, 'K');
            memoryOrder.ToArray<double>().Should().Equal(Enumerable.Range(0, 24).Select(v => -(double)v));
        }

        /// <summary>
        ///     The owner's reach (about 200,000 dimensions): the K-order plan stays linear in the rank, so a permuted
        ///     200,000-dimensional operand keeps its long axes' memory order through a unary and a binary ufunc.
        /// </summary>
        [TestMethod, Misaligned]
        public void TwoHundredThousandDimensions_KeepTheMemoryOrderOfTheLongAxes()
        {
            const int rank = 200_000;
            var dims = Enumerable.Repeat(1L, rank).ToArray();
            dims[0] = 3;
            dims[100_000] = 2;
            dims[rank - 1] = 4;
            using var c = np.arange(24.0).reshape(new Shape(dims));
            using var x = np.swapaxes(c, 0, 100_000);

            using var negative = np.negative(x);
            using var sum = x + x;

            foreach (var r in new[] { negative, sum })
            {
                r.ndim.Should().Be(rank);
                var strides = r.strides;
                strides[100_000].Should().Be(64);
                strides[0].Should().Be(32);
                strides[rank - 1].Should().Be(8);
            }
        }
    }
}
