using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Bugs surfaced by the broadcast-reduction adversarial fuzz
    ///     (<c>benchmark/poc/bcast_consistency.cs</c> + <c>bcast_ax_ref.py</c>/<c>bcast_ax_check.cs</c>,
    ///     run 2026-06-20). Each asserts the CORRECT behavior (NumPy 2.4.2 as oracle, cross-checked
    ///     against NumSharp's own materialized-copy of the same view) and FAILS while the bug exists.
    ///
    ///     IMPORTANT SCOPE NOTE: these are all PRE-EXISTING bugs in code paths DISJOINT from the
    ///     broadcast-reduce fold (commit 878240c3, <c>DefaultEngine.ExecuteElementReduction</c>).
    ///     The fold itself (sum/prod/min/max/mean over broadcast views) was verified bug-free across
    ///     ~6,600 fuzz cases (15 dtypes × layouts × axes × keepdims) vs both NumPy and materialized
    ///     copy. The bugs below live in the NaN-aware engine (<c>Default.Reduction.Nan.cs</c>) and the
    ///     argmax/argmin dtype switch — none of which the fold touches.
    /// </summary>
    public partial class OpenBugs
    {
        // =====================================================================
        // BUG: NaN-aware AXIS reductions on NON-CONTIGUOUS half arrays are wrong.
        //
        // ExecuteNanAxisReduction (Default.Reduction.Nan.cs) hand-rolls the input
        // offset from shape.strides[...] and reads arr.GetAtIndex(baseOffset + i*axisStride).
        // For float/double this resolves correctly, but for Half the index resolution
        // double-applies strides on any non-C-canonical layout (transpose, broadcast,
        // reversed) — so the reduction reads the wrong cells. float32/float64 are correct
        // on the IDENTICAL layouts; only Half breaks. Offset-only slices (still
        // C-canonical strides) are unaffected.
        // =====================================================================
        [TestMethod]
        [OpenBugs]
        public void Bug_NanAxis_Half_NonContiguous_WrongValues()
        {
            // (2,3) half, transpose -> (3,2) non-contiguous (strides [1,3]); NOT broadcast.
            var src = np.array(new double[] { double.NaN, 1.0, 2.0, 3.0, double.NaN, 5.0 })
                        .astype(NPTypeCode.Half).reshape(2, 3);
            var t = src.T; // (3,2), non-contiguous

            // NumPy: np.nansum(t, axis=0) == [3, 8]  (== nansum of the materialized copy)
            var s = np.nansum(t, (int?)0);
            ((double)(Half)s.GetAtIndex(0)).Should().Be(3.0, "nansum over the non-broadcast axis must skip NaN and sum the real cells");
            ((double)(Half)s.GetAtIndex(1)).Should().Be(8.0);

            // NumPy: np.nanmax(t, axis=1) == [3, 1, 5]
            var m = np.nanmax(t, (int?)1);
            ((double)(Half)m.GetAtIndex(0)).Should().Be(3.0);
            ((double)(Half)m.GetAtIndex(1)).Should().Be(1.0);
            ((double)(Half)m.GetAtIndex(2)).Should().Be(5.0);
        }

        // =====================================================================
        // BUG: np.nanmax / np.nanmin on COMPLEX arrays do NOT skip NaN.
        //
        // NanMax/NanMin (Default.Reduction.Nan.cs) gate on Single/Double/Half only and
        // route Complex to the regular ReduceAMax/ReduceAMin, which PROPAGATE NaN. NumPy's
        // nanmax/nanmin skip any element whose real OR imag part is NaN. Reproduces on a
        // plain contiguous array — not broadcast-specific.
        // =====================================================================
        [TestMethod]
        [OpenBugs]
        public void Bug_NanMaxMin_Complex_DoesNotSkipNaN()
        {
            var a = np.array(new Complex[]
            {
                new Complex(double.NaN, 1), new Complex(2, 2), new Complex(-1, 3)
            });

            // NumPy: np.nanmax -> (2+2j), np.nanmin -> (-1+3j)
            var mx = (Complex)np.nanmax(a).GetAtIndex(0);
            mx.Real.Should().Be(2, "nanmax must skip the NaN element and return the max of the rest");
            mx.Imaginary.Should().Be(2);

            var mn = (Complex)np.nanmin(a).GetAtIndex(0);
            mn.Real.Should().Be(-1, "nanmin must skip the NaN element and return the min of the rest");
            mn.Imaginary.Should().Be(3);
        }

        // =====================================================================
        // FIXED: Complex np.nansum AXIS reduction read UNINITIALIZED memory for ndim >= 3.
        //
        // NanSumComplex's axis branch wrote ret.SetAtIndex(sum, iterIndex[0]) — using only
        // the FIRST coordinate of the result incrementor as a flat index. For a >=3-D input
        // the reduced output is multi-D, so only a 1-D subset of positions was ever written;
        // the rest retained uninitialized `new NDArray(...,false)` heap bytes (observed as
        // ~6.95E-310 denormals). The 2-D->1-D case happened to work because the single
        // coordinate IS the flat index. NOT broadcast-specific — reproduced on a contiguous
        // 3-D array.
        //
        // Fix (Default.Reduction.Nan.cs): resolve the FULL output coordinate to its C-order
        // flat offset — ret.SetAtIndex(sum, ret.Shape.GetOffset(iterIndex)).
        // =====================================================================
        [TestMethod]
        [TestCategory("Fixed")]
        public void Bug_NanSum_Complex_Axis_3D_UninitializedMemory()
        {
            var data = new Complex[24];
            for (int i = 0; i < 24; i++)
                data[i] = (i % 3 == 0) ? new Complex(double.NaN, i) : new Complex(i, 1);
            var a = np.array(data).reshape(2, 4, 3); // contiguous, NOT broadcast

            // NumPy np.nansum(a, axis=0).ravel() ==
            //   [0, 14+2j, 16+2j, 0, 20+2j, 22+2j, 0, 26+2j, 28+2j, 0, 32+2j, 34+2j]
            var r = np.nansum(a, (int?)0);
            r.size.Should().Be(12);

            var expReal = new double[] { 0, 14, 16, 0, 20, 22, 0, 26, 28, 0, 32, 34 };
            var expImag = new double[] { 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2 };
            for (long i = 0; i < 12; i++)
            {
                var z = (Complex)r.GetAtIndex(i);
                z.Real.Should().Be(expReal[i], $"nansum axis result[{i}].Real (must match NumPy, not uninitialized memory)");
                z.Imaginary.Should().Be(expImag[i], $"nansum axis result[{i}].Imag");
            }
        }

        // =====================================================================
        // BUG: np.argmax / np.argmin on DECIMAL arrays return the wrong index.
        //
        // The argmax/argmin dtype switch mishandles Decimal: it returns a boundary index
        // (last/first) rather than the index of the extreme value. Reproduces on a plain
        // contiguous array. (Decimal has no NumPy analog; the oracle is float64 with the
        // same values: argmax([3,9,1,9,2,5]) == 1, argmin == 2.)
        // =====================================================================
        [TestMethod]
        [OpenBugs]
        public void Bug_ArgMaxMin_Decimal_WrongIndex()
        {
            var a = np.array(new decimal[] { 3, 9, 1, 9, 2, 5 });
            np.argmax(a).Should().Be(1, "argmax must return the index of the FIRST maximum (9 at index 1)");
            np.argmin(a).Should().Be(2, "argmin must return the index of the FIRST minimum (1 at index 2)");
        }

        // =====================================================================
        // BUG: np.argmax / np.argmin throw NotSupportedException on CHAR arrays.
        //
        // Char maps to NumPy uint16, which DOES support argmax/argmin. NumSharp's
        // argmax/argmin dtype switch omits Char and throws
        // "ArgMax not supported for type Char".
        // =====================================================================
        [TestMethod]
        [OpenBugs]
        public void Bug_ArgMaxMin_Char_NotSupported()
        {
            var a = np.array(new char[] { (char)3, (char)73, (char)1, (char)90, (char)2 });
            np.argmax(a).Should().Be(3, "uint16/char argmax must return index of max code point (90 at index 3)");
            np.argmin(a).Should().Be(2, "uint16/char argmin must return index of min code point (1 at index 2)");
        }
    }
}
