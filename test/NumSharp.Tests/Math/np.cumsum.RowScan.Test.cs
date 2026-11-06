using System;
using System.Linq;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.cumsum / np.cumprod along an axis that is not innermost in memory run by whole rows
    ///     (<c>DefaultEngine.TryScanRows</c>: <c>out[j] = out[j-1] op in[j]</c> over each contiguous block). Every
    ///     position must still see NumPy's sequential scan, so each case is compared byte for byte with the per-position
    ///     route — the same scan along the innermost axis of a C copy whose axes were moved — over special values (a NaN,
    ///     ±inf, two -0.0, integers that wrap under the product), every vector dtype, and C, F, permuted and 4-D layouts.
    ///     NumPy 2.4.2 agrees with both (the probe of the 2026-11-05 parity QA fix, 520 cases, raw bytes).
    /// </summary>
    [TestClass]
    public class CumsumRowScanTests
    {
        private static readonly string[] Dtypes =
            { "float64", "float32", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8" };

        /// <summary>
        ///     The test values: a ramp with a NaN, ±inf and two -0.0 for the floats; for the integers values in [60, 180)
        ///     whose products wrap.
        /// </summary>
        /// <param name="n">The element count (at least 91).</param>
        /// <param name="dtype">The dtype name.</param>
        /// <returns>A fresh 1-D array.</returns>
        private static NDArray Values(int n, string dtype)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = ((i * 37) % 23) - 11 + (i % 5) * 0.25;
            if (dtype.StartsWith("float"))
            {
                v[3] = double.NaN;
                v[17] = -0.0;
                v[40] = double.PositiveInfinity;
                v[41] = -0.0;
                v[90] = double.NegativeInfinity;
                return np.array(v).astype(np.dtype(dtype));
            }
            for (int i = 0; i < n; i++)
                v[i] = (System.Math.Floor(v[i] * 7.0) % 120 + 180) % 120 + 60;
            return np.array(v).astype(np.dtype(dtype));
        }

        /// <summary>The scan through the per-position route: the axis moved innermost in a C copy, scanned, moved back.</summary>
        /// <param name="x">The input.</param>
        /// <param name="axis">The scan axis.</param>
        /// <param name="multiply">True for cumprod.</param>
        /// <returns>The reference result (any layout).</returns>
        private static NDArray Reference(NDArray x, int axis, bool multiply)
        {
            using var moved = np.moveaxis(x, axis, -1);
            using var c = np.ascontiguousarray(moved);
            using var scanned = multiply ? np.cumprod(c, axis: -1) : np.cumsum(c, axis: -1);
            return np.ascontiguousarray(np.moveaxis(scanned, -1, axis));
        }

        /// <summary>The raw bytes of an array in logical C order (NaN payloads and zero signs included).</summary>
        /// <param name="x">The array.</param>
        /// <returns>The bytes.</returns>
        private static byte[] Bytes(NDArray x)
        {
            using var c = np.ascontiguousarray(x);
            using var u8 = c.view(np.uint8);
            return u8.ToArray<byte>();
        }

        /// <summary>Every layout × axis × dtype: the row route's bytes are the per-position route's.</summary>
        [TestMethod]
        public void RowScan_EqualsThePerPositionScan_ByteForByte()
        {
            foreach (var dt in Dtypes)
            {
                var layouts = new (string Name, Func<NDArray> Make)[]
                {
                    ("C", () => Values(3 * 40 * 37, dt).reshape(3, 40, 37)),
                    ("F", () => np.asfortranarray(Values(3 * 40 * 37, dt).reshape(3, 40, 37))),
                    ("P", () => Values(40 * 3 * 37, dt).reshape(40, 3, 37).transpose(new[] { 1, 0, 2 })),
                    ("Q", () => Values(6 * 5 * 4 * 33, dt).reshape(6, 5, 4, 33).transpose(new[] { 2, 0, 3, 1 })),
                };
                foreach (var (name, make) in layouts)
                {
                    using var x = make();
                    for (int axis = 0; axis < x.ndim; axis++)
                        foreach (bool multiply in new[] { false, true })
                        {
                            using var r = multiply ? np.cumprod(x, axis: axis) : np.cumsum(x, axis: axis);
                            using var expected = Reference(x, axis, multiply);
                            r.typecode.Should().Be(expected.typecode);
                            Bytes(r).Should().Equal(Bytes(expected),
                                $"{(multiply ? "cumprod" : "cumsum")} {dt} {name} axis {axis}");
                        }
                }
            }
        }

        /// <summary>
        ///     The row route writes NumPy's layout: a permuted input's result keeps its memory order, and a scan along a
        ///     C array's outer axis stays C (NumPy 2.4.2: <c>np.cumsum(P, axis=1).strides == (296, 888, 8)</c> for
        ///     <c>P = arange(4440.).reshape(40, 3, 37).transpose(1, 0, 2)</c>).
        /// </summary>
        [TestMethod]
        public void RowScan_WritesNumPysLayout()
        {
            using var p = np.arange(4440.0).reshape(40, 3, 37).transpose(new[] { 1, 0, 2 });
            using var c = np.arange(4440.0).reshape(3, 40, 37);
            using var fromP = np.cumsum(p, axis: 1);
            using var fromC = np.cumprod(c, axis: 0);

            fromP.strides.Should().Equal(296L, 888L, 8L);
            fromC.flags.c_contiguous.Should().BeTrue();
        }

        /// <summary>
        ///     A block shorter than the vector loop's threshold, a cast to the accumulator (int32 → int64) and a reversed
        ///     axis keep the per-position route, with the same values.
        /// </summary>
        [TestMethod]
        public void RowScan_Declines_KeepTheSameValues()
        {
            using var narrowInner = Values(7 * 9 * 3, "float64").reshape(7, 9, 3);       // inner block of 3
            using var widening = Values(3 * 40 * 37, "int32").reshape(3, 40, 37);       // int32 accumulates in int64
            using var reversed = Values(3 * 40 * 37, "float64").reshape(3, 40, 37)["::-1"];
            foreach (var (x, axis) in new[] { (narrowInner, 1), (widening, 0), (reversed, 0) })
            {
                using var r = np.cumsum(x, axis: axis);
                using var expected = Reference(x, axis, multiply: false);
                Bytes(r).Should().Equal(Bytes(expected));
            }
        }
    }
}
