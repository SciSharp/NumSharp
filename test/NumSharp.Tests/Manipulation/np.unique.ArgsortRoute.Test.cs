using System;
using NumSharp.Backends;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     np.unique's index and inverse routes over float16 / float32 / float64 / complex128 are NumPy's
    ///     <c>_unique1d</c> statements: <c>perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')</c>,
    ///     <c>aux = ar[perm]</c>, the run starts <c>aux[1:] != aux[:-1]</c>, <c>equal_nan</c> collapsing the trailing NaN run
    ///     from its first NaN, and the index <c>perm[mask]</c>. NumSharp's argsort is NumPy's stable (mergesort) order, so on
    ///     the return_index route every output is NumPy's: the zero a run of <c>±0</c> keeps (the first in input order), the
    ///     bits of the NaN a collapsed run keeps (the first NaN in input order, a signalling NaN unquieted), and the NaN class
    ///     that represents a complex run (NumPy's complex sort order: <c>R+nanj</c> by R, then <c>nan+Rj</c> by R, then
    ///     <c>nan+nanj</c>). The route these replace partitioned the NaNs to the tail unstably and sorted the rest with a key
    ///     that orders <c>-0.0</c> below <c>+0.0</c>: values, index and inverse could each leave NumPy's answer. Expected
    ///     values are NumPy 2.4.2's (raw bits for the floats); the float16 / complex128 counts-only calls take this route too.
    /// </summary>
    [TestClass]
    public class NpUniqueArgsortRouteTest
    {
        private static ulong Bits(double d) => BitConverter.DoubleToUInt64Bits(d);

        private static ulong[] Bits64(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => Bits(a.GetAtIndex<double>(i))).ToArray();

        private static uint[] Bits32(NDArray a) =>
            Enumerable.Range(0, (int)a.size).Select(i => BitConverter.SingleToUInt32Bits(a.GetAtIndex<float>(i))).ToArray();

        private static ushort[] Bits16(NDArray a) =>
            Enumerable.Range(0, (int)a.size).Select(i => BitConverter.HalfToUInt16Bits(a.GetAtIndex<Half>(i))).ToArray();

        /// <summary>Real and imaginary bits of every complex value, interleaved.</summary>
        private static ulong[] BitsC(NDArray a) =>
            Enumerable.Range(0, (int)a.size).SelectMany(i => { var z = a.GetAtIndex<Complex>(i); return new[] { Bits(z.Real), Bits(z.Imaginary) }; }).ToArray();

        private static long[] Ints(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.GetInt64(i)).ToArray();

        private static NDArray[] All(NDArray a, bool equal_nan) =>
            np.unique(a, return_index: true, return_inverse: true, return_counts: true, equal_nan: equal_nan);

        private const ulong Nan = 0x7FF8000000000000, One = 0x3FF0000000000000, Two = 0x4000000000000000;

        /// <summary>
        ///     <c>[nan+0j, 1+nanj, 2+0j]</c>: NumPy sorts <c>1+nanj</c> before <c>nan+0j</c>, so the collapsed run keeps
        ///     <c>1+nanj</c> and its index 1 (the old route kept <c>1+nanj</c>'s value beside <c>nan+0j</c>'s index 0).
        /// </summary>
        [TestMethod]
        public void Complex_NaNClasses_RNanjBeforeNanRj()
        {
            double nan = BitConverter.UInt64BitsToDouble(Nan);
            var a = np.array(new[] { new Complex(nan, 0), new Complex(1, nan), new Complex(2, 0) });

            var r = All(a, equal_nan: true);
            BitsC(r[0]).Should().Equal(Two, 0UL, One, Nan);
            Ints(r[1]).Should().Equal(2L, 1);
            Ints(r[2]).Should().Equal(1L, 1, 0);
            Ints(r[3]).Should().Equal(1L, 2);

            r = All(a, equal_nan: false);
            BitsC(r[0]).Should().Equal(Two, 0UL, One, Nan, Nan, 0UL);
            Ints(r[1]).Should().Equal(2L, 1, 0);
            Ints(r[2]).Should().Equal(2L, 1, 0);
            Ints(r[3]).Should().Equal(1L, 1, 1);

            // Counts alone take the argsort route too (NumPy's in-place complex sort keeps the bits).
            var c = np.unique(a, return_counts: true);
            BitsC(c[0]).Should().Equal(Two, 0UL, One, Nan);
            Ints(c[1]).Should().Equal(1L, 2);
        }

        /// <summary><c>[nan+nanj, nan+1j, 0+nanj, 1+0j]</c>: the classes in NumPy's order, <c>nan+nanj</c> last.</summary>
        [TestMethod]
        public void Complex_NaNClasses_AllFour()
        {
            double nan = BitConverter.UInt64BitsToDouble(Nan);
            var a = np.array(new[] { new Complex(nan, nan), new Complex(nan, 1), new Complex(0, nan), new Complex(1, 0) });

            var r = All(a, equal_nan: true);
            BitsC(r[0]).Should().Equal(One, 0UL, 0UL, Nan);
            Ints(r[1]).Should().Equal(3L, 2);
            Ints(r[2]).Should().Equal(1L, 1, 1, 0);
            Ints(r[3]).Should().Equal(1L, 3);

            r = All(a, equal_nan: false);
            BitsC(r[0]).Should().Equal(One, 0UL, 0UL, Nan, Nan, One, Nan, Nan);
            Ints(r[1]).Should().Equal(3L, 2, 1, 0);
            Ints(r[2]).Should().Equal(3L, 2, 1, 0);
            Ints(r[3]).Should().Equal(1L, 1, 1, 1);
        }

        /// <summary><c>[nan+2j, nan+1j, 1+0j, nan+1j]</c>: <c>nan+Rj</c> values by R, equal ones in input order.</summary>
        [TestMethod]
        public void Complex_NanRj_ByImaginaryPart_TiesInInputOrder()
        {
            double nan = BitConverter.UInt64BitsToDouble(Nan);
            var a = np.array(new[] { new Complex(nan, 2), new Complex(nan, 1), new Complex(1, 0), new Complex(nan, 1) });

            var r = All(a, equal_nan: true);
            BitsC(r[0]).Should().Equal(One, 0UL, Nan, One);
            Ints(r[1]).Should().Equal(2L, 1);
            Ints(r[2]).Should().Equal(1L, 1, 0, 1);
            Ints(r[3]).Should().Equal(1L, 3);

            r = All(a, equal_nan: false);
            BitsC(r[0]).Should().Equal(One, 0UL, Nan, One, Nan, One, Nan, Two);
            Ints(r[1]).Should().Equal(2L, 1, 3, 0);
            Ints(r[2]).Should().Equal(3L, 1, 0, 2);
        }

        /// <summary>
        ///     A run of zeros keeps the FIRST zero in input order on the index and inverse routes (NumPy's mergesort): the
        ///     old route's radix key put <c>-0.0</c> first whatever the input order.
        /// </summary>
        [TestMethod]
        public void SignedZeros_KeepTheFirstZeroInInputOrder()
        {
            foreach (bool en in new[] { true, false })
            {
                // np.unique([0.0, -0.0, 1.0], return_index=True, ...) keeps +0.0 (index 0); [-0.0, 0.0, 1.0] keeps -0.0.
                var r = All(np.array(new[] { 0.0, -0.0, 1.0 }), en);
                Bits64(r[0]).Should().Equal(0UL, One);
                Ints(r[1]).Should().Equal(0L, 2);
                Ints(r[2]).Should().Equal(0L, 0, 1);
                Ints(r[3]).Should().Equal(2L, 1);
                Bits64(All(np.array(new[] { -0.0, 0.0, 1.0 }), en)[0]).Should().Equal(0x8000000000000000UL, One);

                Bits32(All(np.array(new[] { 0.0f, -0.0f, 1.0f }), en)[0]).Should().Equal(0U, 0x3F800000U);
                Bits16(All(np.array(new[] { (Half)0f, BitConverter.UInt16BitsToHalf(0x8000), (Half)1f }), en)[0]).Should().Equal((ushort)0, (ushort)0x3C00);
                BitsC(All(np.array(new[] { new Complex(-0.0, -0.0), new Complex(0, 0), new Complex(1, 0) }), en)[0])
                    .Should().Equal(0x8000000000000000UL, 0x8000000000000000UL, One, 0UL);
            }

            // Forty elements whose first is -0.0 (NumPy: [-0.0, 1, 2, 3, 4] with index [0, 1, 7, 3, 9]); the inverse alone
            // (NumPy's quicksort argsort) agrees here.
            var line = new double[40];
            for (int i = 0; i < 40; i++) line[i] = i % 2 == 0 ? (i % 3 == 0 ? -0.0 : 0.0) : i % 5;
            var all = All(np.array(line), equal_nan: true);
            Bits64(all[0]).Should().Equal(0x8000000000000000UL, One, Two, 0x4008000000000000UL, 0x4010000000000000UL);
            Ints(all[1]).Should().Equal(0L, 1, 7, 3, 9);
            Ints(all[3]).Should().Equal(24L, 4, 4, 4, 4);
            Bits64(np.unique(np.array(line), return_inverse: true)[0])[0].Should().Be(0x8000000000000000UL);
        }

        /// <summary>
        ///     The NaN a collapsed run keeps is the first NaN in input order, bits and all (a payload, a negative sign, a
        ///     signalling NaN): <c>aux = ar[perm]</c> is a gather. Without <c>equal_nan</c> every NaN is a value, in input order.
        /// </summary>
        [TestMethod]
        public void NaNBits_IndexRoute_KeepTheFirstNaNInInputOrder()
        {
            double D(ulong b) => BitConverter.UInt64BitsToDouble(b);
            const ulong Pay = 0x7FF8000000000123, Neg = 0xFFF8000000000000, SNaN = 0x7FF0000000000001;

            var r = All(np.array(new[] { D(Pay), D(Neg), 1.0 }), equal_nan: true);
            Bits64(r[0]).Should().Equal(One, Pay);
            Ints(r[1]).Should().Equal(2L, 0);
            Ints(r[2]).Should().Equal(1L, 1, 0);
            Ints(r[3]).Should().Equal(1L, 2);
            r = All(np.array(new[] { D(Pay), D(Neg), 1.0 }), equal_nan: false);
            Bits64(r[0]).Should().Equal(One, Pay, Neg);
            Ints(r[1]).Should().Equal(2L, 0, 1);
            Ints(r[2]).Should().Equal(1L, 2, 0);

            Bits64(All(np.array(new[] { D(Neg), 1.0, D(Pay) }), equal_nan: true)[0]).Should().Equal(One, Neg);
            r = All(np.array(new[] { D(SNaN), 1.0, D(Nan) }), equal_nan: true);
            Bits64(r[0]).Should().Equal(One, SNaN);
            Ints(r[1]).Should().Equal(1L, 0);

            float S(uint b) => BitConverter.UInt32BitsToSingle(b);
            r = All(np.array(new[] { S(0x7FC00123), S(0xFFC00000), 1.0f }), equal_nan: false);
            Bits32(r[0]).Should().Equal(0x3F800000U, 0x7FC00123U, 0xFFC00000U);
            Ints(r[1]).Should().Equal(2L, 0, 1);

            Half H(ushort b) => BitConverter.UInt16BitsToHalf(b);
            r = All(np.array(new[] { H(0x7E01), H(0xFE00), (Half)1f }), equal_nan: true);
            Bits16(r[0]).Should().Equal((ushort)0x3C00, (ushort)0x7E01);
            Ints(r[1]).Should().Equal(2L, 0);
            // float16 counts alone take this route too (NumPy's float16 in-place sort keeps the bits): the payload NaN,
            // first in input order, represents the run (the permutation route kept -nan).
            var c = np.unique(np.array(new[] { H(0x7E01), H(0xFE00), (Half)1f }), return_counts: true);
            Bits16(c[0]).Should().Equal((ushort)0x3C00, (ushort)0x7E01);
            Ints(c[1]).Should().Equal(1L, 2);
            // float16 values alone too (NumPy's float16 in-place sort, an insertion sort below 17 elements, keeps equal NaNs
            // in input order): [pay, -nan, 1] gives [1, pay], and both NaNs in input order without equal_nan.
            Bits16(np.unique(np.array(new[] { H(0x7E01), H(0xFE00), (Half)1f })).values).Should().Equal((ushort)0x3C00, (ushort)0x7E01);
            Bits16(np.unique(np.array(new[] { H(0x7E01), H(0xFE00), (Half)1f }), equal_nan: false).values)
                .Should().Equal((ushort)0x3C00, (ushort)0x7E01, (ushort)0xFE00);
        }

        /// <summary>
        ///     The three outputs agree with each other on every float / complex route: each value is bitwise the input
        ///     element its index names, and the inverse rebuilds the input (a collapsed NaN run: some NaN). Random lines with
        ///     repeated values, both zeros and NaNs of several payloads, past the radix's insertion thresholds.
        /// </summary>
        [TestMethod]
        public void IndexValuesAndInverse_AreConsistent()
        {
            var rng = new System.Random(20261028);
            double[] pool = { 0.0, -0.0, 1.5, -2.0, double.PositiveInfinity, BitConverter.UInt64BitsToDouble(Nan),
                              BitConverter.UInt64BitsToDouble(0xFFF8000000000000), BitConverter.UInt64BitsToDouble(0x7FF8000000000005) };
            foreach (int n in new[] { 7, 90, 300, 2000 })
            foreach (var dt in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Half, NPTypeCode.Complex })
            foreach (bool en in new[] { true, false })
            {
                var raw = Enumerable.Range(0, n).Select(_ => pool[rng.Next(pool.Length)]).ToArray();
                NDArray a = dt == NPTypeCode.Complex
                    ? np.array(raw.Select((v, i) => new Complex(v, pool[(i * 7 + rng.Next(3)) % pool.Length])).ToArray())
                    : np.array(raw).astype(dt);
                var r = All(a, en);
                var values = r[0]; var index = r[1]; var inverse = r[2];
                for (long k = 0; k < values.size; k++)
                {
                    using var want = a[index.GetInt64(k)];
                    using var got = values[k];
                    got.tobytes().Should().Equal(want.tobytes(), $"{dt} n={n} en={en}: values[{k}] is ar[index[{k}]] bit for bit");
                }
                for (long i = 0; i < n; i++)
                {
                    using var orig = a[i];
                    using var rebuilt = values[inverse.GetInt64(i)];
                    bool origNaN = (bool)np.isnan(orig);
                    if (origNaN) ((bool)np.isnan(rebuilt)).Should().BeTrue();
                    else ((bool)(rebuilt == orig)).Should().BeTrue($"{dt} n={n} en={en}: values[inverse[{i}]] == ar[{i}]");
                }
            }
        }

        /// <summary>
        ///     The long-indexed fallback (above <see cref="Array.MaxLength"/> elements, reached here by reflection on small
        ///     lines) runs the same statements: for every call the managed route answers with the argsort statements, the
        ///     fallback's outputs are byte-identical to np.unique's (values, index, inverse, counts), on the probe lines of the
        ///     other tests and on random lines with ties, both zeros and NaNs of several payloads.
        /// </summary>
        [TestMethod]
        public void LongIndexedFallback_GivesTheManagedOutputs()
        {
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var longFloat = typeof(DefaultEngine).GetMethods(flags).First(m => m.Name == "uniqueFlatSortedLongFloat" && m.IsGenericMethodDefinition);
            var longComplex = typeof(DefaultEngine).GetMethods(flags).First(m => m.Name == "uniqueFlatSortedLongComplex");

            double nan = BitConverter.UInt64BitsToDouble(Nan);
            double D(ulong b) => BitConverter.UInt64BitsToDouble(b);
            var lines = new List<NDArray>
            {
                np.array(new[] { new Complex(nan, 0), new Complex(1, nan), new Complex(2, 0) }),
                np.array(new[] { new Complex(nan, nan), new Complex(nan, 1), new Complex(0, nan), new Complex(1, 0) }),
                np.array(new[] { new Complex(nan, 2), new Complex(nan, 1), new Complex(1, 0), new Complex(nan, 1) }),
                np.array(new[] { new Complex(-0.0, -0.0), new Complex(0, 0), new Complex(1, 0) }),
                np.array(new[] { 0.0, -0.0, 1.0 }),
                np.array(new[] { D(0x7FF8000000000123), D(0xFFF8000000000000), 1.0 }),
                np.array(new[] { D(0x7FF0000000000001), 1.0, D(Nan) }),
                np.array(new[] { BitConverter.UInt32BitsToSingle(0x7FC00123), BitConverter.UInt32BitsToSingle(0xFFC00000), 1.0f }),
                np.array(new[] { BitConverter.UInt16BitsToHalf(0x7E01), BitConverter.UInt16BitsToHalf(0xFE00), (Half)1f }),
            };
            var rng = new System.Random(20261028);
            double[] pool = { 0.0, -0.0, 1.5, -2.0, double.PositiveInfinity, nan, D(0xFFF8000000000000), D(0x7FF8000000000005) };
            foreach (int n in new[] { 5, 130, 1000 })
            {
                var raw = Enumerable.Range(0, n).Select(_ => pool[rng.Next(pool.Length)]).ToArray();
                lines.Add(np.array(raw));
                lines.Add(np.array(raw).astype(np.float32));
                lines.Add(np.array(raw).astype(np.float16));
                lines.Add(np.array(raw.Select((v, i) => new Complex(v, pool[(i * 7 + rng.Next(3)) % pool.Length])).ToArray()));
                lines.Add(np.array(raw).reshape(n / 5, 5)["::-1"]);   // a non-contiguous operand (rows reversed)
            }

            foreach (var a in lines)
            foreach (bool ri in new[] { false, true })
            foreach (bool rv in new[] { false, true })
            foreach (bool rc in new[] { false, true })
            foreach (bool en in new[] { true, false })
            {
                // The routes the managed side answers with the argsort statements: an index or an inverse, every float16
                // call, and complex128 counts (float32 / float64 values or counts alone are NumPy's in-place sort; complex
                // values alone the hash path).
                bool argsorted = ri || rv || a.typecode == NPTypeCode.Half || (rc && a.typecode == NPTypeCode.Complex);
                if (!argsorted) continue;
                NDArray[] want = np.unique(a, return_index: ri, return_inverse: rv, return_counts: rc, equal_nan: en);
                var args = new object[] { a, (long)a.size, en, ri, rv, rc };
                var got = a.typecode == NPTypeCode.Complex
                    ? (NDArray[])longComplex.Invoke(BackendFactory.GetEngine<DefaultEngine>(), args)
                    : (NDArray[])longFloat.MakeGenericMethod(a.dtype.type).Invoke(BackendFactory.GetEngine<DefaultEngine>(), args);
                got.Length.Should().Be(want.Length);
                for (int k = 0; k < got.Length; k++)
                {
                    got[k].shape.Should().Equal(want[k].shape, $"{a.typecode} n={a.size} ri={ri} rv={rv} rc={rc} en={en} output {k}");
                    np.ascontiguousarray(got[k]).tobytes().Should().Equal(np.ascontiguousarray(want[k]).tobytes(),
                        $"{a.typecode} n={a.size} ri={ri} rv={rv} rc={rc} en={en} output {k}");
                }
            }
        }
    }
}
