using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <c>np.unique</c> of complex128 values without return flags is NumPy's C++ hash path (<c>unique.cpp</c>): each
    ///     equality class keeps its FIRST element in C order, every value with a NaN part is one class, and <c>+0.0</c> /
    ///     <c>-0.0</c> are one class. NumSharp's sort path kept the NaN its unstable partition left first. Expected values are
    ///     NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class np_unique_complex_hash_tests
    {
        private static (double re, double im)[] Parts(NDArray a) =>
            a.ToArray<Complex>().Select(c => (c.Real, c.Imaginary)).ToArray();

        /// <summary>The NaN class keeps its first member: <c>nan+0j</c> in one order, <c>1+nanj</c> in the other.</summary>
        [TestMethod]
        public void Unique_ComplexNaNClass_KeepsTheFirstNaN()
        {
            var nan = double.NaN;
            var r = Parts(np.unique(np.array(new[] { new Complex(nan, 0), new Complex(0, 1), new Complex(1, nan),
                                                     new Complex(nan, 0), new Complex(1, nan) })).values);
            r.Length.Should().Be(2);
            r[0].Should().Be((0.0, 1.0));
            double.IsNaN(r[1].re).Should().BeTrue();
            r[1].im.Should().Be(0.0);

            var r2 = Parts(np.unique(np.array(new[] { new Complex(1, nan), new Complex(0, 1), new Complex(nan, 0) })).values);
            r2.Length.Should().Be(2);
            r2[0].Should().Be((0.0, 1.0));
            r2[1].re.Should().Be(1.0);
            double.IsNaN(r2[1].im).Should().BeTrue();
        }

        /// <summary><c>-0.0</c> and <c>+0.0</c> are one class, kept with the first member's bits: <c>[-0+0j, 0+0j, 2j]</c>
        ///     gives <c>[-0+0j, 2j]</c>.</summary>
        [TestMethod]
        public void Unique_ComplexSignedZeros_KeepTheFirstBits()
        {
            var r = np.unique(np.array(new[] { new Complex(-0.0, 0), new Complex(0, 0), new Complex(0, 2) })).values
                      .ToArray<Complex>();
            r.Length.Should().Be(2);
            BitConverter.DoubleToInt64Bits(r[0].Real).Should().BeLessThan(0, "the first zero is -0.0");
            r[1].Imaginary.Should().Be(2.0);
        }

        private const ulong Nan = 0x7FF8000000000000, Pay = 0x7FF8000000000123, Neg = 0xFFF8000000000000;

        private static double D(ulong bits) => BitConverter.UInt64BitsToDouble(bits);

        private static Complex C(double re, double im) => new Complex(re, im);

        /// <summary>The real and imaginary bit patterns of every value, interleaved.</summary>
        private static ulong[] Bits(NDArray a) =>
            Enumerable.Range(0, (int)a.size).SelectMany(i =>
            {
                var z = a.GetAtIndex<Complex>(i);
                return new[] { BitConverter.DoubleToUInt64Bits(z.Real), BitConverter.DoubleToUInt64Bits(z.Imaginary) };
            }).ToArray();

        /// <summary>
        ///     The probe lines: values with a NaN part of every class in different input orders (and two payloads, a
        ///     negative NaN, both zeros), each with NumPy 2.4.2's <c>np.unique(a, equal_nan=False)</c> as bit patterns:
        ///     NumPy's sort of its hash result puts them in its complex order (R+nanj by R, nan+Rj by R, nan+nanj last).
        /// </summary>
        private static readonly (string name, Complex[] line, ulong[] withoutEqualNan)[] Lines =
        {
            ("b_then_c", new[] { C(1, D(Nan)), C(D(Nan), 2), C(3, 0) },
                new ulong[] { 0x4008000000000000, 0, 0x3FF0000000000000, Nan, Nan, 0x4000000000000000 }),
            ("c_then_b", new[] { C(D(Nan), 2), C(1, D(Nan)), C(3, 0) },
                new ulong[] { 0x4008000000000000, 0, 0x3FF0000000000000, Nan, Nan, 0x4000000000000000 }),
            ("d_c_b_a", new[] { C(D(Nan), D(Nan)), C(D(Nan), 5), C(7, D(Nan)), C(1, 1) },
                new ulong[] { 0x3FF0000000000000, 0x3FF0000000000000, 0x401C000000000000, Nan, Nan, 0x4014000000000000, Nan, Nan }),
            ("c_by_imag", new[] { C(D(Nan), 3), C(D(Nan), 1), C(D(Nan), 2), C(0, 0) },
                new ulong[] { 0, 0, Nan, 0x3FF0000000000000, Nan, 0x4000000000000000, Nan, 0x4008000000000000 }),
            ("b_by_real", new[] { C(3, D(Nan)), C(1, D(Nan)), C(2, D(Nan)), C(0, 0) },
                new ulong[] { 0, 0, 0x3FF0000000000000, Nan, 0x4000000000000000, Nan, 0x4008000000000000, Nan }),
            ("payloads", new[] { C(D(Pay), 1), C(D(Neg), 1), C(1, D(Pay)), C(2, 0), C(D(Nan), D(Nan)) },
                new ulong[] { 0x4000000000000000, 0, 0x3FF0000000000000, Pay, Pay, 0x3FF0000000000000, Neg, 0x3FF0000000000000, Nan, Nan }),
            ("first_nan_c_order", new[] { C(D(Nan), 4), C(2, D(Nan)), C(1, 0), C(D(Nan), D(Nan)) },
                new ulong[] { 0x3FF0000000000000, 0, 0x4000000000000000, Nan, Nan, 0x4010000000000000, Nan, Nan }),
        };

        /// <summary>
        ///     Without <c>equal_nan</c> the values-only call is NumPy's hash path too (it used to partition the NaN-part values
        ///     to the tail and leave them in partition order), and np.unique_counts / np.unique_values (both
        ///     <c>equal_nan=False</c>) sort their distinct NaN-part values the same way (they appended them in input order):
        ///     <c>[1+nanj, nan+2j, 3]</c> in either input order gives <c>[3, 1+nanj, nan+2j]</c>.
        /// </summary>
        [TestMethod]
        public void WithoutEqualNan_NaNPartsComeOutInNumPysSortOrder()
        {
            foreach (var (name, line, want) in Lines)
            {
                var a = np.array(line);
                Bits(np.unique(a, equal_nan: false).values).Should().Equal(want, name);
                var counts = np.unique_counts(a);
                Bits(counts.values).Should().Equal(want, name + " unique_counts");
                counts.counts.ToArray<long>().Should().AllBeEquivalentTo(1L, name + " unique_counts");
                Bits(np.unique_values(a)).Should().Equal(want, name + " unique_values (NumSharp sorts)");
            }
        }

        /// <summary>
        ///     With <c>equal_nan</c> the collapsed class keeps its FIRST value in C order (NumPy's hash set), here
        ///     <c>nan+4j</c> though <c>2+nanj</c> sorts first; the counts route (a sort) keeps the first in sorted order.
        /// </summary>
        [TestMethod]
        public void WithEqualNan_TheHashKeepsTheFirstNaNPartInCOrder()
        {
            var a = np.array(new[] { C(D(Nan), 4), C(2, D(Nan)), C(1, 0), C(D(Nan), D(Nan)) });
            Bits(np.unique(a).values).Should().Equal(0x3FF0000000000000UL, 0UL, Nan, 0x4010000000000000UL);
            var r = np.unique(a, return_counts: true);
            Bits(r[0]).Should().Equal(0x3FF0000000000000UL, 0UL, 0x4000000000000000UL, Nan);
            r[1].ToArray<long>().Should().Equal(1L, 3L);
        }

        /// <summary>
        ///     Equal NaN-part values (<c>nan+0j</c> and <c>nan-0j</c>: the zeros are equal) keep their input order. NumPy's
        ///     order of equal values here is its hash table's iteration order, which follows the C++ library's bucket layout
        ///     (NumPy 2.4.2 on win-amd64 gives <c>[nan-0j, nan+0j]</c> for np.unique, while its sort-based np.unique_counts
        ///     gives <c>[nan+0j, nan-0j]</c>); NumSharp gives the input order on both routes.
        /// </summary>
        [TestMethod]
        public void EqualNaNPartValues_KeepTheirInputOrder()
        {
            var a = np.array(new[] { C(-0.0, 0.0), C(0.0, -0.0), C(D(Nan), 0.0), C(D(Nan), -0.0) });
            var want = new ulong[] { 0x8000000000000000, 0, Nan, 0, Nan, 0x8000000000000000 };
            Bits(np.unique(a, equal_nan: false).values).Should().Equal(want);
            Bits(np.unique_counts(a).values).Should().Equal(want);
            np.unique_counts(a).counts.ToArray<long>().Should().Equal(2L, 1L, 1L);
        }

        /// <summary>
        ///     Above 2^28 elements the hash path bails, and the values-only call takes the stable argsort (the managed
        ///     <c>uniqueValuesOnlyComplex</c>, and past <see cref="Array.MaxLength"/> the long-indexed
        ///     <c>uniqueFlatSortedLongComplex</c>): both, called by reflection on small lines, give the hash path's values
        ///     byte for byte, with and without <c>equal_nan</c> (the collapsed class's first value in C order included).
        /// </summary>
        [TestMethod]
        public void AboveTheHashBound_TheArgsortRoutesGiveTheHashAnswer()
        {
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static;
            var managed = typeof(UniqueEngine).GetMethods(flags).First(m => m.Name == "uniqueValuesOnlyComplex");
            var longIndexed = typeof(UniqueEngine).GetMethods(flags).First(m => m.Name == "uniqueFlatSortedLongComplex");

            var lines = Lines.Select(l => l.line).ToList();
            lines.Add(new[] { C(-0.0, 0.0), C(0.0, -0.0), C(D(Nan), 0.0), C(D(Nan), -0.0) });
            var rng = new System.Random(20261028);
            double[] pool = { 0.0, -0.0, 1.0, -2.5, double.PositiveInfinity, D(Nan), D(Neg), D(Pay) };
            foreach (int n in new[] { 9, 200, 3000 })
                lines.Add(Enumerable.Range(0, n).Select(_ => C(pool[rng.Next(pool.Length)], pool[rng.Next(pool.Length)])).ToArray());

            foreach (var line in lines)
            foreach (bool en in new[] { true, false })
            {
                var a = np.array(line);
                var want = Bits(np.unique(a, equal_nan: en).values);
                Bits((NDArray)managed.Invoke(null, new object[] { a, (long)a.size, en })).Should().Equal(want, $"n={a.size} en={en}");
                var r = (NDArray[])longIndexed.Invoke(null, new object[] { a, (long)a.size, en, false, false, false });
                Bits(r[0]).Should().Equal(want, $"long n={a.size} en={en}");
            }
        }
    }
}
