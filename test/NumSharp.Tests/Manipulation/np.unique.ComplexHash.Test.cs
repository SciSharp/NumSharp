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
    }
}
