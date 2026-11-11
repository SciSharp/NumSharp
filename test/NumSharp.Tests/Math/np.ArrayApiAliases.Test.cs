using System;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     NumPy 2.0's Array-API alias names: <c>np.acos</c>, <c>np.asin</c>, <c>np.atan</c>, <c>np.atan2</c>, <c>np.pow</c>,
    ///     <c>np.bitwise_invert</c>, <c>np.bitwise_left_shift</c>, <c>np.bitwise_right_shift</c>. In NumPy each IS its original
    ///     ufunc object (<c>np.acos is np.arccos</c> is True, probed 2.4.2), so the property under test is identity with the
    ///     original - bit for bit over special values, with out= / where= / dtype= passed through - plus NumPy's own values
    ///     for a few points and the one-signature shape of each alias.
    /// </summary>
    [TestClass]
    public class ArrayApiAliasesTests
    {
        /// <summary>Inputs that exercise the float branches: zeros of both signs, the domain edges, NaN and infinities.</summary>
        private static NDArray Specials => np.array(new[]
        {
            0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 1e-300,
        });

        /// <summary>The IEEE bit patterns of a float64 array, so NaN payloads and signed zeros compare exactly.</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>The bits of every element in C order.</returns>
        private static long[] Bits(NDArray a) => a.ToArray<double>().Select(BitConverter.DoubleToInt64Bits).ToArray();

        /// <summary>acos / asin / atan answer bit for bit as arccos / arcsin / arctan, and NumPy's values at a point.</summary>
        [TestMethod]
        public void InverseTrig_Aliases_AreTheOriginals()
        {
            Bits(np.acos(Specials)).Should().Equal(Bits(np.arccos(Specials)));
            Bits(np.asin(Specials)).Should().Equal(Bits(np.arcsin(Specials)));
            Bits(np.atan(Specials)).Should().Equal(Bits(np.arctan(Specials)));
            np.acos(np.array(new[] { 0.5 })).GetDouble(0).Should().Be(1.0471975511965979);
            np.asin(np.array(new[] { 1.0 })).GetDouble(0).Should().Be(1.5707963267948966);
            np.atan(np.array(new[] { 1.0 })).GetDouble(0).Should().Be(0.7853981633974483);
            np.acos(np.array(new[] { 0.5 }), dtype: np.float32).dtype.Should().Be(np.float32);
        }

        /// <summary>atan2 answers as arctan2 over every sign / zero / infinity pairing; NumPy's atan2(1, -1) is 3π/4.</summary>
        [TestMethod]
        public void Atan2_Alias_IsArctan2()
        {
            var y = Specials.reshape(-1, 1);
            var x = Specials.reshape(1, -1);
            Bits(np.atan2(y, x)).Should().Equal(Bits(np.arctan2(y, x)));
            np.atan2(np.array(new[] { 1.0 }), np.array(new[] { -1.0 })).GetDouble(0).Should().Be(2.356194490192345);
        }

        /// <summary>
        ///     pow is power: a C# literal exponent is NumPy's weak Python int (int32 stays int32: <c>[4, 9]</c>), and out= /
        ///     where= behave as the ufunc's (masked-off slots keep out's value, out itself is returned).
        /// </summary>
        [TestMethod]
        public void Pow_Alias_IsPower_WeakLiteralAndOutWhere()
        {
            var r = np.pow(np.array(new[] { 2, 3 }), 2);
            r.dtype.Should().Be(np.int32);
            r.ToArray<int>().Should().Equal(4, 9);
            var o = np.full(new Shape(3), -1.0);
            var w = np.pow(np.array(new[] { 1.0, 2, 3 }), 2, @out: o, where: np.array(new[] { true, false, true }));
            ReferenceEquals(w, o).Should().BeTrue();
            o.ToArray<double>().Should().Equal(1.0, -1.0, 9.0);
            Bits(np.pow(Specials, Specials)).Should().Equal(Bits(np.power(Specials, Specials)));
        }

        /// <summary>The bitwise aliases answer as invert / left_shift / right_shift, NumPy's values; a float operand is
        ///     refused as the originals refuse it.</summary>
        [TestMethod]
        public void BitwiseAliases_AreTheOriginals()
        {
            np.bitwise_invert(np.array(new sbyte[] { 0, 5 })).ToArray<sbyte>().Should().Equal((sbyte)-1, (sbyte)-6);
            np.bitwise_left_shift(np.array(new[] { 1, 2 }), 2).ToArray<int>().Should().Equal(4, 8);
            np.bitwise_right_shift(np.array(new[] { 8, -8 }), 1).ToArray<int>().Should().Equal(4, -4);
            Action floatShift = () => np.bitwise_left_shift(np.array(new[] { 1.0 }), 1);
            floatShift.Should().Throw<TypeError>();
        }

        /// <summary>Each alias has ONE public overload, the ufunc form (no copy of power's / the shifts' extra
        ///     overloads), with its original's parameter names.</summary>
        [TestMethod]
        public void Aliases_HaveOneSignatureEach()
        {
            var expected = new (string Alias, string[] Names)[]
            {
                ("acos", new[] { "x", "out", "where", "dtype" }), ("asin", new[] { "x", "out", "where", "dtype" }),
                ("atan", new[] { "x", "out", "where", "dtype" }), ("atan2", new[] { "y", "x", "out", "where", "dtype" }),
                ("pow", new[] { "x1", "x2", "out", "where", "dtype" }), ("bitwise_invert", new[] { "x", "out", "where", "dtype" }),
                ("bitwise_left_shift", new[] { "x1", "x2", "out", "where", "dtype" }),
                ("bitwise_right_shift", new[] { "x1", "x2", "out", "where", "dtype" }),
            };
            foreach (var (alias, names) in expected)
            {
                var forms = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == alias).ToArray();
                forms.Should().HaveCount(1, alias);
                forms[0].GetParameters().Select(p => p.Name).Should().Equal(names, alias);
            }
        }
    }
}
