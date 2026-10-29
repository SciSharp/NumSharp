using System;
using System.Numerics;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.full and np.full_like read their fill value as NumPy's <c>fill_value = asarray(fill_value)</c> (without a dtype)
    ///     and <c>copyto(a, fill_value, casting='unsafe')</c> read it: a 0-d array keeps its own dtype, a Python int (a C#
    ///     <see cref="BigInteger"/>) is discovered by its value, an empty result converts nothing but a Python scalar, and a
    ///     <c>StringDType</c> result stores a Python int as its text. Every expected value and text was probed against NumPy
    ///     2.4.2 (the np-layer dtype plan's batch 5 replay found the divergences).
    /// </summary>
    [TestClass]
    public class np_full_FillValues_Test
    {
        /// <summary>A 0-d array fill gives its own dtype (it raised ArgumentNullException: an NDArray has no lane).</summary>
        [TestMethod]
        public void ZeroDArrayFill_TakesItsOwnDtype()
        {
            using var five = np.array(5L);
            using var r = np.full(new Shape(3), five);
            r.ToString(true).Should().Be("array([5, 5, 5])");
            using var ab = np.array("ab");
            using var t = np.full(new Shape(3), ab);
            t.ToString(true).Should().Be("array(['ab', 'ab', 'ab'], dtype='<U2')");
        }

        /// <summary>
        ///     A Python int is discovered by its value: int64 when it fits, uint64 past int64, and past 64 bits NumPy's object
        ///     array, which NumSharp refuses (it raised ArgumentNullException for every BigInteger).
        /// </summary>
        [TestMethod]
        public void PythonIntFill_IsDiscoveredByItsValue()
        {
            using var u = np.full(new Shape(3), BigInteger.Pow(2, 63));
            u.ToString(true).Should().Be(
                "array([9223372036854775808, 9223372036854775808, 9223372036854775808],\n      dtype=uint64)");
            using var i = np.full(new Shape(2), new BigInteger(-5));
            i.ToString(true).Should().Be("array([-5, -5])");
            Action big = () => np.full(new Shape(3), BigInteger.Pow(2, 64));
            big.Should().Throw<NotSupportedException>().WithMessage("*object array*");
        }

        /// <summary>
        ///     An empty result converts no text (NumPy casts per element written), but a Python int is still packed into the
        ///     dtype first, so an out-of-range one raises.
        /// </summary>
        [TestMethod]
        public void EmptyResult_ConvertsNoText_ButPacksAPythonInt()
        {
            using var text = np.full(new Shape(0), "ab", np.int64);
            text.ToString(true).Should().Be("array([], dtype=int64)");
            using var bytes = np.full(new Shape(0), new byte[] { 97, 98 }, np.float64);
            bytes.ToString(true).Should().Be("array([], dtype=float64)");
            using var like = np.full_like(np.zeros(new Shape(0, 3), np.complex128), "ab");
            like.size.Should().Be(0);
            Action packed = () => np.full(new Shape(0), 300, np.int8);
            packed.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
        }

        /// <summary>
        ///     A <c>StringDType</c> result stores a Python int as its text: within 64 bits by NumPy's cast (under
        ///     <c>coerce=False</c> too), past them as NumPy's object array (refused under <c>coerce=False</c>). It raised
        ///     "Unable to resolve asanyarray for type BigInteger".
        /// </summary>
        [TestMethod]
        public void StringDTypeResult_StoresAPythonIntAsItsText()
        {
            using var r = np.full(new Shape(2), BigInteger.Pow(2, 70), np.dtypes.StringDType());
            r.ToString(true).Should().Be("array(['1180591620717411303424', '1180591620717411303424'],\n      dtype=StringDType())");
            using var c = np.full(new Shape(2), BigInteger.Pow(2, 63), np.dtypes.StringDType(coerce: false));
            c.ToString(true).Should().Be("array(['9223372036854775808', '9223372036854775808'],\n      dtype=StringDType(coerce=False))");
            using var proto = np.zeros(new Shape(2), np.dtypes.StringDType());
            using var like = np.full_like(proto, BigInteger.Pow(2, 70));
            like.ToString(true).Should().Be("array(['1180591620717411303424', '1180591620717411303424'],\n      dtype=StringDType())");
            Action refused = () => np.full(new Shape(2), BigInteger.Pow(2, 70), np.dtypes.StringDType(coerce: false));
            refused.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
        }

        /// <summary>
        ///     A fill array that does not broadcast into a fixed-width text result raises NumPy's copyto text (it raised the
        ///     copy iterator's "remapped shapes" text), and one with a leading unit axis past the result's rank broadcasts.
        /// </summary>
        [TestMethod]
        public void FixedWidthTextResult_BroadcastsAsCopytoDoes()
        {
            using var row = np.array(new[] { 1.5, 2.5, 3.5 });
            using var u5 = np.zeros(new Shape(3, 4), "U5");
            Action like = () => np.full_like(u5, row);
            like.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,) into shape (3,4)");
            Action full = () => np.full(new Shape(3, 4), row, "S5");
            full.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,) into shape (3,4)");
            using var leading = row.reshape(1, 3);
            using var r = np.full(new Shape(3), leading, "U5");
            r.ToString(true).Should().Be("array(['1.5', '2.5', '3.5'], dtype='<U5')");
        }
    }
}
