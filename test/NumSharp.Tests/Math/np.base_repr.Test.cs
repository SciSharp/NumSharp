using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Pins <see cref="np.binary_repr(long,int?)"/> and <see cref="np.base_repr(long,int,int)"/> against
    ///     NumPy 2.4.2 (numpy/_core/numeric.py). These are SCALAR integer → string formatters with no array
    ///     operand, so they have no differential-fuzz corpus entry (they are classified sibling-owned in the
    ///     oracle surface guard) — this suite IS their gate. Coverage: the two's-complement width path incl.
    ///     the gh-8679 power-of-two boundary, the <c>width or 1</c> zero case, arbitrary-precision magnitudes
    ///     (uint64 max, 2^63, beyond int64 via the BigInteger overload), base 2–36 with padding/sign
    ///     ordering, and the full ValueError taxonomy. Every expected string was produced by running the
    ///     matching NumPy 2.4.2 call.
    /// </summary>
    [TestClass]
    public class BaseReprTest
    {
        // ------------------------------------------------------------------ binary_repr

        /// <summary>Zero renders as a single '0'; <c>width</c> pads it, and NumPy's <c>'0' * (width or 1)</c>
        /// makes width 0 behave as width 1 while a negative width collapses to the empty string.</summary>
        [TestMethod]
        public void BinaryRepr_Zero()
        {
            Assert.AreEqual("0", np.binary_repr(0));
            Assert.AreEqual("0000", np.binary_repr(0, 4));
            Assert.AreEqual("0", np.binary_repr(0, 0));   // width 0 -> 1 ('0' * (0 or 1))
            Assert.AreEqual("", np.binary_repr(0, -1));   // '0' * -1 == ''
        }

        /// <summary>A positive number is its plain binary, left-zero-padded to <c>width</c> when that is larger.</summary>
        [TestMethod]
        public void BinaryRepr_Positive()
        {
            Assert.AreEqual("11", np.binary_repr(3));
            Assert.AreEqual("0011", np.binary_repr(3, 4));
            Assert.AreEqual("11", np.binary_repr(3, 2));      // width == binwidth
            Assert.AreEqual("11111111", np.binary_repr(255));
        }

        /// <summary>A negative number WITHOUT a width gets a leading minus sign in front of the magnitude's binary.</summary>
        [TestMethod]
        public void BinaryRepr_Negative_NoWidth()
        {
            Assert.AreEqual("-11", np.binary_repr(-3));
            Assert.AreEqual("-1", np.binary_repr(-1));
            Assert.AreEqual("-10000000", np.binary_repr(-128));
        }

        /// <summary>A negative number WITH a width is rendered as its two's complement, sign-extended with 1s.</summary>
        [TestMethod]
        public void BinaryRepr_Negative_TwosComplement()
        {
            Assert.AreEqual("101", np.binary_repr(-3, 3));
            Assert.AreEqual("11101", np.binary_repr(-3, 5));
            Assert.AreEqual("11111111", np.binary_repr(-1, 8));
            Assert.AreEqual("10000000", np.binary_repr(-128, 8));
        }

        /// <summary>The gh-8679 fix: a value exactly on a power-of-two boundary needs one fewer bit, so
        /// <c>binary_repr(-8, 4)</c> fits in 4 bits ("1000") instead of demanding a 5th.</summary>
        [TestMethod]
        public void BinaryRepr_PowerOfTwoBoundary_gh8679()
        {
            Assert.AreEqual("1000", np.binary_repr(-8, 4));
            Assert.AreEqual("11000", np.binary_repr(-8, 5));
            Assert.AreEqual("10000", np.binary_repr(-16, 5));
        }

        /// <summary>An insufficient <c>width</c> (smaller than the required bit count, incl. a negative width)
        /// raises NumPy's verbatim ValueError — for both the positive and two's-complement paths.</summary>
        [TestMethod]
        public void BinaryRepr_InsufficientWidth_Throws()
        {
            var e1 = Assert.ThrowsException<ValueError>(() => np.binary_repr(3, 1));
            Assert.AreEqual("Insufficient bit width=1 provided for binwidth=2", e1.Message);
            var e2 = Assert.ThrowsException<ValueError>(() => np.binary_repr(-3, 2));
            Assert.AreEqual("Insufficient bit width=2 provided for binwidth=3", e2.Message);
            Assert.ThrowsException<ValueError>(() => np.binary_repr(-3, -1));
        }

        /// <summary>Arbitrary-precision magnitudes round-trip exactly through the BigInteger overload:
        /// uint64 max (64 ones), 2^63, and values beyond int64.</summary>
        [TestMethod]
        public void BinaryRepr_LargeMagnitudes()
        {
            Assert.AreEqual(new string('1', 64), np.binary_repr(ulong.MaxValue));
            Assert.AreEqual("1" + new string('0', 63), np.binary_repr((BigInteger)1 << 63));
            // 2^100 has 101 binary digits.
            Assert.AreEqual("1" + new string('0', 100), np.binary_repr(BigInteger.Pow(2, 100)));
            // Negative past int64 range, no width: minus sign + magnitude.
            Assert.AreEqual("-1" + new string('0', 64), np.binary_repr(-BigInteger.Pow(2, 64)));
            // Two's complement of a value past int64 range, sign-extended to the requested width.
            Assert.AreEqual("11" + new string('0', 64), np.binary_repr(-BigInteger.Pow(2, 64), 66));
        }

        // ------------------------------------------------------------------ base_repr

        /// <summary>Base 2 renders like <see cref="np.binary_repr(long,int?)"/> (without its two's-complement
        /// behaviour), and zero is a single '0'.</summary>
        [TestMethod]
        public void BaseRepr_Basic()
        {
            Assert.AreEqual("101", np.base_repr(5));
            Assert.AreEqual("0", np.base_repr(0));
            Assert.AreEqual("1111", np.base_repr(15));
        }

        /// <summary>Non-binary bases 2–36 use uppercase A–Z for digits above 9.</summary>
        [TestMethod]
        public void BaseRepr_Bases()
        {
            Assert.AreEqual("11", np.base_repr(6, 5));
            Assert.AreEqual("A", np.base_repr(10, 16));
            Assert.AreEqual("20", np.base_repr(32, 16));
            Assert.AreEqual("Z", np.base_repr(35, 36));
            Assert.AreEqual("777", np.base_repr(511, 8));
        }

        /// <summary><c>padding</c> prepends that many zeros AFTER the sign (so a negative value's minus stays
        /// leftmost), and a non-positive padding adds nothing.</summary>
        [TestMethod]
        public void BaseRepr_Padding()
        {
            Assert.AreEqual("00012", np.base_repr(7, 5, 3));
            Assert.AreEqual("000", np.base_repr(0, 2, 3));      // zero + padding
            Assert.AreEqual("-00012", np.base_repr(-7, 5, 3));  // padding after the sign
            Assert.AreEqual("101", np.base_repr(5, 2, -1));     // negative padding -> none
            Assert.AreEqual("101", np.base_repr(5, 2, 0));
        }

        /// <summary>A negative number gets a leading minus sign (NOT two's complement — that is binary_repr's
        /// job); the boundary value long.MinValue is exact via the BigInteger magnitude.</summary>
        [TestMethod]
        public void BaseRepr_Negative()
        {
            Assert.AreEqual("-101", np.base_repr(-5));
            Assert.AreEqual("-8000000000000000", np.base_repr(long.MinValue, 16));
            Assert.AreEqual(new string('1', 64), np.base_repr(ulong.MaxValue));   // uint64 max in base 2
        }

        /// <summary>A base outside 2–36 raises NumPy's verbatim ValueError (greater-than / less-than texts).</summary>
        [TestMethod]
        public void BaseRepr_BadBase_Throws()
        {
            var hi = Assert.ThrowsException<ValueError>(() => np.base_repr(5, 37));
            Assert.AreEqual("Bases greater than 36 not handled in base_repr.", hi.Message);
            var lo = Assert.ThrowsException<ValueError>(() => np.base_repr(5, 1));
            Assert.AreEqual("Bases less than 2 not handled in base_repr.", lo.Message);
            Assert.ThrowsException<ValueError>(() => np.base_repr(5, 0));
            Assert.ThrowsException<ValueError>(() => np.base_repr(5, -3));
        }

        /// <summary>The BigInteger overloads accept magnitudes beyond int64 (e.g. a ulong or a 90-bit value)
        /// that the long overload cannot represent.</summary>
        [TestMethod]
        public void BaseRepr_LargeMagnitudes()
        {
            Assert.AreEqual("FFFFFFFFFFFFFFFF", np.base_repr(ulong.MaxValue, 16));
            Assert.AreEqual("1" + new string('0', 30), np.base_repr(BigInteger.Pow(2, 30), 2)); // 2^30 base 2
            Assert.AreEqual("-" + "1" + new string('0', 89), np.base_repr(-BigInteger.Pow(2, 89), 2));
        }
    }
}
