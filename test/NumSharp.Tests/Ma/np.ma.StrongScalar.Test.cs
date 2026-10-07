using System;
using System.Numerics;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma's strong operands: <c>getdata</c> turns a Python scalar AND a 0-d array into a strong array, so the masked
    ///     comparisons and arithmetic promote with it fully — where NumSharp's plain <c>np.*</c> reads a 0-d operand against
    ///     an array as a weak Python scalar. Every expected value was probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class MaStrongScalarTests
    {
        private static bool[] Data(NDMaskedArray r) => np.ma.getdata(r).ToArray<bool>();

        private static NDMaskedArray F32() => np.ma.array(np.array(new[] { 0.1f, 0.5f }), mask: np.array(new[] { false, false }));

        /// <summary>The masked comparisons compare a Python float with a float32 array at float64 (strong).</summary>
        [TestMethod]
        public void Comparisons_AreStrong_ForEveryOperandForm()
        {
            // NumPy: np.ma.greater(mx, 0.1) -> [True, True]; mx > 0.1 -> [True, True]; 0.1 >= mx -> [False, False];
            // np.ma.equal(mx, np.array(0.1)) -> [False, False]; mx == 0.1 -> [False, False].
            var mx = F32();
            Data(np.ma.greater(mx, 0.1)).Should().Equal(true, true);
            Data(mx > 0.1).Should().Equal(true, true);
            Data(0.1 >= mx).Should().Equal(false, false);
            Data(np.ma.equal(mx, np.array(0.1))).Should().Equal(false, false);
            Data(np.ma.equal(mx, 0.1)).Should().Equal(false, false);
        }

        /// <summary>A 0-d array operand promotes strongly in the masked arithmetic, where/power and maximum too.</summary>
        [TestMethod]
        public void ZeroDimArrayOperand_PromotesStrongly()
        {
            // NumPy: np.ma.add(mx, np.array(0.1)).dtype, mx + np.array(0.1), np.ma.where(c, mx, np.array(0.1)),
            // np.ma.power(mx, np.array(2.0)), np.ma.maximum(mx, np.array(0.3)): all float64.
            var mx = F32();
            np.ma.add(mx, np.array(0.1)).dtype.Should().Be(np.float64);
            (mx + np.array(0.1)).dtype.Should().Be(np.float64);
            np.ma.where(np.array(new[] { true, false }), mx, np.array(0.1)).dtype.Should().Be(np.float64);
            np.ma.power(mx, np.array(2.0)).dtype.Should().Be(np.float64);
            np.ma.maximum(mx, np.array(0.3)).dtype.Should().Be(np.float64);
        }

        /// <summary>
        ///     A float-only ufunc takes the first float loop each input casts to safely: int16 with a 0-d uint16 is float32,
        ///     where their promotion (int32) would pick float64.
        /// </summary>
        [TestMethod]
        public void FloatOnlyUfunc_ResolvesItsLoopFromTheStrongPair()
        {
            var a = np.ma.array(np.array(new short[] { 1, 2 }));
            var u = np.array((ushort)3);
            np.ma.arctan2(a, u).dtype.Should().Be(np.float32);
            np.ma.hypot(a, u).dtype.Should().Be(np.float32);
            var b = np.ma.array(np.array(new byte[] { 1, 2 }));
            np.ma.arctan2(b, np.array((sbyte)3)).dtype.Should().Be(np.float16);
        }

        /// <summary>The bare C# scalars that are NumPy scalars keep their dtype: a Half is np.float16, a char np.uint16.</summary>
        [TestMethod]
        public void NumPyScalarSpellings_KeepTheirDtype()
        {
            // NumPy: np.ma.add(f32, np.float16(0.5)) -> float32; np.ma.add(int8, np.float16(0.5)) -> float16;
            // np.ma.add(int8, np.uint16(5)) -> int32; np.ma.greater(f32, np.float16(0.1)) -> [True, True].
            var mx = F32();
            var mi8 = np.ma.array(np.array(new sbyte[] { 1, 2 }));
            np.ma.add(mx, (Half)0.5).dtype.Should().Be(np.float32);
            np.ma.add(mi8, (Half)0.5).dtype.Should().Be(np.float16);
            np.ma.add(mi8, (char)5).dtype.Should().Be(np.int32);
            Data(np.ma.greater(mx, (Half)0.1)).Should().Equal(true, true);
        }

        /// <summary>A Python int past int64 is np.array(v)'s uint64, and the comparison with int64 stays exact.</summary>
        [TestMethod]
        public void PythonIntPastInt64_IsUInt64_AndComparesExactly()
        {
            // NumPy: np.ma.equal(np.ma.array(np.array([2**63 - 1], np.int64)), 2**63) -> [False];
            // np.ma.add(np.ma.array(np.array([1, 2], np.uint8)), 2**64 - 1) -> uint64 [0, 1].
            Data(np.ma.equal(np.ma.array(np.array(new[] { long.MaxValue })), 9223372036854775808UL)).Should().Equal(false);
            var r = np.ma.add(np.ma.array(np.array(new byte[] { 1, 2 })), ulong.MaxValue);
            r.dtype.Should().Be(np.uint64);
            np.ma.getdata(r).ToArray<ulong>().Should().Equal(0UL, 1UL);
        }

        /// <summary>allequal / allclose compare strongly (getdata / masked_array of the scalar).</summary>
        [TestMethod]
        public void AllequalAndAllclose_AreStrong()
        {
            // NumPy: np.ma.allequal(np.array([0.1], np.float32), 0.1) -> False;
            // np.ma.allclose(np.array([0.1], np.float32), 0.1, rtol=0, atol=0) -> False.
            np.ma.allequal(np.array(new[] { 0.1f }), 0.1).Should().BeFalse();
            np.ma.allequal(np.array(new[] { 0.1f }), np.array(0.1)).Should().BeFalse();
            np.ma.allclose(np.array(new[] { 0.1f }), 0.1, rtol: 0, atol: 0).Should().BeFalse();
        }

        /// <summary>masked_inside / masked_outside / masked_values compare like NumPy's plain comparisons: weak.</summary>
        [TestMethod]
        public void MaskedInsideOutsideValues_AreWeak()
        {
            // NumPy: np.ma.masked_inside(f32 [0.1, 0.5], 0.0, 0.1) masks [True, False];
            // np.ma.masked_values(f32 [0.1, 0.5], 0.1, rtol=0, atol=0) masks [True, False];
            // np.ma.masked_inside(np.array([2.0**60], np.float32), 0, 2**60 + 2**36 + 1) masks [True] (the bound packs to 2^60).
            var x = np.array(new[] { 0.1f, 0.5f });
            np.ma.getmaskarray(np.ma.masked_inside(x, 0.0, 0.1)).ToArray<bool>().Should().Equal(true, false);
            np.ma.getmaskarray(np.ma.masked_values(x, 0.1, atol: 0, rtol: 0)).ToArray<bool>().Should().Equal(true, false);
            var big = np.array(new[] { (float)(1L << 60) });
            np.ma.getmaskarray(np.ma.masked_inside(big, 0L, (1L << 60) + (1L << 36) + 1)).ToArray<bool>().Should().Equal(true);
        }

        /// <summary>
        ///     The domain of a masked divide sees NumPy's getdata operands, unpromoted: abs(int16 -32768) wraps negative, so
        ///     NumPy leaves that slot unmasked.
        /// </summary>
        [TestMethod]
        public void DomainedDivide_DomainSeesTheUnpromotedData()
        {
            // NumPy: np.ma.floor_divide(np.ma.array(np.array([-32768, 5, 0], np.int16)), 0) -> mask [False, True, True];
            // np.ma.divide(3, np.ma.array(np.array([-128, 1, 0], np.int8))) -> mask [True, False, True].
            var a = np.ma.array(np.array(new short[] { -32768, 5, 0 }));
            np.ma.getmaskarray(np.ma.floor_divide(a, 0)).ToArray<bool>().Should().Equal(false, true, true);
            var b = np.ma.array(np.array(new sbyte[] { -128, 1, 0 }));
            np.ma.getmaskarray(np.ma.divide(3, b)).ToArray<bool>().Should().Equal(true, false, true);
        }

        /// <summary>
        ///     ma.left_shift / right_shift: <c>filled(a)</c> makes a bare value a strong int64 array, a bare shift count stays
        ///     a weak Python int, a 0-d array count is strong.
        /// </summary>
        [TestMethod]
        public void Shift_StrongValue_WeakBareCount()
        {
            // NumPy: ma.left_shift(1, int16 array) -> int64; ma.left_shift(int16 array, 1) -> int16;
            // ma.left_shift(np.array(1), int16 array) -> int64; ma.left_shift(int16 array, np.array(1)) -> int64.
            var i16 = np.array(new short[] { 1, 2 });
            np.ma.left_shift(1, i16).dtype.Should().Be(np.int64);
            np.ma.left_shift(i16, 1).dtype.Should().Be(np.int16);
            np.ma.left_shift(np.array(1L), i16).dtype.Should().Be(np.int64);
            np.ma.left_shift(i16, np.array(1L)).dtype.Should().Be(np.int64);
        }
    }
}
