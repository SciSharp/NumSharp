using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.correlate — cross-correlation of two 1-D sequences. All expected values are from
    /// NumPy 2.4.2 (probed directly). Covers the docstring examples, the swap+reverse of the
    /// non-commutative engine, complex conjugation, every dtype, and the error taxonomy.
    /// </summary>
    [TestClass]
    public class NdArrayCorrelateTest
    {
        // ---------------------------- docstring examples ----------------------------

        [TestMethod]
        public void Correlate_Docstring_Valid_IsDefault()
        {
            // np.correlate([1,2,3],[0,1,0.5]) -> [3.5]   (default mode is 'valid')
            var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 0, 1, 0.5 }));
            r.Data<double>().Should().Equal(new double[] { 3.5 });
        }

        [TestMethod]
        public void Correlate_Docstring_Same()
        {
            var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 0, 1, 0.5 }), "same");
            r.Data<double>().Should().Equal(new double[] { 2.0, 3.5, 3.0 });
        }

        [TestMethod]
        public void Correlate_Docstring_Full()
        {
            var r = np.correlate(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 0, 1, 0.5 }), "full");
            r.Data<double>().Should().Equal(new double[] { 0.5, 2.0, 3.5, 3.0, 0.0 });
        }

        // -------------------- non-commutative swap + output reversal --------------------
        // When len(a) < len(v) the engine correlates(v, a) then reverses the output, so
        // correlate(a, v) is the time-reversal of correlate(v, a).

        [TestMethod]
        public void Correlate_ShorterA_SwapsAndReverses_Valid()
        {
            var r = np.correlate(np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 2, 3, 4 }), "valid");
            r.Data<long>().Should().Equal(new long[] { 11, 8, 5 });
        }

        [TestMethod]
        public void Correlate_ShorterA_SwapsAndReverses_Full()
        {
            var r = np.correlate(np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 2, 3, 4 }), "full");
            r.Data<long>().Should().Equal(new long[] { 4, 11, 8, 5, 2 });
        }

        [TestMethod]
        public void Correlate_IsReverseOf_Swapped()
        {
            // correlate([1,2],[1,2,3,4]) == reverse of correlate([1,2,3,4],[1,2])
            var ab = np.correlate(np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 2, 3, 4 }), "valid");
            var ba = np.correlate(np.array(new long[] { 1, 2, 3, 4 }), np.array(new long[] { 1, 2 }), "valid");
            ab.Data<long>().Should().Equal(new long[] { 11, 8, 5 });
            ba.Data<long>().Should().Equal(new long[] { 5, 8, 11 });
        }

        // ----------------------------- complex conjugation -----------------------------

        [TestMethod]
        public void Correlate_Complex_ConjugatesV()
        {
            // np.correlate([1+1j,2,3-1j],[0,1,0.5j],'full') -> [0.5-0.5j, 1+0j, 1.5-1.5j, 3-1j, 0+0j]
            var a = np.array(new Complex[] { new(1, 1), 2, new(3, -1) });
            var v = np.array(new Complex[] { 0, 1, new(0, 0.5) });
            var r = np.correlate(a, v, "full").ToArray<Complex>();
            r.Should().Equal(new Complex[] { new(0.5, -0.5), new(1, 0), new(1.5, -1.5), new(3, -1), new(0, 0) });
        }

        [TestMethod]
        public void Correlate_Complex_SwappedIsTimeReversedConjugate()
        {
            // np.correlate([0,1,0.5j],[1+1j,2,3-1j],'full') -> [0+0j, 3+1j, 1.5+1.5j, 1+0j, 0.5+0.5j]
            var a = np.array(new Complex[] { 0, 1, new(0, 0.5) });
            var v = np.array(new Complex[] { new(1, 1), 2, new(3, -1) });
            var r = np.correlate(a, v, "full").ToArray<Complex>();
            r.Should().Equal(new Complex[] { new(0, 0), new(3, 1), new(1.5, 1.5), new(1, 0), new(0.5, 0.5) });
        }

        [TestMethod]
        public void Correlate_Complex_Valid()
        {
            var a = np.array(new Complex[] { new(1, 2), new(3, -1), new(0, 2) });
            var v = np.array(new Complex[] { new(0, 1), 2 });
            var r = np.correlate(a, v, "valid").ToArray<Complex>();
            r.Should().Equal(new Complex[] { new(8, -3), new(-1, 1) });
        }

        // -------------------------------- dtype coverage --------------------------------

        [TestMethod]
        public void Correlate_Int32_ValidAndFull()
        {
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var v = np.array(new int[] { 1, 0, -1 });
            np.correlate(a, v, "valid").Data<int>().Should().Equal(new[] { -2, -2, -2 });
            np.correlate(a, v, "full").Data<int>().Should().Equal(new[] { -1, -2, -2, -2, -2, 4, 5 });
            np.correlate(a, v).GetTypeCode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void Correlate_UInt8_Wraps()
        {
            var a = np.array(new byte[] { 10, 20, 30 });
            var v = np.array(new byte[] { 2, 1 });
            np.correlate(a, v, "valid").Data<byte>().Should().Equal(new byte[] { 40, 70 });
        }

        [TestMethod]
        public void Correlate_Float32_Same()
        {
            var a = np.array(new float[] { 1, 2, 3, 4 });
            var v = np.array(new float[] { 0.5f, 1f, 0.5f });
            np.correlate(a, v, "same").Data<float>().Should().Equal(new float[] { 2f, 4f, 6f, 5.5f });
        }

        [TestMethod]
        public void Correlate_Float16_Valid()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 });
            var v = np.array(new Half[] { (Half)1, (Half)1 });
            np.correlate(a, v, "valid").Data<Half>().Should().Equal(new Half[] { (Half)3, (Half)5, (Half)7 });
        }

        [TestMethod]
        public void Correlate_Bool_OrOfAnds()
        {
            var a = np.array(new bool[] { true, false, true });
            var v = np.array(new bool[] { true, true });
            np.correlate(a, v, "full").Data<bool>().Should().Equal(new bool[] { true, true, true, true });
        }

        [TestMethod]
        public void Correlate_Char_RoutesThroughUShortKernel()
        {
            // 'A'=65 etc.  valid corr with [1,1] -> [A+B, B+C, C+D] = [131,133,135]
            var a = np.array(new char[] { 'A', 'B', 'C', 'D' });
            var v = np.array(new char[] { (char)1, (char)1 });
            var r = np.correlate(a, v, "valid");
            r.GetTypeCode.Should().Be(NPTypeCode.Char);
            r.ToArray<char>().Select(c => (int)c).Should().Equal(new[] { 131, 133, 135 });
        }

        [TestMethod]
        public void Correlate_DtypePromotion()
        {
            // NEP50: int32 + float32 promotes to float64 (int32 exceeds float32's exact range).
            np.correlate(np.array(new int[] { 1 }), np.array(new float[] { 1 })).GetTypeCode.Should().Be(NPTypeCode.Double);
            np.correlate(np.array(new byte[] { 1 }), np.array(new sbyte[] { 1 })).GetTypeCode.Should().Be(NPTypeCode.Int16);
            np.correlate(np.array(new float[] { 1 }), np.array(new Complex[] { 1 })).GetTypeCode.Should().Be(NPTypeCode.Complex);
        }

        // ------------------------------ non-contiguous inputs ------------------------------

        [TestMethod]
        public void Correlate_StridedInput_MatchesContiguous()
        {
            // a[::2] strided view over arange(20) correlated with [1,2,0.5], full.
            var baseA = np.arange(20).astype(NPTypeCode.Double);
            var v = np.array(new double[] { 1, 2, 0.5 });
            var strided = np.correlate(baseA["::2"], v, "full");
            var copied = np.correlate(baseA["::2"].copy(), v, "full");
            strided.Data<double>().Should().Equal(copied.Data<double>().ToArray());
            strided.GetDouble(2).Should().Be(6.0); // matches numpy
        }

        [TestMethod]
        public void Correlate_ReversedView_MatchesCopy()
        {
            var baseA = np.arange(20).astype(NPTypeCode.Double);
            var v = np.array(new double[] { 1, 2, 0.5 });
            var rev = baseA["10:16"]["::-1"];
            np.correlate(rev, v, "same").Data<double>().Should().Equal(np.correlate(rev.copy(), v, "same").Data<double>().ToArray());
        }

        // ---------------------------------- error taxonomy ----------------------------------

        [TestMethod]
        public void Correlate_EmptyA_Throws()
        {
            Action act = () => np.correlate(np.array(new double[0]), np.array(new double[] { 1 }));
            act.Should().Throw<ArgumentException>().WithMessage("first array argument cannot be empty*");
        }

        [TestMethod]
        public void Correlate_EmptyV_Throws()
        {
            Action act = () => np.correlate(np.array(new double[] { 1 }), np.array(new double[0]));
            act.Should().Throw<ArgumentException>().WithMessage("second array argument cannot be empty*");
        }

        [TestMethod]
        public void Correlate_BadMode_Throws()
        {
            Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), "bogus");
            act.Should().Throw<ArgumentException>().WithMessage("mode must be one of 'valid', 'same', or 'full' (got 'bogus')*");
        }

        [TestMethod]
        public void Correlate_ModeIsCaseSensitive()
        {
            // NumPy's parser is case-sensitive: 'VALID' is a near-miss, not an alias.
            Action act = () => np.correlate(np.array(new double[] { 1, 2 }), np.array(new double[] { 1 }), "VALID");
            act.Should().Throw<ArgumentException>().WithMessage("Use one of 'valid', 'same', or 'full' for convolve/correlate mode*");
        }

        [TestMethod]
        public void Correlate_TwoDimensional_Throws()
        {
            Action act = () => np.correlate(np.arange(6).reshape(2, 3).astype(NPTypeCode.Double), np.array(new double[] { 1 }));
            act.Should().Throw<IncorrectShapeException>();
        }
    }
}
