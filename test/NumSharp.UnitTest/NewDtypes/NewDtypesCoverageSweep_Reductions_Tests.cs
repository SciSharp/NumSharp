using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Round 14 — Reductions sweep for Half / Complex / SByte, battletested against
    /// NumPy 2.4.2. 80-case matrix, 100% parity after closing 10 open bugs.
    ///
    /// Bugs closed:
    ///   B1  — Half min/max elementwise returned ±∞ (IL OpCodes.Bgt/Blt don't work on Half).
    ///   B2  — Complex mean axis returned Double instead of Complex.
    ///   B4  — np.prod(Half) and np.prod(Complex) threw NotSupportedException.
    ///   B5  — np.min/max(SByte, axis=N) threw NotSupportedException (missing from identity table).
    ///   B6  — np.cumsum(Half|Complex, axis=N) threw at kernel execution time.
    ///   B7  — np.argmax/argmin(Half|Complex|SByte, axis=N) threw NotSupportedException.
    ///   B8  — np.min/max(Complex) elementwise threw NotSupportedException.
    ///   B12 — np.argmax(Complex) returned wrong index (IL kernel tiebreak broken).
    ///   B15 — np.nansum(Complex) fell through to regular sum, propagating NaN.
    ///   B16 — np.std/var(Half, axis=N) returned Double instead of Half.
    /// </summary>
    [TestClass]
    public class NewDtypesCoverageSweep_Reductions_Tests
    {
        private const double HalfTol = 1e-3;
        private static Complex C(double r, double i) => new Complex(r, i);

        #region B1 — Half min/max elementwise

        [TestMethod]
        public void B1_Half_Min_Elementwise()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2.5f, (Half)(-3), (Half)4.5f, (Half)0 });
            ((double)np.min(a).GetAtIndex<Half>(0)).Should().Be(-3.0);
        }

        [TestMethod]
        public void B1_Half_Max_Elementwise()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2.5f, (Half)(-3), (Half)4.5f, (Half)0 });
            ((double)np.max(a).GetAtIndex<Half>(0)).Should().Be(4.5);
        }

        [TestMethod]
        public void B1_Half_Min_NaNPropagates()
        {
            var a = np.array(new Half[] { (Half)1, Half.NaN, (Half)3 });
            Half.IsNaN(np.min(a).GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void B1_Half_Amin_Amax()
        {
            var a = np.array(new Half[] { (Half)5, (Half)1, (Half)3 });
            ((double)np.amin(a).GetAtIndex<Half>(0)).Should().Be(1.0);
            ((double)np.amax(a).GetAtIndex<Half>(0)).Should().Be(5.0);
        }

        #endregion

        #region B2 — Complex mean axis preserves dtype

        [TestMethod]
        public void B2_Complex_MeanAxis_PreservesComplexDtype()
        {
            // NumPy: np.mean([[1+0j, 0+2j], [3+1j, 1-1j]], axis=0) == [2+0.5j, 0.5+0.5j]
            var a = np.array(new Complex[,] { { C(1, 0), C(0, 2) }, { C(3, 1), C(1, -1) } });
            var r = np.mean(a, 0);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 0.5));
            r.GetAtIndex<Complex>(1).Should().Be(C(0.5, 0.5));
        }

        [TestMethod]
        public void B2_Half_MeanAxis_PreservesHalfDtype()
        {
            // NumPy: mean(half2d, axis=0) returns half
            var a = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var r = np.mean(a, 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.5, HalfTol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(3.5, HalfTol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(4.5, HalfTol);
        }

        #endregion

        #region B4 — Half/Complex prod

        [TestMethod]
        public void B4_Half_Prod()
        {
            // NumPy: prod([1,2,-3,4.5,0]) with float16 = -0.0 (zero wins)
            var a = np.array(new Half[] { (Half)1, (Half)2.5f, (Half)(-3), (Half)4.5f, (Half)0 });
            ((double)np.prod(a).GetAtIndex<Half>(0)).Should().Be(-0.0);
        }

        [TestMethod]
        public void B4_Complex_Prod()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, -1), C(0, 0), C(-2, 3) });
            // any zero kills the product
            np.prod(a).GetAtIndex<Complex>(0).Should().Be(C(0, 0));
        }

        [TestMethod]
        public void B4_Complex_Prod_NoZero()
        {
            // (1+2j) * (2+1j) = 2+1j + 4j+2j^2 = 2+1j+4j-2 = 0+5j
            var a = np.array(new Complex[] { C(1, 2), C(2, 1) });
            np.prod(a).GetAtIndex<Complex>(0).Should().Be(C(0, 5));
        }

        [TestMethod]
        public void B4_Half_Prod_Axis()
        {
            var a = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var r = np.prod(a, 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(4.0);
            ((double)r.GetAtIndex<Half>(1)).Should().Be(10.0);
            ((double)r.GetAtIndex<Half>(2)).Should().Be(18.0);
        }

        #endregion

        #region B5 — SByte axis reduction (min/max/sum/prod)

        [TestMethod]
        public void B5_SByte_MinAxis()
        {
            var a = np.array(new sbyte[,] { { 10, 20, 30 }, { -10, -20, -30 } });
            var r = np.min(a, 0);
            r.typecode.Should().Be(NPTypeCode.SByte);
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-10));
            r.GetAtIndex<sbyte>(2).Should().Be((sbyte)(-30));
        }

        [TestMethod]
        public void B5_SByte_MaxAxis()
        {
            var a = np.array(new sbyte[,] { { 10, 20, 30 }, { -10, -20, -30 } });
            var r = np.max(a, 0);
            r.typecode.Should().Be(NPTypeCode.SByte);
            r.GetAtIndex<sbyte>(0).Should().Be((sbyte)10);
            r.GetAtIndex<sbyte>(2).Should().Be((sbyte)30);
        }

        #endregion

        #region B6 — Half/Complex cumsum axis

        [TestMethod]
        public void B6_Half_CumSumAxis()
        {
            var a = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var r = np.cumsum(a, 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            r.shape.Should().Equal(new long[] { 2, 3 });
            ((double)r.GetAtIndex<Half>(3)).Should().Be(5.0); // 1+4
            ((double)r.GetAtIndex<Half>(4)).Should().Be(7.0); // 2+5
            ((double)r.GetAtIndex<Half>(5)).Should().Be(9.0); // 3+6
        }

        [TestMethod]
        public void B6_Complex_CumSumAxis_PreservesImaginary()
        {
            var a = np.array(new Complex[,] { { C(1, 0), C(0, 2) }, { C(3, 1), C(1, -1) } });
            var r = np.cumsum(a, 0);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(4, 1)); // 1+3, 0+1
            r.GetAtIndex<Complex>(3).Should().Be(C(1, 1)); // 0+1, 2-1
        }

        #endregion

        #region B7 — argmax/argmin axis for Half/Complex/SByte

        [TestMethod]
        public void B7_Half_ArgmaxAxis()
        {
            var a = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var r = np.argmax(a, 0);
            r.GetAtIndex<long>(0).Should().Be(1L);
            r.GetAtIndex<long>(1).Should().Be(1L);
            r.GetAtIndex<long>(2).Should().Be(1L);
        }

        [TestMethod]
        public void B7_Complex_ArgmaxAxis()
        {
            var a = np.array(new Complex[,] { { C(1, 0), C(0, 2) }, { C(3, 1), C(1, -1) } });
            // Real-first lex compare: col 0 max = row 1 (3>1), col 1 max = row 1 (1>0)
            var r = np.argmax(a, 0);
            r.GetAtIndex<long>(0).Should().Be(1L);
            r.GetAtIndex<long>(1).Should().Be(1L);
        }

        [TestMethod]
        public void B7_SByte_ArgmaxAxis()
        {
            var a = np.array(new sbyte[,] { { 10, 20, 30 }, { -10, -20, -30 } });
            var r = np.argmax(a, 0);
            r.GetAtIndex<long>(0).Should().Be(0L);
            r.GetAtIndex<long>(1).Should().Be(0L);
            r.GetAtIndex<long>(2).Should().Be(0L);
        }

        #endregion

        #region B8 — Complex min/max elementwise

        [TestMethod]
        public void B8_Complex_Min_LexicographicCompare()
        {
            // NumPy lex max: real-first, imag as tie-break
            var a = np.array(new Complex[] { C(1, 2), C(3, -1), C(0, 0), C(-2, 3) });
            // min = (-2, 3) (smallest real)
            np.min(a).GetAtIndex<Complex>(0).Should().Be(C(-2, 3));
        }

        [TestMethod]
        public void B8_Complex_Max_LexicographicCompare()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, -1), C(0, 0), C(-2, 3) });
            // max = (3, -1) (largest real)
            np.max(a).GetAtIndex<Complex>(0).Should().Be(C(3, -1));
        }

        [TestMethod]
        public void B8_Complex_Min_TiebreakByImag()
        {
            // Same real: 1+0j vs 1+2j — min by lex is 1+0j (smaller imag)
            var a = np.array(new Complex[] { C(1, 2), C(1, 0) });
            np.min(a).GetAtIndex<Complex>(0).Should().Be(C(1, 0));
        }

        [TestMethod]
        public void B8_Complex_NaN_PropagatesThroughMin()
        {
            var a = np.array(new Complex[] { C(1, 2), C(double.NaN, 0), C(3, 1) });
            var r = np.min(a).GetAtIndex<Complex>(0);
            double.IsNaN(r.Real).Should().BeTrue();
            double.IsNaN(r.Imaginary).Should().BeTrue();
        }

        #endregion

        #region B12 — Complex argmax tiebreak (lex compare)

        [TestMethod]
        public void B12_Complex_Argmax_ReturnsLexMaxIndex()
        {
            // cplx = [1+2j, 3-1j, 0+0j, -2+3j] — lex max is 3-1j at index 1
            var a = np.array(new Complex[] { C(1, 2), C(3, -1), C(0, 0), C(-2, 3) });
            np.argmax(a).Should().Be(1L);
        }

        [TestMethod]
        public void B12_Complex_Argmin_ReturnsLexMinIndex()
        {
            // lex min is -2+3j at index 3
            var a = np.array(new Complex[] { C(1, 2), C(3, -1), C(0, 0), C(-2, 3) });
            np.argmin(a).Should().Be(3L);
        }

        #endregion

        #region B15 — Complex nansum skips NaN entries

        [TestMethod]
        public void B15_Complex_NanSum_SkipsNaN()
        {
            // NumPy: np.nansum([1+2j, nan+0j, 3+1j]) = 4+3j (skips the nan entry)
            var a = np.array(new Complex[] { C(1, 2), C(double.NaN, 0), C(3, 1) });
            np.nansum(a).GetAtIndex<Complex>(0).Should().Be(C(4, 3));
        }

        [TestMethod]
        public void B15_Complex_NanSum_AllNaN_ReturnsZero()
        {
            var a = np.array(new Complex[] { C(double.NaN, 0), C(double.NaN, 0) });
            np.nansum(a).GetAtIndex<Complex>(0).Should().Be(C(0, 0));
        }

        [TestMethod]
        public void B15_Complex_NanSum_NoNaN_BehavesAsSum()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, 1) });
            np.nansum(a).GetAtIndex<Complex>(0).Should().Be(C(4, 3));
        }

        #endregion

        #region B16 — Half std/var axis preserve input dtype

        [TestMethod]
        public void B16_Half_StdAxis_PreservesHalfDtype()
        {
            var a = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var r = np.std(a, 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            // std([1,4]) = 1.5
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(1.5, HalfTol);
            ((double)r.GetAtIndex<Half>(1)).Should().BeApproximately(1.5, HalfTol);
            ((double)r.GetAtIndex<Half>(2)).Should().BeApproximately(1.5, HalfTol);
        }

        [TestMethod]
        public void B16_Half_VarAxis_PreservesHalfDtype()
        {
            var a = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            var r = np.var(a, 0);
            r.typecode.Should().Be(NPTypeCode.Half);
            ((double)r.GetAtIndex<Half>(0)).Should().BeApproximately(2.25, HalfTol);
        }

        [TestMethod]
        public void B16_Complex_VarAxis_ReturnsDouble()
        {
            // NumPy: variance of complex always returns real float (std/var is non-negative real)
            var a = np.array(new Complex[,] { { C(1, 0), C(0, 2) }, { C(3, 1), C(1, -1) } });
            var r = np.var(a, 0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        #endregion

        #region Round 14 smoke tests

        [TestMethod]
        public void Sum_Half()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2.5f, (Half)(-3), (Half)4.5f, (Half)0 });
            ((double)np.sum(a).GetAtIndex<Half>(0)).Should().BeApproximately(5.0, HalfTol);
        }

        [TestMethod]
        public void Sum_Complex()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, -1), C(0, 0), C(-2, 3) });
            np.sum(a).GetAtIndex<Complex>(0).Should().Be(C(2, 4));
        }

        [TestMethod]
        public void Any_All_Complex_NonZero()
        {
            var a = np.array(new Complex[] { C(1, 0), C(0, 1), C(2, 2) });
            np.all(a).Should().BeTrue();
            np.any(a).Should().BeTrue();
        }

        [TestMethod]
        public void CountNonzero_Complex()
        {
            var a = np.array(new Complex[] { C(1, 0), C(0, 0), C(2, 2) });
            // count = 2 (skips (0,0))
            np.count_nonzero(a).Should().Be(2L);
        }

        [TestMethod]
        public void ArgmaxAxis_SByte()
        {
            var a = np.array(new sbyte[,] { { 10, 20, 30 }, { -10, -20, -30 } });
            var r = np.argmax(a, 0);
            // All columns: row 0 wins
            r.GetAtIndex<long>(0).Should().Be(0L);
            r.GetAtIndex<long>(1).Should().Be(0L);
            r.GetAtIndex<long>(2).Should().Be(0L);
        }

        #endregion

        #region B9 — np.unique(Complex)

        [TestMethod]
        public void B9_Unique_Complex_BasicDedup()
        {
            // NumPy: np.unique([1+2j, 1+2j, 3+0j, 0+0j, 3+0j]) = [0+0j, 1+2j, 3+0j]
            var a = np.array(new Complex[] { C(1, 2), C(1, 2), C(3, 0), C(0, 0), C(3, 0) });
            var r = np.unique(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.size.Should().Be(3);
            r.GetAtIndex<Complex>(0).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 0));
        }

        [TestMethod]
        public void B9_Unique_Complex_AlreadySorted()
        {
            var a = np.array(new Complex[] { C(0, 0), C(1, 2), C(3, 0) });
            var r = np.unique(a);
            r.size.Should().Be(3);
            r.GetAtIndex<Complex>(0).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 0));
        }

        [TestMethod]
        public void B9_Unique_Complex_ReverseOrder()
        {
            var a = np.array(new Complex[] { C(3, 0), C(1, 2), C(0, 0) });
            var r = np.unique(a);
            r.size.Should().Be(3);
            r.GetAtIndex<Complex>(0).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 0));
        }

        [TestMethod]
        public void B9_Unique_Complex_AllDuplicates()
        {
            var a = np.array(new Complex[] { C(1, 2), C(1, 2), C(1, 2) });
            var r = np.unique(a);
            r.size.Should().Be(1);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
        }

        [TestMethod]
        public void B9_Unique_Complex_SingleElement()
        {
            var a = np.array(new Complex[] { C(5, 5) });
            var r = np.unique(a);
            r.size.Should().Be(1);
            r.GetAtIndex<Complex>(0).Should().Be(C(5, 5));
        }

        [TestMethod]
        public void B9_Unique_Complex_SameRealDifferentImag()
        {
            // Lex sort: (1,1) < (1,2) < (1,3)
            var a = np.array(new Complex[] { C(1, 3), C(1, 2), C(1, 2), C(1, 1) });
            var r = np.unique(a);
            r.size.Should().Be(3);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(1, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(1, 3));
        }

        [TestMethod]
        public void B9_Unique_Complex_NaNSortsToEnd()
        {
            // NaN any-component sorts to end; non-NaN lex-sorted first
            var a = np.array(new Complex[] { C(1, 2), C(double.NaN, 0), C(1, 2) });
            var r = np.unique(a);
            r.size.Should().Be(2);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
            var last = r.GetAtIndex<Complex>(1);
            double.IsNaN(last.Real).Should().BeTrue();
        }

        [TestMethod]
        public void B9_Unique_Complex_PureImagNaN()
        {
            // NaN in imag component also triggers NaN-at-end classification
            var a = np.array(new Complex[] { C(2, 0), C(1, double.NaN), C(0, 0) });
            var r = np.unique(a);
            r.size.Should().Be(3);
            r.GetAtIndex<Complex>(0).Should().Be(C(0, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 0));
            var last = r.GetAtIndex<Complex>(2);
            double.IsNaN(last.Imaginary).Should().BeTrue();
        }

        [TestMethod]
        public void B9_Unique_Complex_NonContiguousView()
        {
            // Non-contiguous flat path (strided slice)
            var full = np.array(new Complex[] { C(3, 0), C(1, 2), C(5, 0), C(1, 2), C(3, 0), C(0, 0) });
            var view = full["::2"];  // [3+0j, 5+0j, 3+0j]
            var r = np.unique(view);
            r.size.Should().Be(2);
            r.GetAtIndex<Complex>(0).Should().Be(C(3, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(5, 0));
        }

        #endregion

        #region B13 — Complex argmax/argmin with NaN

        [TestMethod]
        public void B13_ArgMax_Complex_NaNInMiddle()
        {
            // NumPy: np.argmax([1+2j, nan+0j, 3+1j]) == 1 (first NaN wins)
            var a = np.array(new Complex[] { C(1, 2), C(double.NaN, 0), C(3, 1) });
            np.argmax(a).Should().Be(1L);
        }

        [TestMethod]
        public void B13_ArgMax_Complex_NaNFirst()
        {
            var a = np.array(new Complex[] { C(double.NaN, 0), C(1, 2), C(3, 1) });
            np.argmax(a).Should().Be(0L);
        }

        [TestMethod]
        public void B13_ArgMax_Complex_NaNLast()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, 0), C(double.NaN, 1) });
            np.argmax(a).Should().Be(2L);
        }

        [TestMethod]
        public void B13_ArgMax_Complex_NaNInImagOnly()
        {
            // Imag NaN also counts as "NaN" for argmax purposes
            var a = np.array(new Complex[] { C(1, 2), C(3, double.NaN), C(5, 1) });
            np.argmax(a).Should().Be(1L);
        }

        [TestMethod]
        public void B13_ArgMin_Complex_NaNInMiddle()
        {
            var a = np.array(new Complex[] { C(3, 1), C(double.NaN, 0), C(1, 2) });
            np.argmin(a).Should().Be(1L);
        }

        [TestMethod]
        public void B13_ArgMin_Complex_NaNFirst()
        {
            var a = np.array(new Complex[] { C(double.NaN, 0), C(1, 2) });
            np.argmin(a).Should().Be(0L);
        }

        [TestMethod]
        public void B13_ArgMax_Complex_NoNaN_Regression_B12()
        {
            // Regression: B12 lex compare must still work when no NaN present
            var a = np.array(new Complex[] { C(1, 0), C(1, 5), C(1, 3) });
            np.argmax(a).Should().Be(1L);  // (1,5) has highest imag
        }

        [TestMethod]
        public void B13_ArgMin_Complex_NoNaN_Regression_B12()
        {
            var a = np.array(new Complex[] { C(1, 0), C(1, -5), C(1, 3) });
            np.argmin(a).Should().Be(1L);  // (1,-5) has lowest imag
        }

        [TestMethod]
        public void B13_ArgMax_Complex_Axis_NaNPropagates()
        {
            // Axis variant: B7 fallback uses argmax_elementwise_il per slice → NaN semantics preserved
            var a = np.array(new Complex[,] {
                { C(1, 0), C(5, 0) },
                { C(double.NaN, 0), C(2, 0) },
                { C(3, 0), C(1, 0) }
            });
            var r = np.argmax(a, 0);
            r.GetAtIndex<long>(0).Should().Be(1L);  // NaN at row 1 wins column 0
            r.GetAtIndex<long>(1).Should().Be(0L);  // column 1: 5 > 2 > 1
        }

        #endregion
    }
}
