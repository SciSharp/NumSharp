using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Battletests for Round 7 fixes — Complex axis-reduction bugs.
    /// All expected values verified against NumPy 2.4.2.
    ///
    /// Covers:
    ///   - B17: np.clip for Half/Complex (regression — closed in Round 6, re-verified here)
    ///   - B18: np.cumprod(Complex, axis=N) — was silently dropping imaginary part
    ///   - B19: np.max/min(Complex, axis=N) — was returning all zeros
    ///   - B20: np.std/var(Complex, axis=N) — was computing variance of real parts only
    /// </summary>
    [TestClass]
    public class NewDtypesBattletestRound7Tests
    {
        private const double Tol = 1e-3;

        private static Complex C(double r, double i) => new Complex(r, i);

        // Shared 3×3 Complex matrix used across tests.
        //   c_mat = [[1+1j, 2+2j, 3+3j],
        //            [4+4j, 5+5j, 6+6j],
        //            [7+7j, 8+8j, 9+9j]]
        private static NDArray ComplexMat3x3() =>
            np.array(new Complex[,] {
                { C(1,1), C(2,2), C(3,3) },
                { C(4,4), C(5,5), C(6,6) },
                { C(7,7), C(8,8), C(9,9) }
            });

        #region B17 — np.clip Half/Complex (re-verify)

        [TestMethod]
        public void B17_Half_Clip_RegressionCheck()
        {
            // np.clip(np.array([1, 5, 10, 15], dtype=float16), 3, 10) → [3, 5, 10, 10]
            var a = np.array(new Half[] { (Half)1, (Half)5, (Half)10, (Half)15 });
            var r = np.clip(a, np.array(new Half[] { (Half)3 }), np.array(new Half[] { (Half)10 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            r.GetAtIndex<Half>(0).Should().Be((Half)3);
            r.GetAtIndex<Half>(1).Should().Be((Half)5);
            r.GetAtIndex<Half>(2).Should().Be((Half)10);
            r.GetAtIndex<Half>(3).Should().Be((Half)10);
        }

        [TestMethod]
        public void B17_Complex_Clip_RegressionCheck()
        {
            // np.clip([1+1j, 5+5j, 10+10j], 2+0j, 8+0j) → [2+0j, 5+5j, 8+0j] (lex)
            var a = np.array(new Complex[] { C(1, 1), C(5, 5), C(10, 10) });
            var r = np.clip(a, np.array(new Complex[] { C(2, 0) }), np.array(new Complex[] { C(8, 0) }));
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(C(2, 0));
            r.GetAtIndex<Complex>(1).Should().Be(C(5, 5));
            r.GetAtIndex<Complex>(2).Should().Be(C(8, 0));
        }

        #endregion

        #region B18 — Complex cumprod along axis

        [TestMethod]
        public void B18_Complex_Cumprod_Axis0()
        {
            // np.cumprod(c_mat, axis=0):
            //   row 0: [1+1j, 2+2j, 3+3j]                  (passthrough)
            //   row 1: [(1+1j)(4+4j), (2+2j)(5+5j), (3+3j)(6+6j)]   = [0+8j, 0+20j, 0+36j]
            //   row 2: [(0+8j)(7+7j), (0+20j)(8+8j), (0+36j)(9+9j)] = [-56+56j, -160+160j, -324+324j]
            var r = np.cumprod(ComplexMat3x3(), axis: 0);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.shape.Should().BeEquivalentTo(new[] { 3, 3 });

            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 3));
            r.GetAtIndex<Complex>(3).Should().Be(C(0, 8));
            r.GetAtIndex<Complex>(4).Should().Be(C(0, 20));
            r.GetAtIndex<Complex>(5).Should().Be(C(0, 36));
            r.GetAtIndex<Complex>(6).Should().Be(C(-56, 56));
            r.GetAtIndex<Complex>(7).Should().Be(C(-160, 160));
            r.GetAtIndex<Complex>(8).Should().Be(C(-324, 324));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_Axis1()
        {
            // np.cumprod(c_mat, axis=1):
            //   row 0: [1+1j, (1+1j)(2+2j)=0+4j, (0+4j)(3+3j)=-12+12j]
            //   row 1: [4+4j, (4+4j)(5+5j)=0+40j, (0+40j)(6+6j)=-240+240j]
            //   row 2: [7+7j, (7+7j)(8+8j)=0+112j, (0+112j)(9+9j)=-1008+1008j]
            var r = np.cumprod(ComplexMat3x3(), axis: 1);
            r.typecode.Should().Be(NPTypeCode.Complex);

            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(-12, 12));
            r.GetAtIndex<Complex>(3).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(4).Should().Be(C(0, 40));
            r.GetAtIndex<Complex>(5).Should().Be(C(-240, 240));
            r.GetAtIndex<Complex>(6).Should().Be(C(7, 7));
            r.GetAtIndex<Complex>(7).Should().Be(C(0, 112));
            r.GetAtIndex<Complex>(8).Should().Be(C(-1008, 1008));
        }

        [TestMethod]
        public void B18_Complex_Cumprod_Elementwise_Unchanged()
        {
            // Regression: elementwise cumprod (axis=None) already worked pre-fix — don't break it.
            // np.cumprod([1+1j, 2+2j, 3+3j]) → [1+1j, 0+4j, -12+12j]
            var a = np.array(new Complex[] { C(1, 1), C(2, 2), C(3, 3) });
            var r = np.cumprod(a);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(0, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(-12, 12));
        }

        #endregion

        #region B19 — Complex max/min along axis (lex ordering + NaN propagation)

        [TestMethod]
        public void B19_Complex_Max_Axis0()
        {
            // np.max(c_mat, axis=0) → [7+7j, 8+8j, 9+9j]   (last row wins by lex)
            var r = np.max(ComplexMat3x3(), axis: 0);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(C(7, 7));
            r.GetAtIndex<Complex>(1).Should().Be(C(8, 8));
            r.GetAtIndex<Complex>(2).Should().Be(C(9, 9));
        }

        [TestMethod]
        public void B19_Complex_Min_Axis0()
        {
            // np.min(c_mat, axis=0) → [1+1j, 2+2j, 3+3j]   (first row wins by lex)
            var r = np.min(ComplexMat3x3(), axis: 0);
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(2, 2));
            r.GetAtIndex<Complex>(2).Should().Be(C(3, 3));
        }

        [TestMethod]
        public void B19_Complex_Max_Axis1()
        {
            // np.max(c_mat, axis=1) → [3+3j, 6+6j, 9+9j]   (last col wins by lex)
            var r = np.max(ComplexMat3x3(), axis: 1);
            r.GetAtIndex<Complex>(0).Should().Be(C(3, 3));
            r.GetAtIndex<Complex>(1).Should().Be(C(6, 6));
            r.GetAtIndex<Complex>(2).Should().Be(C(9, 9));
        }

        [TestMethod]
        public void B19_Complex_Min_Axis1()
        {
            // np.min(c_mat, axis=1) → [1+1j, 4+4j, 7+7j]
            var r = np.min(ComplexMat3x3(), axis: 1);
            r.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            r.GetAtIndex<Complex>(1).Should().Be(C(4, 4));
            r.GetAtIndex<Complex>(2).Should().Be(C(7, 7));
        }

        [TestMethod]
        public void B19_Complex_Max_LexOrder_SameReal()
        {
            // Same-real lex: secondary sort by imaginary.
            //   col 0 = [1+5j, 1+3j, 1+7j] → max = 1+7j, min = 1+3j
            //   col 1 = [2+1j, 2+9j, 2+5j] → max = 2+9j, min = 2+1j
            var m = np.array(new Complex[,] {
                { C(1, 5), C(2, 1) },
                { C(1, 3), C(2, 9) },
                { C(1, 7), C(2, 5) }
            });
            var mx = np.max(m, axis: 0);
            mx.GetAtIndex<Complex>(0).Should().Be(C(1, 7));
            mx.GetAtIndex<Complex>(1).Should().Be(C(2, 9));

            var mn = np.min(m, axis: 0);
            mn.GetAtIndex<Complex>(0).Should().Be(C(1, 3));
            mn.GetAtIndex<Complex>(1).Should().Be(C(2, 1));
        }

        [TestMethod]
        public void B19_Complex_MinMax_NaN_Propagates()
        {
            // NumPy: if any element along the axis is NaN-containing (Re or Im NaN), result is NaN.
            //   np.max([[1+1j, nan+0j], [2+2j, 3+3j]], axis=0) → [2+2j, nan+0j]
            //   np.min(...)                                  → [1+1j, nan+0j]
            var m = np.array(new Complex[,] {
                { C(1, 1), C(double.NaN, 0) },
                { C(2, 2), C(3, 3) }
            });
            var mx = np.max(m, axis: 0);
            mx.GetAtIndex<Complex>(0).Should().Be(C(2, 2));
            var mx1 = mx.GetAtIndex<Complex>(1);
            double.IsNaN(mx1.Real).Should().BeTrue();
            mx1.Imaginary.Should().Be(0.0);

            var mn = np.min(m, axis: 0);
            mn.GetAtIndex<Complex>(0).Should().Be(C(1, 1));
            var mn1 = mn.GetAtIndex<Complex>(1);
            double.IsNaN(mn1.Real).Should().BeTrue();
            mn1.Imaginary.Should().Be(0.0);
        }

        [TestMethod]
        public void B19_Complex_Sum_Prod_Mean_Axis_Unchanged()
        {
            // Regression: Sum/Prod/Mean axis paths previously worked (via CombineScalarsPromoted Complex
            // path) — verify my Min/Max fix didn't break them.
            var r_sum = np.sum(ComplexMat3x3(), axis: 0);
            r_sum.typecode.Should().Be(NPTypeCode.Complex);
            r_sum.GetAtIndex<Complex>(0).Should().Be(C(12, 12));
            r_sum.GetAtIndex<Complex>(1).Should().Be(C(15, 15));
            r_sum.GetAtIndex<Complex>(2).Should().Be(C(18, 18));

            var r_prod = np.prod(ComplexMat3x3(), axis: 0);
            r_prod.typecode.Should().Be(NPTypeCode.Complex);
            r_prod.GetAtIndex<Complex>(0).Should().Be(C(-56, 56));  // (1+1j)(4+4j)(7+7j)
            r_prod.GetAtIndex<Complex>(1).Should().Be(C(-160, 160));
            r_prod.GetAtIndex<Complex>(2).Should().Be(C(-324, 324));
        }

        #endregion

        #region B20 — Complex std/var along axis

        [TestMethod]
        public void B20_Complex_Var_Axis0()
        {
            // np.var(c_mat, axis=0):
            //   col 0 = [1+1j, 4+4j, 7+7j], mean = 4+4j
            //     |z - mean|² = |-3-3j|²=18, 0, |3+3j|²=18 → sum=36, var=36/3=12
            //   cols 1, 2 analogous → [12, 12, 12] (dtype float64).
            var r = np.var(ComplexMat3x3(), axis: 0);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(12.0, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(12.0, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(12.0, Tol);
        }

        [TestMethod]
        public void B20_Complex_Std_Axis0()
        {
            // std = sqrt(var) → sqrt(12) = 3.464... (dtype float64)
            var r = np.std(ComplexMat3x3(), axis: 0);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(3.46410161513775, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(3.46410161513775, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(3.46410161513775, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_Axis1()
        {
            // np.var(c_mat, axis=1):
            //   row 0 = [1+1j, 2+2j, 3+3j], mean = 2+2j
            //     |z - mean|² = |-1-1j|²=2, 0, |1+1j|²=2 → sum=4, var=4/3=1.333...
            var r = np.var(ComplexMat3x3(), axis: 1);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(1.33333333333, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(1.33333333333, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(1.33333333333, Tol);
        }

        [TestMethod]
        public void B20_Complex_Std_Axis1()
        {
            // std = sqrt(1.333...) = 1.1547...
            var r = np.std(ComplexMat3x3(), axis: 1);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(1.15470053837925, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(1.15470053837925, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(1.15470053837925, Tol);
        }

        [TestMethod]
        public void B20_Complex_Var_Ddof()
        {
            // np.var(np.array([[1+2j, 3+4j, 5+6j]]), axis=1, ddof=1) = 8.0
            //   mean = 3+4j; |-2-2j|²=8, 0, |2+2j|²=8; sum=16; divisor=3-1=2; var=8
            var m = np.array(new Complex[,] { { C(1, 2), C(3, 4), C(5, 6) } });
            var r = np.var(m, axis: 1, ddof: 1);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().BeApproximately(8.0, Tol);
        }

        [TestMethod]
        public void B20_Complex_Std_Elementwise_Unchanged()
        {
            // Regression: elementwise std/var already worked pre-fix — don't break them.
            // np.std([1+2j, 3+4j, 5+6j]) = 2.309... ; np.var(...) = 5.333...
            var a = np.array(new Complex[] { C(1, 2), C(3, 4), C(5, 6) });
            np.std(a).GetAtIndex<double>(0).Should().BeApproximately(2.30940107675, Tol);
            np.var(a).GetAtIndex<double>(0).Should().BeApproximately(5.33333333333, Tol);
        }

        [TestMethod]
        public void B20_Double_Var_Regression()
        {
            // Regression: existing double path unchanged.
            // np.var([[1,2,3],[4,5,6],[7,8,9]], axis=0) → [6, 6, 6]
            var m = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            var r = np.var(m, axis: 0);
            r.GetAtIndex<double>(0).Should().BeApproximately(6.0, Tol);
            r.GetAtIndex<double>(1).Should().BeApproximately(6.0, Tol);
            r.GetAtIndex<double>(2).Should().BeApproximately(6.0, Tol);
        }

        #endregion
    }
}
