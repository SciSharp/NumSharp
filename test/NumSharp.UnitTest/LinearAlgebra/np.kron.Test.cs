using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    /// Parity tests for np.kron (Kronecker product). Every expected value/shape/dtype was produced by
    /// running NumPy 2.4.2 and replicated here. Covers the docstring examples, mixed ranks, scalars,
    /// dtype promotion, empties, non-contiguous inputs, and both internal dispatch paths (direct
    /// broadcast-multiply and the tile fast-path for short inner runs).
    /// </summary>
    [TestClass]
    public class np_kron_Test : TestClass
    {
        // ---- docstring / 1-D ----------------------------------------------------------------

        [TestMethod]
        public void Kron_1D_Basic()
        {
            // np.kron([1,10,100],[5,6,7]) -> [5,6,7,50,60,70,500,600,700]
            np.kron(np.array(new[] { 1, 10, 100 }), np.array(new[] { 5, 6, 7 }))
                .Should().BeShaped(9)
                .And.BeOfValues(5, 6, 7, 50, 60, 70, 500, 600, 700);
        }

        [TestMethod]
        public void Kron_1D_OrderMatters()
        {
            // np.kron([5,6,7],[1,10,100]) -> [5,50,500,6,60,600,7,70,700]
            np.kron(np.array(new[] { 5, 6, 7 }), np.array(new[] { 1, 10, 100 }))
                .Should().BeShaped(9)
                .And.BeOfValues(5, 50, 500, 6, 60, 600, 7, 70, 700);
        }

        [TestMethod]
        public void Kron_2D_EyeOnes()
        {
            // np.kron(np.eye(2), np.ones((2,2))) -> block-diagonal ones
            np.kron(np.eye(2), np.ones(new Shape(2, 2)))
                .Should().BeShaped(4, 4)
                .And.BeOfValues(1d, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1);
        }

        [TestMethod]
        public void Kron_2D_QuantumGate()
        {
            // kron(I2, X) where X = [[0,1],[1,0]] — a common tensor-product pattern.
            var x = np.array(new[,] { { 0d, 1 }, { 1, 0 } });
            np.kron(np.eye(2), x)
                .Should().BeShaped(4, 4)
                .And.BeOfValues(0d, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0);
        }

        [TestMethod]
        public void Kron_SingleElementVectors()
        {
            // np.kron([2],[3]) -> [6]
            np.kron(np.array(new[] { 2 }), np.array(new[] { 3 }))
                .Should().BeShaped(1)
                .And.BeOfValues(6);
        }

        // ---- scalars ------------------------------------------------------------------------

        [TestMethod]
        public void Kron_BothScalar_Is0d()
        {
            // np.kron(3,4) -> array(12) shape ()
            var r = np.kron((NDArray)3, (NDArray)4);
            r.ndim.Should().Be(0);
            r.size.Should().Be(1);
            r.GetInt32(0).Should().Be(12);
        }

        [TestMethod]
        public void Kron_ScalarB_IsElementwiseScale()
        {
            // np.kron([1,2,3], 3) -> [3,6,9]
            np.kron(np.array(new[] { 1, 2, 3 }), (NDArray)3)
                .Should().BeShaped(3)
                .And.BeOfValues(3, 6, 9);
        }

        [TestMethod]
        public void Kron_ScalarA_IsElementwiseScale()
        {
            // np.kron(3, [1,2,3]) -> [3,6,9]
            np.kron((NDArray)3, np.array(new[] { 1, 2, 3 }))
                .Should().BeShaped(3)
                .And.BeOfValues(3, 6, 9);
        }

        // ---- mixed rank (ndmin promotion) ---------------------------------------------------

        [TestMethod]
        public void Kron_1D_by_2D()
        {
            // np.kron([1,2], [[1,2,3],[4,5,6]]) -> shape (2,6)
            var b = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            np.kron(np.array(new[] { 1, 2 }), b)
                .Should().BeShaped(2, 6)
                .And.BeOfValues(1, 2, 3, 2, 4, 6, 4, 5, 6, 8, 10, 12);
        }

        [TestMethod]
        public void Kron_2D_by_1D()
        {
            // np.kron([[1,2,3],[4,5,6]], [1,2]) -> shape (2,6)
            var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            np.kron(a, np.array(new[] { 1, 2 }))
                .Should().BeShaped(2, 6)
                .And.BeOfValues(1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12);
        }

        [TestMethod]
        public void Kron_1D_by_3D()
        {
            // np.kron([1,2], arange(8).reshape(2,2,2)) -> shape (2,2,4)
            np.kron(np.array(new[] { 1, 2 }), np.arange(8).reshape(2, 2, 2))
                .Should().BeShaped(2, 2, 4)
                .And.BeOfValues(0, 1, 0, 2, 2, 3, 4, 6, 4, 5, 8, 10, 6, 7, 12, 14);
        }

        [TestMethod]
        public void Kron_4D_by_3D_Shape()
        {
            // docstring: a=(2,5,2,5), b=(2,3,4) -> c.shape == (2,10,6,20)
            var a = np.arange(100).reshape(2, 5, 2, 5);
            var b = np.arange(24).reshape(2, 3, 4);
            np.kron(a, b).Should().BeShaped(2, 10, 6, 20);
        }

        // ---- dtype promotion (NEP50, delegated to multiply) ---------------------------------

        [TestMethod]
        public void Kron_Promotion_Int32_Float64()
        {
            // np.kron(int32[1,2], float64[1.5,2.5]) -> float64 [1.5,2.5,3,5]
            var r = np.kron(np.array(new[] { 1, 2 }).astype(NPTypeCode.Int32), np.array(new[] { 1.5, 2.5 }));
            r.dtype.Should().Be(typeof(double));
            r.Should().BeShaped(4).And.BeOfValues(1.5, 2.5, 3.0, 5.0);
        }

        [TestMethod]
        public void Kron_Promotion_UInt8_Int8_To_Int16()
        {
            var r = np.kron(np.array(new byte[] { 1, 2 }), np.array(new sbyte[] { 3, 4 }));
            r.dtype.Should().Be(typeof(short));
            r.Should().BeShaped(4).And.BeOfValues((short)3, 4, 6, 8);
        }

        [TestMethod]
        public void Kron_Complex()
        {
            // np.kron([1+2j, 1j], [3, 1+1j]) -> [3+6j, -1+3j, 3j, -1+1j]
            var a = np.array(new[] { new Complex(1, 2), new Complex(0, 1) });
            var b = np.array(new[] { new Complex(3, 0), new Complex(1, 1) });
            var r = np.kron(a, b);
            r.dtype.Should().Be(typeof(Complex));
            r.Should().BeShaped(4);
            // BeOfValues has no Complex overload — assert the flat elements directly.
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(3, 6));
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(-1, 3));
            r.GetAtIndex<Complex>(2).Should().Be(new Complex(0, 3));
            r.GetAtIndex<Complex>(3).Should().Be(new Complex(-1, 1));
        }

        // ---- empty --------------------------------------------------------------------------

        [TestMethod]
        public void Kron_Empty1D()
        {
            // np.kron([], [1,2]) -> shape (0), float64
            np.kron(np.array(new double[] { }), np.array(new[] { 1.0, 2.0 }))
                .Should().BeShaped(0);
        }

        [TestMethod]
        public void Kron_Empty2D()
        {
            // np.kron(zeros((0,3)), ones((2,2))) -> shape (0,6)
            np.kron(np.zeros(new Shape(0, 3)), np.ones(new Shape(2, 2)))
                .Should().BeShaped(0, 6);
        }

        [TestMethod]
        public void Kron_EmptyInnerDim()
        {
            // np.kron(zeros((2,0)), ones((2,2))) -> shape (4,0)
            np.kron(np.zeros(new Shape(2, 0)), np.ones(new Shape(2, 2)))
                .Should().BeShaped(4, 0);
        }

        // ---- non-contiguous inputs (reshape materialises C-order, values must match) --------

        [TestMethod]
        public void Kron_TransposedA()
        {
            // a = arange(6).reshape(2,3).T -> (3,2) F-contiguous; kron with b(2,2).
            var a = np.arange(6).reshape(2, 3).T;                 // [[0,3],[1,4],[2,5]]
            var b = np.array(new[,] { { 1, 1 }, { 1, 1 } });
            // Each element scaled over a 2x2 ones block: row0 = [0,0,3,3], etc.
            np.kron(a, b)
                .Should().BeShaped(6, 4)
                .And.BeOfValues(
                    0, 0, 3, 3, 0, 0, 3, 3,
                    1, 1, 4, 4, 1, 1, 4, 4,
                    2, 2, 5, 5, 2, 2, 5, 5);
        }

        [TestMethod]
        public void Kron_StridedB()
        {
            // b = arange(12).reshape(3,4)[:, ::2] -> (3,2) strided; a = ones(2,2).
            var a = np.array(new[,] { { 1, 1 }, { 1, 1 } });
            var b = np.arange(12).reshape(3, 4)["..., ::2"];      // [[0,2],[4,6],[8,10]]
            np.kron(a, b)
                .Should().BeShaped(6, 4)
                .And.BeOfValues(
                    0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10,
                    0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10);
        }

        // ---- dispatch paths exercised at size (values hand-computable via block structure) ---

        [TestMethod]
        public void Kron_TilePath_ShortRun_LargeOutput()
        {
            // ones(64,64) (x) [[1,2],[3,4]] -> (128,128), run=2, >8192 elems => tile fast-path.
            // Every 2x2 block equals [[1,2],[3,4]], so result[i,j] = block[i%2, j%2].
            var block = np.array(new[,] { { 1d, 2 }, { 3, 4 } });
            var r = np.kron(np.ones(new Shape(64, 64)), block);
            r.Should().BeShaped(128, 128);
            ((double)r[0, 0]).Should().Be(1);
            ((double)r[0, 1]).Should().Be(2);
            ((double)r[1, 0]).Should().Be(3);
            ((double)r[1, 1]).Should().Be(4);
            ((double)r[100, 101]).Should().Be(2);   // even,odd -> block[0,1]
            ((double)r[127, 127]).Should().Be(4);   // odd,odd  -> block[1,1]
        }

        [TestMethod]
        public void Kron_DirectPath_LongRun_LargeOutput()
        {
            // [[1,2],[3,4]] (x) ones(64,64) -> (128,128), run=64 => direct path.
            // result[i,j] = a[i//64, j//64].
            var a = np.array(new[,] { { 1d, 2 }, { 3, 4 } });
            var r = np.kron(a, np.ones(new Shape(64, 64)));
            r.Should().BeShaped(128, 128);
            ((double)r[0, 0]).Should().Be(1);
            ((double)r[0, 64]).Should().Be(2);
            ((double)r[64, 0]).Should().Be(3);
            ((double)r[64, 64]).Should().Be(4);
            ((double)r[127, 127]).Should().Be(4);
        }

        // ---- result is a fresh, writeable, C-contiguous array (NumPy guarantee) --------------

        [TestMethod]
        public void Kron_Result_IsWriteable()
        {
            var r = np.kron(np.eye(2), np.ones(new Shape(2, 2)));
            r[0, 0] = 99;                       // must not throw and must stick
            ((double)r[0, 0]).Should().Be(99);
        }
    }
}
