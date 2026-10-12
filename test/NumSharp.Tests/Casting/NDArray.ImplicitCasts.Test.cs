using System;
using System.Linq;

namespace NumSharp.Tests.Casting
{
    [TestClass]
    public class ImplicitCastTester
    {
        [TestMethod]
        public void ConvertFromJagged()
        {
            double[][] a = new double[3][];
            for (int idx = 0; idx < a.Length; idx++)
                a[idx] = new double[2];


            for (int idx = 0; idx < 3; idx++)
            for (int jdx = 0; jdx < 2; jdx++)
                a[idx][jdx] = 10 * idx + jdx;

            NDArray b = a;

            var c = b.MakeGeneric<double>();

            for (int idx = 0; idx < 3; idx++)
            for (int jdx = 0; jdx < 2; jdx++)
                Assert.IsTrue(c[idx, jdx] == a[idx][jdx]);
        }

        [TestMethod]
        public void FromDotNetVector()
        {
            NDArray nd = new double[] {1, 2, 3, 4};

            Assert.IsTrue(((double)nd[0]) == 1);
            Assert.IsTrue(((double)nd[1]) == 2);
            Assert.IsTrue(((double)nd[2]) == 3);
            Assert.IsTrue(((double)nd[3]) == 4);
        }

        [TestMethod]
        public void FromDotNetMatrix()
        {
            NDArray nd = new double[,] {{1, 2, 3}, {4, 5, 6}};

            var doubleMatr = new double[,] {{1, 2, 3}, {4, 5, 6}};

            for (int idx = 0; idx < doubleMatr.GetLength(0); idx++)
            for (int jdx = 0; jdx < doubleMatr.GetLength(1); jdx++)
                Assert.IsTrue((double)nd[idx, jdx] == doubleMatr[idx, jdx]);
        }

        [TestMethod]
        public void FromAndToDotNetMatrix()
        {
            NDArray nd = new double[,] {{1, 2, 3}, {4, 5, 6}};

            double[,] nd_ = new double[,] {{1, 2, 3}, {4, 5, 6}};

            Array arr = (Array)nd;

            double[,] doubleMatr = (double[,])arr;

            for (int idx = 0; idx < doubleMatr.GetLength(0); idx++)
            {
                for (int jdx = 0; jdx < doubleMatr.GetLength(1); jdx++)
                {
                    Assert.IsTrue((double)nd[idx, jdx] == doubleMatr[idx, jdx]);
                    Assert.IsTrue(nd_[idx, jdx] == doubleMatr[idx, jdx]);
                }
            }
        }

        /// <summary>
        ///     A C# string converts to NumPy's <c>np.array(str)</c>: a 0-d <c>U</c> array holding the text, never the numbers
        ///     the text spells (probed against NumPy 2.4.2: <c>np.array("[1,2,3;4,5,6]")</c> is <c>array('[1,2,3;4,5,6]',
        ///     dtype='&lt;U13')</c>). The MATLAB-style numeric parse this conversion once ran was removed (the owner's
        ///     decision D6 of the strings phase-1 plan).
        /// </summary>
        [TestMethod]
        public void StringLiteral_IsA0dStrArray_NotParsedNumbers()
        {
            NDArray nd = "[1,2,3;4,5,6]";
            nd.ndim.Should().Be(0);
            nd.dtype.str.Should().Be("<U13");
            ((string)nd.item()).Should().Be("[1,2,3;4,5,6]");
        }

        /// <summary>
        ///     The numbers of a text are reached by an explicit cast, which parses each element with <c>float()</c>, as
        ///     NumPy's <c>astype</c> does: a 0-d text that is not one number raises NumPy's ValueError, a text array of numbers
        ///     parses element by element.
        /// </summary>
        [TestMethod]
        public void StringLiteral_ParsesOnlyThroughACast()
        {
            NDArray nd = "[3,1,1,2]";
            nd.dtype.str.Should().Be("<U9");
            Action whole = () => nd.astype(np.float64);
            whole.Should().Throw<ValueError>().WithMessage("could not convert string to float: np.str_('[3,1,1,2]')");
            np.array(new[] { "3", "1", "1", "2" }).astype(np.float64).ToArray<double>().Should().Equal(3.0, 1.0, 1.0, 2.0);
        }
    }
}
