using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using System;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_append_Test : TestClass
    {
        // ---------------- axis=None (ravel both) ----------------

        [TestMethod]
        public void OneD_AxisNone()
        {
            np.append(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 4, 5, 6 }))
                .array_equal(np.array(new long[] { 1, 2, 3, 4, 5, 6 })).Should().BeTrue();
        }

        [TestMethod]
        public void Scalar_Value()
        {
            np.append(np.array(new long[] { 1, 2, 3 }), (object)4L)
                .array_equal(np.array(new long[] { 1, 2, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_AxisNone_FlattensBoth()
        {
            var arr = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var vals = np.array(new long[,] { { 7, 8, 9 } });
            np.append(arr, vals)
                .array_equal(np.array(new long[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 })).Should().BeTrue();
        }

        // ---------------- explicit axis ----------------

        [TestMethod]
        public void TwoD_Axis0()
        {
            var arr = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var vals = np.array(new long[,] { { 7, 8, 9 } });
            np.append(arr, vals, axis: 0)
                .array_equal(np.array(new long[,]
                    { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_AxisNegative()
        {
            var arr = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            var vals = np.array(new long[,] { { 5, 6 }, { 7, 8 } });
            np.append(arr, vals, axis: -1)
                .array_equal(np.array(new long[,]
                    { { 1, 2, 5, 6 }, { 3, 4, 7, 8 } })).Should().BeTrue();
        }

        // ---------------- shape mismatch ----------------

        [TestMethod]
        public void AxisGiven_NdimMismatch_Throws()
        {
            var arr = np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            // 1D values with axis= ⇒ concatenate raises ndim mismatch.
            Action act = () => np.append(arr, np.array(new long[] { 7, 8, 9 }), axis: 0);
            act.Should().Throw<IncorrectShapeException>();
        }

        // ---------------- empty / dtype promotion ----------------

        [TestMethod]
        public void Empty_Values()
        {
            np.append(np.array(new long[] { 1, 2 }), np.array(new long[0]))
                .array_equal(np.array(new long[] { 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Empty_Arr()
        {
            np.append(np.array(new long[0]), np.array(new long[] { 1, 2, 3 }))
                .array_equal(np.array(new long[] { 1, 2, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void DtypePromotion_IntFloat()
        {
            var r = np.append(np.array(new long[] { 1, 2, 3 }), np.array(new double[] { 4.5 }));
            r.dtype.Should().Be<double>();
            r.array_equal(np.array(new double[] { 1.0, 2.0, 3.0, 4.5 })).Should().BeTrue();
        }

        // ---------------- dtype coverage ----------------

        [TestMethod]
        public void Dtype_Double()
        {
            np.append(np.array(new double[] { 1.0, 2.0 }), np.array(new double[] { 3.0, 4.0 }))
                .array_equal(np.array(new double[] { 1.0, 2.0, 3.0, 4.0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Dtype_Byte()
        {
            np.append(np.array(new byte[] { 1, 2 }), (object)(byte)3)
                .array_equal(np.array(new byte[] { 1, 2, 3 })).Should().BeTrue();
        }
    }
}
