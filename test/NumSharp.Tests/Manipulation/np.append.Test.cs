using System;

namespace NumSharp.Tests.Manipulation
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

        /// <summary>A C# long literal is NumPy's Python int. NumPy 2.4.2: <c>np.append(int64[1, 2, 3], 4)</c> is int64
        ///     <c>[1, 2, 3, 4]</c>.</summary>
        [TestMethod]
        public void Scalar_Value()
        {
            var r = np.append(np.array(new long[] { 1, 2, 3 }), 4L);
            r.dtype.Should().Be<long>();
            r.array_equal(np.array(new long[] { 1, 2, 3, 4 })).Should().BeTrue();
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

        /// <summary>A uint8 array value keeps uint8 (a C# byte LITERAL is a Python int, whose array is int64:
        ///     <c>NumPyKeywordParity.Batch12</c>). NumPy 2.4.2: <c>np.append(uint8[1, 2], np.array([3], np.uint8))</c> is
        ///     uint8 <c>[1, 2, 3]</c>.</summary>
        [TestMethod]
        public void Dtype_Byte()
        {
            var r = np.append(np.array(new byte[] { 1, 2 }), np.array(new byte[] { 3 }));
            r.dtype.Should().Be<byte>();
            r.array_equal(np.array(new byte[] { 1, 2, 3 })).Should().BeTrue();
        }

        // ---------------- lifetime: the caller's arrays survive the call ----------------

        /// <summary>
        ///     <c>NDArray.append</c> is <c>[NDScoped]</c> and converts a C# literal to NumPy's array of the Python number
        ///     (<see cref="NDArray"/>'s <c>PythonScalarAsArray</c>); an array argument comes back from that conversion
        ///     unchanged, built before the scope opened, so the scope never disposes it. (The removed <c>object</c>
        ///     overload once disposed its conversion unconditionally, which released the CALLER'S array.) NumPy 2.4.2:
        ///     <c>np.append([1, 2, 3], [4, 5])</c> → <c>[1, 2, 3, 4, 5]</c>.
        /// </summary>
        [TestMethod]
        public void NDArrayValue_SurvivesTheCall()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var values = np.array(new long[] { 4, 5 });

            var r = np.append(arr, values);

            LeakGuards.StillUsable(values, " — np.append must not dispose the caller's NDArray");
            LeakGuards.StillUsable(arr);
            r.array_equal(np.array(new long[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
            // A second call reads the same live array, not a recycled buffer.
            np.append(arr, values).array_equal(r).Should().BeTrue();
            values.ToArray<long>().Should().Equal(4L, 5L);
        }

        /// <summary>
        ///     The axis path of the same call. NumPy 2.4.2:
        ///     <c>np.append([[1, 2], [3, 4]], [[5, 6]], axis=0)</c> → <c>[[1, 2], [3, 4], [5, 6]]</c>.
        /// </summary>
        [TestMethod]
        public void NDArrayValue_Axis0_SurvivesTheCall()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 }).reshape(2, 2);
            var values = np.array(new long[] { 5, 6 }).reshape(1, 2);

            var r = np.append(arr, values, axis: 0);

            LeakGuards.StillUsable(values, " — np.append must not dispose the caller's NDArray");
            r.array_equal(np.array(new long[] { 1, 2, 3, 4, 5, 6 }).reshape(3, 2)).Should().BeTrue();
        }
    }
}
