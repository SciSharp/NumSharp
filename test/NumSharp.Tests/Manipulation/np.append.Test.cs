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

        // ---------------- lifetime: an NDArray reaching the object overload ----------------

        /// <summary>
        ///     An <see cref="NDArray"/> whose static type is <see cref="object"/> binds the object overload, and
        ///     <c>np.asanyarray</c> hands it back unchanged. The hand-written overload disposed that conversion
        ///     unconditionally, which released the CALLER'S array: after <c>np.append(a, (object)v)</c>,
        ///     <c>v</c> was disposed and its buffer back in the pool (a second call then read recycled memory).
        ///     The overload is now <c>[NDScoped]</c>, and <c>v</c> was built before its scope opened, so the scope
        ///     never touches it. NumPy 2.4.2: <c>np.append([1, 2, 3], [4, 5])</c> → <c>[1, 2, 3, 4, 5]</c>.
        /// </summary>
        [TestMethod]
        public void ObjectOverload_NDArrayValue_SurvivesTheCall()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var values = np.array(new long[] { 4, 5 });

            var r = np.append(arr, (object)values);

            LeakGuards.StillUsable(values, " — the object overload must not dispose the caller's NDArray");
            LeakGuards.StillUsable(arr);
            r.array_equal(np.array(new long[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
            // A second call reads the same live array, not a recycled buffer.
            np.append(arr, (object)values).array_equal(r).Should().BeTrue();
            values.ToArray<long>().Should().Equal(4L, 5L);
        }

        /// <summary>
        ///     The axis path of the same overload. NumPy 2.4.2:
        ///     <c>np.append([[1, 2], [3, 4]], [[5, 6]], axis=0)</c> → <c>[[1, 2], [3, 4], [5, 6]]</c>.
        /// </summary>
        [TestMethod]
        public void ObjectOverload_NDArrayValue_Axis0_SurvivesTheCall()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 }).reshape(2, 2);
            var values = np.array(new long[] { 5, 6 }).reshape(1, 2);

            var r = np.append(arr, (object)values, axis: 0);

            LeakGuards.StillUsable(values, " — the object overload must not dispose the caller's NDArray");
            r.array_equal(np.array(new long[] { 1, 2, 3, 4, 5, 6 }).reshape(3, 2)).Should().BeTrue();
        }
    }
}
