using System;

namespace NumSharp.Tests.Manipulation
{
    [TestClass]
    public class np_insert_Test : TestClass
    {
        // ---------------- scalar-int obj path (single-index) ----------------

        [TestMethod]
        public void ScalarObj_ScalarValue()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            np.insert(arr, 1, (object)99L)
                .array_equal(np.array(new long[] { 1, 99, 2, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void ScalarObj_ArrayValue()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            np.insert(arr, 1, np.array(new long[] { 99, 98 }))
                .array_equal(np.array(new long[] { 1, 99, 98, 2, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void ScalarObj_AtEnd_Allowed()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            // index == N is the append-at-end position (NumPy allows it for insert,
            // unlike delete which raises).
            np.insert(arr, 4, (object)99L)
                .array_equal(np.array(new long[] { 1, 2, 3, 4, 99 })).Should().BeTrue();
        }

        [TestMethod]
        public void ScalarObj_Negative()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            np.insert(arr, -1, (object)99L)
                .array_equal(np.array(new long[] { 1, 2, 3, 99, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void ScalarObj_OutOfBounds_Throws()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            Action act = () => np.insert(arr, 5, (object)99L);
            act.Should().Throw<IndexOutOfRangeException>();
        }

        // ---------------- multi-index path ----------------

        [TestMethod]
        public void MultiObj_TwoIndices()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            np.insert(arr, new long[] { 1, 3 }, np.array(new long[] { 99, 98 }))
                .array_equal(np.array(new long[] { 1, 99, 2, 3, 98, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void MultiObj_Duplicates_StableOrder()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            // Both inserts at position 1 in input order (stable).
            np.insert(arr, new long[] { 1, 1 }, np.array(new long[] { 99, 98 }))
                .array_equal(np.array(new long[] { 1, 99, 98, 2, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void MultiObj_Unsorted_ReordersValues()
        {
            // Sort indices=[3,0,2] ⇒ order=[1,2,0]; sorted=[0,2,3] with values=[88,77,99].
            // Pieces: []|88|[0,1]|77|[2]|99|[3,4] → [88,0,1,77,2,99,3,4].
            var arr = np.array(new long[] { 0, 1, 2, 3, 4 });
            np.insert(arr, new long[] { 3, 0, 2 }, np.array(new long[] { 99, 88, 77 }))
                .array_equal(np.array(new long[] { 88, 0, 1, 77, 2, 99, 3, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void MultiObj_NegativeIndices()
        {
            var arr = np.array(new long[] { 0, 1, 2, 3, 4 });
            np.insert(arr, new long[] { -1, 0, 2 }, np.array(new long[] { 99, 88, 77 }))
                .array_equal(np.array(new long[] { 88, 0, 1, 77, 2, 3, 99, 4 })).Should().BeTrue();
        }

        // ---------------- slice obj path ----------------

        [TestMethod]
        public void SliceObj_Simple()
        {
            // slice(2, 4) → indices [2, 3] (treated as multi-index).
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            np.insert(arr, new Slice(2, 4), np.array(new long[] { 99, 88 }))
                .array_equal(np.array(new long[] { 1, 2, 99, 3, 88, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void SliceObj_WithStep()
        {
            // slice(0, 6, 2) → indices [0, 2, 4].
            var arr = np.array(new long[] { 0, 1, 2, 3, 4, 5 });
            np.insert(arr, new Slice(0, 6, 2), np.array(new long[] { 99, 88, 77 }))
                .array_equal(np.array(new long[] { 99, 0, 1, 88, 2, 3, 77, 4, 5 })).Should().BeTrue();
        }

        // ---------------- 2-D / axis paths ----------------

        [TestMethod]
        public void TwoD_AxisNone_Flattens()
        {
            // axis=None ravels the 2-D arr first.
            var arr2d = np.arange(6).reshape(3, 2);
            np.insert(arr2d, 1, (object)6)
                .array_equal(np.array(new int[] { 0, 6, 1, 2, 3, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_Axis0_ScalarBroadcast()
        {
            var arr2d = np.arange(6).reshape(3, 2);
            np.insert(arr2d, 1, (object)6, axis: 0)
                .array_equal(np.array(new int[,] { { 0, 1 }, { 6, 6 }, { 2, 3 }, { 4, 5 } })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_Axis1_ScalarBroadcast()
        {
            var arr2d = np.arange(6).reshape(3, 2);
            np.insert(arr2d, 1, (object)6, axis: 1)
                .array_equal(np.array(new int[,] { { 0, 6, 1 }, { 2, 6, 3 }, { 4, 6, 5 } })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_Axis1_1DValues()
        {
            // 1D values [7,8,9] with scalar obj=1 axis=1 ⇒ each row gets vᵢ inserted at col 1.
            var arr2d = np.arange(6).reshape(3, 2);
            np.insert(arr2d, 1, np.array(new int[] { 7, 8, 9 }), axis: 1)
                .array_equal(np.array(new int[,] { { 0, 7, 1 }, { 2, 8, 3 }, { 4, 9, 5 } })).Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_ArrayObj_vs_ScalarObj_DiffersByMoveaxisQuirk()
        {
            // NumPy parity: scalar obj triggers moveaxis(values, 0, axis); array obj does not.
            // With values shape (3,1) and obj=[1]: result is (3,3) — moveaxis NOT applied.
            // With values shape (3,1) and obj=1:   result is (3,5) — moveaxis (3,1)→(1,3),
            //                                       3 values broadcast across all 3 rows.
            var arr2d = np.arange(6).reshape(3, 2);
            var valsCol = np.array(new int[,] { { 7 }, { 8 }, { 9 } });

            var arrayObj = np.insert(arr2d, new int[] { 1 }, valsCol, axis: 1);
            arrayObj.array_equal(np.array(new int[,] { { 0, 7, 1 }, { 2, 8, 3 }, { 4, 9, 5 } }))
                .Should().BeTrue();

            var scalarObj = np.insert(arr2d, 1, valsCol, axis: 1);
            scalarObj.array_equal(np.array(new int[,] { { 0, 7, 8, 9, 1 }, { 2, 7, 8, 9, 3 }, { 4, 7, 8, 9, 5 } }))
                .Should().BeTrue();
        }

        [TestMethod]
        public void TwoD_MultiObj_Axis1_ScalarBroadcast()
        {
            var arr2d = np.arange(12).reshape(3, 4);
            np.insert(arr2d, new long[] { 1, 3 }, (object)99, axis: 1)
                .array_equal(np.array(new int[,]
                {
                    { 0, 99, 1, 2, 99, 3 },
                    { 4, 99, 5, 6, 99, 7 },
                    { 8, 99, 9, 10, 99, 11 }
                })).Should().BeTrue();
        }

        // ---------------- bool obj path ----------------

        [TestMethod]
        public void BoolObj_ConvertsViaFlatnonzero()
        {
            // [T,F,T] → flatnonzero → [0, 2] → multi-index path.
            var arr = np.array(new long[] { 1, 2, 3 });
            np.insert(arr, np.array(new bool[] { true, false, true }), np.array(new long[] { 99, 98 }))
                .array_equal(np.array(new long[] { 99, 1, 2, 98, 3 })).Should().BeTrue();
        }

        // ---------------- type cast ----------------

        [TestMethod]
        public void Values_DowncastToArrDtype()
        {
            // values [7.13, 0.0] cast to int → [7, 0].
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            np.insert(arr, new int[] { 1, 2 }, np.array(new double[] { 7.13, 0.0 }))
                .array_equal(np.array(new long[] { 1, 7, 2, 0, 3, 4 })).Should().BeTrue();
        }

        // ---------------- empty arr / empty values ----------------

        [TestMethod]
        public void EmptyArr_InsertAtZero()
        {
            var arr = np.array(new long[0]);
            np.insert(arr, 0, np.array(new long[] { 1, 2, 3 }))
                .array_equal(np.array(new long[] { 1, 2, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void EmptyObj_NoOp()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 });
            np.insert(arr, new long[0], np.array(new long[0]))
                .array_equal(arr).Should().BeTrue();
        }

        // ---------------- NDArray-obj dispatch ----------------

        [TestMethod]
        public void NDArrayObj_ZeroD_TreatedAsScalar()
        {
            // 0-D ndarray obj → scalar single-index path (moveaxis quirk applies).
            var arr2d = np.arange(6).reshape(3, 2);
            var obj0d = NDArray.Scalar(1L);
            var vals = np.array(new int[,] { { 7 }, { 8 }, { 9 } });
            np.insert(arr2d, obj0d, vals, axis: 1)
                .array_equal(np.array(new int[,]
                {
                    { 0, 7, 8, 9, 1 }, { 2, 7, 8, 9, 3 }, { 4, 7, 8, 9, 5 }
                })).Should().BeTrue();
        }

        [TestMethod]
        public void NDArrayObj_SingleElem1D_NoMoveaxis()
        {
            // size-1 1-D ndarray obj → single-index path with scalarObj=false.
            var arr2d = np.arange(6).reshape(3, 2);
            var obj1d = np.array(new long[] { 1 });
            var vals = np.array(new int[,] { { 7 }, { 8 }, { 9 } });
            np.insert(arr2d, obj1d, vals, axis: 1)
                .array_equal(np.array(new int[,] { { 0, 7, 1 }, { 2, 8, 3 }, { 4, 9, 5 } }))
                .Should().BeTrue();
        }

        // ---------------- dtype coverage ----------------

        [TestMethod]
        public void Dtype_Double()
        {
            np.insert(np.array(new double[] { 1.0, 2.0, 3.0 }), 1, (object)99.5)
                .array_equal(np.array(new double[] { 1.0, 99.5, 2.0, 3.0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Dtype_Byte()
        {
            np.insert(np.array(new byte[] { 1, 2, 3 }), 1, (object)(byte)99)
                .array_equal(np.array(new byte[] { 1, 99, 2, 3 })).Should().BeTrue();
        }

        // ---------------- lifetime: an NDArray reaching the object-value overloads ----------------

        /// <summary>
        ///     Every object-value <c>insert</c> overload (int, long, Slice, int[], long[] and NDArray obj) used to
        ///     dispose its value conversion unconditionally, but <c>ScalarValueAsArray</c> returns an
        ///     <see cref="NDArray"/> value as-is, so the caller's array was released. The overloads are now
        ///     <c>[NDScoped]</c>; the caller's array was built before the scope opened and survives. NumPy 2.4.2:
        ///     <c>np.insert([1, 2, 3], obj, [9])</c> → <c>[1, 9, 2, 3]</c> for obj = <c>1</c>, <c>slice(1, 2)</c>,
        ///     <c>[1]</c> and <c>np.array([1])</c>.
        /// </summary>
        [TestMethod]
        public void ObjectValueOverloads_NDArrayValue_SurvivesTheCall()
        {
            var arr = np.array(new long[] { 1, 2, 3 });
            var expected = np.array(new long[] { 1, 9, 2, 3 });
            var calls = new (string name, Func<NDArray, NDArray> call)[]
            {
                ("int obj", v => np.insert(arr, 1, (object)v)),
                ("long obj", v => np.insert(arr, 1L, (object)v)),
                ("Slice obj", v => np.insert(arr, new Slice(1, 2), (object)v)),
                ("int[] obj", v => np.insert(arr, new[] { 1 }, (object)v)),
                ("long[] obj", v => np.insert(arr, new long[] { 1 }, (object)v)),
                ("NDArray obj", v => np.insert(arr, np.array(new long[] { 1 }), (object)v)),
            };

            foreach (var (name, call) in calls)
            {
                var values = np.array(new long[] { 9 });
                var r = call(values);

                LeakGuards.StillUsable(values, $" — insert({name}, object) must not dispose the caller's NDArray");
                r.array_equal(expected).Should().BeTrue(name);
                // A second call reads the same live array, not a recycled buffer.
                call(values).array_equal(expected).Should().BeTrue(name + " (second call)");
            }
            LeakGuards.StillUsable(arr);
        }

        /// <summary>
        ///     The 2-D single-index path hands a dtype-matched, already slot-shaped value (here <c>(1, 2)</c>
        ///     int64 into a <c>(2, 2)</c> int64 array) straight to the concatenation: the coercion helpers no longer
        ///     wrap it in a second NDArray. The result must still be a copy. NumPy 2.4.2:
        ///     <c>np.insert([[1, 2], [3, 4]], 1, [[9, 8]], axis=0)</c> → <c>[[1, 2], [9, 8], [3, 4]]</c>.
        /// </summary>
        [TestMethod]
        public void SingleIndex_SlotShapedValues_AreCopiedNotAliased()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4 }).reshape(2, 2);
            var values = np.array(new long[] { 9, 8 }).reshape(1, 2);

            var r = np.insert(arr, 1L, values, axis: 0);

            r.array_equal(np.array(new long[] { 1, 2, 9, 8, 3, 4 }).reshape(3, 2)).Should().BeTrue();
            LeakGuards.StillUsable(values);
            // The result owns a copy: writing the input afterwards does not reach it.
            values[0, 0] = 0L;
            ((long)r[1, 0]).Should().Be(9L);
        }
    }
}
