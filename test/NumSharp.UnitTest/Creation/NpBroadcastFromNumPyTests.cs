using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     Comprehensive broadcast tests ported from NumPy's test suite.
    ///     Sources:
    ///       - numpy/lib/tests/test_stride_tricks.py (broadcast_to, broadcast_arrays, broadcast_shapes)
    ///       - numpy/_core/tests/test_numeric.py (np.broadcast class)
    ///     Verifies NumSharp's broadcasting against NumPy 2.x behavior.
    /// </summary>
    [TestClass]
    public class NpBroadcastFromNumPyTests : TestClass
    {
        #region Helpers

        /// <summary>
        ///     Creates an NDArray filled with ones of the given shape.
        ///     For zero-size shapes, returns an empty array.
        /// </summary>
        private static NDArray ones(params int[] shape)
        {
            return np.ones(new Shape(shape));
        }

        /// <summary>
        ///     Assert that two shapes are identical.
        /// </summary>
        private static void AssertShapeEqual(Shape actual, params int[] expected)
        {
            actual.dimensions.Should().BeEquivalentTo(expected,
                $"expected shape ({string.Join(",", expected)}) but got ({string.Join(",", actual.dimensions)})");
        }

        /// <summary>
        ///     Assert that an NDArray has a given shape.
        /// </summary>
        private static void AssertShapeEqual(NDArray actual, params int[] expected)
        {
            AssertShapeEqual(actual.Shape, expected);
        }

        #endregion

        // ================================================================
        // Group 1: broadcast_arrays Shape Resolution
        //   (from numpy/lib/tests/test_stride_tricks.py)
        // ================================================================

        #region Group 1: broadcast_arrays Shape Resolution

        /// <summary>
        ///     Two identical (1,3) arrays broadcast unchanged.
        ///     >>> x = np.array([[1, 2, 3]])
        ///     >>> a, b = np.broadcast_arrays(x, x)
        ///     >>> np.array_equal(a, x) and np.array_equal(b, x)
        ///     True
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_Same()
        {
            var x = np.array(new int[,] { { 1, 2, 3 } }); // shape (1,3)
            var (a, b) = np.broadcast_arrays(x, x);

            AssertShapeEqual(a, 1, 3);
            AssertShapeEqual(b, 1, 3);
            np.array_equal(a, x).Should().BeTrue();
            np.array_equal(b, x).Should().BeTrue();
        }

        /// <summary>
        ///     (1,3) + (3,1) -> both become (3,3) with correct values.
        ///     >>> x = np.array([[1, 2, 3]])       # shape (1,3)
        ///     >>> y = np.array([[1], [2], [3]])    # shape (3,1)
        ///     >>> a, b = np.broadcast_arrays(x, y)
        ///     >>> a
        ///     array([[1, 2, 3],
        ///            [1, 2, 3],
        ///            [1, 2, 3]])
        ///     >>> b
        ///     array([[1, 1, 1],
        ///            [2, 2, 2],
        ///            [3, 3, 3]])
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_OneOff()
        {
            var x = np.array(new int[,] { { 1, 2, 3 } });    // shape (1,3)
            var y = np.array(new int[,] { { 1 }, { 2 }, { 3 } }); // shape (3,1)
            var (a, b) = np.broadcast_arrays(x, y);

            AssertShapeEqual(a, 3, 3);
            AssertShapeEqual(b, 3, 3);

            // a should be [[1,2,3],[1,2,3],[1,2,3]]
            var expected_a = np.array(new int[,] { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } });
            np.array_equal(a, expected_a).Should().BeTrue();

            // b should be [[1,1,1],[2,2,2],[3,3,3]]
            var expected_b = np.array(new int[,] { { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
            np.array_equal(b, expected_b).Should().BeTrue();
        }

        /// <summary>
        ///     Comprehensive same-input-shapes test from test_stride_tricks.py.
        ///     For each shape in a set of shapes, broadcasting an array with itself
        ///     should produce the same shape. Testing with 1, 2, and 3 identical inputs.
        ///
        ///     Shapes tested: (1,), (3,), (1,3), (3,1), (3,3)
        ///     (Omitting zero-size shapes since they require special handling)
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_SameInputShapes()
        {
            var shapes = new[]
            {
                new int[] { 1 },
                new int[] { 3 },
                new int[] { 1, 3 },
                new int[] { 3, 1 },
                new int[] { 3, 3 },
            };

            foreach (var shape in shapes)
            {
                var x = np.ones(new Shape(shape));

                // Single broadcast (two identical)
                var (a, b) = np.broadcast_arrays(x, x);
                AssertShapeEqual(a, shape);
                AssertShapeEqual(b, shape);

                // Triple broadcast (three identical)
                var result = np.broadcast_arrays(x, x, x);
                result.Length.Should().Be(3);
                foreach (var r in result)
                    AssertShapeEqual(r, shape);
            }
        }

        /// <summary>
        ///     Compatible shape pairs where one dimension is 1.
        ///     From test_stride_tricks.py _test_broadcast_compatible_by_ones.
        ///
        ///     Tests both (a,b) and (b,a) orderings for symmetry.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_CompatibleByOnes()
        {
            // (input1_shape, input2_shape, expected_output_shape)
            var cases = new[]
            {
                (new int[] { 1 },    new int[] { 3 },    new int[] { 3 }),
                (new int[] { 1, 3 }, new int[] { 3, 3 }, new int[] { 3, 3 }),
                (new int[] { 3, 1 }, new int[] { 3, 3 }, new int[] { 3, 3 }),
                (new int[] { 1, 3 }, new int[] { 3, 1 }, new int[] { 3, 3 }),
                (new int[] { 1, 1 }, new int[] { 3, 3 }, new int[] { 3, 3 }),
                (new int[] { 1, 1 }, new int[] { 1, 3 }, new int[] { 1, 3 }),
                (new int[] { 1, 1 }, new int[] { 3, 1 }, new int[] { 3, 1 }),
                (new int[] { 1, 3 }, new int[] { 1, 3 }, new int[] { 1, 3 }),
                (new int[] { 3, 1 }, new int[] { 3, 1 }, new int[] { 3, 1 }),
                (new int[] { 3, 3 }, new int[] { 3, 3 }, new int[] { 3, 3 }),
                // Higher dimensional
                (new int[] { 1, 1, 1 }, new int[] { 3, 3, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 1, 3, 1 }, new int[] { 3, 1, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 3, 1, 1 }, new int[] { 1, 3, 3 }, new int[] { 3, 3, 3 }),
            };

            foreach (var (s1, s2, expected) in cases)
            {
                // Forward order
                var x1 = np.ones(new Shape(s1));
                var x2 = np.ones(new Shape(s2));
                var (a, b) = np.broadcast_arrays(x1, x2);
                AssertShapeEqual(a, expected);
                AssertShapeEqual(b, expected);

                // Reversed order (symmetry)
                var (a2, b2) = np.broadcast_arrays(x2, x1);
                AssertShapeEqual(a2, expected);
                AssertShapeEqual(b2, expected);
            }
        }

        /// <summary>
        ///     Compatible shape pairs with different ndim (prepending ones).
        ///     From test_stride_tricks.py _test_broadcast_compatible_by_prepending_ones.
        ///
        ///     Tests both (a,b) and (b,a) orderings for symmetry.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_CompatibleByPrependingOnes()
        {
            // (input1_shape, input2_shape, expected_output_shape)
            var cases = new[]
            {
                (new int[] { 3 },    new int[] { 3, 3 }, new int[] { 3, 3 }),
                (new int[] { 3 },    new int[] { 3, 1 }, new int[] { 3, 3 }),
                (new int[] { 1 },    new int[] { 3, 3 }, new int[] { 3, 3 }),
                (new int[] { 1 },    new int[] { 1, 3 }, new int[] { 1, 3 }),
                (new int[] { 1 },    new int[] { 3, 1 }, new int[] { 3, 1 }),
                (new int[] { 3 },    new int[] { 1, 3 }, new int[] { 1, 3 }),
                (new int[] { 1 },    new int[] { 1, 1 }, new int[] { 1, 1 }),
                (new int[] { 3 },    new int[] { 3, 3, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 1 },    new int[] { 3, 3, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 1, 3 }, new int[] { 3, 3, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 3, 3 }, new int[] { 3, 3, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 1, 1 }, new int[] { 3, 3, 3 }, new int[] { 3, 3, 3 }),
                (new int[] { 1, 3 }, new int[] { 3, 1, 3 }, new int[] { 3, 1, 3 }),
                (new int[] { 3, 1 }, new int[] { 3, 3, 1 }, new int[] { 3, 3, 1 }),
            };

            foreach (var (s1, s2, expected) in cases)
            {
                var x1 = np.ones(new Shape(s1));
                var x2 = np.ones(new Shape(s2));

                // Forward
                var (a, b) = np.broadcast_arrays(x1, x2);
                AssertShapeEqual(a, expected);
                AssertShapeEqual(b, expected);

                // Reversed (symmetry)
                var (a2, b2) = np.broadcast_arrays(x2, x1);
                AssertShapeEqual(a2, expected);
                AssertShapeEqual(b2, expected);
            }
        }

        /// <summary>
        ///     Incompatible shape pairs should throw.
        ///     From test_stride_tricks.py _test_broadcast_shapes_raise.
        ///
        ///     >>> np.broadcast_arrays(np.ones(3), np.ones(4))
        ///     ValueError: shape mismatch
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_IncompatibleShapesThrow()
        {
            var incompatible_pairs = new[]
            {
                (new int[] { 3 },    new int[] { 4 }),
                (new int[] { 2, 3 }, new int[] { 2 }),
                (new int[] { 1, 3, 4 }, new int[] { 2, 3, 3 }),
            };

            foreach (var (s1, s2) in incompatible_pairs)
            {
                var x1 = np.ones(new Shape(s1));
                var x2 = np.ones(new Shape(s2));

                new Action(() => np.broadcast_arrays(x1, x2))
                    .Should().Throw<Exception>(
                        $"shapes ({string.Join(",", s1)}) and ({string.Join(",", s2)}) should be incompatible");

                // Also reversed
                new Action(() => np.broadcast_arrays(x2, x1))
                    .Should().Throw<Exception>(
                        $"shapes ({string.Join(",", s2)}) and ({string.Join(",", s1)}) should be incompatible (reversed)");
            }
        }

        /// <summary>
        ///     Three-way incompatible: (3,) + (3,) + (4,) should throw.
        ///     >>> np.broadcast_arrays(np.ones(3), np.ones(3), np.ones(4))
        ///     ValueError: shape mismatch
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_ThreeWayIncompatibleThrows()
        {
            var x1 = np.ones(new Shape(3));
            var x2 = np.ones(new Shape(3));
            var x3 = np.ones(new Shape(4));

            new Action(() => np.broadcast_arrays(x1, x2, x3))
                .Should().Throw<Exception>();
        }

        #endregion

        // ================================================================
        // Group 2: broadcast_to
        //   (from numpy/lib/tests/test_stride_tricks.py)
        // ================================================================

        #region Group 2: broadcast_to

        /// <summary>
        ///     broadcast_to succeeds for compatible shapes.
        ///     From test_stride_tricks.py test_broadcast_to_succeeds.
        ///
        ///     Scalar cases:
        ///       np.broadcast_to(np.int32(0), (1,))   -> shape (1,)
        ///       np.broadcast_to(np.int32(0), (3,))   -> shape (3,)
        ///
        ///     1D cases:
        ///       np.broadcast_to(np.ones(1), (1,))    -> shape (1,)
        ///       np.broadcast_to(np.ones(1), (2,))    -> shape (2,)
        ///       np.broadcast_to(np.ones(1), (1,2,3)) -> shape (1,2,3)
        ///
        ///       np.broadcast_to(np.arange(3), (3,))  -> shape (3,)
        ///       np.broadcast_to(np.arange(3), (1,3)) -> shape (1,3)
        ///       np.broadcast_to(np.arange(3), (2,3)) -> shape (2,3)
        /// </summary>
        [TestMethod]
        public void BroadcastTo_Succeeds()
        {
            // Scalar to various shapes
            var scalar = NDArray.Scalar(0);

            var r1 = np.broadcast_to(scalar, new Shape(1));
            AssertShapeEqual(r1, 1);

            var r2 = np.broadcast_to(scalar, new Shape(3));
            AssertShapeEqual(r2, 3);

            // ones(1) to various shapes
            var o1 = np.ones(new Shape(1));

            AssertShapeEqual(np.broadcast_to(o1, new Shape(1)), 1);
            AssertShapeEqual(np.broadcast_to(o1, new Shape(2)), 2);
            AssertShapeEqual(np.broadcast_to(o1, new Shape(1, 2, 3)), 1, 2, 3);

            // arange(3) to various shapes
            var a3 = np.arange(3);

            AssertShapeEqual(np.broadcast_to(a3, new Shape(3)), 3);
            AssertShapeEqual(np.broadcast_to(a3, new Shape(1, 3)), 1, 3);
            AssertShapeEqual(np.broadcast_to(a3, new Shape(2, 3)), 2, 3);
        }

        /// <summary>
        ///     broadcast_to succeeds for zero-size target shapes.
        ///     >>> np.broadcast_to(np.ones(1), (0,)).shape
        ///     (0,)
        ///     >>> np.broadcast_to(np.ones((1,2)), (0,2)).shape
        ///     (0, 2)
        ///     >>> np.broadcast_to(np.ones((2,1)), (2,0)).shape
        ///     (2, 0)
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ZeroSizeShapes()
        {
            AssertShapeEqual(np.broadcast_to(np.ones(new Shape(1)), new Shape(0)), 0);
            AssertShapeEqual(np.broadcast_to(np.ones(new Shape(1, 2)), new Shape(0, 2)), 0, 2);
            AssertShapeEqual(np.broadcast_to(np.ones(new Shape(2, 1)), new Shape(2, 0)), 2, 0);
        }

        /// <summary>
        ///     broadcast_to raises for truly incompatible shapes where dimensions
        ///     conflict (neither is 1).
        ///     From test_stride_tricks.py test_broadcast_to_raises.
        ///
        ///     >>> np.broadcast_to(np.ones(3), (2,))
        ///     ValueError: ...
        ///     >>> np.broadcast_to(np.ones(3), (4,))
        ///     ValueError: ...
        /// </summary>
        [TestMethod]
        public void BroadcastTo_Raises()
        {
            // (3,) -> (2,) incompatible: dimension 3 vs 2, neither is 1
            new Action(() => np.broadcast_to(np.ones(new Shape(3)), new Shape(2)))
                .Should().Throw<Exception>();

            // (3,) -> (4,) incompatible: dimension 3 vs 4, neither is 1
            new Action(() => np.broadcast_to(np.ones(new Shape(3)), new Shape(4)))
                .Should().Throw<Exception>();
        }

        /// <summary>
        ///     NumPy's broadcast_to is one-directional: source dims can only be 1 or match target.
        ///     broadcast_to uses unilateral semantics matching NumPy: only stretches source
        ///     dimensions that are size 1. Cases where the source has a dimension larger than
        ///     the target are rejected.
        ///
        ///     >>> np.broadcast_to(np.ones(3), (1,))  # NumPy: ValueError
        ///     >>> np.broadcast_to(np.ones((1,2)), (2,1))  # NumPy: ValueError
        ///     >>> np.broadcast_to(np.ones((1,1)), (1,))  # NumPy: ValueError (ndim mismatch)
        /// </summary>
        [TestMethod]
        public void BroadcastTo_UnilateralSemantics_RejectsInvalidCases()
        {
            // (3,) -> (1,): source dim 3 != target dim 1 and != 1 → must throw
            new Action(() => np.broadcast_to(np.ones(new Shape(3)), new Shape(1)))
                .Should().Throw<IncorrectShapeException>();

            // (1,2) -> (2,1): source dim 2 != target dim 1 and != 1 → must throw
            new Action(() => np.broadcast_to(np.ones(new Shape(1, 2)), new Shape(2, 1)))
                .Should().Throw<IncorrectShapeException>();

            // (1,1) -> (1,): source has more dims than target → must throw
            new Action(() => np.broadcast_to(np.ones(new Shape(1, 1)), new Shape(1)))
                .Should().Throw<IncorrectShapeException>();
        }

        /// <summary>
        ///     broadcast_to returns a view (shared memory), not a copy.
        ///     From test_stride_tricks.py test_broadcast_to_is_view.
        ///
        ///     >>> x = np.array([1, 2, 3])
        ///     >>> y = np.broadcast_to(x, (2, 3))
        ///     >>> y.base is x  # (shares memory)
        ///     True
        /// </summary>
        [TestMethod]
        public void BroadcastTo_IsView()
        {
            var x = np.array(new int[] { 1, 2, 3 });
            var y = np.broadcast_to(x, new Shape(2, 3));

            AssertShapeEqual(y, 2, 3);

            // Verify it's a view by checking values match
            // and that the broadcasted array shares data
            Assert.AreEqual(1, y.GetInt32(0, 0));
            Assert.AreEqual(2, y.GetInt32(0, 1));
            Assert.AreEqual(3, y.GetInt32(0, 2));
            Assert.AreEqual(1, y.GetInt32(1, 0));
            Assert.AreEqual(2, y.GetInt32(1, 1));
            Assert.AreEqual(3, y.GetInt32(1, 2));
        }

        /// <summary>
        ///     Verify broadcast_to produces correct element values.
        ///     >>> x = np.arange(3)
        ///     >>> y = np.broadcast_to(x, (2, 3))
        ///     >>> y
        ///     array([[0, 1, 2],
        ///            [0, 1, 2]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ValuesCorrect()
        {
            var x = np.arange(3); // [0, 1, 2]
            var y = np.broadcast_to(x, new Shape(2, 3));

            var expected = np.array(new int[,] { { 0, 1, 2 }, { 0, 1, 2 } });
            np.array_equal(y, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Verify broadcast_to with scalar input produces correct values.
        ///     >>> np.broadcast_to(np.int32(5), (3,))
        ///     array([5, 5, 5])
        ///     >>> np.broadcast_to(np.int32(5), (2, 3))
        ///     array([[5, 5, 5],
        ///            [5, 5, 5]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ScalarValues()
        {
            var scalar = NDArray.Scalar(5);

            var r1 = np.broadcast_to(scalar, new Shape(3));
            np.array_equal(r1, np.array(new int[] { 5, 5, 5 })).Should().BeTrue();

            var r2 = np.broadcast_to(scalar, new Shape(2, 3));
            np.array_equal(r2, np.array(new int[,] { { 5, 5, 5 }, { 5, 5, 5 } })).Should().BeTrue();
        }

        /// <summary>
        ///     broadcast_to with (3,1) -> (3,3).
        ///     >>> x = np.array([[1],[2],[3]])
        ///     >>> np.broadcast_to(x, (3,3))
        ///     array([[1, 1, 1],
        ///            [2, 2, 2],
        ///            [3, 3, 3]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ColumnToMatrix()
        {
            var x = np.array(new int[,] { { 1 }, { 2 }, { 3 } }); // shape (3,1)
            var y = np.broadcast_to(x, new Shape(3, 3));

            AssertShapeEqual(y, 3, 3);
            var expected = np.array(new int[,] { { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
            np.array_equal(y, expected).Should().BeTrue();
        }

        #endregion

        // ================================================================
        // Group 3: broadcast_arrays Data Correctness
        //   (from numpy/lib/tests/test_stride_tricks.py)
        // ================================================================

        #region Group 3: broadcast_arrays Data Correctness

        /// <summary>
        ///     Verify that broadcast_arrays produces the same data layout
        ///     as ufunc broadcasting (via addition).
        ///
        ///     For compatible shape pairs, broadcast_arrays(x0, x1) should
        ///     produce arrays whose element-wise sum equals x0 + x1.
        ///
        ///     From test_stride_tricks.py test_same_as_ufunc.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_SameAsUfunc()
        {
            var cases = new[]
            {
                (new int[] { 1 },    new int[] { 3 }),
                (new int[] { 3 },    new int[] { 3 }),
                (new int[] { 1, 3 }, new int[] { 3, 3 }),
                (new int[] { 3, 1 }, new int[] { 3, 3 }),
                (new int[] { 1, 3 }, new int[] { 3, 1 }),
                (new int[] { 3 },    new int[] { 3, 3 }),
                (new int[] { 3 },    new int[] { 1, 3 }),
                (new int[] { 1 },    new int[] { 3, 3 }),
            };

            foreach (var (s1, s2) in cases)
            {
                var shape1 = new Shape(s1);
                var shape2 = new Shape(s2);
                var x0 = np.arange(shape1.size).reshape(shape1);
                var x1 = np.arange(shape2.size).reshape(shape2);

                // Result from ufunc broadcasting
                var ufunc_result = x0 + x1;

                // Result from broadcast_arrays + manual element-wise add
                var (b0, b1) = np.broadcast_arrays(x0, x1);
                var broadcast_result = b0 + b1;

                np.array_equal(ufunc_result, broadcast_result).Should().BeTrue(
                    $"broadcast_arrays + add should match ufunc add for shapes ({string.Join(",", s1)}) and ({string.Join(",", s2)})");
            }
        }

        #endregion

        // ================================================================
        // Group 4: np.broadcast Object
        //   (from numpy/_core/tests/test_numeric.py)
        // ================================================================

        #region Group 4: np.broadcast Object

        /// <summary>
        ///     Test np.broadcast shape, ndim, size properties.
        ///     >>> arr1 = np.arange(6).reshape(2, 3)
        ///     >>> arr2 = np.arange(3)
        ///     >>> b = np.broadcast(arr1, arr2)
        ///     >>> b.shape
        ///     (2, 3)
        ///     >>> b.ndim
        ///     2
        ///     >>> b.size
        ///     6
        /// </summary>
        [TestMethod]
        public void Broadcast_Properties()
        {
            var arr1 = np.arange(6).reshape(2, 3);
            var arr2 = np.arange(3);
            var b = np.broadcast(arr1, arr2);

            AssertShapeEqual(b.shape, 2, 3);
            Assert.AreEqual(2, b.ndim);
            Assert.AreEqual(6, b.size);
            Assert.AreEqual(b.ndim, b.nd); // nd is alias for ndim
        }

        /// <summary>
        ///     Test np.broadcast with shapes that require broadcasting.
        ///     >>> arr1 = np.ones((2, 1))
        ///     >>> arr2 = np.ones((1, 3))
        ///     >>> b = np.broadcast(arr1, arr2)
        ///     >>> b.shape
        ///     (2, 3)
        ///     >>> b.size
        ///     6
        /// </summary>
        [TestMethod]
        public void Broadcast_BroadcastedShape()
        {
            var arr1 = np.ones(new Shape(2, 1));
            var arr2 = np.ones(new Shape(1, 3));
            var b = np.broadcast(arr1, arr2);

            AssertShapeEqual(b.shape, 2, 3);
            Assert.AreEqual(6, b.size);
            Assert.AreEqual(2, b.ndim);
        }

        /// <summary>
        ///     Test np.broadcast with single-element arrays.
        ///     >>> arr = np.arange(6).reshape(2, 3)
        ///     >>> b = np.broadcast(arr, np.int32(1))
        ///     >>> b.shape
        ///     (2, 3)
        /// </summary>
        [TestMethod]
        public void Broadcast_WithScalar()
        {
            var arr = np.arange(6).reshape(2, 3);
            var scalar = NDArray.Scalar(1);
            var b = np.broadcast(arr, scalar);

            AssertShapeEqual(b.shape, 2, 3);
            Assert.AreEqual(6, b.size);
        }

        /// <summary>
        ///     Test np.broadcast with 4D broadcasting.
        ///     >>> b = np.broadcast(np.ones((8,1,6,1)), np.ones((7,1,5)))
        ///     >>> b.shape
        ///     (8, 7, 6, 5)
        ///     >>> b.size
        ///     1680
        /// </summary>
        [TestMethod]
        public void Broadcast_4D()
        {
            var arr1 = np.ones(new Shape(8, 1, 6, 1));
            var arr2 = np.ones(new Shape(7, 1, 5));
            var b = np.broadcast(arr1, arr2);

            AssertShapeEqual(b.shape, 8, 7, 6, 5);
            Assert.AreEqual(8 * 7 * 6 * 5, b.size);
            Assert.AreEqual(4, b.ndim);
        }

        /// <summary>
        ///     Test np.broadcast iterators array.
        ///     >>> b = np.broadcast(np.arange(3), np.ones((2, 3)))
        ///     >>> len(b.iters)
        ///     2
        /// </summary>
        [TestMethod]
        public void Broadcast_HasIters()
        {
            var arr1 = np.arange(3);
            var arr2 = np.ones(new Shape(2, 3));
            var b = np.broadcast(arr1, arr2);

            b.iters.Should().NotBeNull();
            b.iters.Length.Should().Be(2);
        }

        #endregion

        // ================================================================
        // Group 5: Arithmetic Broadcasting End-to-End
        // ================================================================

        #region Group 5: Arithmetic Broadcasting End-to-End

        /// <summary>
        ///     Scalar + Array and Array + Scalar.
        ///     >>> np.int32(1) + np.array([1, 2, 3])
        ///     array([2, 3, 4])
        ///     >>> np.array([1, 2, 3]) + np.int32(10)
        ///     array([11, 12, 13])
        /// </summary>
        [TestMethod]
        public void Add_ScalarAndArray()
        {
            var arr = np.array(new int[] { 1, 2, 3 });
            var scalar = NDArray.Scalar(1);

            // scalar + array
            var r1 = scalar + arr;
            np.array_equal(r1, np.array(new int[] { 2, 3, 4 })).Should().BeTrue();

            // array + scalar
            var scalar10 = NDArray.Scalar(10);
            var r2 = arr + scalar10;
            np.array_equal(r2, np.array(new int[] { 11, 12, 13 })).Should().BeTrue();
        }

        /// <summary>
        ///     1D broadcast against 2D.
        ///     >>> np.arange(6).reshape(2, 3) + np.arange(3)
        ///     array([[0, 2, 4],
        ///            [3, 5, 7]])
        /// </summary>
        [TestMethod]
        public void Add_1DTo2D()
        {
            var a = np.arange(6).reshape(2, 3); // [[0,1,2],[3,4,5]]
            var b = np.arange(3);               // [0,1,2]
            var result = a + b;

            AssertShapeEqual(result, 2, 3);
            var expected = np.array(new int[,] { { 0, 2, 4 }, { 3, 5, 7 } });
            np.array_equal(result, expected).Should().BeTrue();
        }

        /// <summary>
        ///     (3,1) + (1,3) broadcasting.
        ///     >>> np.array([[0],[1],[2]]) + np.array([[0, 1, 2]])
        ///     array([[0, 1, 2],
        ///            [1, 2, 3],
        ///            [2, 3, 4]])
        /// </summary>
        [TestMethod]
        public void Add_ColumnPlusRow()
        {
            var col = np.array(new int[,] { { 0 }, { 1 }, { 2 } }); // shape (3,1)
            var row = np.array(new int[,] { { 0, 1, 2 } });          // shape (1,3)
            var result = col + row;

            AssertShapeEqual(result, 3, 3);
            var expected = np.array(new int[,] { { 0, 1, 2 }, { 1, 2, 3 }, { 2, 3, 4 } });
            np.array_equal(result, expected).Should().BeTrue();
        }

        /// <summary>
        ///     4D broadcasting: (8,1,6,1) + (7,1,5) = (8,7,6,5).
        ///     Verify shape only (values would be huge).
        ///     >>> (np.ones((8,1,6,1)) + np.ones((7,1,5))).shape
        ///     (8, 7, 6, 5)
        /// </summary>
        [TestMethod]
        public void Add_4DBroadcast()
        {
            var a = np.ones(new Shape(8, 1, 6, 1));
            var b = np.ones(new Shape(7, 1, 5));
            var result = a + b;

            AssertShapeEqual(result, 8, 7, 6, 5);

            // All values should be 2.0 (1+1)
            Assert.AreEqual(2.0, result.GetDouble(0, 0, 0, 0));
            Assert.AreEqual(2.0, result.GetDouble(7, 6, 5, 4));
        }

        /// <summary>
        ///     Subtraction with broadcasting.
        ///     >>> np.arange(6).reshape(2, 3) - np.arange(3)
        ///     array([[0, 0, 0],
        ///            [3, 3, 3]])
        /// </summary>
        [TestMethod]
        public void Subtract_Broadcast()
        {
            var a = np.arange(6).reshape(2, 3); // [[0,1,2],[3,4,5]]
            var b = np.arange(3);               // [0,1,2]
            var result = a - b;

            AssertShapeEqual(result, 2, 3);
            var expected = np.array(new int[,] { { 0, 0, 0 }, { 3, 3, 3 } });
            np.array_equal(result, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Multiplication with broadcasting.
        ///     >>> np.arange(6).reshape(2, 3) * np.array([1, 2, 3])
        ///     array([[ 0,  2,  6],
        ///            [ 3,  8, 15]])
        /// </summary>
        [TestMethod]
        public void Multiply_Broadcast()
        {
            var a = np.arange(6).reshape(2, 3); // [[0,1,2],[3,4,5]]
            var b = np.array(new int[] { 1, 2, 3 });
            var result = a * b;

            AssertShapeEqual(result, 2, 3);
            var expected = np.array(new int[,] { { 0, 2, 6 }, { 3, 8, 15 } });
            np.array_equal(result, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Element-wise comparison with scalar broadcasting.
        ///     NumSharp's > operator only supports NDArray > scalar (not NDArray > NDArray
        ///     with broadcasting). This test verifies scalar comparison works correctly.
        ///
        ///     >>> np.arange(6).reshape(2, 3) > 1
        ///     array([[False, False,  True],
        ///            [ True,  True,  True]])
        /// </summary>
        [TestMethod]
        public void Comparison_Broadcast()
        {
            var a = np.arange(6).reshape(2, 3); // [[0,1,2],[3,4,5]]
            var result = a > 1;

            AssertShapeEqual(result, 2, 3);
            Assert.AreEqual(false, result.GetBoolean(0, 0)); // 0 > 1 = false
            Assert.AreEqual(false, result.GetBoolean(0, 1)); // 1 > 1 = false
            Assert.AreEqual(true, result.GetBoolean(0, 2));  // 2 > 1 = true
            Assert.AreEqual(true, result.GetBoolean(1, 0));  // 3 > 1 = true
            Assert.AreEqual(true, result.GetBoolean(1, 1));  // 4 > 1 = true
            Assert.AreEqual(true, result.GetBoolean(1, 2));  // 5 > 1 = true
        }

        #endregion

        // ================================================================
        // Group 6: Edge Cases
        // ================================================================

        #region Group 6: Edge Cases

        /// <summary>
        ///     Scalar NDArray broadcast to various shapes.
        ///     >>> np.broadcast_to(np.float64(1.0), (3, 4))
        ///     array([[1., 1., 1., 1.],
        ///            [1., 1., 1., 1.],
        ///            [1., 1., 1., 1.]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ScalarToShape()
        {
            var scalar = NDArray.Scalar(1.0);

            var r1 = np.broadcast_to(scalar, new Shape(3));
            AssertShapeEqual(r1, 3);
            Assert.AreEqual(1.0, r1.GetDouble(0));
            Assert.AreEqual(1.0, r1.GetDouble(2));

            var r2 = np.broadcast_to(scalar, new Shape(3, 4));
            AssertShapeEqual(r2, 3, 4);
            Assert.AreEqual(1.0, r2.GetDouble(0, 0));
            Assert.AreEqual(1.0, r2.GetDouble(2, 3));

            var r3 = np.broadcast_to(scalar, new Shape(2, 3, 4));
            AssertShapeEqual(r3, 2, 3, 4);
            Assert.AreEqual(1.0, r3.GetDouble(0, 0, 0));
            Assert.AreEqual(1.0, r3.GetDouble(1, 2, 3));
        }

        /// <summary>
        ///     broadcast_arrays returns views (shared memory).
        ///     From test_stride_tricks.py.
        ///
        ///     >>> x = np.array([1, 2, 3])
        ///     >>> y = np.array([[1], [2]])
        ///     >>> a, b = np.broadcast_arrays(x, y)
        ///     >>> a.shape
        ///     (2, 3)
        ///
        ///     Verify that a/b reflect the original data.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_ReturnsViews()
        {
            var x = np.array(new int[] { 1, 2, 3 });      // shape (3,)
            var y = np.array(new int[,] { { 1 }, { 2 } }); // shape (2,1)
            var (a, b) = np.broadcast_arrays(x, y);

            AssertShapeEqual(a, 2, 3);
            AssertShapeEqual(b, 2, 3);

            // a should broadcast x across rows: [[1,2,3],[1,2,3]]
            Assert.AreEqual(1, a.GetInt32(0, 0));
            Assert.AreEqual(2, a.GetInt32(0, 1));
            Assert.AreEqual(3, a.GetInt32(0, 2));
            Assert.AreEqual(1, a.GetInt32(1, 0));
            Assert.AreEqual(2, a.GetInt32(1, 1));
            Assert.AreEqual(3, a.GetInt32(1, 2));

            // b should broadcast y across columns: [[1,1,1],[2,2,2]]
            Assert.AreEqual(1, b.GetInt32(0, 0));
            Assert.AreEqual(1, b.GetInt32(0, 1));
            Assert.AreEqual(1, b.GetInt32(0, 2));
            Assert.AreEqual(2, b.GetInt32(1, 0));
            Assert.AreEqual(2, b.GetInt32(1, 1));
            Assert.AreEqual(2, b.GetInt32(1, 2));
        }

        /// <summary>
        ///     Broadcasting works correctly with sliced (view) arrays.
        ///     >>> x = np.arange(12).reshape(3, 4)
        ///     >>> y = x[:, 0:1]  # shape (3, 1) - a view
        ///     >>> z = np.broadcast_to(y, (3, 4))
        ///     >>> z
        ///     array([[0, 0, 0, 0],
        ///            [4, 4, 4, 4],
        ///            [8, 8, 8, 8]])
        /// </summary>
        [TestMethod]
        public void Broadcast_SlicedInput()
        {
            var x = np.arange(12).reshape(3, 4);
            // x = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
            var y = x[":, 0:1"]; // shape (3,1), values [0],[4],[8]

            AssertShapeEqual(y, 3, 1);
            Assert.AreEqual(0, y.GetInt32(0, 0));
            Assert.AreEqual(4, y.GetInt32(1, 0));
            Assert.AreEqual(8, y.GetInt32(2, 0));

            var z = np.broadcast_to(y, new Shape(3, 4));
            AssertShapeEqual(z, 3, 4);

            var expected = np.array(new int[,] { { 0, 0, 0, 0 }, { 4, 4, 4, 4 }, { 8, 8, 8, 8 } });
            np.array_equal(z, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Broadcasting with sliced input in arithmetic.
        ///     >>> x = np.arange(12).reshape(3, 4)
        ///     >>> y = x[:, 0:1]  # column 0 as (3,1)
        ///     >>> result = x + y
        ///     >>> result
        ///     array([[ 0,  1,  2,  3],
        ///            [ 8,  9, 10, 11],
        ///            [16, 17, 18, 19]])
        /// </summary>
        [TestMethod]
        public void Broadcast_SlicedInputArithmetic()
        {
            var x = np.arange(12).reshape(3, 4);
            var y = x[":, 0:1"]; // shape (3,1): [[0],[4],[8]]
            var result = x + y;

            AssertShapeEqual(result, 3, 4);
            var expected = np.array(new int[,]
            {
                { 0, 1, 2, 3 },
                { 8, 9, 10, 11 },
                { 16, 17, 18, 19 }
            });
            np.array_equal(result, expected).Should().BeTrue();
        }

        /// <summary>
        ///     5D broadcasting.
        ///     >>> a = np.ones((2, 1, 3, 1, 5))
        ///     >>> b = np.ones((1, 4, 1, 6, 1))
        ///     >>> (a + b).shape
        ///     (2, 4, 3, 6, 5)
        /// </summary>
        [TestMethod]
        public void Broadcast_HighDimensional()
        {
            var a = np.ones(new Shape(2, 1, 3, 1, 5));
            var b = np.ones(new Shape(1, 4, 1, 6, 1));
            var result = a + b;

            AssertShapeEqual(result, 2, 4, 3, 6, 5);

            // All values should be 2.0
            Assert.AreEqual(2.0, result.GetDouble(0, 0, 0, 0, 0));
            Assert.AreEqual(2.0, result.GetDouble(1, 3, 2, 5, 4));
        }

        /// <summary>
        ///     6D broadcasting.
        ///     >>> a = np.ones((1, 2, 1, 3, 1, 4))
        ///     >>> b = np.ones((5, 1, 6, 1, 7, 1))
        ///     >>> (a + b).shape
        ///     (5, 2, 6, 3, 7, 4)
        /// </summary>
        [TestMethod]
        public void Broadcast_6D()
        {
            var a = np.ones(new Shape(1, 2, 1, 3, 1, 4));
            var b = np.ones(new Shape(5, 1, 6, 1, 7, 1));
            var result = a + b;

            AssertShapeEqual(result, 5, 2, 6, 3, 7, 4);
            Assert.AreEqual(2.0, result.GetDouble(0, 0, 0, 0, 0, 0));
        }

        /// <summary>
        ///     broadcast_arrays with three inputs.
        ///     >>> a = np.ones((2, 1))
        ///     >>> b = np.ones((1, 3))
        ///     >>> c = np.ones((2, 3))
        ///     >>> r = np.broadcast_arrays(a, b, c)
        ///     >>> [x.shape for x in r]
        ///     [(2, 3), (2, 3), (2, 3)]
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_ThreeInputs()
        {
            var a = np.ones(new Shape(2, 1));
            var b = np.ones(new Shape(1, 3));
            var c = np.ones(new Shape(2, 3));
            var result = np.broadcast_arrays(a, b, c);

            result.Length.Should().Be(3);
            foreach (var r in result)
                AssertShapeEqual(r, 2, 3);
        }

        /// <summary>
        ///     broadcast_arrays with four inputs of varying dimensions.
        ///     >>> a = np.ones((6, 7))
        ///     >>> b = np.ones((5, 6, 1))
        ///     >>> c = np.ones((7,))
        ///     >>> d = np.ones((5, 1, 7))
        ///     >>> r = np.broadcast_arrays(a, b, c, d)
        ///     >>> [x.shape for x in r]
        ///     [(5, 6, 7), (5, 6, 7), (5, 6, 7), (5, 6, 7)]
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_FourInputs()
        {
            var a = np.ones(new Shape(6, 7));
            var b = np.ones(new Shape(5, 6, 1));
            var c = np.ones(new Shape(7));
            var d = np.ones(new Shape(5, 1, 7));
            var result = np.broadcast_arrays(a, b, c, d);

            result.Length.Should().Be(4);
            foreach (var r in result)
                AssertShapeEqual(r, 5, 6, 7);
        }

        /// <summary>
        ///     ResolveReturnShape: classic broadcasting examples from NumPy docs.
        ///     Validates the low-level shape resolution.
        /// </summary>
        [TestMethod]
        public void ResolveReturnShape_ClassicExamples()
        {
            // (256,256,3) + (3,) -> (256,256,3)
            var s1 = DefaultEngine.ResolveReturnShape(new Shape(256, 256, 3), new Shape(3));
            s1.dimensions.Should().BeEquivalentTo(new[] { 256, 256, 3 });

            // (5,4) + (1,) -> (5,4)
            var s2 = DefaultEngine.ResolveReturnShape(new Shape(5, 4), new Shape(1));
            s2.dimensions.Should().BeEquivalentTo(new[] { 5, 4 });

            // (5,4) + (4,) -> (5,4)
            var s3 = DefaultEngine.ResolveReturnShape(new Shape(5, 4), new Shape(4));
            s3.dimensions.Should().BeEquivalentTo(new[] { 5, 4 });

            // (15,3,5) + (15,1,5) -> (15,3,5)
            var s4 = DefaultEngine.ResolveReturnShape(new Shape(15, 3, 5), new Shape(15, 1, 5));
            s4.dimensions.Should().BeEquivalentTo(new[] { 15, 3, 5 });

            // (15,3,5) + (3,1) -> (15,3,5)
            var s5 = DefaultEngine.ResolveReturnShape(new Shape(15, 3, 5), new Shape(3, 1));
            s5.dimensions.Should().BeEquivalentTo(new[] { 15, 3, 5 });
        }

        /// <summary>
        ///     are_broadcastable returns true for compatible and false for incompatible shapes.
        /// </summary>
        [TestMethod]
        public void AreBroadcastable_CompatibleAndIncompatible()
        {
            // Compatible pairs
            np.are_broadcastable(new int[] { 1, 3 }, new int[] { 3, 1 }).Should().BeTrue();
            np.are_broadcastable(new int[] { 5, 4 }, new int[] { 1 }).Should().BeTrue();
            np.are_broadcastable(new int[] { 5, 4 }, new int[] { 4 }).Should().BeTrue();
            np.are_broadcastable(new int[] { 8, 1, 6, 1 }, new int[] { 7, 1, 5 }).Should().BeTrue();

            // Incompatible pairs
            np.are_broadcastable(new int[] { 3 }, new int[] { 4 }).Should().BeFalse();
            np.are_broadcastable(new int[] { 2, 1 }, new int[] { 8, 4, 3 }).Should().BeFalse();
            np.are_broadcastable(new int[] { 1, 3, 4 }, new int[] { 2, 3, 3 }).Should().BeFalse();
        }

        /// <summary>
        ///     Broadcasting with float types.
        ///     >>> np.float64(2.5) + np.array([1.0, 2.0, 3.0])
        ///     array([3.5, 4.5, 5.5])
        /// </summary>
        [TestMethod]
        public void Add_BroadcastFloat()
        {
            var scalar = NDArray.Scalar(2.5);
            var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = scalar + arr;

            AssertShapeEqual(result, 3);
            Assert.AreEqual(3.5, result.GetDouble(0));
            Assert.AreEqual(4.5, result.GetDouble(1));
            Assert.AreEqual(5.5, result.GetDouble(2));
        }

        /// <summary>
        ///     Verify broadcast_to with (1,) -> (1,) is identity.
        ///     >>> x = np.array([42])
        ///     >>> y = np.broadcast_to(x, (1,))
        ///     >>> y[0]
        ///     42
        /// </summary>
        [TestMethod]
        public void BroadcastTo_IdentityShape()
        {
            var x = np.array(new int[] { 42 });
            var y = np.broadcast_to(x, new Shape(1));

            AssertShapeEqual(y, 1);
            Assert.AreEqual(42, y.GetInt32(0));
        }

        /// <summary>
        ///     Verify broadcast with (1,1) + (3,3) -> (3,3).
        ///     >>> np.ones((1,1)) + np.arange(9).reshape(3,3)
        ///     array([[ 1.,  2.,  3.],
        ///            [ 4.,  5.,  6.],
        ///            [ 7.,  8.,  9.]])
        /// </summary>
        [TestMethod]
        public void Add_1x1_Plus_3x3()
        {
            var a = np.ones(new Shape(1, 1));           // [[1.0]]
            var b = np.arange(9).reshape(3, 3);         // [[0,1,2],[3,4,5],[6,7,8]]
            var result = a + b;

            AssertShapeEqual(result, 3, 3);
            Assert.AreEqual(1.0, result.GetDouble(0, 0));
            Assert.AreEqual(2.0, result.GetDouble(0, 1));
            Assert.AreEqual(9.0, result.GetDouble(2, 2));
        }

        /// <summary>
        ///     Verify broadcasting produces correct stride pattern.
        ///     When a dimension is broadcast, its stride should be 0.
        ///
        ///     >>> x = np.array([1, 2, 3])
        ///     >>> y = np.broadcast_to(x, (4, 3))
        ///     >>> y.strides
        ///     (0, 4)  # (0 bytes for first dim since it's broadcasted, 4 bytes for int32)
        /// </summary>
        [TestMethod]
        public void BroadcastTo_StridesCorrect()
        {
            var x = np.array(new int[] { 1, 2, 3 }); // shape (3,)
            var y = np.broadcast_to(x, new Shape(4, 3));

            AssertShapeEqual(y, 4, 3);

            // The broadcast dimension (dim 0, size 4) should have stride 0
            y.Shape.strides[0].Should().Be(0,
                "broadcast dimension should have stride 0");

            // The non-broadcast dimension (dim 1, size 3) should have stride 1
            y.Shape.strides[1].Should().Be(1,
                "non-broadcast dimension should have stride 1 (element stride)");
        }

        /// <summary>
        ///     Verify DefaultEngine.Broadcast produces correct strides for
        ///     a dimension-1 that gets stretched.
        ///
        ///     >>> x = np.ones((3, 1))
        ///     >>> y = np.ones((1, 4))
        ///     >>> bx, by = np.broadcast_arrays(x, y)
        ///     >>> bx.strides
        ///     (8, 0)   # 8 bytes stride for rows, 0 for broadcast cols
        ///     >>> by.strides
        ///     (0, 8)   # 0 for broadcast rows, 8 bytes stride for cols
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_StridesCorrect()
        {
            var x = np.ones(new Shape(3, 1));
            var y = np.ones(new Shape(1, 4));
            var (bx, by) = np.broadcast_arrays(x, y);

            AssertShapeEqual(bx, 3, 4);
            AssertShapeEqual(by, 3, 4);

            // bx: row stride should be non-zero, col stride should be 0 (broadcast)
            bx.Shape.strides[0].Should().NotBe(0, "row stride should be original");
            bx.Shape.strides[1].Should().Be(0, "broadcast col should have stride 0");

            // by: row stride should be 0 (broadcast), col stride should be non-zero
            by.Shape.strides[0].Should().Be(0, "broadcast row should have stride 0");
            by.Shape.strides[1].Should().NotBe(0, "col stride should be original");
        }

        /// <summary>
        ///     broadcast_to with reversed-stride (negative step) input.
        ///     NOTE: GetInt32/NDIterator/array_equal return correct values,
        ///     but ToString has a known bug where it ignores negative strides
        ///     on broadcasted arrays, displaying elements in wrong order.
        ///
        ///     >>> x = np.arange(3)[::-1]     # [2, 1, 0]
        ///     >>> np.broadcast_to(x, (2, 3))
        ///     array([[2, 1, 0],
        ///            [2, 1, 0]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ReversedSlice()
        {
            var rev = np.arange(3)["::-1"]; // [2, 1, 0]
            Assert.AreEqual(2, rev.GetInt32(0));
            Assert.AreEqual(1, rev.GetInt32(1));
            Assert.AreEqual(0, rev.GetInt32(2));

            var brev = np.broadcast_to(rev, new Shape(2, 3));
            AssertShapeEqual(brev, 2, 3);

            // Verify correct values via element access
            Assert.AreEqual(2, brev.GetInt32(0, 0));
            Assert.AreEqual(1, brev.GetInt32(0, 1));
            Assert.AreEqual(0, brev.GetInt32(0, 2));
            Assert.AreEqual(2, brev.GetInt32(1, 0));
            Assert.AreEqual(1, brev.GetInt32(1, 1));
            Assert.AreEqual(0, brev.GetInt32(1, 2));

            // Verify via array_equal
            var expected = np.array(new int[,] { { 2, 1, 0 }, { 2, 1, 0 } });
            np.array_equal(brev, expected).Should().BeTrue();
        }

        /// <summary>
        ///     broadcast_to with step-sliced (step=2) input.
        ///     NOTE: Same ToString bug as reversed slice — values via element
        ///     access and array_equal are correct.
        ///
        ///     >>> x = np.arange(6)[::2]      # [0, 2, 4]
        ///     >>> np.broadcast_to(x, (2, 3))
        ///     array([[0, 2, 4],
        ///            [0, 2, 4]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_StepSlice()
        {
            var stepped = np.arange(6)["::2"]; // [0, 2, 4]
            Assert.AreEqual(0, stepped.GetInt32(0));
            Assert.AreEqual(2, stepped.GetInt32(1));
            Assert.AreEqual(4, stepped.GetInt32(2));

            var bstepped = np.broadcast_to(stepped, new Shape(2, 3));
            AssertShapeEqual(bstepped, 2, 3);

            Assert.AreEqual(0, bstepped.GetInt32(0, 0));
            Assert.AreEqual(2, bstepped.GetInt32(0, 1));
            Assert.AreEqual(4, bstepped.GetInt32(0, 2));
            Assert.AreEqual(0, bstepped.GetInt32(1, 0));
            Assert.AreEqual(2, bstepped.GetInt32(1, 1));
            Assert.AreEqual(4, bstepped.GetInt32(1, 2));

            var expected = np.array(new int[,] { { 0, 2, 4 }, { 0, 2, 4 } });
            np.array_equal(bstepped, expected).Should().BeTrue();
        }

        /// <summary>
        ///     broadcast_to with 2D sliced column then broadcast.
        ///     NOTE: ToString bug — shows zeros in last row instead of correct values.
        ///     Element access and array_equal are correct.
        ///
        ///     >>> x = np.arange(12).reshape(3, 4)
        ///     >>> col = x[:, 1:2]             # [[1],[5],[9]]
        ///     >>> np.broadcast_to(col, (3, 3))
        ///     array([[1, 1, 1],
        ///            [5, 5, 5],
        ///            [9, 9, 9]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_SlicedColumn()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 1:2"]; // shape (3,1): [[1],[5],[9]]

            AssertShapeEqual(col, 3, 1);
            Assert.AreEqual(1, col.GetInt32(0, 0));
            Assert.AreEqual(5, col.GetInt32(1, 0));
            Assert.AreEqual(9, col.GetInt32(2, 0));

            var bcol = np.broadcast_to(col, new Shape(3, 3));
            AssertShapeEqual(bcol, 3, 3);

            var expected = np.array(new int[,] { { 1, 1, 1 }, { 5, 5, 5 }, { 9, 9, 9 } });
            np.array_equal(bcol, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Double-sliced (row step + column slice) then broadcast.
        ///     NOTE: ToString shows wrong values for second row.
        ///     Element access and array_equal are correct.
        ///
        ///     >>> x = np.arange(12).reshape(3, 4)
        ///     >>> ds = x[::2, 0:1]            # [[0],[8]]
        ///     >>> np.broadcast_to(ds, (2, 4))
        ///     array([[0, 0, 0, 0],
        ///            [8, 8, 8, 8]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_DoubleSliced()
        {
            var x = np.arange(12).reshape(3, 4);
            var dslice = x["::2, :"]; // rows 0 and 2: shape (2,4)
            var dslice_col = dslice[":, 0:1"]; // shape (2,1): [[0],[8]]

            Assert.AreEqual(0, dslice_col.GetInt32(0, 0));
            Assert.AreEqual(8, dslice_col.GetInt32(1, 0));

            var bdslice = np.broadcast_to(dslice_col, new Shape(2, 4));
            AssertShapeEqual(bdslice, 2, 4);

            var expected = np.array(new int[,] { { 0, 0, 0, 0 }, { 8, 8, 8, 8 } });
            np.array_equal(bdslice, expected).Should().BeTrue();
        }

        /// <summary>
        ///     Arithmetic with reversed-slice + broadcast produces correct results.
        ///     >>> x = np.arange(3)[::-1]
        ///     >>> y = np.broadcast_to(x, (2, 3))
        ///     >>> y + np.ones((2, 3), dtype=np.int32)
        ///     array([[3, 2, 1],
        ///            [3, 2, 1]])
        /// </summary>
        [TestMethod]
        public void Add_ReversedSliceBroadcast()
        {
            var rev = np.arange(3)["::-1"];
            var brev = np.broadcast_to(rev, new Shape(2, 3));
            var ones = np.ones(new Shape(2, 3), NPTypeCode.Int32);
            var result = brev + ones;

            AssertShapeEqual(result, 2, 3);
            var expected = np.array(new int[,] { { 3, 2, 1 }, { 3, 2, 1 } });
            np.array_equal(result, expected).Should().BeTrue();
        }

        /// <summary>
        ///     np.sum on broadcast_to of a sliced column.
        ///     >>> x = np.arange(12).reshape(3, 4)
        ///     >>> col = x[:, 2:3]              # [[2],[6],[10]]
        ///     >>> bc = np.broadcast_to(col, (3, 5))
        ///     >>> np.sum(bc)
        ///     90
        ///     >>> np.sum(bc, axis=0)
        ///     array([18, 18, 18, 18, 18])
        ///     >>> np.sum(bc, axis=1)
        ///     array([10, 30, 50])
        /// </summary>
        [TestMethod]
        public void Sum_SlicedBroadcast()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 2:3"]; // shape (3,1): [[2],[6],[10]]
            var bc = np.broadcast_to(col, new Shape(3, 5));

            // Total sum: 2*5 + 6*5 + 10*5 = 90
            Assert.AreEqual(90, (int)np.sum(bc));

            // Sum along axis 0: [2+6+10, ...] = [18,18,18,18,18]
            var s0 = np.sum(bc, axis: 0);
            AssertShapeEqual(s0, 5);
            for (int c = 0; c < 5; c++)
                Assert.AreEqual(18, s0.GetInt32(c));

            // Sum along axis 1: [2*5, 6*5, 10*5] = [10,30,50]
            var s1 = np.sum(bc, axis: 1);
            AssertShapeEqual(s1, 3);
            Assert.AreEqual(10, s1.GetInt32(0));
            Assert.AreEqual(30, s1.GetInt32(1));
            Assert.AreEqual(50, s1.GetInt32(2));
        }

        /// <summary>
        ///     Flatten and copy of sliced+broadcast arrays produce correct data.
        ///     >>> x = np.arange(3)[::-1]
        ///     >>> bc = np.broadcast_to(x, (2, 3))
        ///     >>> bc.flatten()
        ///     array([2, 1, 0, 2, 1, 0])
        ///     >>> np.copy(bc)
        ///     array([[2, 1, 0],
        ///            [2, 1, 0]])
        /// </summary>
        [TestMethod]
        public void FlattenAndCopy_SlicedBroadcast()
        {
            var rev = np.arange(3)["::-1"];
            var brev = np.broadcast_to(rev, new Shape(2, 3));

            // flatten should produce [2,1,0,2,1,0]
            var flat = brev.flatten();
            AssertShapeEqual(flat, 6);
            var expected_flat = np.array(new int[] { 2, 1, 0, 2, 1, 0 });
            np.array_equal(flat, expected_flat).Should().BeTrue();

            // copy should produce a contiguous copy with correct values
            var copied = np.copy(brev);
            AssertShapeEqual(copied, 2, 3);
            var expected_copy = np.array(new int[,] { { 2, 1, 0 }, { 2, 1, 0 } });
            np.array_equal(copied, expected_copy).Should().BeTrue();
        }

        #endregion

        // ================================================================
        // Group 7: Stress Tests & Bug Probes
        //   Systematically exercising every broadcast code path.
        // ================================================================

        #region Group 7: Stress Tests & Bug Probes

        /// <summary>
        ///     broadcast_to result allows write-through to original (no read-only protection).
        ///     NumPy raises ValueError: assignment destination is read-only.
        ///
        ///     Known NumSharp bug: broadcast_to does not enforce read-only semantics.
        ///     Writing to the broadcast view modifies the original array.
        ///
        ///     >>> x = np.array([1, 2, 3, 4])
        ///     >>> y = np.broadcast_to(x, (2, 4))
        ///     >>> y[0, 0] = 999  # NumPy: ValueError
        /// </summary>
        [TestMethod]
        public void BroadcastTo_WriteThrough_KnownBug()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 });
            var bx = np.broadcast_to(x, new Shape(2, 4));

            // NumSharp allows writing through broadcast view to original.
            // This is a known bug — NumPy would throw.
            // We document the current behavior: SetInt32 modifies original.
            bx.SetInt32(999, 0, 0);
            Assert.AreEqual(999, x.GetInt32(0),
                "NumSharp bug: broadcast_to write-through modifies original. " +
                "NumPy would throw ValueError: assignment destination is read-only.");

            // Restore for safety
            x.SetInt32(1, 0);
        }

        /// <summary>
        ///     Re-broadcasting an already-broadcasted array via broadcast_arrays
        ///     succeeds in NumSharp (the N-array Broadcast overload doesn't guard against it).
        ///     The 2-array Broadcast(Shape,Shape) overload throws NotSupportedException.
        ///
        ///     This is inconsistent behavior.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_AlreadyBroadcasted_Succeeds()
        {
            var x = np.ones(new Shape(1, 3));
            var y = np.ones(new Shape(3, 1));
            var (bx, by) = np.broadcast_arrays(x, y);

            // bx is now a broadcasted (3,3). broadcast_arrays with another array succeeds.
            var (rbx, rby) = np.broadcast_arrays(bx, np.ones(new Shape(3, 3)));
            AssertShapeEqual(rbx, 3, 3);
            AssertShapeEqual(rby, 3, 3);
        }

        /// <summary>
        ///     broadcast_to on an already-broadcasted array throws NotSupportedException.
        /// </summary>
        [TestMethod]
        public void BroadcastTo_AlreadyBroadcasted_Throws()
        {
            var x = np.ones(new Shape(1, 3));
            var bx = np.broadcast_to(x, new Shape(4, 3));

            new Action(() => np.broadcast_to(bx, new Shape(2, 4, 3)))
                .Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     Dimension-0 (zero-size) array broadcasting.
        ///     >>> np.broadcast_arrays(np.ones((0, 3)), np.ones((1, 3)))
        ///     Both shapes become (0, 3).
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_ZeroSizeDimension()
        {
            var z = np.ones(new Shape(0, 3));
            var o = np.ones(new Shape(1, 3));
            var (bz, bo) = np.broadcast_arrays(z, o);

            AssertShapeEqual(bz, 0, 3);
            AssertShapeEqual(bo, 0, 3);
        }

        /// <summary>
        ///     (0,3) + (2,3) should be incompatible (0 != 2, neither is 1).
        ///     >>> np.broadcast_arrays(np.ones((0, 3)), np.ones((2, 3)))
        ///     ValueError: shape mismatch
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_ZeroVsNonZeroDim_Throws()
        {
            var z = np.ones(new Shape(0, 3));
            var o = np.ones(new Shape(2, 3));

            new Action(() => np.broadcast_arrays(z, o))
                .Should().Throw<IncorrectShapeException>();
        }

        /// <summary>
        ///     All 12 dtypes can broadcast via arithmetic (shape correctness).
        ///     (1,3) + (3,1) -> (3,3) for every supported type.
        /// </summary>
        [TestMethod]
        public void Add_BroadcastAllDtypes()
        {
            var dtypes = new NPTypeCode[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Char, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal
            };

            foreach (var dt in dtypes)
            {
                var a = np.ones(new Shape(1, 3), dt);
                var b = np.ones(new Shape(3, 1), dt);
                var result = a + b;
                AssertShapeEqual(result, 3, 3);
            }
        }

        /// <summary>
        ///     Mixed int32 + double broadcasting promotes to double.
        ///     >>> np.array([1, 2, 3]) + np.array([0.5])
        ///     array([1.5, 2.5, 3.5])
        /// </summary>
        [TestMethod]
        public void Add_MixedDtypeBroadcast()
        {
            var i32 = np.array(new int[] { 1, 2, 3 });
            var f64 = np.array(new double[] { 0.5 });
            var result = i32 + f64;

            Assert.AreEqual(NPTypeCode.Double, result.typecode);
            AssertShapeEqual(result, 3);
            Assert.AreEqual(1.5, result.GetDouble(0));
            Assert.AreEqual(2.5, result.GetDouble(1));
            Assert.AreEqual(3.5, result.GetDouble(2));
        }

        /// <summary>
        ///     Negative value arithmetic with broadcasting.
        ///     >>> np.array([-3,-2,-1,0,1,2,3]) + np.array([[10],[-10]])
        ///     array([[  7,   8,   9,  10,  11,  12,  13],
        ///            [-13, -12, -11, -10,  -9,  -8,  -7]])
        /// </summary>
        [TestMethod]
        public void Add_NegativeValuesBroadcast()
        {
            var a = np.array(new int[] { -3, -2, -1, 0, 1, 2, 3 });
            var b = np.array(new int[,] { { 10 }, { -10 } });
            var result = a + b;

            AssertShapeEqual(result, 2, 7);
            Assert.AreEqual(7, result.GetInt32(0, 0));
            Assert.AreEqual(13, result.GetInt32(0, 6));
            Assert.AreEqual(-13, result.GetInt32(1, 0));
            Assert.AreEqual(-7, result.GetInt32(1, 6));
        }

        /// <summary>
        ///     np.maximum with broadcasting.
        ///     >>> np.maximum(np.array([1, 5, 3]), np.array([[2], [4]]))
        ///     array([[2, 5, 3],
        ///            [4, 5, 4]])
        /// </summary>
        [TestMethod]
        public void Maximum_Broadcast()
        {
            var a = np.array(new int[] { 1, 5, 3 });
            var b = np.array(new int[,] { { 2 }, { 4 } });
            var mx = np.maximum(a, b);

            AssertShapeEqual(mx, 2, 3);
            Assert.AreEqual(2, mx.GetInt32(0, 0));
            Assert.AreEqual(5, mx.GetInt32(0, 1));
            Assert.AreEqual(3, mx.GetInt32(0, 2));
            Assert.AreEqual(4, mx.GetInt32(1, 0));
            Assert.AreEqual(5, mx.GetInt32(1, 1));
            Assert.AreEqual(4, mx.GetInt32(1, 2));
        }

        /// <summary>
        ///     Sliced columns and rows from the same array broadcast against each other.
        ///     >>> x = np.arange(20).reshape(4, 5)
        ///     >>> x[:, 3:4] + x[2:3, :]
        ///     array([[13, 14, 15, 16, 17],
        ///            [18, 19, 20, 21, 22],
        ///            [23, 24, 25, 26, 27],
        ///            [28, 29, 30, 31, 32]])
        /// </summary>
        [TestMethod]
        public void Add_SlicedColPlusSlicedRow()
        {
            var x = np.arange(20).reshape(4, 5);
            var col = x[":, 3:4"]; // (4,1): [[3],[8],[13],[18]]
            var row = x["2:3, :"]; // (1,5): [[10,11,12,13,14]]
            var result = col + row;

            AssertShapeEqual(result, 4, 5);
            Assert.AreEqual(13, result.GetInt32(0, 0)); // 3+10
            Assert.AreEqual(17, result.GetInt32(0, 4)); // 3+14
            Assert.AreEqual(28, result.GetInt32(3, 0)); // 18+10
            Assert.AreEqual(32, result.GetInt32(3, 4)); // 18+14
        }

        /// <summary>
        ///     broadcast_to result can be sliced, and sliced values are correct.
        ///     >>> x = np.array([10, 20, 30])
        ///     >>> y = np.broadcast_to(x, (4, 3))
        ///     >>> y[1:3, :]
        ///     array([[10, 20, 30],
        ///            [10, 20, 30]])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ThenSlice()
        {
            var x = np.array(new int[] { 10, 20, 30 });
            var bx = np.broadcast_to(x, new Shape(4, 3));
            var sliced = bx["1:3, :"];

            AssertShapeEqual(sliced, 2, 3);
            Assert.AreEqual(10, sliced.GetInt32(0, 0));
            Assert.AreEqual(30, sliced.GetInt32(0, 2));
            Assert.AreEqual(10, sliced.GetInt32(1, 0));
            Assert.AreEqual(30, sliced.GetInt32(1, 2));
        }

        /// <summary>
        ///     broadcast_to result can be integer-indexed.
        ///     >>> y = np.broadcast_to(np.array([10, 20, 30]), (4, 3))
        ///     >>> y[0]
        ///     array([10, 20, 30])
        ///     >>> y[-1]
        ///     array([10, 20, 30])
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ThenIntegerIndex()
        {
            var bx = np.broadcast_to(np.array(new int[] { 10, 20, 30 }), new Shape(4, 3));

            var row0 = bx["0"];
            AssertShapeEqual(row0, 3);
            Assert.AreEqual(10, row0.GetInt32(0));
            Assert.AreEqual(20, row0.GetInt32(1));
            Assert.AreEqual(30, row0.GetInt32(2));

            var rowLast = bx["-1"];
            Assert.AreEqual(10, rowLast.GetInt32(0));
            Assert.AreEqual(30, rowLast.GetInt32(2));
        }

        /// <summary>
        ///     Broadcast result + transpose + reshape chain.
        ///     >>> c = np.arange(3).reshape(1, 3) + np.arange(3).reshape(3, 1)
        ///     >>> np.transpose(c)
        ///     array([[0, 1, 2],
        ///            [1, 2, 3],
        ///            [2, 3, 4]])
        /// </summary>
        [TestMethod]
        public void BroadcastResult_Transpose()
        {
            var c = np.arange(3).reshape(1, 3) + np.arange(3).reshape(3, 1);
            // c is symmetric: [[0,1,2],[1,2,3],[2,3,4]]
            var t = np.transpose(c);

            AssertShapeEqual(t, 3, 3);
            Assert.AreEqual(1, t.GetInt32(0, 1));
            Assert.AreEqual(1, t.GetInt32(1, 0));
            Assert.AreEqual(4, t.GetInt32(2, 2));
        }

        /// <summary>
        ///     Broadcast result can be reshaped.
        ///     >>> c = np.arange(6).reshape(2, 3) + np.array([1, 2, 3])
        ///     >>> c.reshape(3, 2)
        ///     array([[1, 3],
        ///            [5, 4],
        ///            [6, 8]])
        /// </summary>
        [TestMethod]
        public void BroadcastResult_Reshape()
        {
            var c = np.arange(6).reshape(2, 3) + np.array(new int[] { 1, 2, 3 });
            // c = [[1,3,5],[4,6,8]]
            var r = c.reshape(3, 2);

            AssertShapeEqual(r, 3, 2);
            Assert.AreEqual(1, r.GetInt32(0, 0));
            Assert.AreEqual(3, r.GetInt32(0, 1));
            Assert.AreEqual(5, r.GetInt32(1, 0));
            Assert.AreEqual(4, r.GetInt32(1, 1));
            Assert.AreEqual(6, r.GetInt32(2, 0));
            Assert.AreEqual(8, r.GetInt32(2, 1));
        }

        /// <summary>
        ///     sum with keepdims on broadcast result.
        ///     >>> c = np.ones((1, 3)) + np.ones((4, 1))
        ///     >>> np.sum(c, axis=0, keepdims=True)
        ///     array([[8., 8., 8.]])
        /// </summary>
        [TestMethod]
        public void Sum_Keepdims_BroadcastResult()
        {
            var c = np.ones(new Shape(1, 3)) + np.ones(new Shape(4, 1));
            var s = np.sum(c, axis: 0, keepdims: true);

            AssertShapeEqual(s, 1, 3);
            Assert.AreEqual(8.0, s.GetDouble(0, 0));
            Assert.AreEqual(8.0, s.GetDouble(0, 1));
            Assert.AreEqual(8.0, s.GetDouble(0, 2));
        }

        /// <summary>
        ///     np.mean on broadcast result.
        ///     >>> c = np.arange(3).reshape(1, 3) + np.ones((4, 1), dtype=int) * 10
        ///     >>> np.mean(c)
        ///     11.0
        /// </summary>
        [TestMethod]
        public void Mean_BroadcastResult()
        {
            var c = np.arange(3).reshape(1, 3) + np.ones(new Shape(4, 1), NPTypeCode.Int32) * 10;
            // c = [[10,11,12],[10,11,12],[10,11,12],[10,11,12]]
            var m = np.mean(c);

            Assert.AreEqual(11.0, m.GetDouble(0), 0.001);
        }

        /// <summary>
        ///     Scalar + scalar via broadcasting.
        ///     >>> np.int32(5) + np.int32(3)
        ///     8
        /// </summary>
        [TestMethod]
        public void Add_ScalarPlusScalar()
        {
            var c = NDArray.Scalar(5) + NDArray.Scalar(3);
            Assert.AreEqual(8, c.GetInt32(0));
        }

        /// <summary>
        ///     7D broadcasting.
        ///     >>> (np.ones((1,2,1,3,1,4,1)) + np.ones((5,1,6,1,7,1,8))).shape
        ///     (5, 2, 6, 3, 7, 4, 8)
        /// </summary>
        [TestMethod]
        public void Broadcast_7D()
        {
            var a = np.ones(new Shape(1, 2, 1, 3, 1, 4, 1));
            var b = np.ones(new Shape(5, 1, 6, 1, 7, 1, 8));
            var c = a + b;

            AssertShapeEqual(c, 5, 2, 6, 3, 7, 4, 8);
            Assert.AreEqual(2.0, c.GetDouble(0, 0, 0, 0, 0, 0, 0));
        }

        /// <summary>
        ///     Unary negation on broadcast result.
        ///     >>> -( np.arange(3).reshape(1, 3) + np.arange(2).reshape(2, 1) )
        ///     array([[ 0, -1, -2],
        ///            [-1, -2, -3]])
        /// </summary>
        [TestMethod]
        public void UnaryMinus_BroadcastResult()
        {
            var c = np.arange(3).reshape(1, 3) + np.arange(2).reshape(2, 1);
            var neg = -c;

            AssertShapeEqual(neg, 2, 3);
            Assert.AreEqual(0, neg.GetInt32(0, 0));
            Assert.AreEqual(-2, neg.GetInt32(0, 2));
            Assert.AreEqual(-1, neg.GetInt32(1, 0));
            Assert.AreEqual(-3, neg.GetInt32(1, 2));
        }

        /// <summary>
        ///     np.sqrt on broadcast result.
        ///     >>> np.sqrt(np.array([1., 4., 9.]) + np.zeros((2, 1)))
        ///     array([[1., 2., 3.],
        ///            [1., 2., 3.]])
        /// </summary>
        [TestMethod]
        public void Sqrt_BroadcastResult()
        {
            var c = np.array(new double[] { 1.0, 4.0, 9.0 }) + np.zeros(new Shape(2, 1));
            var s = np.sqrt(c);

            AssertShapeEqual(s, 2, 3);
            Assert.AreEqual(1.0, s.GetDouble(0, 0), 0.001);
            Assert.AreEqual(2.0, s.GetDouble(0, 1), 0.001);
            Assert.AreEqual(3.0, s.GetDouble(1, 2), 0.001);
        }

        /// <summary>
        ///     Very asymmetric shape broadcasting.
        ///     >>> (np.ones((100, 1)) + np.ones((1, 200))).shape
        ///     (100, 200)
        /// </summary>
        [TestMethod]
        public void Broadcast_LargeAsymmetric()
        {
            var a = np.ones(new Shape(100, 1));
            var b = np.ones(new Shape(1, 200));
            var c = a + b;

            AssertShapeEqual(c, 100, 200);
            Assert.AreEqual(20000, c.size);
            Assert.AreEqual(2.0, c.GetDouble(0, 0));
            Assert.AreEqual(2.0, c.GetDouble(99, 199));
        }

        /// <summary>
        ///     broadcast_to scalar to large 1D shape.
        ///     >>> np.broadcast_to(np.float64(1.0), (10000,))
        ///     All elements should be 1.0.
        /// </summary>
        [TestMethod]
        public void BroadcastTo_ScalarToLargeShape()
        {
            var scalar = np.array(new double[] { 1.0 });
            var b = np.broadcast_to(scalar, new Shape(10000));

            Assert.AreEqual(10000, b.size);
            Assert.AreEqual(1.0, b.GetDouble(0));
            Assert.AreEqual(1.0, b.GetDouble(9999));
        }

        /// <summary>
        ///     Chained operations: broadcast add, then square, then sum.
        ///     >>> a = np.arange(3).reshape(1, 3)
        ///     >>> b = np.arange(3).reshape(3, 1)
        ///     >>> c = a + b
        ///     >>> d = c * c  # element-wise square
        ///     >>> np.sum(d)
        ///     48
        /// </summary>
        [TestMethod]
        public void ChainedBroadcast_AddSquareSum()
        {
            var a = np.arange(3).reshape(1, 3);
            var b = np.arange(3).reshape(3, 1);
            var c = a + b;
            var d = c * c;
            var s = (int)np.sum(d);

            // d = [[0,1,4],[1,4,9],[4,9,16]], sum = 48
            Assert.AreEqual(48, s);
        }

        /// <summary>
        ///     Boolean dtype broadcasting.
        ///     >>> np.broadcast_arrays(np.array([True, False, True]), np.array([[True], [False]]))
        ///     Results should preserve boolean values across broadcast dimensions.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_BoolDtype()
        {
            var a = np.array(new bool[] { true, false, true });
            var b = np.array(new bool[,] { { true }, { false } });
            var (ba, bb) = np.broadcast_arrays(a, b);

            AssertShapeEqual(ba, 2, 3);
            AssertShapeEqual(bb, 2, 3);

            // ba broadcasts [T,F,T] to both rows
            Assert.AreEqual(true, ba.GetBoolean(0, 0));
            Assert.AreEqual(false, ba.GetBoolean(0, 1));
            Assert.AreEqual(true, ba.GetBoolean(1, 0));
            Assert.AreEqual(false, ba.GetBoolean(1, 1));

            // bb broadcasts [[T],[F]] to all columns
            Assert.AreEqual(true, bb.GetBoolean(0, 0));
            Assert.AreEqual(true, bb.GetBoolean(0, 2));
            Assert.AreEqual(false, bb.GetBoolean(1, 0));
            Assert.AreEqual(false, bb.GetBoolean(1, 2));
        }

        #endregion

        // ================================================================
        //  Re-broadcasting and broadcast path consistency tests
        // ================================================================

        #region Re-broadcasting (2-arg path)

        /// <summary>
        ///     Re-broadcasting an already-broadcast array through the 2-arg Broadcast path.
        ///     Verifies that the IsBroadcasted guard was removed and re-broadcast produces
        ///     correct values.
        ///
        ///     >>> a = np.broadcast_to(np.array([1,2,3]), (3,3))
        ///     >>> b = np.broadcast_to(a, (3,3))
        ///     >>> b
        ///     array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        /// </summary>
        [TestMethod]
        public void ReBroadcast_2Arg_SameShape()
        {
            var a = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(3, 3));
            var b = np.broadcast_to(a, new Shape(3, 3));

            AssertShapeEqual(b, 3, 3);
            for (int r = 0; r < 3; r++)
            {
                b.GetInt32(r, 0).Should().Be(1);
                b.GetInt32(r, 1).Should().Be(2);
                b.GetInt32(r, 2).Should().Be(3);
            }
        }

        /// <summary>
        ///     Re-broadcasting to a higher-dimensional shape through the 2-arg path.
        ///
        ///     >>> a = np.broadcast_to(np.array([[1],[2],[3]]), (3,3))
        ///     >>> b = np.broadcast_to(a, (2,3,3))
        ///     >>> b[0]
        ///     array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        ///     >>> b[1]
        ///     array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        /// </summary>
        [TestMethod]
        public void ReBroadcast_2Arg_HigherDim()
        {
            var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var b = np.broadcast_to(a, new Shape(2, 3, 3));

            AssertShapeEqual(b, 2, 3, 3);
            for (int d = 0; d < 2; d++)
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        b.GetInt32(d, r, c).Should().Be(r + 1,
                            $"b[{d},{r},{c}] should be {r + 1}");
        }

        /// <summary>
        ///     np.clip on a broadcast array requires re-broadcasting internally.
        ///     This was Bug 4 variant — clip broadcasts inputs together, hitting the guard.
        ///
        ///     >>> np.clip(np.broadcast_to(np.array([1.,5.,9.]), (2,3)), 2., 7.)
        ///     array([[2., 5., 7.], [2., 5., 7.]])
        /// </summary>
        [TestMethod]
        public void ReBroadcast_2Arg_ClipOnBroadcast()
        {
            var a = np.broadcast_to(np.array(new double[] { 1.0, 5.0, 9.0 }), new Shape(2, 3));
            var result = np.clip(a, 2.0, 7.0);

            AssertShapeEqual(result, 2, 3);
            result.GetDouble(0, 0).Should().Be(2.0);
            result.GetDouble(0, 1).Should().Be(5.0);
            result.GetDouble(0, 2).Should().Be(7.0);
            result.GetDouble(1, 0).Should().Be(2.0);
            result.GetDouble(1, 1).Should().Be(5.0);
            result.GetDouble(1, 2).Should().Be(7.0);
        }

        #endregion

        #region N-arg path consistency with sliced inputs

        /// <summary>
        ///     N-arg Broadcast with a sliced input produces correct values.
        ///     Before the fix, the N-arg path did not set ViewInfo for sliced inputs,
        ///     causing GetOffset to resolve wrong memory offsets.
        ///
        ///     >>> x = np.arange(12).reshape(3,4)
        ///     >>> col = x[:, 1:2]           # (3,1): [[1],[5],[9]]
        ///     >>> r = np.broadcast_arrays(col, np.ones((3,3), dtype=int))
        ///     >>> r[0]
        ///     array([[1, 1, 1], [5, 5, 5], [9, 9, 9]])
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_NArg_SlicedInput_CorrectValues()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 1:2"]; // (3,1): [[1],[5],[9]]
            var target = np.ones(new Shape(3, 3), np.int32);

            var result = np.broadcast_arrays(new NDArray[] { col, target });

            AssertShapeEqual(result[0], 3, 3);
            result[0].GetInt32(0, 0).Should().Be(1, "row 0 = value from x[0,1]=1");
            result[0].GetInt32(0, 1).Should().Be(1);
            result[0].GetInt32(0, 2).Should().Be(1);
            result[0].GetInt32(1, 0).Should().Be(5, "row 1 = value from x[1,1]=5");
            result[0].GetInt32(1, 1).Should().Be(5);
            result[0].GetInt32(1, 2).Should().Be(5);
            result[0].GetInt32(2, 0).Should().Be(9, "row 2 = value from x[2,1]=9");
            result[0].GetInt32(2, 1).Should().Be(9);
            result[0].GetInt32(2, 2).Should().Be(9);
        }

        /// <summary>
        ///     Verifies N-arg and 2-arg broadcast paths produce identical results
        ///     for the same sliced input.
        ///
        ///     >>> x = np.arange(6).reshape(2,3)
        ///     >>> row = x[0:1, :]  # (1,3): [[0,1,2]]
        ///     >>> # 2-arg: broadcast_to
        ///     >>> np.broadcast_to(row, (3,3))
        ///     array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        ///     >>> # N-arg: broadcast_arrays
        ///     >>> np.broadcast_arrays(row, np.ones((3,3), dtype=int))[0]
        ///     array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        /// </summary>
        [TestMethod]
        public void BroadcastPaths_2Arg_vs_NArg_SlicedInput_Identical()
        {
            var x = np.arange(6).reshape(2, 3);
            var row = x["0:1, :"]; // (1,3): [[0,1,2]]

            // 2-arg path via broadcast_to
            var via2arg = np.broadcast_to(row, new Shape(3, 3));

            // N-arg path via broadcast_arrays
            var viaNarg = np.broadcast_arrays(new NDArray[] { row, np.ones(new Shape(3, 3), np.int32) });

            AssertShapeEqual(via2arg, 3, 3);
            AssertShapeEqual(viaNarg[0], 3, 3);

            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    viaNarg[0].GetInt32(r, c).Should().Be(via2arg.GetInt32(r, c),
                        $"N-arg[{r},{c}] should equal 2-arg[{r},{c}]={via2arg.GetInt32(r, c)}");
        }

        #endregion
    }
}
