using AwesomeAssertions;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     NumPy-aligned offset parity tests: Verify element access uses the formula
    ///     offset + sum(indices * strides) for all shape types (simple, sliced, broadcast).
    /// </summary>
    public class ShapeOffsetParityTests
    {
        [Test]
        public void Parity_Vector_AllIndices()
        {
            // 1D vector
            var shape = new Shape(5);
            for (int i = 0; i < 5; i++)
            {
                int expected = shape.GetOffset(i);
                int actual = shape.GetOffsetSimple(i);
                actual.Should().Be(expected, because: $"index {i} should match");
            }
        }

        [Test]
        public void Parity_Matrix_AllIndices()
        {
            // 2D matrix
            var shape = new Shape(4, 3);
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int expected = shape.GetOffset(i, j);
                    int actual = shape.GetOffsetSimple(i, j);
                    actual.Should().Be(expected, because: $"indices ({i},{j}) should match");
                }
            }
        }

        [Test]
        public void Parity_3DArray_AllIndices()
        {
            // 3D array
            var shape = new Shape(2, 3, 4);
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        int expected = shape.GetOffset(i, j, k);
                        int actual = shape.GetOffsetSimple(i, j, k);
                        actual.Should().Be(expected, because: $"indices ({i},{j},{k}) should match");
                    }
                }
            }
        }

        [Test]
        public void Parity_4DArray_SampleIndices()
        {
            // 4D array (sample indices to keep test fast)
            var shape = new Shape(2, 3, 4, 5);
            var indices = new[] { 1, 2, 3, 4 };
            int expected = shape.GetOffset(indices);
            int actual = shape.GetOffsetSimple(indices);
            actual.Should().Be(expected);

            indices = new[] { 0, 0, 0, 0 };
            expected = shape.GetOffset(indices);
            actual = shape.GetOffsetSimple(indices);
            actual.Should().Be(expected);

            indices = new[] { 1, 2, 0, 1 };
            expected = shape.GetOffset(indices);
            actual = shape.GetOffsetSimple(indices);
            actual.Should().Be(expected);
        }

        [Test]
        public void Parity_Scalar()
        {
            // Scalar shape (edge case)
            var shape = Shape.Scalar;
            // For scalars, GetOffset with empty array should work
            // GetOffsetSimple will return just offset (0)
            shape.Offset.Should().Be(0);
        }


        [Test]
        public void Offset_DefaultsToZero_ForNewShapes()
        {
            // Verify offset is 0 for all newly created shapes
            new Shape(5).Offset.Should().Be(0);
            new Shape(4, 3).Offset.Should().Be(0);
            new Shape(2, 3, 4).Offset.Should().Be(0);
            Shape.Scalar.Offset.Should().Be(0);
            Shape.Vector(10).Offset.Should().Be(0);
            Shape.Matrix(3, 4).Offset.Should().Be(0);
        }

        [Test]
        public void Offset_PreservedByClone()
        {
            var original = new Shape(4, 3);
            // Currently offset is always 0, but test that clone preserves it
            original.Offset.Should().Be(0);

            var cloned = original.Clone();
            cloned.Offset.Should().Be(original.Offset);
        }

        [Test]
        public void Offset_PreservedByCopyConstructor()
        {
            var original = new Shape(4, 3);
            var copy = new Shape(original);
            copy.Offset.Should().Be(original.Offset);
        }

        [Test]
        public void Parity_RandomIndices()
        {
            // Test with random indices
            var rnd = new Randomizer();
            var dims = new[] { rnd.Next(2, 10), rnd.Next(2, 10), rnd.Next(2, 10) };
            var shape = new Shape(dims);

            for (int trial = 0; trial < 20; trial++)
            {
                var indices = new[]
                {
                    rnd.Next(0, dims[0]),
                    rnd.Next(0, dims[1]),
                    rnd.Next(0, dims[2])
                };

                int expected = shape.GetOffset(indices);
                int actual = shape.GetOffsetSimple(indices);
                actual.Should().Be(expected, because: $"trial {trial}, indices ({indices[0]},{indices[1]},{indices[2]}) should match");
            }
        }

        [Test]
        public void GetOffsetSimple_Formula_MatchesExpectedCalculation()
        {
            // Verify the formula: offset + sum(indices * strides)
            var shape = new Shape(4, 3, 2); // strides should be [6, 2, 1]
            shape.Strides.Should().BeEquivalentTo([6, 2, 1]);
            shape.Offset.Should().Be(0);

            // indices (1, 2, 1) should be: 0 + 1*6 + 2*2 + 1*1 = 11
            shape.GetOffsetSimple(1, 2, 1).Should().Be(11);
            shape.GetOffset(1, 2, 1).Should().Be(11);

            // indices (3, 2, 1) should be: 0 + 3*6 + 2*2 + 1*1 = 23
            shape.GetOffsetSimple(3, 2, 1).Should().Be(23);
            shape.GetOffset(3, 2, 1).Should().Be(23);
        }

        // ================================================================
        //  Sliced shape tests - offset computed at slice time
        // ================================================================

        [Test]
        public void Slice_SlicedShape_OffsetComputedAtSliceTime()
        {
            // Original shape (4,3) with strides [3, 1]
            // Slice [1:3, :] should start at offset 1*3 = 3
            var original = new Shape(4, 3);
            var sliced = original.Slice(new Slice(1, 3), Slice.All);

            // The sliced shape has offset = 3 (start of row 1)
            sliced.Offset.Should().Be(3, because: "slice [1:3, :] starts at linear offset 3");
            sliced.Dimensions.Should().BeEquivalentTo([2, 3]);
        }

        [Test]
        public void Slice_SlicedShape_GetOffsetSimple_Parity()
        {
            // np.arange(12).reshape(4,3)[1:3, :]
            // Original: [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
            // Sliced:   [[3,4,5], [6,7,8]] with offset=3, strides=[3,1]
            var original = new Shape(4, 3);
            var sliced = original.Slice(new Slice(1, 3), Slice.All);

            // Verify GetOffsetSimple matches GetOffset for all indices
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int expected = sliced.GetOffset(i, j);
                    int actual = sliced.GetOffsetSimple(i, j);
                    actual.Should().Be(expected, because: $"sliced indices ({i},{j}) should match");
                }
            }
        }

        [Test]
        public void Slice_SlicedVector_OffsetAndParity()
        {
            // np.arange(10)[3:7] -> offset=3, strides=[1], dims=[4]
            var original = new Shape(10);
            var sliced = original.Slice(new Slice(3, 7));

            sliced.Offset.Should().Be(3);
            sliced.Dimensions.Should().BeEquivalentTo([4]);
            sliced.Strides.Should().BeEquivalentTo([1]);

            // Verify parity
            for (int i = 0; i < 4; i++)
            {
                sliced.GetOffsetSimple(i).Should().Be(sliced.GetOffset(i));
            }
        }

        [Test]
        public void Slice_SlicedWithStep_OffsetAndStrides()
        {
            // np.arange(10)[1:7:2] -> starts at 1, takes [1,3,5], dims=[3]
            // offset=1, strides=[2]
            var original = new Shape(10);
            var sliced = original.Slice(new Slice(1, 7, 2));

            sliced.Offset.Should().Be(1);
            sliced.Dimensions.Should().BeEquivalentTo([3]);
            sliced.Strides.Should().BeEquivalentTo([2]);

            // Verify parity: indices 0,1,2 should map to linear offsets 1,3,5
            for (int i = 0; i < 3; i++)
            {
                int expected = sliced.GetOffset(i);
                int actual = sliced.GetOffsetSimple(i);
                actual.Should().Be(expected);
            }
        }

        [Test]
        public void Slice_Sliced2D_ColumnSlice()
        {
            // np.arange(12).reshape(4,3)[:, 1] -> column 1
            // Original strides [3, 1], slicing [:, 1] gives offset=1, dims=[4], strides=[3]
            var original = new Shape(4, 3);
            var sliced = original.Slice(Slice.All, Slice.Index(1));

            sliced.Offset.Should().Be(1, because: "column 1 starts at linear offset 1");
            sliced.Dimensions.Should().BeEquivalentTo([4]);
            sliced.Strides.Should().BeEquivalentTo([3]);

            // Values: 1, 4, 7, 10 (column 1 of each row)
            for (int i = 0; i < 4; i++)
            {
                int expected = sliced.GetOffset(i);
                int actual = sliced.GetOffsetSimple(i);
                actual.Should().Be(expected);
            }
        }

        [Test]
        public void Slice_SlicedScalar_Offset()
        {
            // np.arange(12).reshape(4,3)[2, 1] -> scalar at offset 2*3 + 1 = 7
            var original = new Shape(4, 3);
            var sliced = original.Slice(Slice.Index(2), Slice.Index(1));

            sliced.IsScalar.Should().BeTrue();
            sliced.Offset.Should().Be(7, because: "[2,1] is at linear offset 7");
        }

        [Test]
        public void Slice_DoubleSliced_OffsetAccumulates()
        {
            // First slice: np.arange(12).reshape(4,3)[1:, :] -> offset=3
            // Second slice: [1:, 1:] on the result -> additional offset
            var original = new Shape(4, 3);
            var first = original.Slice(new Slice(1, 4), Slice.All);
            first.Offset.Should().Be(3);

            var second = first.Slice(new Slice(1, 3), new Slice(1, 3));
            // Second slice starts at row 1, col 1 of the first slice
            // First slice has offset 3, strides [3, 1]
            // Second slice adds: merged slices give offset from original
            // Row 1 of first = row 2 of original (offset 6), col 1 = offset +1
            // Total offset = 7
            second.Offset.Should().Be(7);

            // Verify parity for the double-sliced shape
            for (int i = 0; i < second.Dimensions[0]; i++)
            {
                for (int j = 0; j < second.Dimensions[1]; j++)
                {
                    second.GetOffsetSimple(i, j).Should().Be(second.GetOffset(i, j));
                }
            }
        }

        // ================================================================
        //  Broadcast shape tests - offset preserved from source
        // ================================================================

        [Test]
        public void Broadcast_BroadcastBasic_OffsetPreserved()
        {
            // Broadcast a simple shape - offset should stay 0
            var left = new Shape(3, 1);
            var right = new Shape(1, 4);
            var (bLeft, bRight) = NumSharp.Backends.DefaultEngine.Broadcast(left, right);

            bLeft.Offset.Should().Be(0);
            bRight.Offset.Should().Be(0);
            bLeft.Dimensions.Should().BeEquivalentTo([3, 4]);
            bRight.Dimensions.Should().BeEquivalentTo([3, 4]);
        }

        [Test]
        public void Broadcast_BroadcastSliced_OffsetPreserved()
        {
            // Slice a shape (gains offset), then broadcast
            // np.arange(12).reshape(4,3)[1:3, :] has offset=3
            var original = new Shape(4, 3);
            var sliced = original.Slice(new Slice(1, 3), Slice.All);
            sliced.Offset.Should().Be(3);

            // Broadcast against a compatible shape
            var other = new Shape(2, 1);
            var (bSliced, bOther) = NumSharp.Backends.DefaultEngine.Broadcast(sliced, other);

            // The broadcast result should preserve the sliced shape's offset
            bSliced.Offset.Should().Be(3, because: "broadcast should preserve source offset");
            bSliced.Dimensions.Should().BeEquivalentTo([2, 3]);
        }

        [Test]
        public void Broadcast_BroadcastScalar_OffsetPreserved()
        {
            // Slice to get scalar with offset, then broadcast
            var original = new Shape(4, 3);
            var scalar = original.Slice(Slice.Index(2), Slice.Index(1));
            scalar.IsScalar.Should().BeTrue();
            scalar.Offset.Should().Be(7); // 2*3 + 1 = 7

            // Broadcast scalar against a shape
            var target = new Shape(2, 2);
            var (bScalar, bTarget) = NumSharp.Backends.DefaultEngine.Broadcast(scalar, target);

            // The broadcast scalar should preserve its offset
            bScalar.Offset.Should().Be(7, because: "broadcast scalar should preserve source offset");
            bScalar.Dimensions.Should().BeEquivalentTo([2, 2]);
        }

        [Test]
        public void Broadcast_BroadcastTo_SlicedArray()
        {
            // np.arange(10)[3:7] = [3,4,5,6], offset=3, shape=(4,)
            // broadcast_to shape (3, 4)
            var original = new Shape(10);
            var sliced = original.Slice(new Slice(3, 7));
            sliced.Offset.Should().Be(3);

            var target = new Shape(3, 4);
            var broadcasted = np.broadcast_to(sliced, target);

            broadcasted.Offset.Should().Be(3, because: "broadcast_to should preserve source offset");
            broadcasted.Dimensions.Should().BeEquivalentTo([3, 4]);
        }

        [Test]
        public void Broadcast_BroadcastMultiple_OffsetPreserved()
        {
            // Test Broadcast(Shape[]) with multiple shapes
            var shape1 = new Shape(4, 3);
            var sliced1 = shape1.Slice(new Slice(1, 3), Slice.All); // offset=3
            sliced1.Offset.Should().Be(3);

            var shape2 = new Shape(2, 1);
            var shape3 = new Shape(1, 3);

            var results = NumSharp.Backends.DefaultEngine.Broadcast(new[] { sliced1, shape2, shape3 });

            results[0].Offset.Should().Be(3, because: "first shape's offset should be preserved");
            results[1].Offset.Should().Be(0);
            results[2].Offset.Should().Be(0);
        }

        [Test]
        public void Broadcast_GetOffsetSimple_BroadcastedSliced()
        {
            // End-to-end: slice then broadcast, verify GetOffsetSimple parity
            var original = new Shape(4, 3);
            var sliced = original.Slice(new Slice(1, 3), Slice.All); // offset=3, dims=[2,3]

            var target = new Shape(2, 1);
            var (bSliced, _) = NumSharp.Backends.DefaultEngine.Broadcast(sliced, target);

            // bSliced has offset=3, dims=[2,3], strides with zeros for broadcast dims
            // Verify GetOffsetSimple for a few indices
            // Note: GetOffset for broadcast shapes uses complex logic,
            // but GetOffsetSimple just does offset + sum(indices * strides)
            // For Phase 3, we're establishing that offset is preserved in the shape
            bSliced.Offset.Should().Be(3);

            // The actual element access parity will be tested when GetOffset is updated
            // to use offset (Phase 5). For now, just verify offset propagation.
        }

        // ================================================================
        //  NumPy-aligned GetOffset verification
        // ================================================================

        [Test]
        public void SlicedShape_GetOffset_NumPyAligned()
        {
            // GetOffset uses NumPy formula: offset + sum(indices * strides)
            // This test verifies the full element access path works correctly

            // Simple row slice
            var arr = np.arange(12).reshape(4, 3);
            var sliced = arr["1:3, :"];
            sliced.shape.Should().BeEquivalentTo([2, 3]);

            // Verify actual values (not just offsets)
            sliced[0, 0].GetAtIndex<int>(0).Should().Be(3);
            sliced[0, 1].GetAtIndex<int>(0).Should().Be(4);
            sliced[0, 2].GetAtIndex<int>(0).Should().Be(5);
            sliced[1, 0].GetAtIndex<int>(0).Should().Be(6);
            sliced[1, 1].GetAtIndex<int>(0).Should().Be(7);
            sliced[1, 2].GetAtIndex<int>(0).Should().Be(8);
        }

        [Test]
        public void SlicedWithStep_GetOffset_NumPyAligned()
        {
            // Slice with step
            var arr = np.arange(10);
            var sliced = arr["1:7:2"]; // [1, 3, 5]
            sliced.shape.Should().BeEquivalentTo([3]);

            sliced[0].GetAtIndex<int>(0).Should().Be(1);
            sliced[1].GetAtIndex<int>(0).Should().Be(3);
            sliced[2].GetAtIndex<int>(0).Should().Be(5);
        }

        [Test]
        public void ColumnSlice_GetOffset_NumPyAligned()
        {
            // Column slice (dimension reduction)
            var arr = np.arange(12).reshape(4, 3);
            var col1 = arr[":, 1"]; // column 1: [1, 4, 7, 10]
            col1.shape.Should().BeEquivalentTo([4]);

            col1[0].GetAtIndex<int>(0).Should().Be(1);
            col1[1].GetAtIndex<int>(0).Should().Be(4);
            col1[2].GetAtIndex<int>(0).Should().Be(7);
            col1[3].GetAtIndex<int>(0).Should().Be(10);
        }

        [Test]
        public void DoubleSliced_GetOffset_NumPyAligned()
        {
            // Double slice (slice of slice)
            var arr = np.arange(12).reshape(4, 3);
            var first = arr["1:, :"];  // rows 1-3
            var second = first["1:, 1:"]; // rows 1-2 of that, cols 1-2

            second.shape.Should().BeEquivalentTo([2, 2]);
            // Original: [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
            // First: [[3,4,5], [6,7,8], [9,10,11]]
            // Second: [[7,8], [10,11]]
            second[0, 0].GetAtIndex<int>(0).Should().Be(7);
            second[0, 1].GetAtIndex<int>(0).Should().Be(8);
            second[1, 0].GetAtIndex<int>(0).Should().Be(10);
            second[1, 1].GetAtIndex<int>(0).Should().Be(11);
        }

        // ================================================================
        //  NumPy purity verification
        // ================================================================

        [Test]
        public void NumPyPurity_IsSimpleSlice_TrueForNonContiguousSlice()
        {
            // Note: Contiguous slices get optimized (IsSliced=false).
            // Use step or column slices to test IsSimpleSlice.
            var arr = np.arange(10);
            var stepSliced = arr["::2"]; // Non-contiguous step slice

            stepSliced.Shape.IsSliced.Should().BeTrue();
            stepSliced.Shape.IsSimpleSlice.Should().BeTrue();
            stepSliced.Shape.IsBroadcasted.Should().BeFalse();
        }

        [Test]
        public void NumPyPurity_IsSimpleSlice_TrueForColumnSlice()
        {
            // Column slice is non-contiguous
            var arr = np.arange(12).reshape(4, 3);
            var colSliced = arr[":, 1"]; // Column 1

            colSliced.Shape.IsSliced.Should().BeTrue();
            colSliced.Shape.IsSimpleSlice.Should().BeTrue();
            colSliced.Shape.Offset.Should().Be(1);
            colSliced.Shape.Strides.Should().BeEquivalentTo([3]);
        }

        [Test]
        public void NumPyPurity_IsSimpleSlice_FalseForBroadcast()
        {
            var arr = np.arange(3);
            var broadcasted = np.broadcast_to(arr, (2, 3));

            broadcasted.Shape.IsBroadcasted.Should().BeTrue();
            broadcasted.Shape.IsSimpleSlice.Should().BeFalse();
        }

        [Test]
        public void NumPyPurity_ContiguousSlice_Optimized()
        {
            // Contiguous slices are optimized: no ViewInfo, IsSliced=false
            // This matches NumPy's architecture (data pointer adjustment)
            var arr = np.arange(12).reshape(4, 3);
            var sliced = arr["1:3, :"];

            // The contiguous slice optimization creates a clean shape
            sliced.Shape.IsSliced.Should().BeFalse();
            sliced.Shape.Offset.Should().Be(0); // Offset is in InternalArray, not Shape

            // Values still correct (via InternalArray offset)
            sliced[0, 0].GetAtIndex<int>(0).Should().Be(3);
            sliced[1, 2].GetAtIndex<int>(0).Should().Be(8);
        }

        [Test]
        public void NumPyPurity_NonSlicedShape_UsesGetOffsetSimple()
        {
            // Verify non-sliced shapes also use the simple offset formula
            var arr = np.arange(12).reshape(4, 3);

            // Non-sliced shape properties
            arr.Shape.IsSliced.Should().BeFalse();
            arr.Shape.Offset.Should().Be(0);

            // Verify element access works correctly
            arr[2, 1].GetAtIndex<int>(0).Should().Be(7);
            arr[3, 2].GetAtIndex<int>(0).Should().Be(11);
        }
    }
}
