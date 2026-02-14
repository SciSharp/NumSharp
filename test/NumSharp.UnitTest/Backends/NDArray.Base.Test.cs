using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    /// Tests for NDArray.base property - NumPy compatibility.
    /// Port of NumPy's base-related tests from:
    /// - numpy/_core/tests/test_multiarray.py
    /// - numpy/_core/tests/test_indexing.py
    /// </summary>
    public class NDArray_Base_Test
    {
        #region NumPy Behavior Tests (Ported)

        /// <summary>
        /// NumPy: a = np.arange(10); a.base is None
        /// Original array owns its data.
        /// </summary>
        [Test]
        public void Base_OriginalArray_IsNull()
        {
            var a = np.arange(10);
            a.@base.Should().BeNull();
        }

        /// <summary>
        /// NumPy: b = a[2:5]; b.base is a
        /// Slice creates a view pointing to original.
        /// </summary>
        [Test]
        public void Base_Slice_PointsToOriginal()
        {
            var a = np.arange(10);
            var b = a["2:5"];

            b.@base.Should().NotBeNull();
            b.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: c = b[1:2]; c.base is a (chains to ORIGINAL, not b)
        /// Slice of slice chains to ultimate owner.
        /// </summary>
        [Test]
        public void Base_SliceOfSlice_ChainsToOriginal()
        {
            var a = np.arange(10);
            var b = a["2:8"];
            var c = b["1:3"];

            // c.base should point to a's storage, not b's
            c.@base.Should().NotBeNull();
            c.@base!.Storage.Should().BeSameAs(a.Storage);

            // Verify it's not pointing to b
            // (b.Storage is an alias of a.Storage, so we check the _baseStorage field)
            c.Storage._baseStorage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: d = a.copy(); d.base is None
        /// Copy owns its data.
        /// </summary>
        [Test]
        public void Base_Copy_IsNull()
        {
            var a = np.arange(10);
            var d = np.copy(a);

            d.@base.Should().BeNull();
        }

        /// <summary>
        /// NumPy: d = a.reshape(2,5); d.base is a
        /// Reshape returns view with base.
        /// </summary>
        [Test]
        public void Base_Reshape_PointsToOriginal()
        {
            var a = np.arange(10);
            var e = a.reshape(2, 5);

            e.@base.Should().NotBeNull();
            e.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: f = e.T; f.base is a (still chains to original)
        /// Transpose of reshape chains to original.
        /// </summary>
        [Test]
        public void Base_TransposeOfReshape_ChainsToOriginal()
        {
            var a = np.arange(10);
            var e = a.reshape(2, 5);
            var f = e.T;

            f.@base.Should().NotBeNull();
            f.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: g = np.broadcast_to(a, (3, 10)); g.base is a
        /// Broadcast creates view with base.
        /// </summary>
        [Test]
        public void Base_Broadcast_PointsToOriginal()
        {
            var a = np.arange(10);
            var g = np.broadcast_to(a, new Shape(3, 10));

            g.@base.Should().NotBeNull();
            g.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: h = np.expand_dims(a, 0); h.base is a
        /// expand_dims creates view with base.
        /// </summary>
        [Test]
        public void Base_ExpandDims_PointsToOriginal()
        {
            var a = np.arange(10);
            var h = np.expand_dims(a, 0);

            h.@base.Should().NotBeNull();
            h.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: a[...].base is a
        /// Ellipsis subscript creates a view.
        /// </summary>
        [Test]
        public void Base_EllipsisSubscript_PointsToOriginal()
        {
            var a = np.arange(10);
            var view = a["..."];

            view.@base.Should().NotBeNull();
            view.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: a[:].base is a
        /// Full slice creates a view.
        /// </summary>
        [Test]
        public void Base_FullSlice_PointsToOriginal()
        {
            var a = np.arange(10);
            var view = a[":"];

            view.@base.Should().NotBeNull();
            view.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: a.view().base is a
        /// view() creates a view with base.
        /// </summary>
        [Test]
        public void Base_View_PointsToOriginal()
        {
            var a = np.arange(10);
            var v = a.view();

            v.@base.Should().NotBeNull();
            v.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// NumPy: a.flatten().base is None
        /// flatten() creates a copy.
        /// </summary>
        [Test]
        public void Base_Flatten_IsNull()
        {
            var a = np.arange(12).reshape(3, 4);
            var f = a.flatten();

            f.@base.Should().BeNull();
        }

        /// <summary>
        /// NumPy: a.ravel().base is a (when contiguous)
        /// ravel() returns view when possible.
        /// </summary>
        [Test]
        public void Base_Ravel_Contiguous_PointsToOriginal()
        {
            // Start with an owned array (not a view)
            var a = np.arange(12);
            var reshaped = a.reshape(3, 4);
            var r = reshaped.ravel();

            // ravel of a view should chain to the original owner
            r.@base.Should().NotBeNull();
            r.@base!.Storage.Should().BeSameAs(a.Storage);

            // Also verify reshaped chains to a
            reshaped.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        #endregion

        #region Multiple View Operations

        /// <summary>
        /// Test a chain of operations all pointing to original.
        /// </summary>
        [Test]
        public void Base_ChainedOperations_AllPointToOriginal()
        {
            var original = np.arange(24);
            var reshaped = original.reshape(2, 3, 4);
            var sliced = reshaped["0, :, :"];
            var transposed = sliced.T;
            var expanded = np.expand_dims(transposed, 0);

            // All should chain to original
            reshaped.@base!.Storage.Should().BeSameAs(original.Storage);
            sliced.@base!.Storage.Should().BeSameAs(original.Storage);
            transposed.@base!.Storage.Should().BeSameAs(original.Storage);
            expanded.@base!.Storage.Should().BeSameAs(original.Storage);
        }

        /// <summary>
        /// Test that copies break the chain.
        /// </summary>
        [Test]
        public void Base_CopyBreaksChain()
        {
            var a = np.arange(10);
            var b = a["2:8"];
            var c = b.copy();  // Explicit copy
            var d = c["1:3"];

            // a -> b chains
            b.@base!.Storage.Should().BeSameAs(a.Storage);

            // c is a copy - owns its data
            c.@base.Should().BeNull();

            // d chains to c (the copy), not to a
            d.@base!.Storage.Should().BeSameAs(c.Storage);
            d.@base!.Storage.Should().NotBeSameAs(a.Storage);
        }

        #endregion

        #region Memory Safety Tests

        /// <summary>
        /// Verify that views keep the original data alive.
        /// This tests that the shared memory isn't prematurely freed.
        /// </summary>
        [Test]
        public void Base_ViewKeepsDataAlive()
        {
            NDArray view;
            int expectedValue;

            // Create original in inner scope
            {
                var original = np.arange(10);
                view = original["2:5"];
                expectedValue = (int)original.GetInt32(2);
            }

            // Force GC
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // View should still have valid data
            view.GetInt32(0).Should().Be(expectedValue);
        }

        /// <summary>
        /// Verify nested views keep data alive through the chain.
        /// </summary>
        [Test]
        public void Base_NestedViews_KeepDataAlive()
        {
            NDArray deepView;

            {
                var a = np.arange(100);
                var b = a["10:90"];
                var c = b["10:70"];
                deepView = c["10:50"];
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // deepView should still be valid
            deepView.size.Should().Be(40);
            deepView.GetInt32(0).Should().Be(30); // a[30]
        }

        /// <summary>
        /// Verify broadcast views keep original data alive.
        /// </summary>
        [Test]
        public void Base_BroadcastView_KeepsDataAlive()
        {
            NDArray broadcasted;
            int[] expectedValues;

            {
                var original = np.array(new[] { 1, 2, 3 });
                broadcasted = np.broadcast_to(original, new Shape(3, 3));
                expectedValues = new[] { 1, 2, 3 };
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Verify broadcasted data is still valid
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    broadcasted.GetInt32(row, col).Should().Be(expectedValues[col]);
                }
            }
        }

        /// <summary>
        /// Verify reshape views keep original data alive.
        /// </summary>
        [Test]
        public void Base_ReshapeView_KeepsDataAlive()
        {
            NDArray reshaped;

            {
                var original = np.arange(12);
                reshaped = original.reshape(3, 4);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Verify data integrity
            int expected = 0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    reshaped.GetInt32(i, j).Should().Be(expected++);
                }
            }
        }

        /// <summary>
        /// Verify transpose views keep original data alive.
        /// </summary>
        [Test]
        public void Base_TransposeView_KeepsDataAlive()
        {
            NDArray transposed;

            {
                var original = np.arange(6).reshape(2, 3);
                transposed = original.T;
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Verify transposed data
            transposed.Shape.dimensions.Should().BeEquivalentTo(new[] { 3, 2 });
            transposed.GetInt32(0, 0).Should().Be(0);
            transposed.GetInt32(0, 1).Should().Be(3);
            transposed.GetInt32(1, 0).Should().Be(1);
            transposed.GetInt32(1, 1).Should().Be(4);
        }

        #endregion

        #region View Detection Tests

        /// <summary>
        /// Test that we can detect if an array is a view using .base
        /// </summary>
        [Test]
        public void Base_CanDetectView()
        {
            var owner = np.arange(10);
            var view = owner["2:5"];
            var copy = owner.copy();

            IsView(owner).Should().BeFalse();
            IsView(view).Should().BeTrue();
            IsView(copy).Should().BeFalse();
        }

        private static bool IsView(NDArray arr) => arr.Storage._baseStorage != null;

        #endregion

        #region Broadcast Arrays Tests

        /// <summary>
        /// Test broadcast_arrays returns views with proper base.
        /// </summary>
        [Test]
        public void Base_BroadcastArrays_ReturnViews()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[,] { { 1 }, { 2 }, { 3 } });

            var (ba, bb) = np.broadcast_arrays(a, b);

            // Both should be views
            ba.@base.Should().NotBeNull();
            bb.@base.Should().NotBeNull();

            // ba chains to a
            ba.@base!.Storage.Should().BeSameAs(a.Storage);

            // bb chains to b
            bb.@base!.Storage.Should().BeSameAs(b.Storage);
        }

        #endregion

        #region Reduction With Keepdims Tests

        /// <summary>
        /// When reduction with keepdims on axis with size 1, returns view.
        /// </summary>
        [Test]
        public void Base_ReductionKeepdims_Size1Axis_ReturnsView()
        {
            // Start with an owned array (not a view)
            var original = np.arange(6);
            var a = original.reshape(1, 6);

            // sum along axis 0 (size 1) with keepdims should return view
            var result = np.sum(a, axis: 0, keepdims: true);

            // Result should be a view chaining to the original owner
            result.@base.Should().NotBeNull();
            result.@base!.Storage.Should().BeSameAs(original.Storage);
        }

        #endregion

        #region Storage-Level Verification Tests

        /// <summary>
        /// Verify _baseStorage is set correctly at storage level.
        /// </summary>
        [Test]
        public void BaseStorage_Alias_SetsBaseStorage()
        {
            var original = np.arange(10);
            var alias = original.Storage.Alias();

            alias._baseStorage.Should().BeSameAs(original.Storage);
        }

        /// <summary>
        /// Verify _baseStorage chains correctly through multiple aliases.
        /// </summary>
        [Test]
        public void BaseStorage_MultipleAliases_ChainToOriginal()
        {
            var original = np.arange(10);
            var alias1 = original.Storage.Alias();
            var alias2 = alias1.Alias();
            var alias3 = alias2.Alias();

            // All should chain to original
            alias1._baseStorage.Should().BeSameAs(original.Storage);
            alias2._baseStorage.Should().BeSameAs(original.Storage);
            alias3._baseStorage.Should().BeSameAs(original.Storage);
        }

        /// <summary>
        /// Verify Clone does not set _baseStorage.
        /// </summary>
        [Test]
        public void BaseStorage_Clone_IsNull()
        {
            var original = np.arange(10);
            var cloned = original.Storage.Clone();

            cloned._baseStorage.Should().BeNull();
        }

        /// <summary>
        /// Verify broadcast_to uses CreateBroadcastedUnsafe which sets _baseStorage.
        /// </summary>
        [Test]
        public void BaseStorage_BroadcastTo_SetsBaseStorage()
        {
            var original = np.arange(3);
            var broadcasted = np.broadcast_to(original, new Shape(3, 3));

            // The NDArray wraps storage with _baseStorage set
            broadcasted.Storage._baseStorage.Should().BeSameAs(original.Storage);
        }

        #endregion

        #region Edge Cases

        /// <summary>
        /// Scalar arrays should work correctly.
        /// </summary>
        [Test]
        public void Base_Scalar_Works()
        {
            var scalar = NDArray.Scalar(42);

            scalar.@base.Should().BeNull(); // Owns its data

            var view = scalar[":"];
            view.@base.Should().NotBeNull();
            view.@base!.Storage.Should().BeSameAs(scalar.Storage);
        }

        /// <summary>
        /// Empty array handling.
        /// </summary>
        [Test]
        public void Base_EmptyArray_Works()
        {
            var empty = np.empty(new Shape(0));

            empty.@base.Should().BeNull();
        }

        /// <summary>
        /// 0-size dimension handling.
        /// </summary>
        [Test]
        public void Base_ZeroSizeDimension_Works()
        {
            var a = np.empty(new Shape(3, 0, 4));

            a.@base.Should().BeNull();

            // Slicing should still work
            var view = a[":, :, :"];
            view.@base.Should().NotBeNull();
        }

        /// <summary>
        /// Test with various dtypes.
        /// </summary>
        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        public void Base_AllDTypes_Work(NPTypeCode dtype)
        {
            var a = np.zeros(new Shape(10), dtype);
            var view = a["2:5"];

            view.@base.Should().NotBeNull();
            view.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        #endregion

        #region Negative Stride Tests

        /// <summary>
        /// Reversed array creates view.
        /// </summary>
        [Test]
        public void Base_ReversedArray_PointsToOriginal()
        {
            var a = np.arange(10);
            var reversed = a["::-1"];

            reversed.@base.Should().NotBeNull();
            reversed.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        /// <summary>
        /// Step slicing creates view.
        /// </summary>
        [Test]
        public void Base_StepSlice_PointsToOriginal()
        {
            var a = np.arange(20);
            var stepped = a["::3"];

            stepped.@base.Should().NotBeNull();
            stepped.@base!.Storage.Should().BeSameAs(a.Storage);
        }

        #endregion
    }
}
