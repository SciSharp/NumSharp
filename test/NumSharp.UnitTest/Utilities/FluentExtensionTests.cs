using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Utilities
{
    /// <summary>
    /// Infrastructure tests for the custom <see cref="FluentExtension"/> assertions.
    /// Verifies that each assertion method produces correct pass/fail behavior
    /// and that error messages contain meaningful information.
    /// </summary>
    [TestClass]
    public class FluentExtensionTests
    {
        #region ShapeAssertions

        [TestMethod]
        public void Shape_BeOfSize_Passes_WhenCorrect()
        {
            new Shape(3, 4).Should().BeOfSize(12);
        }

        [TestMethod]
        public void Shape_BeOfSize_Fails_WhenWrong()
        {
            Action act = () => new Shape(3, 4).Should().BeOfSize(10);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_NotBeOfSize_Passes_WhenDifferent()
        {
            new Shape(3, 4).Should().NotBeOfSize(10);
        }

        [TestMethod]
        public void Shape_NotBeOfSize_Fails_WhenEqual()
        {
            Action act = () => new Shape(3, 4).Should().NotBeOfSize(12);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_BeShaped_Passes_WhenMatch()
        {
            new Shape(2, 3, 4).Should().BeShaped(2, 3, 4);
        }

        [TestMethod]
        public void Shape_BeShaped_Fails_WhenMismatch()
        {
            Action act = () => new Shape(2, 3, 4).Should().BeShaped(2, 4, 3);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_Be_Passes_WhenEqual()
        {
            var s = new Shape(5, 10);
            s.Should().Be(new Shape(5, 10));
        }

        [TestMethod]
        public void Shape_Be_Fails_WhenDifferent()
        {
            var s = new Shape(5, 10);
            Action act = () => s.Should().Be(new Shape(10, 5));
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_NotBe_Passes_WhenDifferent()
        {
            new Shape(5, 10).Should().NotBe(new Shape(10, 5));
        }

        [TestMethod]
        public void Shape_NotBe_Fails_WhenEqual()
        {
            Action act = () => new Shape(5, 10).Should().NotBe(new Shape(5, 10));
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_HaveNDim_Passes()
        {
            new Shape(2, 3, 4).Should().HaveNDim(3);
        }

        [TestMethod]
        public void Shape_HaveNDim_Fails()
        {
            Action act = () => new Shape(2, 3, 4).Should().HaveNDim(2);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_BeNDim_IsAliasForHaveNDim()
        {
            // BeNDim delegates to HaveNDim — verify both work identically
            new Shape(3, 4).Should().BeNDim(2);
            new Shape(3, 4).Should().HaveNDim(2);
        }

        [TestMethod]
        public void Shape_BeSliced_Passes_WhenSliced()
        {
            var a = np.arange(10);
            var sliced = a["2:8"];
            sliced.Shape.Should().BeSliced();
        }

        [TestMethod]
        public void Shape_NotBeSliced_Passes_WhenNotSliced()
        {
            np.arange(5).Shape.Should().NotBeSliced();
        }

        [TestMethod]
        public void Shape_BeScalar_Passes()
        {
            Shape.NewScalar().Should().BeScalar();
        }

        [TestMethod]
        public void Shape_NotBeScalar_Passes()
        {
            new Shape(3).Should().NotBeScalar();
        }

        [TestMethod]
        public void Shape_BeBroadcasted_Passes_WhenBroadcasted()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(3, 3));
            b.Shape.Should().BeBroadcasted();
        }

        [TestMethod]
        public void Shape_NotBeBroadcasted_Passes_WhenNotBroadcasted()
        {
            np.arange(6).reshape(2, 3).Shape.Should().NotBeBroadcasted();
        }

        [TestMethod]
        public void Shape_BeEquivalentTo_Validates_All_Parameters()
        {
            new Shape(2, 3).Should().BeEquivalentTo(size: 6, ndim: 2, shape: (2, 3));
        }

        [TestMethod]
        public void Shape_BeEquivalentTo_Fails_WhenDimensionCountMismatch()
        {
            // shape tuple has 3 elements but actual shape has 2 dimensions
            Action act = () => new Shape(2, 3).Should().BeEquivalentTo(shape: (2, 3, 1));
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Shape_Chaining_Works()
        {
            new Shape(2, 3).Should()
                .BeOfSize(6)
                .And.HaveNDim(2)
                .And.BeShaped(2, 3)
                .And.NotBeScalar()
                .And.NotBeBroadcasted();
        }

        #endregion

        #region NDArrayAssertions — Shape/Structure

        [TestMethod]
        public void NDArray_BeShaped_Passes()
        {
            np.arange(6).reshape(2, 3).Should().BeShaped(2, 3);
        }

        [TestMethod]
        public void NDArray_BeShaped_Fails()
        {
            Action act = () => np.arange(6).reshape(2, 3).Should().BeShaped(3, 2);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_BeShaped_WithSizeNdimTuple()
        {
            np.arange(6).reshape(2, 3).Should().BeShaped(size: 6, ndim: 2, shape: (2, 3));
        }

        [TestMethod]
        public void NDArray_BeShaped_ITuple_FailsOnDimensionCountMismatch()
        {
            // Bug 4 fix: should fail cleanly rather than IndexOutOfRangeException
            Action act = () => np.arange(6).reshape(2, 3).Should().BeShaped(shape: (2, 3, 1));
            act.Should().Throw<Exception>().Which.Should().NotBeOfType<IndexOutOfRangeException>();
        }

        [TestMethod]
        public void NDArray_BeOfSize_Passes()
        {
            np.arange(12).Should().BeOfSize(12);
        }

        [TestMethod]
        public void NDArray_HaveNDim_Passes()
        {
            np.arange(24).reshape(2, 3, 4).Should().HaveNDim(3);
        }

        [TestMethod]
        public void NDArray_BeNDim_IsAlias()
        {
            np.arange(6).reshape(2, 3).Should().BeNDim(2);
        }

        [TestMethod]
        public void NDArray_BeScalar_Passes()
        {
            NDArray.Scalar(42).Should().BeScalar();
        }

        [TestMethod]
        public void NDArray_BeScalar_WithValue_Passes()
        {
            NDArray.Scalar(42).Should().BeScalar(42);
        }

        [TestMethod]
        public void NDArray_BeScalar_WithValue_Fails()
        {
            Action act = () => NDArray.Scalar(42).Should().BeScalar(99);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_NotBeScalar_Passes()
        {
            np.arange(3).Should().NotBeScalar();
        }

        [TestMethod]
        public void NDArray_BeSliced_Passes()
        {
            var a = np.arange(10);
            a["2:5"].Should().BeSliced();
        }

        [TestMethod]
        public void NDArray_NotBeSliced_Passes()
        {
            np.arange(5).Should().NotBeSliced();
        }

        [TestMethod]
        public void NDArray_BeBroadcasted_Passes()
        {
            np.broadcast_to(np.array(new[] { 1, 2, 3 }), new Shape(3, 3)).Should().BeBroadcasted();
        }

        [TestMethod]
        public void NDArray_NotBeBroadcasted_Passes()
        {
            np.arange(6).Should().NotBeBroadcasted();
        }

        [TestMethod]
        public void NDArray_BeOfType_NPTypeCode_Passes()
        {
            np.arange(3).Should().BeOfType(NPTypeCode.Int32);
        }

        [TestMethod]
        public void NDArray_BeOfType_SystemType_Passes()
        {
            np.arange(3.0).Should().BeOfType(typeof(double));
        }

        [TestMethod]
        public void NDArray_BeOfType_Generic_Passes()
        {
            np.array(new float[] { 1, 2, 3 }).Should().BeOfType<float>();
        }

        #endregion

        #region NDArrayAssertions — Value Assertions

        [TestMethod]
        public void NDArray_Be_Passes_WhenEqual()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 2, 3 });
            a.Should().Be(b);
        }

        [TestMethod]
        public void NDArray_Be_Fails_WhenDifferent()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 2, 4 });
            Action act = () => a.Should().Be(b);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_BeOfValues_Int32_Passes()
        {
            np.array(new[] { 10, 20, 30 }).Should().BeOfValues(10, 20, 30);
        }

        [TestMethod]
        public void NDArray_BeOfValues_Int32_Fails_OnMismatch()
        {
            Action act = () => np.array(new[] { 10, 20, 30 }).Should().BeOfValues(10, 20, 99);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_BeOfValues_Double_Passes()
        {
            np.array(new[] { 1.5, 2.5, 3.5 }).Should().BeOfValues(1.5, 2.5, 3.5);
        }

        [TestMethod]
        public void NDArray_BeOfValues_Boolean_Passes()
        {
            var a = np.array(new[] { true, false, true });
            a.Should().BeOfValues(true, false, true);
        }

        [TestMethod]
        public void NDArray_BeOfValues_SizeMismatch_Fails()
        {
            Action act = () => np.array(new[] { 1, 2, 3 }).Should().BeOfValues(1, 2);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_BeOfValues_Float_Passes()
        {
            np.array(new float[] { 1f, 2f, 3f }).Should().BeOfValues(1f, 2f, 3f);
        }

        [TestMethod]
        public void NDArray_BeOfValues_Byte_Passes()
        {
            np.array(new byte[] { 0, 127, 255 }).Should().BeOfValues((byte)0, (byte)127, (byte)255);
        }

        [TestMethod]
        public void NDArray_BeOfValues_Int64_Passes()
        {
            np.array(new long[] { 100L, 200L, 300L }).Should().BeOfValues(100L, 200L, 300L);
        }

        [TestMethod]
        public void NDArray_AllValuesBe_Passes()
        {
            np.full(42, new Shape(3, 3)).Should().AllValuesBe(42);
        }

        [TestMethod]
        public void NDArray_AllValuesBe_Fails()
        {
            Action act = () => np.arange(9).Should().AllValuesBe(0);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_AllValuesBe_Double_Passes()
        {
            np.full(3.14, new Shape(2, 2)).Should().AllValuesBe(3.14);
        }

        [TestMethod]
        public void NDArray_BeOfValuesApproximately_Passes()
        {
            np.array(new[] { 1.001, 2.002, 3.003 }).Should()
                .BeOfValuesApproximately(0.01, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void NDArray_BeOfValuesApproximately_Fails_WhenOutOfTolerance()
        {
            Action act = () => np.array(new[] { 1.0, 2.0, 3.5 }).Should()
                .BeOfValuesApproximately(0.01, 1.0, 2.0, 3.0);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_BeOfValuesApproximately_Float_Passes()
        {
            np.array(new float[] { 1.001f, 2.002f }).Should()
                .BeOfValuesApproximately(0.01, 1.0f, 2.0f);
        }

        #endregion

        #region NDArrayAssertions — Error Message Quality

        [TestMethod]
        public void AllValuesBe_ErrorMessage_ContainsIndex_And_Values()
        {
            // Verifies Bug 1 fix: error messages used to show literal "0", "1", "2"
            // instead of actual expected value, actual value, and index.
            var a = np.array(new[] { 1, 2, 999 });
            try
            {
                a.Should().AllValuesBe(1);
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                // The message should contain the actual wrong value (2 or 999)
                // and the index where it diverged (1 or 2), not literal "0", "1", "2"
                Assert.IsTrue(msg.Contains("1th") || msg.Contains("2th"),
                    $"Error message should contain the diverging index, got: {msg}");
                Assert.IsTrue(msg.Contains("Subject"),
                    $"Error message should contain the Subject array, got: {msg}");
            }
        }

        [TestMethod]
        public void BeOfValues_ErrorMessage_ContainsIndex_And_Values()
        {
            var a = np.array(new[] { 10, 20, 30 });
            try
            {
                a.Should().BeOfValues(10, 20, 99);
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                Assert.IsTrue(msg.Contains("2th"),
                    $"Error message should contain the index (2), got: {msg}");
                Assert.IsTrue(msg.Contains("99"),
                    $"Error message should contain the expected value (99), got: {msg}");
                Assert.IsTrue(msg.Contains("30"),
                    $"Error message should contain the actual value (30), got: {msg}");
                Assert.IsTrue(msg.Contains("dtype: Int32"),
                    $"Error message should contain dtype, got: {msg}");
            }
        }

        [TestMethod]
        public void BeOfValuesApproximately_ErrorMessage_ShowsCorrectDtype()
        {
            // Verifies Bug 2 fix: all branches used to say "dtype: Boolean"
            var a = np.array(new double[] { 1.0, 2.0, 100.0 });
            try
            {
                a.Should().BeOfValuesApproximately(0.01, 1.0, 2.0, 3.0);
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                Assert.IsTrue(msg.Contains("dtype: Double"),
                    $"Error message should say 'dtype: Double' not 'dtype: Boolean', got: {msg}");
            }
        }

        [TestMethod]
        public void BeOfValuesApproximately_ErrorMessage_Int32_ShowsCorrectDtype()
        {
            var a = np.array(new int[] { 1, 2, 100 });
            try
            {
                a.Should().BeOfValuesApproximately(0.5, 1, 2, 3);
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                Assert.IsTrue(msg.Contains("dtype: Int32"),
                    $"Error message should say 'dtype: Int32', got: {msg}");
            }
        }

        [TestMethod]
        public void BeOfValuesApproximately_ErrorMessage_Single_ShowsCorrectDtype()
        {
            var a = np.array(new float[] { 1f, 2f, 100f });
            try
            {
                a.Should().BeOfValuesApproximately(0.01, 1f, 2f, 3f);
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                Assert.IsTrue(msg.Contains("dtype: Single"),
                    $"Error message should say 'dtype: Single', got: {msg}");
            }
        }

        #endregion

        #region NDArrayAssertions — Chaining

        [TestMethod]
        public void NDArray_Chaining_Values_And_Shape()
        {
            np.arange(6).reshape(2, 3).Should()
                .BeOfValues(0, 1, 2, 3, 4, 5)
                .And.BeShaped(2, 3)
                .And.HaveNDim(2)
                .And.NotBeScalar()
                .And.BeOfType(NPTypeCode.Int32);
        }

        [TestMethod]
        public void NDArray_Chaining_AllValuesBe_And_Shape()
        {
            np.full(7, new Shape(3, 3)).Should()
                .AllValuesBe(7)
                .And.BeShaped(3, 3)
                .And.BeOfSize(9);
        }

        [TestMethod]
        public void NDArray_Chaining_Approximate_And_Shape()
        {
            np.array(new[] { 1.001, 2.002 }).Should()
                .BeOfValuesApproximately(0.01, 1.0, 2.0)
                .And.BeShaped(2)
                .And.BeOfType<double>();
        }

        #endregion

        #region NDArrayAssertions — View/Broadcast Combinations

        [TestMethod]
        public void NDArray_SlicedArray_BeOfValues()
        {
            var a = np.arange(10);
            a["2:5"].Should().BeOfValues(2, 3, 4).And.BeSliced();
        }

        [TestMethod]
        public void NDArray_BroadcastedArray_AllValuesBe()
        {
            var a = np.broadcast_to(np.array(new[] { 5 }), new Shape(3));
            a.Should().AllValuesBe(5).And.BeBroadcasted();
        }

        [TestMethod]
        public void NDArray_2D_BeOfValues_RowMajorOrder()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            a.Should().BeOfValues(1, 2, 3, 4).And.BeShaped(2, 2);
        }

        #endregion

        #region UnmanagedStorage — Extension Entry Point

        [TestMethod]
        public void UnmanagedStorage_Should_ReturnsNDArrayAssertions()
        {
            var a = np.array(new[] { 1, 2, 3 });
            a.Storage.Should().BeOfValues(1, 2, 3);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void NDArray_BeOfValues_SingleElement()
        {
            np.array(new[] { 42 }).Should().BeOfValues(42);
        }

        [TestMethod]
        public void NDArray_AllValuesBe_SingleElement()
        {
            np.array(new[] { 42 }).Should().AllValuesBe(42);
        }

        [TestMethod]
        public void NDArray_BeOfValues_AllDtypes()
        {
            // Verify type switch covers all supported dtypes
            np.array(new bool[] { true, false }).Should().BeOfValues(true, false);
            np.array(new byte[] { 1, 2 }).Should().BeOfValues((byte)1, (byte)2);
            np.array(new short[] { 1, 2 }).Should().BeOfValues((short)1, (short)2);
            np.array(new ushort[] { 1, 2 }).Should().BeOfValues((ushort)1, (ushort)2);
            np.array(new int[] { 1, 2 }).Should().BeOfValues(1, 2);
            np.array(new uint[] { 1, 2 }).Should().BeOfValues(1u, 2u);
            np.array(new long[] { 1, 2 }).Should().BeOfValues(1L, 2L);
            np.array(new ulong[] { 1, 2 }).Should().BeOfValues(1UL, 2UL);
            np.array(new char[] { 'a', 'b' }).Should().BeOfValues('a', 'b');
            np.array(new double[] { 1.0, 2.0 }).Should().BeOfValues(1.0, 2.0);
            np.array(new float[] { 1f, 2f }).Should().BeOfValues(1f, 2f);
            np.array(new decimal[] { 1m, 2m }).Should().BeOfValues(1m, 2m);
        }

        [TestMethod]
        public void NDArray_AllValuesBe_AllDtypes()
        {
            np.full(true, new Shape(2)).Should().AllValuesBe(true);
            np.full((byte)5, new Shape(2)).Should().AllValuesBe((byte)5);
            np.full((short)5, new Shape(2)).Should().AllValuesBe((short)5);
            np.full((ushort)5, new Shape(2)).Should().AllValuesBe((ushort)5);
            np.full(5, new Shape(2)).Should().AllValuesBe(5);
            np.full(5u, new Shape(2)).Should().AllValuesBe(5u);
            np.full(5L, new Shape(2)).Should().AllValuesBe(5L);
            np.full(5UL, new Shape(2)).Should().AllValuesBe(5UL);
            np.full('x', new Shape(2)).Should().AllValuesBe('x');
            np.full(5.0, new Shape(2)).Should().AllValuesBe(5.0);
            np.full(5f, new Shape(2)).Should().AllValuesBe(5f);
            np.full(5m, new Shape(2)).Should().AllValuesBe(5m);
        }

        #endregion

        #region New Capabilities

        [TestMethod]
        public void NDArray_BeContiguous_Passes_ForFreshArray()
        {
            np.arange(6).Should().BeContiguous();
        }

        [TestMethod]
        public void NDArray_NotBeContiguous_Passes_ForSlicedWithStep()
        {
            np.arange(10)["::2"].Should().NotBeContiguous();
        }

        [TestMethod]
        public void Shape_BeContiguous_Passes()
        {
            np.arange(6).reshape(2, 3).Shape.Should().BeContiguous();
        }

        [TestMethod]
        public void Shape_HaveStrides_Passes()
        {
            // (2,3) C-order strides are (3,1)
            np.arange(6).reshape(2, 3).Shape.Should().HaveStrides(3, 1);
        }

        [TestMethod]
        public void Shape_HaveStrides_Fails_WhenWrong()
        {
            Action act = () => np.arange(6).reshape(2, 3).Shape.Should().HaveStrides(1, 3);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_HaveStrides_Passes()
        {
            np.arange(6).reshape(2, 3).Should().HaveStrides(3, 1);
        }

        [TestMethod]
        public void NDArray_BeEmpty_Passes()
        {
            new NDArray(NPTypeCode.Int32).Should().BeEmpty();
        }

        [TestMethod]
        public void NDArray_BeEmpty_Fails_WhenNotEmpty()
        {
            Action act = () => np.arange(3).Should().BeEmpty();
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_NotBeOfType_Passes()
        {
            np.arange(3).Should().NotBeOfType(NPTypeCode.Double);
        }

        [TestMethod]
        public void NDArray_NotBeOfType_Fails_WhenMatch()
        {
            Action act = () => np.arange(3).Should().NotBeOfType(NPTypeCode.Int32);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void NDArray_NotBeOfType_Generic_Passes()
        {
            np.arange(3).Should().NotBeOfType<double>();
        }

        [TestMethod]
        public void NDArray_NotBe_Passes_WhenDifferent()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 2, 4 });
            a.Should().NotBe(b);
        }

        [TestMethod]
        public void NDArray_NotBe_Fails_WhenEqual()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 2, 3 });
            Action act = () => a.Should().NotBe(b);
            act.Should().Throw<Exception>();
        }

        #endregion

        #region Correctness — Error Message Fixes

        [TestMethod]
        public void Shape_NotBe_ErrorMessage_SaysDidNotExpect()
        {
            // Verifies the inverted error message fix
            var s = new Shape(2, 3);
            try
            {
                s.Should().NotBe(new Shape(2, 3));
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains("Did not expect"),
                    $"Error message should say 'Did not expect', got: {ex.Message}");
            }
        }

        [TestMethod]
        public void NDArray_NotBeShaped_ErrorMessage_SaysDidNotExpect()
        {
            var a = np.arange(6).reshape(2, 3);
            try
            {
                a.Should().NotBeShaped(new Shape(2, 3));
                Assert.Fail("Should have thrown");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.Message.Contains("Did not expect"),
                    $"Error message should say 'Did not expect', got: {ex.Message}");
            }
        }

        [TestMethod]
        public void BeOfValuesApproximately_UInt64_BothDirections()
        {
            // Verifies UInt64 overflow fix: 3UL vs 5UL should have distance 2, not wrap around
            var a = np.array(new ulong[] { 5, 3 });
            a.Should().BeOfValuesApproximately(3.0, 3UL, 5UL);
        }

        #endregion
    }
}

