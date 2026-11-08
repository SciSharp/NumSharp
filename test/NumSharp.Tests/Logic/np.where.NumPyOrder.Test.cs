using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    ///     np.where(condition, x, y) as NumPy's <c>PyArray_Where</c> computes it, found when the np-layer dtype plan's P3
    ///     batch 7 moved its body to <c>NDArray.where</c> and replayed the snapshot against NumPy 2.4.2: the result dtype is
    ///     <c>PyArray_ResultType</c> under NEP 50 (a C# number literal is a weak Python scalar, every array - a 0-d one
    ///     included - and a NumPy scalar are strong), that dtype is resolved before the operands are broadcast, and a
    ///     broadcast failure carries the text of NumPy's iterator. Every expectation is NumPy 2.4.2's output.
    /// </summary>
    [TestClass]
    public class np_where_numpy_order_tests
    {
        /// <summary>A 0-d array is a strong operand (NEP 50): np.where(c, np.array(2), int8 array) is int64, where the
        ///     legacy rule NumSharp used read it as a weak scalar and returned int8.</summary>
        [TestMethod]
        public void ZeroDArray_IsStrong()
        {
            // np.where([True, False, True], np.array(2), np.array([5, 6, 7], np.int8)) -> int64 [2, 6, 2]
            var r = np.where(np.array(new[] { true, false, true }), np.array(2L), np.array(new sbyte[] { 5, 6, 7 }));
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(2L, 6L, 2L);
        }

        /// <summary>Two weak Python ints have the default integer dtype: np.where(c, 7, 7) is int64 (NumSharp returned the
        ///     C# literals' int32).</summary>
        [TestMethod]
        public void TwoLiterals_AreTheDefaultInteger()
        {
            // np.where([True, False, True], 7, 7) -> int64 [7, 7, 7]
            var r = np.where(np.array(new[] { true, false, true }), (object)7, (object)7);
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(7L, 7L, 7L);
        }

        /// <summary>A NumPy scalar is strong and a Python int adopts its dtype: np.where(c, np.float16(1.5), 7) is float16
        ///     (NumSharp returned float64).</summary>
        [TestMethod]
        public void NumPyScalar_IsStrong_ThePythonIntAdoptsIt()
        {
            // np.where([True, False, True], np.float16(1.5), 7) -> float16 [1.5, 7.0, 1.5]
            var r = np.where(np.array(new[] { true, false, true }), (object)(Half)1.5, (object)7);
            r.dtype.Should().Be(np.float16);
            r.ToArray<Half>().Should().Equal((Half)1.5, (Half)7, (Half)1.5);
        }

        /// <summary>A weak Python int takes the array's dtype and is cast into it unsafely, as NumPy's iterator casts it:
        ///     np.where([True], 300, int8 array) is int8 44.</summary>
        [TestMethod]
        public void WeakLiteral_WrapsIntoTheArrayDtype()
        {
            // np.where(np.array([True]), 300, np.zeros(1, np.int8)) -> int8 [44]
            var r = np.where(np.array(new[] { true }), 300, np.zeros(new Shape(1), np.int8));
            r.dtype.Should().Be(np.int8);
            r.ToArray<sbyte>().Should().Equal((sbyte)44);
        }

        /// <summary>A shape that does not broadcast is refused with NumPy's iterator text, every operand's shape listed and a
        ///     trailing space (NumSharp's was np.broadcast_arrays' "shape mismatch: objects cannot be broadcast to a single
        ///     shape").</summary>
        [TestMethod]
        public void BroadcastFailure_IsNumPysIteratorText()
        {
            // np.where(np.ones((3, 4), bool), np.zeros(7), np.zeros(())) ->
            //   ValueError: operands could not be broadcast together with shapes (3,4) (7,) ()
            var e = new Action(() => np.where(np.ones(new Shape(3, 4), np.@bool), np.zeros(new Shape(7)), np.zeros(Shape.Scalar)))
                .Should().Throw<IncorrectShapeException>().Which;
            e.Message.Should().Be("operands could not be broadcast together with shapes (3,4) (7,) () ");
        }

        /// <summary>The result dtype is resolved before the broadcast: an int32 x of a shape that does not broadcast beside a
        ///     StringDType y raises NumPy's DTypePromotionError, not the shape error.</summary>
        [TestMethod]
        public void PromotionError_ComesBeforeTheShapeError()
        {
            // np.where(np.ones((3, 4), bool), np.zeros(7, np.int32), <StringDType (3, 4)>) -> DTypePromotionError
            var text = np.array(new[] { "a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c", "d" })
                .astype(np.dtype("T")).reshape(new Shape(3, 4));
            var e = new Action(() => np.where(np.ones(new Shape(3, 4), np.@bool), np.zeros(new Shape(7), np.int32), text))
                .Should().Throw<DTypePromotionError>().Which;
            e.Message.Should().Be(
                "The DType <class 'numpy.dtypes.Int32DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>. " +
                "This means that no common DType exists for the given inputs. For example they cannot be stored in a single " +
                "array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.Int32DType'>, " +
                "<class 'numpy.dtypes.StringDType'>)");
        }
    }
}
