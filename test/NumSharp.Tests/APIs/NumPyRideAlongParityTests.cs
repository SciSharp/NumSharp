using System;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     Numeric NumPy-parity fixes found while validating the string dtypes: the shape-decided answers run the numeric
    ///     path, and the text calls exposed the numeric path's order and layout. Each test pins one NumPy 2.4.2 behavior
    ///     (probed) that NumSharp did not have before.
    /// </summary>
    [TestClass]
    public class NumPyRideAlongParityTests
    {
        private static NDArray F() => np.array(new[] { 1.5, 2.0, 3.0 });

        // ---------------------------------------------------------------- sorting, unique, concatenate, reductions

        [TestMethod]
        public void Argsort_Of0d_RavelsFirst()
        {
            // np.argsort(np.array(5.0)) -> array([0]) (shape (1,)).
            var r = np.argsort(np.array(5.0));
            r.shape.Should().Equal(new long[] { 1 });
            r.GetInt64(0).Should().Be(0L);
        }

        [TestMethod]
        public void Unique_OfAnEmpty2D_InverseHasTheInputShape_AsAView()
        {
            // np.unique(np.zeros((0, 3)), return_inverse=True)[1]: shape (0, 3), a view (flags.owndata False).
            var r = (NDArray[])np.unique(np.zeros(new Shape(0L, 3L)), false, true);
            r[1].shape.Should().Equal(new long[] { 0, 3 });
            r[1].dtype.Should().Be(np.int64);
            r[1].flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void Unique_1DWithAxis0_IsTheFlatPath()
        {
            // np.unique([3, 1, 3], True, True, True, axis=0) -> ([1, 3], [1, 0], [1, 0, 1], [1, 2]); the inverse is a view.
            var r = (NDArray[])np.unique(np.array(new long[] { 3, 1, 3 }), true, true, true, 0);
            r[0].ToArray<long>().Should().Equal(1L, 3L);
            r[1].ToArray<long>().Should().Equal(1L, 0L);
            r[2].ToArray<long>().Should().Equal(1L, 0L, 1L);
            r[3].ToArray<long>().Should().Equal(1L, 2L);
            r[2].flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void Concatenate_AxisNone_Accepts0dOperands()
        {
            // np.concatenate((np.array(1), np.array(2)), axis=None) -> [1, 2]; axis=0 refuses 0-d operands.
            var r = np.concatenate(new[] { NDArray.Scalar(1L), NDArray.Scalar(2L) }, null);
            r.ToArray<long>().Should().Equal(1L, 2L);
            Action act = () => np.concatenate(new[] { NDArray.Scalar(1L), NDArray.Scalar(2L) }, 0);
            act.Should().Throw<ValueError>().WithMessage("zero-dimensional arrays cannot be concatenated");
        }

        [TestMethod]
        public void ArgMax_AlongALengthOneAxis_IsAFreshArray()
        {
            // np.argmax([[1.0], [2.0]], axis=1) -> [0, 0], owning its data.
            var r = np.argmax(np.array(new[,] { { 1.0 }, { 2.0 } }), 1);
            r.ToArray<long>().Should().Equal(0L, 0L);
            r.flags.owndata.Should().BeTrue();
        }

        [TestMethod]
        public void AnyAll_Keepdims_AreFreshArrays()
        {
            // np.any(np.ones((2, 3)), keepdims=True) -> [[True]] shape (1, 1), owning its data; np.all the same.
            var any = np.any(np.ones(new Shape(2L, 3L)), keepdims: true);
            any.shape.Should().Equal(new long[] { 1, 1 });
            any.GetBoolean(0, 0).Should().BeTrue();
            any.flags.owndata.Should().BeTrue();
            var all = np.all(np.ones(new Shape(2L, 3L)), keepdims: true);
            all.flags.owndata.Should().BeTrue();
        }

        // ---------------------------------------------------------------- place / putmask values

        [TestMethod]
        public void PlacePutmask_AListConvertsDirectly_AnArrayMeetsTheSafeRule()
        {
            // np.place(int32, mask, [1.9, 2.9]) truncates (a Python list converts straight into the target dtype);
            // np.place(int32, mask, np.array([1.9, 2.9])) -> TypeError (PyArray_FromAny without FORCECAST is 'safe').
            var mask = np.array(new[] { true, true, true, true, true });
            var a = np.arange(5).astype(np.int32);
            np.place(a, mask, new[] { 1.9, 2.9 });
            a.ToArray<int>().Should().Equal(1, 2, 1, 2, 1);

            var b = np.arange(5).astype(np.int32);
            np.putmask(b, mask, new[] { 1.9, 2.9 });
            b.ToArray<int>().Should().Equal(1, 2, 1, 2, 1);

            var c = np.arange(5).astype(np.int32);
            Action place = () => np.place(c, mask, np.array(new[] { 1.9, 2.9 }));
            place.Should().Throw<InvalidCastException>()
                 .WithMessage("Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'");
            c.ToArray<int>().Should().Equal(0, 1, 2, 3, 4);
            Action putmask = () => np.putmask(c, mask, np.array(new[] { 1.9, 2.9 }));
            putmask.Should().Throw<InvalidCastException>()
                   .WithMessage("Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'");
        }

        // ---------------------------------------------------------------- trim_zeros, insert, delete

        [TestMethod]
        public void TrimZeros_ArgumentErrors_AreValueErrors()
        {
            Action trim = () => np.trim_zeros(np.array(new[] { 0, 1, 0 }), "x");
            trim.Should().Throw<ValueError>().WithMessage("unexpected character(s) in `trim`: 'x'");
            Action axes = () => np.trim_zeros(np.zeros(new Shape(2L, 2L)), "fb", new[] { 0, 0 });
            axes.Should().Throw<ValueError>().WithMessage("repeated axis in `axis` argument");
        }

        [TestMethod]
        public void InsertDelete_AxisOutOfRange_IsAxisError()
        {
            Action insert = () => np.insert(F(), 1, 9.0, 9);
            insert.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
            Action delete = () => np.delete(F(), 1, 9);
            delete.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void Insert_OneElementSequence_InsertsEveryValue()
        {
            // np.insert([1.5, 2.0, 3.0], [1], [7, 8, 9]) -> [1.5, 7, 8, 9, 2, 3] (a one-element sequence obj takes all
            // the values, unlike a scalar obj).
            np.insert(F(), new[] { 1 }, np.array(new[] { 7.0, 8.0, 9.0 })).ToArray<double>()
              .Should().Equal(1.5, 7.0, 8.0, 9.0, 2.0, 3.0);
        }

        [TestMethod]
        public void Delete_ArrayObj_IsRavelledAndMustBeInteger()
        {
            // np.delete(F, np.array([[0]])) -> [2, 3]; [[0, 2]] -> [2.]; int8 [1] -> [1.5, 3]; float [1.] -> IndexError.
            np.delete(F(), np.array(new long[,] { { 0 } })).ToArray<double>().Should().Equal(2.0, 3.0);
            np.delete(F(), np.array(new long[,] { { 0, 2 } })).ToArray<double>().Should().Equal(2.0);
            np.delete(F(), np.array(new sbyte[] { 1 })).ToArray<double>().Should().Equal(1.5, 3.0);
            Action act = () => np.delete(F(), np.array(new[] { 1.0 }));
            act.Should().Throw<IndexError>().WithMessage("arrays used as indices must be of integer (or boolean) type");
        }

        // ---------------------------------------------------------------- index conversions, nanargmax out=

        [TestMethod]
        public void IndexConversions_OnlyIntIndicesArePermitted()
        {
            Action unravel = () => np.unravel_index(np.array(new[] { 1.0 }), new[] { 3 });
            unravel.Should().Throw<TypeError>().WithMessage("only int indices permitted");
            Action ravel = () => np.ravel_multi_index(new[] { np.array(new[] { 1.0 }) }, new[] { 3 });
            ravel.Should().Throw<TypeError>().WithMessage("only int indices permitted");
            Action length = () => np.ravel_multi_index(new[] { np.array(new long[] { 1 }), np.array(new long[] { 1 }) }, new[] { 3 });
            length.Should().Throw<ValueError>().WithMessage("parameter multi_index must be a sequence of length 1");
            // A bool index array is an integer one (False 0, True 1).
            np.unravel_index(np.array(new[] { true, false }), new[] { 3 })[0].ToArray<long>().Should().Equal(1L, 0L);
        }

        [TestMethod]
        public void UnravelIndex_ShapeProductOfMinusOne_IsTheOverflowSentinel()
        {
            // PyArray_OverflowMultiplyList returns -1 for an overflow, and a product that IS -1 reads the same.
            Action act = () => np.unravel_index(np.array(new long[] { 0 }), new[] { -1 });
            act.Should().Throw<ValueError>().WithMessage(
                "dimensions are too large; arrays and shapes with a total size greater than 'intp' are not supported.");
        }

        [TestMethod]
        public void NanArgMax_Out_IsCheckedBeforeTheComputation()
        {
            // The out's shape first, then its safe cast to intp (a float out refuses, an int8 out is accepted).
            Action shape = () => np.nanargmax(F(), null, np.zeros(new Shape(3L), np.int64));
            shape.Should().Throw<ValueError>().WithMessage("output array does not match result of np.argmax.");
            Action cast = () => np.nanargmax(F(), null, np.zeros(Shape.NewScalar(), np.float64));
            cast.Should().Throw<TypeError>().WithMessage("Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'safe'");
            var o = np.zeros(Shape.NewScalar(), np.int8);
            ReferenceEquals(np.nanargmax(F(), null, o), o).Should().BeTrue();
            ((long)o.GetAtIndex<sbyte>(0)).Should().Be(2L);
        }

        // ---------------------------------------------------------------- dtype predicates

        [TestMethod]
        public void MinScalarType_OfAnArray_IsItsDtype_ExceptA0dValue()
        {
            // np.min_scalar_type(F) -> float64; a 0-d array is its value: np.array(300) -> uint16, np.array(True) -> bool.
            np.min_scalar_type(F()).Should().Be(np.float64);
            np.min_scalar_type(NDArray.Scalar(300L)).Should().Be(np.uint16);
            np.min_scalar_type(NDArray.Scalar(true)).Should().Be(np.@bool);
            np.min_scalar_type(np.array(new[] { "ab", "c" })).str.Should().Be("<U2");
        }

        [TestMethod]
        public void CommonType_OfANonNumericArray_Refuses()
        {
            Action text = () => np.common_type(np.array(new[] { "a" }));
            text.Should().Throw<TypeError>().WithMessage("can't get common type for non-numeric array");
            Action boolean = () => np.common_type(np.array(new[] { true }));
            boolean.Should().Throw<TypeError>().WithMessage("can't get common type for non-numeric array");
        }
    }
}
