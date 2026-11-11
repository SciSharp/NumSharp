using System;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     NumPy's scalar C-API seams over NumSharp's scalar model (<c>NDArray.ScalarApi.cs</c>): a 0-d array is NumPy's
    ///     scalar unless <c>np.array(x)</c> / <c>np.asarray(x)</c> built it as an explicit array
    ///     (<see cref="NDArray.IsNumPyScalar"/>), and <see cref="NDArray.PyArray_Return"/> makes a fresh 0-d result the
    ///     immutable scalar NumPy hands out. Every expectation is NumPy 2.4.2's, probed live: <c>np.isscalar(np.array(5))</c>
    ///     is False, <c>np.int32(5).flags.num</c> is 263 (read-only) where <c>np.array(5)</c>'s is 1287, and converting a
    ///     scalar into an array (<c>np.asarray</c>, <c>np.array</c>, <c>np.asanyarray</c>) gives a NEW writeable ndarray.
    /// </summary>
    [TestClass]
    public class NDArrayScalarApiTests
    {
        /// <summary>
        ///     The scalar predicate: computed results and <see cref="NDArray.Scalar{T}(T)"/> are NumPy scalars, the explicit
        ///     0-d arrays of <c>np.array</c> / <c>np.asarray</c> are not, and an array with dimensions never is.
        /// </summary>
        [TestMethod]
        public void IsNumPyScalar_IsZeroDimAndNotAnExplicitArray()
        {
            var a = np.arange(6).reshape(2, 3);

            NDArray.Scalar(1.5).IsNumPyScalar.Should().BeTrue();
            a.sum().IsNumPyScalar.Should().BeTrue();
            np.array(1.5).IsNumPyScalar.Should().BeFalse();
            np.asarray(5).IsNumPyScalar.Should().BeFalse();
            np.arange(3).IsNumPyScalar.Should().BeFalse();
            np.sum(a, keepdims: true).IsNumPyScalar.Should().BeFalse();
        }

        /// <summary>
        ///     The accepted difference of the model: a 0-d array allocated another way than the two explicit spellings
        ///     reads as a scalar (NumPy's <c>np.zeros(())</c> is an ndarray), yet stays writeable, so <c>fill</c> and stores
        ///     keep working on it.
        /// </summary>
        [TestMethod, Misaligned]
        public void IsNumPyScalar_ZerosOfEmptyShape_IsAScalarButWriteable()
        {
            var z = np.zeros(new Shape(), np.float64);
            z.IsNumPyScalar.Should().BeTrue("NumSharp's accepted difference: only np.array / np.asarray mark a 0-d ARRAY");
            z.flags.writeable.Should().BeTrue();
            z.SetDouble(2.5);
            z.GetDouble().Should().Be(2.5);
        }

        /// <summary>
        ///     <see cref="NDArray.PyArray_Return"/> on a fresh 0-d array: the same object comes back read-only, NumPy's scalar
        ///     flags (<c>flags.num</c> 263), and no longer an explicit array.
        /// </summary>
        [TestMethod]
        public void PyArray_Return_ZeroDim_BecomesTheReadOnlyScalar()
        {
            var fresh = np.array(7L);
            fresh.flags.writeable.Should().BeTrue();
            fresh.IsNumPyScalar.Should().BeFalse();

            var scalar = fresh.PyArray_Return();

            ReferenceEquals(scalar, fresh).Should().BeTrue("the seam changes the fresh array in place in this phase");
            scalar.flags.writeable.Should().BeFalse();
            scalar.flags.num.Should().Be(263);
            scalar.IsNumPyScalar.Should().BeTrue();
            scalar.GetInt64().Should().Be(7L);
        }

        /// <summary><see cref="NDArray.PyArray_Return"/> hands an array with dimensions back untouched and writeable.</summary>
        [TestMethod]
        public void PyArray_Return_WithDimensions_IsANoOp()
        {
            var arr = np.arange(4);
            var same = arr.PyArray_Return();
            ReferenceEquals(same, arr).Should().BeTrue();
            same.flags.writeable.Should().BeTrue();
            same.flags.num.Should().Be(np.arange(4).flags.num);
        }

        /// <summary>
        ///     NumPy's <c>np.isscalar</c>: False for the explicit 0-d array <c>np.array(5)</c> and for arrays with
        ///     dimensions, True for scalars (a reduction's result, <c>np.int32(5)</c>) and for C# values.
        /// </summary>
        [TestMethod]
        public void IsScalar_ExplicitZeroDimArray_IsFalse_AsInNumPy()
        {
            var a = np.arange(6).reshape(2, 3);
            np.isscalar(np.array(5)).Should().BeFalse();
            np.isscalar(NDArray.Scalar(5)).Should().BeTrue();
            np.isscalar(a.sum()).Should().BeTrue();
            np.isscalar(5).Should().BeTrue();
            np.isscalar(np.arange(3)).Should().BeFalse();
        }

        /// <summary>
        ///     The scalar-class <c>isinstance</c> reads the same predicate: <c>isinstance(np.array(1.5), np.floating)</c> is
        ///     False, <c>isinstance(np.float64(1.5), np.floating)</c> and <c>isinstance(a.sum(), np.integer)</c> are True.
        /// </summary>
        [TestMethod]
        public void InstanceCheck_ReadsTheScalarPredicate()
        {
            var a = np.arange(6).reshape(2, 3);
            np.floating.__instancecheck__(np.array(1.5)).Should().BeFalse();
            np.floating.__instancecheck__(NDArray.Scalar(1.5)).Should().BeTrue();
            np.integer.__instancecheck__(a.sum()).Should().BeTrue();
        }

        /// <summary>
        ///     <see cref="NDArray.Scalar{T}(T)"/> and its overloads build NumPy's immutable scalar: read-only, flags 263
        ///     (<c>np.int32(5).flags.num</c>), owning its value. A full reduction's result is the same kind of object.
        /// </summary>
        [TestMethod]
        public void Scalar_Factories_BuildTheReadOnlyScalar()
        {
            NDArray.Scalar(5).flags.num.Should().Be(263);
            NDArray.Scalar((object)2.5).flags.writeable.Should().BeFalse();
            NDArray.Scalar(3, np.int16).flags.writeable.Should().BeFalse();
            NDArray.Scalar<float>((object)1.5).flags.writeable.Should().BeFalse();
            np.arange(6).sum().flags.num.Should().Be(263);

            var s = NDArray.Scalar(5);
            new Action(() => s.SetInt32(6)).Should().Throw<ValueError>();
            s.GetInt32().Should().Be(5);
        }

        /// <summary>
        ///     Converting a scalar into an ARRAY builds a new writeable ndarray, as NumPy's <c>PyArray_FromAny</c> of a scalar
        ///     object does: <c>np.asarray(s)</c> and <c>np.array(s)</c> (flags 1287, <c>np.isscalar</c> False, the scalar left
        ///     untouched) and <c>np.asanyarray(5)</c> of a C# value.
        /// </summary>
        [TestMethod]
        public void ScalarIntoArray_IsANewWriteableExplicitArray()
        {
            var s = NDArray.Scalar(7L);

            var viaAsarray = np.asarray(s);
            ReferenceEquals(viaAsarray, s).Should().BeFalse();
            viaAsarray.flags.num.Should().Be(1287);
            np.isscalar(viaAsarray).Should().BeFalse();

            var viaArray = np.array(s);
            viaArray.flags.num.Should().Be(1287);
            np.isscalar(viaArray).Should().BeFalse();
            viaArray.SetInt64(8L);
            s.GetInt64().Should().Be(7L);

            var any = np.asanyarray(5);
            any.flags.writeable.Should().BeTrue();
            any.flags.num.Should().Be(1287);
        }

        /// <summary>
        ///     A keepdims reduction is an ARRAY in NumPy (writeable), while the same reduction without keepdims is the
        ///     read-only scalar: the engine builds its 0-d intermediate writeable and converts it only at the scalar exit.
        /// </summary>
        [TestMethod]
        public void Reduction_KeepdimsArray_StaysWriteable_ScalarExitIsReadOnly()
        {
            var a = np.arange(6.0).reshape(2, 3);
            np.sum(a, keepdims: true).flags.writeable.Should().BeTrue();
            np.mean(a, keepdims: true).flags.writeable.Should().BeTrue();
            np.sum(a).flags.writeable.Should().BeFalse();
            np.mean(a).flags.writeable.Should().BeFalse();
        }

        /// <summary>
        ///     numpy.ma converts a scalar's data into a new writeable array (<c>getdata</c>'s <c>np.array(a, copy=None)</c>), so a
        ///     masked array over a NumPy scalar can be written: <c>np.ma.array(np.int64(5), mask=True).put(0, 9)</c> stores 9.
        /// </summary>
        [TestMethod]
        public void MaskedArray_OverAScalar_HasWriteableData()
        {
            var m = np.ma.array(NDArray.Scalar(5L), mask: true);
            m.data.flags.writeable.Should().BeTrue();
            m.put(np.array(0L), 9);
            m.data.GetInt64().Should().Be(9L);
        }
    }
}
