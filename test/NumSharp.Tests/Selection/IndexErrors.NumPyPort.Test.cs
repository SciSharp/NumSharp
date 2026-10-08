using System;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     The IndexErrors NumPy's <c>prepare_index</c> (<c>mapping.c</c>) raises for a boolean index that does not fit and
    ///     for an index array of a non-index dtype, on the getter and the setter. Texts are NumPy 2.4.2's, verbatim.
    /// </summary>
    /// <remarks>
    ///     Found as a neighbour of numpy.ma's put (2026-10-08), whose N-D hard-mask error IS this boolean-index error: the
    ///     single-mask indexer raised an <see cref="IndexOutOfRangeException"/> whose text always named axis 0 and the
    ///     mask's first length ("size of axis is 2 but size of boolean index is 2" for a 2-D mask on a 1-D array), the
    ///     integer-index normalizer raised an <see cref="IndexOutOfRangeException"/> for a float array, and an EMPTY float
    ///     index array became an empty fancy index instead of the error.
    /// </remarks>
    [TestClass]
    public class index_errors_numpy_port_tests
    {
        private static void GetAndSetRaise(NDArray a, NDArray index, string text)
        {
            Action get = () => { var _ = a[index]; };
            get.Should().Throw<IndexError>().Where(e => e.Message == text);
            var b = a.copy();
            Action set = () => b[index] = NDArray.Scalar(9L);
            set.Should().Throw<IndexError>().Where(e => e.Message == text);
        }

        /// <summary>A mask with more dimensions than the array: each mask axis consumes one array axis.</summary>
        [TestMethod]
        public void BoolMask_MoreDimensionsThanTheArray_IsTooManyIndices()
        {
            GetAndSetRaise(np.arange(4), np.zeros(new Shape(2, 3), np.@bool),
                "too many indices for array: array is 1-dimensional, but 2 were indexed");
            GetAndSetRaise(np.arange(4), np.zeros(new Shape(4, 1), np.@bool),
                "too many indices for array: array is 1-dimensional, but 2 were indexed");
            GetAndSetRaise(np.arange(6).reshape(2, 3), np.zeros(new Shape(2, 3, 1), np.@bool),
                "too many indices for array: array is 2-dimensional, but 3 were indexed");
            // The too-many check comes first, an empty mask axis included.
            GetAndSetRaise(np.arange(4), np.zeros(new Shape(0, 3), np.@bool),
                "too many indices for array: array is 1-dimensional, but 2 were indexed");
        }

        /// <summary>The FIRST mask axis whose (non-zero) length differs, named by its array axis.</summary>
        [TestMethod]
        public void BoolMask_LengthMismatch_NamesTheAxis()
        {
            GetAndSetRaise(np.arange(4), np.array(new[] { true, false }),
                "boolean index did not match indexed array along axis 0; size of axis is 4 but size of corresponding boolean axis is 2");
            GetAndSetRaise(np.arange(6).reshape(2, 3), np.zeros(new Shape(2, 4), np.@bool),
                "boolean index did not match indexed array along axis 1; size of axis is 3 but size of corresponding boolean axis is 4");
            GetAndSetRaise(np.arange(6).reshape(2, 3), np.zeros(new Shape(2, 1), np.@bool),
                "boolean index did not match indexed array along axis 1; size of axis is 3 but size of corresponding boolean axis is 1");
            GetAndSetRaise(np.arange(24).reshape(2, 3, 4), np.zeros(new Shape(2, 3, 5), np.@bool),
                "boolean index did not match indexed array along axis 2; size of axis is 4 but size of corresponding boolean axis is 5");
            GetAndSetRaise(np.zeros(new Shape(0, 3)), np.zeros(new Shape(0, 4), np.@bool),
                "boolean index did not match indexed array along axis 1; size of axis is 3 but size of corresponding boolean axis is 4");
        }

        /// <summary>A zero-length mask axis matches an axis of any length (NumPy skips the comparison for it).</summary>
        [TestMethod]
        public void BoolMask_ZeroLengthAxis_MatchesAnyLength()
        {
            np.arange(4)[np.zeros(0, np.@bool)].shape.Should().Equal(0L);
            np.arange(6).reshape(2, 3)[np.zeros(0, np.@bool)].shape.Should().Equal(0L, 3L);
            np.zeros(new Shape(0, 3))[np.zeros(new Shape(0, 3), np.@bool)].shape.Should().Equal(0L);
        }

        /// <summary>
        ///     An index array that is neither integer nor bool is refused whatever its size: an empty or a 0-d float array
        ///     too (NumSharp used to take an empty one as an empty fancy index), complex and decimal alike.
        /// </summary>
        [TestMethod]
        public void NonIndexDtypeArrays_AreRefused_EvenEmpty()
        {
            const string text = "arrays used as indices must be of integer (or boolean) type";
            GetAndSetRaise(np.arange(4), np.array(new[] { 0.0, 1.0 }), text);
            GetAndSetRaise(np.arange(4), np.array(new double[0]), text);
            GetAndSetRaise(np.arange(4), np.zeros(new Shape(0, 2)), text);
            GetAndSetRaise(np.arange(4), NDArray.Scalar(0.5), text);
            GetAndSetRaise(np.arange(4), np.array(new[] { Complex.Zero }), text);
            GetAndSetRaise(np.arange(4), np.array(new[] { 1m }), text);
            // The dtype is checked before a 0-d target's too-many check, an empty index array included (probed 2.4.2:
            // np.array(5.0)[np.array([], float)] is this error, not "too many indices").
            GetAndSetRaise(NDArray.Scalar(7L), np.array(new[] { 0.0 }), text);
            GetAndSetRaise(NDArray.Scalar(7L), np.array(new double[0]), text);
            GetAndSetRaise(NDArray.Scalar(7L), np.zeros(new Shape(0, 2)), text);
            // An empty INTEGER array is the empty fancy index.
            np.arange(4)[np.zeros(new Shape(0, 3), np.int64)].shape.Should().Equal(0L, 3L);
        }

        /// <summary>The same refusal inside an index tuple (the tuple classifier, NumPy's prepare_index walk).</summary>
        [TestMethod]
        public void NonIndexDtypeArrays_InATuple_AreRefused()
        {
            var a = np.arange(12).reshape(3, 4);
            Action get = () => { var _ = a[np.array(new double[0]), 0]; };
            get.Should().Throw<IndexError>().WithMessage("arrays used as indices must be of integer (or boolean) type");
            // prepare_index walks every item before its too-many check, so a non-index array anywhere in a tuple that is
            // also too long is the dtype error (probed 2.4.2, the array first and last).
            Action first = () => { var _ = a[np.array(new double[0]), 0, 0]; };
            first.Should().Throw<IndexError>().WithMessage("arrays used as indices must be of integer (or boolean) type");
            Action last = () => { var _ = a[0, 0, np.array(new double[0])]; };
            last.Should().Throw<IndexError>().WithMessage("arrays used as indices must be of integer (or boolean) type");
            // An empty bool array in a tuple stays a mask: its zero-length axis matches.
            a[np.zeros(0, np.@bool), 0].shape.Should().Equal(0L);
            // A 2-D empty mask in a tuple consumes two axes (the force-cast to an empty integer index used to take one).
            Action two = () => { var _ = a[np.zeros(new Shape(0, 3), np.@bool), Slice.All]; };
            two.Should().Throw<IndexError>().WithMessage("too many indices for array: array is 2-dimensional, but 3 were indexed");
        }
    }
}
