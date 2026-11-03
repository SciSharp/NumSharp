using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.array</c> / <c>np.asarray</c> of a jagged C# array and of an <c>NDArray[]</c>, read as NumPy reads nested lists
    ///     and a list of arrays (parity QA-45): the shape is NumPy's discovery, every row checked, so a ragged input is NumPy's
    ///     inhomogeneous-shape ValueError (the readers took the FIRST row's length: a longer later row was cut silently, a
    ///     shorter one threw IndexOutOfRange), the dims end at the first empty level, and an <c>NDArray[]</c> is stacked with
    ///     NumPy's promotion where the typed route refused the element type. Every expectation is NumPy 2.4.2's answer for the
    ///     Python spelling.
    /// </summary>
    [TestClass]
    public class NpArrayJaggedAndListOfArraysTests
    {
        /// <summary>NumPy's text for an input whose children disagree after one dimension of (2,).</summary>
        private const string AfterOne =
            "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. " +
            "The detected shape was (2,) + inhomogeneous part.";

        /// <summary>
        ///     <c>np.array([[1], [2, 3]])</c> and <c>np.array([[1, 2], [3]])</c> raise NumPy's ValueError, through the typed
        ///     overload, the dtype= overload (<c>np.array(jagged, dtype)</c>, the general Array route) and np.asarray (the
        ///     implicit conversion); nothing is truncated any more.
        /// </summary>
        [TestMethod]
        public void RaggedRows_RaiseNumPysInhomogeneousError()
        {
            Action longerLater = () => np.array(new int[][] { new[] { 1 }, new[] { 2, 3 } }).Dispose();
            longerLater.Should().Throw<ValueError>().WithMessage(AfterOne);
            Action shorterLater = () => np.array(new int[][] { new[] { 1, 2 }, new[] { 3 } }).Dispose();
            shorterLater.Should().Throw<ValueError>().WithMessage(AfterOne);
            Action withDtype = () => np.array(new int[][] { new[] { 1 }, new[] { 2, 3 } }, np.float64).Dispose();
            withDtype.Should().Throw<ValueError>().WithMessage(AfterOne);
            Action asarray = () => np.asarray(new long[][] { new[] { 1L }, new[] { 2L, 3L } }).Dispose();
            asarray.Should().Throw<ValueError>().WithMessage(AfterOne);
        }

        /// <summary>
        ///     The reported depth is the one the children agreed to: <c>[[[1, 2], [3, 4]], [[5, 6], [7]]]</c> is "after 2
        ///     dimensions ... (2, 2)", and a null row (NumPy's None beside sequences) is "after 1 dimensions ... (2,)".
        /// </summary>
        [TestMethod]
        public void DeeperRaggedness_AndNullRows_NameTheAgreedShape()
        {
            Action deep = () => np.array(new int[][][]
            {
                new[] { new[] { 1, 2 }, new[] { 3, 4 } },
                new[] { new[] { 5, 6 }, new[] { 7 } },
            }).Dispose();
            deep.Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. " +
                "The detected shape was (2, 2) + inhomogeneous part.");

            Action nullRow = () => np.array(new int[][] { new[] { 1 }, null }).Dispose();
            nullRow.Should().Throw<ValueError>().WithMessage(AfterOne);
        }

        /// <summary>
        ///     NumPy's reading ends at the first empty level: <c>[]</c> is <c>(0,)</c>, <c>[[], []]</c> is <c>(2, 0)</c>,
        ///     <c>[[[], []], [[], []]]</c> is <c>(2, 2, 0)</c> (an empty outer jagged array used to throw IndexOutOfRange); the
        ///     element type (or the requested dtype) is kept.
        /// </summary>
        [TestMethod]
        public void EmptyLevels_EndTheDims()
        {
            using var none = np.array(new int[0][]);
            none.shape.Should().Equal(0L);
            none.typecode.Should().Be(NPTypeCode.Int32);

            using var emptyRows = np.array(new int[][] { new int[0], new int[0] });
            emptyRows.shape.Should().Equal(2L, 0L);

            using var emptyLeaves = np.array(new double[][][] { new[] { new double[0], new double[0] }, new[] { new double[0], new double[0] } });
            emptyLeaves.shape.Should().Equal(2L, 2L, 0L);

            using var noLeaves = np.array(new int[][][] { new int[0][], new int[0][] });
            noLeaves.shape.Should().Equal(2L, 0L);

            using var withDtype = np.array(new int[0][], np.float64);
            withDtype.shape.Should().Equal(0L);
            withDtype.typecode.Should().Be(NPTypeCode.Double);
        }

        /// <summary>Regular jagged arrays read exactly as before: their values, shapes and element types.</summary>
        [TestMethod]
        public void RegularJaggedArrays_StillRead()
        {
            using var two = np.array(new int[][] { new[] { 1, 2 }, new[] { 3, 4 } });
            two.shape.Should().Equal(2L, 2L);
            two.ToArray<int>().Should().Equal(1, 2, 3, 4);

            using var three = np.array(new long[][][] { new[] { new[] { 1L }, new[] { 2L } }, new[] { new[] { 3L }, new[] { 4L } } });
            three.shape.Should().Equal(2L, 2L, 1L);
            three.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L);

            using var cast = np.array(new int[][] { new[] { 1, 2 }, new[] { 3, 4 } }, np.float32);
            cast.typecode.Should().Be(NPTypeCode.Single);
            cast.ToArray<float>().Should().Equal(1f, 2f, 3f, 4f);
        }

        /// <summary>
        ///     <c>np.array([a, b])</c> and <c>np.asarray([a, b])</c> of two <c>(3,)</c> int64 arrays are <c>(2, 3)</c> int64;
        ///     int8 beside int16 promotes to int16, 0-d arrays stack to <c>(2,)</c> (1 and 2.5: float64), 2-D arrays to
        ///     <c>(2, 2, 2)</c>, two empty arrays to <c>(2, 0)</c>; unequal shapes are the inhomogeneous ValueError.
        /// </summary>
        [TestMethod]
        public void ListOfArrays_StacksWithNumPysPromotion()
        {
            using var a = np.arange(3);
            using var b = np.arange(3, 6);

            using var viaArray = np.array(new[] { a, b });
            viaArray.shape.Should().Equal(2L, 3L);
            viaArray.typecode.Should().Be(NPTypeCode.Int64);
            viaArray.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            using var viaAsarray = np.asarray(new[] { a, b });
            viaAsarray.shape.Should().Equal(2L, 3L);
            viaAsarray.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            using var i8 = np.array(new sbyte[] { 1, 2 });
            using var i16 = np.array(new short[] { 3, 4 });
            using var promoted = np.array(new[] { i8, i16 });
            promoted.typecode.Should().Be(NPTypeCode.Int16);

            using var one = np.array(1L);
            using var half = np.array(2.5);
            using var scalars = np.array(new[] { one, half });
            scalars.shape.Should().Equal(2L);
            scalars.typecode.Should().Be(NPTypeCode.Double);

            using var ones = np.ones(new Shape(2, 2));
            using var zeros = np.zeros(new Shape(2, 2));
            using var cube = np.array(new[] { ones, zeros });
            cube.shape.Should().Equal(2L, 2L, 2L);

            using var e1 = np.ones(new Shape(0));
            using var e2 = np.ones(new Shape(0));
            using var empties = np.asarray(new[] { e1, e2 });
            empties.shape.Should().Equal(2L, 0L);

            using var short2 = np.arange(2);
            Action unequal = () => np.array(new[] { a, short2 }).Dispose();
            unequal.Should().Throw<ValueError>().WithMessage(AfterOne);
        }
    }
}
