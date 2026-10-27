using System;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     np.insert as NumPy 2.4.2's statements (<c>numpy/lib/_function_base_impl.py</c>), through the C# argument forms of
    ///     its twelve overloads; every expected value and text was probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_insert_NumPyPort_Test
    {
        /// <summary>
        ///     A value given as an object (a C# number, string or collection) is converted where NumPy converts its values,
        ///     after the axis and the single-index bounds checks (<c>array(values, ndmin=arr.ndim, dtype=arr.dtype)</c>) and
        ///     in the multi-index branch's setitem: an axis or index error comes first, then the value's own packing error.
        ///     NumSharp converted the object first, so an out-of-range literal, a text that does not parse or a complex into a
        ///     real array raised its conversion error where NumPy raises the AxisError / IndexError.
        /// </summary>
        [TestMethod]
        public void ObjectValue_IsConvertedAfterTheAxisAndIndexChecks()
        {
            // >>> np.insert(np.zeros(3, np.int8), 10, 1000)  IndexError: index 10 is out of bounds for axis 0 with size 3
            Action oobLiteral = () => np.insert(np.zeros(new Shape(3), np.int8), 10, (object)1000);
            oobLiteral.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 3");
            // >>> np.insert(np.zeros(3, np.uint8), 1, 1000, axis=1)  AxisError: axis 1 is out of bounds for array of dimension 1
            Action axisLiteral = () => np.insert(np.zeros(new Shape(3), np.uint8), 1, (object)1000, axis: 1);
            axisLiteral.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
            // >>> np.insert(np.zeros(3, np.int32), 10, 'x')  IndexError (not int()'s ValueError)
            Action oobText = () => np.insert(np.zeros(new Shape(3), np.int32), 10, (object)"x");
            oobText.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 3");
            // >>> np.insert(np.zeros(3, np.int32), 10, [7, 'x'])  IndexError
            Action oobList = () => np.insert(np.zeros(new Shape(3), np.int32), 10, new object[] { 7, "x" });
            oobList.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 3");
            // >>> np.insert(np.zeros(3), 10, 1+2j)  IndexError (not float()'s TypeError)
            Action oobComplex = () => np.insert(np.zeros(new Shape(3)), 10, (object)new System.Numerics.Complex(1, 2));
            oobComplex.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 3");

            // Within bounds the conversion errors stay NumPy's.
            // >>> np.insert(np.zeros(3, np.int8), 1, 1000)  OverflowError: Python integer 1000 out of bounds for int8
            Action overflow = () => np.insert(np.zeros(new Shape(3), np.int8), 1, (object)1000);
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            // >>> np.insert(np.zeros(3, np.int32), 1, 'x')  ValueError: invalid literal for int() with base 10: 'x'
            Action badText = () => np.insert(np.zeros(new Shape(3), np.int32), 1, (object)"x");
            badText.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'x'");
            // >>> np.insert(np.zeros(3, np.int8), [], 1000)  OverflowError: the empty index still converts the values
            Action emptyIndex = () => np.insert(np.zeros(new Shape(3), np.int8), Array.Empty<int>(), (object)1000);
            emptyIndex.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");

            // And a value that converts is stored as NumPy stores it.
            // >>> np.insert(np.zeros(3, np.int8), 1, 7)  array([0, 7, 0, 0], dtype=int8)
            var r = np.insert(np.zeros(new Shape(3), np.int8), 1, (object)7);
            r.dtype.Should().Be(np.int8);
            r.array_equal(np.array(new sbyte[] { 0, 7, 0, 0 })).Should().BeTrue();
            // >>> np.insert(np.zeros(3, np.float64), 2, '1.5')  array([0. , 0. , 1.5, 0. ])
            np.insert(np.zeros(new Shape(3)), 2, (object)"1.5").array_equal(np.array(new[] { 0.0, 0.0, 1.5, 0.0 }))
                .Should().BeTrue();
        }

        /// <summary>
        ///     NumPy enters the single-index branch on <c>indices.size == 1</c> whatever the positions argument is, so a slice
        ///     or a boolean mask that names one position inserts EVERY value there, the values keeping their shape (the
        ///     slice's arange and the mask's flatnonzero are 1-D: no moveaxis). NumSharp sent every slice and every mask to
        ///     the multi-index branch, which refuses two values for one position.
        /// </summary>
        [TestMethod]
        public void OnePositionSliceOrMask_TakesTheSingleIndexBranch()
        {
            var a = np.array(new long[] { 0, 1, 2, 3 });
            var b = np.array(new long[] { 0, 1, 2, 3, 4, 5 }).reshape(2, 3);
            // >>> np.insert(np.arange(4), slice(1, 2), [7, 8])  array([0, 7, 8, 1, 2, 3])
            np.insert(a, new Slice(1, 2), np.array(new long[] { 7, 8 }))
                .array_equal(np.array(new long[] { 0, 7, 8, 1, 2, 3 })).Should().BeTrue();
            // >>> np.insert(b, slice(1, 2), [[7], [8]], axis=1)  array([[0, 7, 1, 2], [3, 8, 4, 5]])
            np.insert(b, new Slice(1, 2), np.array(new long[] { 7, 8 }).reshape(2, 1), axis: 1)
                .array_equal(np.array(new long[,] { { 0, 7, 1, 2 }, { 3, 8, 4, 5 } })).Should().BeTrue();
            // >>> np.insert(b, slice(1, 2), [7, 8], axis=1)  array([[0, 7, 8, 1, 2], [3, 7, 8, 4, 5]])
            np.insert(b, new Slice(1, 2), np.array(new long[] { 7, 8 }), axis: 1)
                .array_equal(np.array(new long[,] { { 0, 7, 8, 1, 2 }, { 3, 7, 8, 4, 5 } })).Should().BeTrue();
            // >>> np.insert(b, slice(0, 1), [7, 8, 9], axis=0)  array([[7, 8, 9], [0, 1, 2], [3, 4, 5]])
            np.insert(b, new Slice(0, 1), np.array(new long[] { 7, 8, 9 }), axis: 0)
                .array_equal(np.array(new long[,] { { 7, 8, 9 }, { 0, 1, 2 }, { 3, 4, 5 } })).Should().BeTrue();
            // >>> np.insert(np.arange(4), np.array([False, True]), [7, 8])  array([0, 7, 8, 1, 2, 3])
            np.insert(a, np.array(new[] { false, true }), np.array(new long[] { 7, 8 }))
                .array_equal(np.array(new long[] { 0, 7, 8, 1, 2, 3 })).Should().BeTrue();
            // >>> np.insert(np.arange(4), np.array([False, False, False, False, True]), 9)  array([0, 1, 2, 3, 9])
            np.insert(a, np.array(new[] { false, false, false, false, true }), (object)9)
                .array_equal(np.array(new long[] { 0, 1, 2, 3, 9 })).Should().BeTrue();
        }

        /// <summary>
        ///     The single-index branch's IndexError prints the positions ARGUMENT, as NumPy's <c>f"index {obj} ..."</c> does:
        ///     a number or a 0-d array as itself, a one-element sequence or array as <c>[13]</c>, a boolean mask whole.
        ///     NumSharp printed the position, so <c>np.insert(a, [10], 7)</c> read "index 10" where NumPy reads "index [10]".
        /// </summary>
        [TestMethod]
        public void SingleIndexBoundsError_PrintsThePositionsArgument()
        {
            var a = np.array(new long[] { 0, 1, 2, 3 });
            var seven = np.array(7L);
            // >>> np.insert(np.arange(4), [10], 7)  IndexError: index [10] is out of bounds for axis 0 with size 4
            Action intList = () => np.insert(a, new[] { 10 }, seven);
            intList.Should().Throw<IndexOutOfRangeException>().WithMessage("index [10] is out of bounds for axis 0 with size 4");
            // >>> np.insert(np.arange(4), [-13], 7)
            Action longList = () => np.insert(a, new long[] { -13 }, seven);
            longList.Should().Throw<IndexOutOfRangeException>().WithMessage("index [-13] is out of bounds for axis 0 with size 4");
            // >>> np.insert(np.arange(4), np.array([10]), 7) and np.array([10], np.uint8)
            Action array = () => np.insert(a, np.array(new long[] { 10 }), seven);
            array.Should().Throw<IndexOutOfRangeException>().WithMessage("index [10] is out of bounds for axis 0 with size 4");
            Action uint8 = () => np.insert(a, np.array(new byte[] { 10 }), seven);
            uint8.Should().Throw<IndexOutOfRangeException>().WithMessage("index [10] is out of bounds for axis 0 with size 4");
            // >>> np.insert(np.arange(4), np.array(10), 7) and np.insert(np.arange(4), 10, 7)
            Action zeroD = () => np.insert(a, np.array(10L), seven);
            zeroD.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 4");
            Action number = () => np.insert(a, 10, seven);
            number.Should().Throw<IndexOutOfRangeException>().WithMessage("index 10 is out of bounds for axis 0 with size 4");
            // >>> np.insert(np.arange(4), np.array([False] * 6 + [True]), 7)
            // IndexError: index [False False False False False False  True] is out of bounds for axis 0 with size 4
            Action mask = () => np.insert(a, np.array(new[] { false, false, false, false, false, false, true }), seven);
            mask.Should().Throw<IndexOutOfRangeException>()
                .WithMessage("index [False False False False False False  True] is out of bounds for axis 0 with size 4");
        }

        /// <summary>
        ///     The single-index branch writes its values with <c>new[..., index:index+numnew, ...] = values</c>, a view
        ///     assignment: leading extent-1 axes beyond the slot's rank are stripped, then the values must broadcast to the
        ///     slot, and a refusal is NumPy's ValueError with the stripped shape. NumSharp broadcast them with
        ///     <c>np.broadcast_to</c>, which refused every extra axis and raised its own IncorrectShapeException text.
        /// </summary>
        [TestMethod]
        public void SingleIndexValues_AreAssignedAsNumPysViewAssignment()
        {
            var a = np.array(new long[] { 0, 1, 2, 3 });
            var b = np.array(new long[] { 0, 1, 2, 3, 4, 5 }).reshape(2, 3);
            // >>> np.insert(np.arange(4), 1, [[7, 8]])
            // ValueError: could not broadcast input array from shape (2,) into shape (1,)
            Action stripped = () => np.insert(a, 1, np.array(new long[] { 7, 8 }).reshape(1, 2));
            stripped.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,) into shape (1,)");
            // >>> np.insert(np.arange(4), [1], [[7], [8]])
            // ValueError: could not broadcast input array from shape (2,1) into shape (2,)
            Action column = () => np.insert(a, new[] { 1 }, np.array(new long[] { 7, 8 }).reshape(2, 1));
            column.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,1) into shape (2,)");
            // >>> np.insert(b, 1, [7, 8, 9], axis=1)  (the scalar obj moves the values' first axis to the insertion axis)
            // ValueError: could not broadcast input array from shape (3,1) into shape (2,1)
            Action moved = () => np.insert(b, 1, np.array(new long[] { 7, 8, 9 }), axis: 1);
            moved.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,1) into shape (2,1)");
            // >>> np.insert(b, 1, [[[7, 8]]], axis=1)
            // ValueError: could not broadcast input array from shape (1,2) into shape (2,1)
            Action extraAxes = () => np.insert(b, 1, np.array(new long[] { 7, 8 }).reshape(1, 1, 2), axis: 1);
            extraAxes.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (1,2) into shape (2,1)");
            // >>> np.insert(b, 1, np.zeros(0), axis=1)  (an empty value fills no slot of two rows)
            // ValueError: could not broadcast input array from shape (0,1) into shape (2,1)
            Action empty = () => np.insert(b, 1, np.zeros(new Shape(0)), axis: 1);
            empty.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (0,1) into shape (2,1)");
            // >>> np.insert(np.arange(4), np.array([1.0]), [[7], [8]])  (a float index: the values are converted, then the
            // slice built from the float refuses before they are assigned)
            // TypeError: slice indices must be integers or None or have an __index__ method
            Action floatIndex = () => np.insert(a, np.array(new[] { 1.0 }), np.array(new long[] { 7, 8 }).reshape(2, 1));
            floatIndex.Should().Throw<TypeError>().WithMessage("slice indices must be integers or None or have an __index__ method");

            // >>> np.insert(np.arange(4), 1, [[7]])  array([0, 7, 1, 2, 3])
            np.insert(a, 1, np.array(new long[] { 7 }).reshape(1, 1))
                .array_equal(np.array(new long[] { 0, 7, 1, 2, 3 })).Should().BeTrue();
            // >>> np.insert(b, 1, [[[7, 8, 9]]], axis=0)  array([[0, 1, 2], [7, 8, 9], [3, 4, 5]])
            np.insert(b, 1, np.array(new long[] { 7, 8, 9 }).reshape(1, 1, 3), axis: 0)
                .array_equal(np.array(new long[,] { { 0, 1, 2 }, { 7, 8, 9 }, { 3, 4, 5 } })).Should().BeTrue();
        }
    }
}
