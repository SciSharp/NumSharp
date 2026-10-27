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

        /// <summary>
        ///     The multi-index branch's positions are NumPy's: <c>indices[indices &lt; 0] += N</c> once, then each index
        ///     shifted by its rank (<c>indices[order] += np.arange(numnew)</c>), then <c>old_mask[indices] = False</c> over the
        ///     <c>N + numnew</c> slots, which wraps a negative position and refuses one past the end. So an index below
        ///     <c>-N</c> lands at the END, an index past <c>N</c> is an IndexError naming the SHIFTED position, two indices may
        ///     name one slot (the later value wins), and the array must then fill one slot more (only a one-slab array can).
        ///     NumSharp checked the indices against <c>[-N, N]</c> before the shift and interleaved the sorted indices.
        /// </summary>
        [TestMethod]
        public void MultiIndex_PositionsAreNumPysShiftedIndices()
        {
            var a = np.arange(12L);
            var v = np.array(new long[] { 100, 200 });
            // >>> np.insert(np.arange(12), [-13, 1], [100, 200])
            // array([  0,   1, 200,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11, 100])
            np.insert(a, new long[] { -13, 1 }, v)
                .array_equal(np.array(new long[] { 0, 1, 200, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100 })).Should().BeTrue();
            // >>> np.insert(np.arange(12), [1, -26], [100, 200])
            // array([200,   0, 100,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11])
            np.insert(a, new long[] { 1, -26 }, v)
                .array_equal(np.array(new long[] { 200, 0, 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })).Should().BeTrue();
            // >>> np.insert(np.arange(12), [1, 13], [100, 200])  IndexError: index 14 is out of bounds for axis 0 with size 14
            Action past = () => np.insert(a, new long[] { 1, 13 }, v);
            past.Should().Throw<IndexOutOfRangeException>().WithMessage("index 14 is out of bounds for axis 0 with size 14");
            // >>> np.insert(np.arange(12), [13, 14], [100, 200])  IndexError: index 15 ... (13 shifted to 13 is in bounds)
            Action first = () => np.insert(a, new[] { 13, 14 }, v);
            first.Should().Throw<IndexOutOfRangeException>().WithMessage("index 15 is out of bounds for axis 0 with size 14");
            // >>> np.insert(np.arange(12), [1, -27], [100, 200])  IndexError: index -15 is out of bounds for axis 0 with size 14
            Action below = () => np.insert(a, new long[] { 1, -27 }, v);
            below.Should().Throw<IndexOutOfRangeException>().WithMessage("index -15 is out of bounds for axis 0 with size 14");
            // >>> np.insert(np.arange(2), [-3, 0, 1, 2], [100, 200, 300, 400])  (-3 wraps onto the slot of 2: three slots
            // for two elements) ValueError: NumPy boolean array indexing assignment cannot assign 2 input values to the 3
            // output values where the mask is true
            Action collide = () => np.insert(np.arange(2L), new long[] { -3, 0, 1, 2 }, np.array(new long[] { 100, 200, 300, 400 }));
            collide.Should().Throw<ValueError>().WithMessage(
                "NumPy boolean array indexing assignment cannot assign 2 input values to the 3 output values where the mask is true");
            // >>> np.insert(np.arange(6).reshape(2, 3), [-7, 0, 1, 2], 9, axis=1)
            // ValueError: shape mismatch: value array of shape (2,3) could not be broadcast to indexing result of shape (2,4)
            Action collide2d = () => np.insert(np.arange(6L).reshape(2, 3), new long[] { -7, 0, 1, 2 }, (object)9, axis: 1);
            collide2d.Should().Throw<ValueError>().WithMessage(
                "shape mismatch: value array of shape (2,3) could not be broadcast to indexing result of shape (2,4)");
            // >>> np.insert(np.arange(1), [-3, 0, 1, 1], [100, 200, 300, 400])  (one slab fills both free slots; the
            // later of the two values for slot 3 wins)  array([  0, 200,   0, 300, 400])
            np.insert(np.arange(1L), new long[] { -3, 0, 1, 1 }, np.array(new long[] { 100, 200, 300, 400 }))
                .array_equal(np.array(new long[] { 0, 200, 0, 300, 400 })).Should().BeTrue();
            // >>> np.insert(np.arange(3).reshape(3, 1), [-2, 0], [100, 200], axis=1)
            // array([[  0, 200, 100], [  1, 200, 100], [  2, 200, 100]])
            np.insert(np.arange(3L).reshape(3, 1), new long[] { -2, 0 }, v, axis: 1)
                .array_equal(np.array(new long[,] { { 0, 200, 100 }, { 1, 200, 100 }, { 2, 200, 100 } })).Should().BeTrue();
        }

        /// <summary>
        ///     The multi-index branch's arithmetic runs in NumPy's index dtype: <c>indices &lt; 0</c> meets the Python int
        ///     <c>N</c>, which NEP 50 refuses for a dtype that cannot hold it, and the in-place add of <c>np.arange</c> cannot
        ///     cast its int64 result into an unsigned dtype (an empty index array too). NumSharp widened every index array to
        ///     int64 first.
        /// </summary>
        [TestMethod]
        public void MultiIndex_IndexArithmeticRunsInTheIndexDtype()
        {
            var v = np.array(new long[] { 100, 200 });
            // >>> np.insert(np.arange(12), np.array([0, 2], np.uint8), [100, 200])
            // UFuncTypeError: Cannot cast ufunc 'add' output from dtype('int64') to dtype('uint8') with casting rule 'same_kind'
            // (NumSharp's ufunc output-cast refusal is the house ArgumentException with NumPy's text.)
            Action unsigned = () => np.insert(np.arange(12L), np.array(new byte[] { 0, 2 }), v);
            unsigned.Should().ThrowExactly<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'add' output from dtype('int64') to dtype('uint8') with casting rule 'same_kind'");
            Action emptyUnsigned = () => np.insert(np.arange(12L), np.array(new byte[0]), v);
            emptyUnsigned.Should().ThrowExactly<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'add' output from dtype('int64') to dtype('uint8') with casting rule 'same_kind'");
            // >>> np.insert(np.arange(300), np.array([0, 2], np.uint8), [100, 200])
            // OverflowError: Python integer 300 out of bounds for uint8
            Action overflowU = () => np.insert(np.arange(300L), np.array(new byte[] { 0, 2 }), v);
            overflowU.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for uint8");
            // >>> np.insert(np.arange(200), np.array([-1, 2], np.int8), [100, 200])
            // OverflowError: Python integer 200 out of bounds for int8
            Action overflowS = () => np.insert(np.arange(200L), np.array(new sbyte[] { -1, 2 }), v);
            overflowS.Should().Throw<OverflowException>().WithMessage("Python integer 200 out of bounds for int8");
            // >>> np.insert(np.arange(12), np.array([3, 1], np.int32), [100, 200])
            // array([  0, 200,   1,   2, 100,   3,   4,   5,   6,   7,   8,   9,  10,  11])
            np.insert(np.arange(12L), np.array(new[] { 3, 1 }), v)
                .array_equal(np.array(new long[] { 0, 200, 1, 2, 100, 3, 4, 5, 6, 7, 8, 9, 10, 11 })).Should().BeTrue();
        }

        /// <summary>
        ///     The multi-index branch assigns its values with <c>new[..., indices, ...] = values</c>, a fancy assignment, so
        ///     a value that does not fit the indexing result is MapIter's ValueError — an empty index included, whose
        ///     result has no slot (<c>np.insert(a, [], [7, 8, 9])</c> raises where NumSharp returned a copy).
        /// </summary>
        [TestMethod]
        public void MultiIndex_ValuesFitTheIndexingResultAsNumPysFancyAssignment()
        {
            var a = np.arange(12L);
            var b = np.arange(6L).reshape(2, 3);
            var three = np.array(new long[] { 7, 8, 9 });
            // >>> np.insert(np.arange(12), [1, 3], [7, 8, 9])
            // ValueError: shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,)
            Action tooMany = () => np.insert(a, new long[] { 1, 3 }, three);
            tooMany.Should().Throw<ValueError>().WithMessage(
                "shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,)");
            // >>> np.insert(b, [1, 3], [7, 8, 9], axis=1)  ... of shape (3,) ... indexing result of shape (2,2)
            Action rows = () => np.insert(b, new long[] { 1, 3 }, three, axis: 1);
            rows.Should().Throw<ValueError>().WithMessage(
                "shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,2)");
            // >>> np.insert(np.arange(12), [], [7, 8, 9])  ... of shape (3,) ... indexing result of shape (0,)
            Action emptyIndex = () => np.insert(a, new long[0], three);
            emptyIndex.Should().Throw<ValueError>().WithMessage(
                "shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (0,)");
            // >>> np.insert(b, [], [7, 8, 9], axis=1)  ... of shape (3,) ... indexing result of shape (2,0)
            Action emptyIndex2d = () => np.insert(b, new long[0], three, axis: 1);
            emptyIndex2d.Should().Throw<ValueError>().WithMessage(
                "shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,0)");
            // >>> np.insert(np.arange(12), [], 7)  is a copy (a 0-d value fits (0,))
            np.insert(a, new long[0], (object)7).array_equal(a).Should().BeTrue();
            // >>> np.insert(b, [1, 3], [[7], [8]], axis=1)  array([[0, 7, 1, 2, 7], [3, 8, 4, 5, 8]])
            np.insert(b, new long[] { 1, 3 }, np.array(new long[] { 7, 8 }).reshape(2, 1), axis: 1)
                .array_equal(np.array(new long[,] { { 0, 7, 1, 2, 7 }, { 3, 8, 4, 5, 8 } })).Should().BeTrue();
        }
    }
}
