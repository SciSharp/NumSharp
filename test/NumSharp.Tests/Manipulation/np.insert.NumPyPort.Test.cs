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
    }
}
