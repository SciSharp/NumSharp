using System;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     How NumPy's <c>array_assign_subscript</c> converts the value of a store: with the array's own descriptor, after the
    ///     index. A full integer index (one integer, or 0-d integer array, per axis) packs the value into its one element
    ///     with the dtype's setitem (<c>array_assign_item</c> → <c>PyArray_Pack</c>); None and a Python int are packed the
    ///     same way for any selection, before the store looks at it; an index array's values are bounds-checked after the
    ///     conversion. Expected outcomes probed against NumPy 2.4.2 (the int64 / uint64 overflow texts on Linux, the LP64
    ///     texts NumSharp pins; Windows NumPy says "int too big to convert" there).
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): <c>__setitem__</c> of a Python int
    ///     past every C# integer raised <c>NotSupportedException</c> ("Unable to resolve asanyarray for type BigInteger"); a
    ///     list into one element of an integer array raised the sequence ValueError and into a StringDType element stored
    ///     <c>'[1 2]'</c>; None was converted before the index was checked, or not at all for an empty selection; an array
    ///     with dimensions into one element was refused alike for every dtype; a uint64 index read a fixed-width text element
    ///     as a copy and missed the element store; a 0-d integer array was a fancy index.
    /// </remarks>
    [TestClass]
    public class setitem_value_conversion_numpy_port_tests
    {
        private static readonly BigInteger Big = BigInteger.Pow(2, 70);

        /// <summary>The array's elements read back as strings (any dtype), for value checks.</summary>
        /// <param name="a">The array.</param>
        /// <returns>Its repr, NumPy's text.</returns>
        private static string Repr(NDArray a) => a.ToString(true);

        /// <summary>
        ///     A Python int past every C# integer is stored with the array's descriptor: its rounded double, its truth, its
        ///     digits (truncated to a fixed width), or NumPy's OverflowError for an integer dtype (LP64 texts).
        /// </summary>
        [TestMethod]
        public void PythonIntPastInt64_IsPackedWithTheArraysDtype()
        {
            var f = np.arange(3.0);
            f.__setitem__(0, Big);
            f.GetDouble(0).Should().Be(1.1805916207174113e21);

            var b = np.zeros(new Shape(3), np.@bool);
            b.__setitem__(0, Big);
            b.GetBoolean(0).Should().BeTrue();

            var u = np.zeros(new Shape(3), np.dtype("U5"));
            u.__setitem__(0, Big);
            Repr(u).Should().Be("array(['11805', '', ''], dtype='<U5')");

            var t = np.zeros(new Shape(3), np.dtypes.StringDType());
            t.__setitem__(0, Big);
            Repr(t).Should().Be("array(['1180591620717411303424', '', ''], dtype=StringDType())");

            var s = np.arange(3.0);
            s.__setitem__("1:", Big);
            Repr(s).Should().Be("array([0.00000000e+00, 1.18059162e+21, 1.18059162e+21])");

            foreach (var dt in new[] { "int8", "int32", "int64", "uint64" })
            {
                var i = np.zeros(new Shape(3), np.dtype(dt));
                Action store = () => i.__setitem__(0, Big);
                store.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            }
        }

        /// <summary>
        ///     NumPy's order for a Python int: the writeability, the index (an integer's bounds), the conversion, then an index
        ///     array's bounds — so a fancy out-of-bounds index meets the OverflowError first for an integer dtype and the
        ///     IndexError for a float one.
        /// </summary>
        [TestMethod]
        public void PythonInt_ChecksTheIndexBeforeAndTheIndexArrayAfterTheConversion()
        {
            Action readOnly = () => np.broadcast_to(np.arange(3.0), new Shape(2, 3)).__setitem__(0, Big);
            readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");

            Action integerOut = () => np.zeros(new Shape(3), np.int32).__setitem__(99, Big);
            integerOut.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 0 with size 3");

            Action fancyInt = () => np.zeros(new Shape(3), np.int32).__setitem__(new long[] { 0, 99 }, Big);
            fancyInt.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");

            Action fancyFloat = () => np.arange(3.0).__setitem__(new long[] { 0, 99 }, Big);
            fancyFloat.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 0 with size 3");
        }

        /// <summary>A Python list into one element is the dtype's setitem's to answer.</summary>
        [TestMethod]
        public void PythonList_IntoOneElement_IsTheSetitemsAnswer()
        {
            object list() => new object[] { 1, 2 };
            Action int8 = () => np.zeros(new Shape(3), np.int8).__setitem__(0, list());
            int8.Should().Throw<TypeError>().WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'list'");
            Action complex = () => np.zeros(new Shape(3), np.complex128).__setitem__(0, list());
            complex.Should().Throw<TypeError>().WithMessage("must be real number, not list");
            Action f64 = () => np.zeros(new Shape(3)).__setitem__(0, list());
            f64.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.");
            Action u5 = () => np.zeros(new Shape(3), np.dtype("U5")).__setitem__(0, list());
            u5.Should().Throw<ValueError>().Where(e => e.Message == "setting an array element with a sequence");

            var b = np.zeros(new Shape(3), np.@bool);
            b.__setitem__(0, list());
            b.GetBoolean(0).Should().BeTrue();

            var t = np.zeros(new Shape(3), np.dtypes.StringDType());
            t.__setitem__(0, list());
            Repr(t).Should().Be("array(['[1, 2]', '', ''], dtype=StringDType())");
        }

        /// <summary>None is converted after the index's checks and before the store looks at the selection.</summary>
        [TestMethod]
        public void None_IsConvertedAfterTheIndexAndForAnEmptySelection()
        {
            const string intNone = "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'";
            Action empty = () => np.zeros(new Shape(3), np.int32).__setitem__(false, null);
            empty.Should().Throw<TypeError>().WithMessage(intNone);
            Action badIndex = () => np.zeros(new Shape(3), np.int32).__setitem__(1.5, null);
            badIndex.Should().Throw<IndexError>().Where(e => e.Message ==
                "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices");
            Action outOfBounds = () => np.zeros(new Shape(3), np.int32).__setitem__(99, null);
            outOfBounds.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 0 with size 3");
        }

        /// <summary>An array with dimensions into one element is the dtype's setitem's to answer too.</summary>
        [TestMethod]
        public void ArrayWithDimensions_IntoOneElement_IsTheSetitemsAnswer()
        {
            var b = np.ones(new Shape(3), np.@bool);
            b[0] = np.zeros(new Shape(1));
            b.GetBoolean(0).Should().BeFalse("bool takes a one-element array's truth");

            Action complex = () => np.zeros(new Shape(3), np.complex128)[0] = np.zeros(new Shape(1));
            complex.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
            Action f64 = () => np.zeros(new Shape(3))[0] = np.zeros(new Shape(1));
            f64.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.");
            Action u5 = () => np.zeros(new Shape(3), np.dtype("U5"))[0] = np.zeros(new Shape(1), np.dtype("U5"));
            u5.Should().Throw<ValueError>().Where(e => e.Message == "setting an array element with a sequence");

            var t = np.zeros(new Shape(3), np.dtypes.StringDType());
            t[0] = np.array(new[] { 0.5 });
            Repr(t).Should().Be("array(['[0.5]', '', ''], dtype=StringDType())");
        }

        /// <summary>
        ///     A uint64 index is a basic integer (a view of a fixed-width text element, and NumPy's element store), and a 0-d
        ///     integer array per axis is a full integer index (NumPy cleans <c>HAS_SCALAR_ARRAY</c> off it).
        /// </summary>
        [TestMethod]
        public void Uint64AndScalarArrayIndices_AreFullIntegerIndices()
        {
            var u = np.array(new[] { "ab", "cd", "ef" });
            var view = u[1UL];
            view.flags.owndata.Should().BeFalse("a uint64 index is basic: a view");
            view.flags.writeable.Should().BeTrue();
            Action ulongStore = () => u[1UL] = np.zeros(new Shape(1), np.dtype("U5"));
            ulongStore.Should().Throw<ValueError>().Where(e => e.Message == "setting an array element with a sequence");

            var t = np.array(new[] { "a", "b", "c" }).astype(np.dtypes.StringDType());
            t[np.array(1L)] = np.array(new[] { 0.5 });
            Repr(t).Should().Be("array(['a', '[0.5]', 'c'], dtype=StringDType())");

            Action scalarArrayStore = () => np.array(new[] { "ab", "cd", "ef" })[np.array(1L)] = np.zeros(new Shape(1), np.dtype("U5"));
            scalarArrayStore.Should().Throw<ValueError>().Where(e => e.Message == "setting an array element with a sequence");
        }
    }
}
