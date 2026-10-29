using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     A Python LIST stored into a selection of several elements, as NumPy's <c>array_assign_subscript</c> converts it:
    ///     with the array's own descriptor (every item by its setitem — a Python int range-checked, a str parsed, None NaN or
    ///     <c>int()</c>'s TypeError), by the route of the index: a view assignment's <c>PyArray_CopyObject</c>, whose shape
    ///     discovery stops at the view's rank; for a <c>StringDType</c> array's fancy index the same CopyObject into MapIter's
    ///     temporary of the result shape; for any other fancy index or a mask <c>PyArray_FromAny(value, descr)</c>, unlimited.
    ///     Expected outcomes probed against NumPy 2.4.2 (C# <c>object[]</c> is the Python list; a nested list is spelled with
    ///     explicit <c>new object[] { ... }</c>, since a lone <c>object[]</c> argument to a <c>params object[]</c> parameter
    ///     would be the params array itself).
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan) and a probe around it: the list
    ///     became the array NumSharp's coercion discovers (int64 for Python ints), cast by the store — <c>uint8[0:2] = [1, -1]</c>
    ///     stored 255 where NumPy raises —, None in a list was refused, a list nested deeper than a view was broadcast, and a
    ///     StringDType fancy store raised MapIter's text instead of CopyObject's.
    /// </remarks>
    [TestClass]
    public class setitem_list_value_numpy_port_tests
    {
        /// <summary>A Python list: its items as given.</summary>
        /// <param name="items">The items.</param>
        /// <returns>The list.</returns>
        private static object[] L(params object[] items) => items;

        /// <summary>The items are stored by the array's setitem, on every route: range checks, None, parsing.</summary>
        [TestMethod]
        public void ListIntoASelection_IsConvertedWithTheArraysDtype()
        {
            var mask = np.array(new[] { true, false, true, false });
            Action view = () => np.zeros(new Shape(4), np.uint8).__setitem__("0:2", L(1, -1));
            view.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            Action fancy = () => np.zeros(new Shape(4), np.uint8).__setitem__(new long[] { 0, 1 }, L(1, -1));
            fancy.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            Action masked = () => np.zeros(new Shape(4), np.uint8).__setitem__(mask, L(1, -1));
            masked.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            Action int8 = () => np.zeros(new Shape(4), np.int8).__setitem__("0:2", L(1, 1000));
            int8.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");

            Action none = () => np.zeros(new Shape(4), np.int8).__setitem__("0:2", L(1, null));
            none.Should().Throw<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            var f = np.zeros(new Shape(4));
            f.__setitem__("0:2", L(1, null));
            double.IsNaN(f.GetDouble(1)).Should().BeTrue("None into a float array is NaN");

            Action text = () => np.zeros(new Shape(4)).__setitem__("0:2", L(1, "x"));
            text.Should().Throw<ValueError>().WithMessage("could not convert string to float: 'x'");
        }

        /// <summary>On a view, the list is discovered with at most the view's rank (PyArray_CopyObject's max_dims).</summary>
        [TestMethod]
        public void ListIntoAView_IsDiscoveredWithAtMostTheViewsRank()
        {
            const string deep1 = "setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 1.";
            Action row = () => np.zeros(new Shape(4)).__setitem__("1:", new object[] { L(1, 2, 3) });
            row.Should().Throw<ValueError>().Where(e => e.Message == deep1);
            Action columns = () => np.zeros(new Shape(4)).__setitem__("1:3", L(L(1), L(2)));
            columns.Should().Throw<ValueError>().Where(e => e.Message == deep1);
            Action zeroD = () => np.zeros(new Shape(4)).__setitem__(L(Slice.Ellipsis, 0), L(1));
            zeroD.Should().Throw<ValueError>().Where(e => e.Message ==
                "setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 0.");
            Action emptyIntoZeroD = () => np.zeros(new Shape(4)).__setitem__(L(Slice.Ellipsis, 0), new object[0]);
            emptyIntoZeroD.Should().Throw<ValueError>().Where(e => e.Message ==
                "setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 0.");
            Action raggedAtTheLimit = () => np.zeros(new Shape(3, 4)).__setitem__("0:2", L(L(1, L(2)), L(3, 4)));
            raggedAtTheLimit.Should().Throw<ValueError>().Where(e => e.Message ==
                "setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 2.");

            var broadcast = np.zeros(new Shape(3, 4));
            broadcast.__setitem__("0:2", new object[] { L(1, 2, 3, 4) });
            broadcast["0:2"].flatten().ToArray<double>().Should().Equal(1, 2, 3, 4, 1, 2, 3, 4);
        }

        /// <summary>
        ///     A StringDType array's fancy store fills MapIter's temporary of the result shape with CopyObject: the list is
        ///     bounded by the result's rank and must broadcast to its shape (CopyObject's text, not MapIter's). Any other
        ///     dtype converts the list unbounded (a (1, 2) list fits a (2,) result).
        /// </summary>
        [TestMethod]
        public void ListIntoAStringDTypeFancyIndex_FillsTheResultShapedTemporary()
        {
            var T = np.dtypes.StringDType();
            Action deep = () => np.zeros(new Shape(4), T).__setitem__(new long[] { 0, 1 }, new object[] { L("a", "b") });
            deep.Should().Throw<ValueError>().Where(e => e.Message ==
                "setting an array element with a sequence. The requested array would exceed the maximum number of dimension of 1.");
            Action three = () => np.zeros(new Shape(4), T).__setitem__(new long[] { 0, 1 }, L("a", "b", "c"));
            three.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,) into shape (2,)");
            Action viaBool = () => np.zeros(new Shape(3, 4), T).__setitem__(true, L("a", "b"));
            viaBool.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,) into shape (1,3,4)");
            // On a 0-d array a bool is the full mask (HAS_BOOL): no MapIter, the list converted unbounded and checked by the
            // boolean assignment.
            Action fullMask = () => np.zeros(Shape.Scalar, T).__setitem__(true, L("a", "b"));
            fullMask.Should().Throw<ValueError>().WithMessage(
                "NumPy boolean array indexing assignment cannot assign 2 input values to the 1 output values where the mask is true");

            var t = np.zeros(new Shape(4), T);
            t.__setitem__(new long[] { 0, 1 }, L("a", "b"));
            t.ToString(true).Should().Be("array(['a', 'b', '', ''], dtype=StringDType())");

            var f = np.zeros(new Shape(4));
            f.__setitem__(new long[] { 0, 1 }, new object[] { L(1, 2) });
            f.ToArray<double>().Should().Equal(1, 2, 0, 0);
        }
    }
}
