using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     An ARRAY stored through a view index into a 0-d view: NumPy's <c>PyArray_CopyObject</c> →
    ///     <c>PyArray_AssignArray</c>, which strips the array's leading unit dimensions while it has more than the view ("a
    ///     special case for backwards compatibility") and broadcasts the rest, quoting the stripped shape when it cannot.
    ///     Expected outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): the "..., 0" key on a 1-D array
    ///     is a 0-d VIEW (an ellipsis makes it no full integer index), and NumSharp's copy loop refused every array value with
    ///     dimensions there ("setting an array element with a sequence."), or for a <c>StringDType</c> array stored the
    ///     value's <c>str()</c>.
    /// </remarks>
    [TestClass]
    public class view_assignment_numpy_port_tests
    {
        /// <summary>The key <c>(..., 0)</c>: on a 1-D array, a 0-d view of element 0.</summary>
        private static readonly object[] ZeroDView = { Slice.Ellipsis, 0 };

        /// <summary>A (1,) or (1, 1) array fills a 0-d view; anything with more dimensions after the strip cannot broadcast.</summary>
        [TestMethod]
        public void ArrayIntoA0dView_StripsLeadingUnitDims()
        {
            var a = np.arange(4.0);
            a[ZeroDView] = np.full(new Shape(1), 7.0);
            a.GetDouble(0).Should().Be(7.0);
            a[ZeroDView] = np.full(new Shape(1, 1), 8.0);
            a.GetDouble(0).Should().Be(8.0);

            Action five = () => np.arange(4.0)[ZeroDView] = np.arange(5.0);
            five.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (5,) into shape ()");
            Action square = () => np.arange(4.0)[ZeroDView] = np.zeros(new Shape(2, 2));
            square.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,2) into shape ()");
            Action row = () => np.arange(4.0)[ZeroDView] = np.zeros(new Shape(1, 5));
            row.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (5,) into shape ()");
        }

        /// <summary>A text array's 0-d view takes the element of a one-element text array, not its <c>str()</c>.</summary>
        [TestMethod]
        public void TextArrayIntoA0dView_StoresTheElement()
        {
            var t = np.zeros(new Shape(3), np.dtypes.StringDType());
            t[ZeroDView] = np.array(new[] { "x" }).astype(np.dtypes.StringDType());
            t.ToString(true).Should().Be("array(['x', '', ''], dtype=StringDType())");

            var u = np.zeros(new Shape(3), np.dtype("U5"));
            u[ZeroDView] = np.array(new[] { "x" });
            u.ToString(true).Should().Be("array(['x', '', ''], dtype='<U5')");
        }
    }
}
