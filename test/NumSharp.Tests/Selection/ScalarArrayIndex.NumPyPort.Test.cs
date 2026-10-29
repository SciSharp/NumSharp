using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     A 0-d integer array in an index with no index array beside it, as NumPy's <c>prepare_index</c> reads it
    ///     (<c>HAS_SCALAR_ARRAY</c>): the integer it holds, and the index stays a VIEW — copied by the getter
    ///     (<c>PyArray_NewCopy(view, NPY_KEEPORDER)</c>, so the copy keeps the source's memory order) and assigned through by
    ///     the setter (<c>PyArray_CopyObject</c>, whose broadcast text is the view's). Expected outcomes probed against NumPy
    ///     2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): such a key was a fancy index, so
    ///     a store checked its value with MapIter's text ("shape mismatch: value array of shape (5,) could not be broadcast
    ///     to indexing result of shape (4,)") and the getter's copy was C-ordered whatever the source's order. Values were
    ///     right either way.
    /// </remarks>
    [TestClass]
    public class scalar_array_index_numpy_port_tests
    {
        /// <summary>A (3, 4) float64 array holding 0 to 11 in C order.</summary>
        private static NDArray A2() => np.arange(12.0).reshape(3, 4);

        /// <summary>A (2, 3, 4) float64 array holding 0 to 23 in C order.</summary>
        private static NDArray A3() => np.arange(24.0).reshape(2, 3, 4);

        /// <summary>The array's values in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values as doubles.</returns>
        private static double[] Values(NDArray a) => a.flatten().ToArray<double>();

        /// <summary>The getter copies the view of the 0-d array's integer, keeping an F-ordered source's order.</summary>
        [TestMethod]
        public void ScalarArrayBesideBasicEntries_Get_IsAKeepOrderCopy()
        {
            var f = np.asfortranarray(A3());
            var plane = f[np.array(1L)];
            plane.shape.Should().Equal(3L, 4L);
            plane.flags.owndata.Should().BeTrue("NumPy copies a HAS_SCALAR_ARRAY index");
            plane.flags.f_contiguous.Should().BeTrue("the copy keeps the source's memory order (KEEPORDER)");
            plane.flags.c_contiguous.Should().BeFalse();
            Values(plane).Should().Equal(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);

            var rows = f[Slice.All, np.array(1L)];
            rows.shape.Should().Equal(2L, 4L);
            rows.flags.f_contiguous.Should().BeTrue();
            Values(rows).Should().Equal(4, 5, 6, 7, 16, 17, 18, 19);

            var part = A2()[np.array(1L), "1:3"];
            part.flags.owndata.Should().BeTrue();
            Values(part).Should().Equal(5, 6);
        }

        /// <summary>The setter assigns through the view: a value that does not broadcast raises the view's text.</summary>
        [TestMethod]
        public void ScalarArrayBesideBasicEntries_Set_IsAViewAssignment()
        {
            Action row = () => A2()[np.array(1L)] = np.arange(5.0);
            row.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (5,) into shape (4,)");
            Action withInteger = () => A3()[np.array(1L), 0] = np.arange(5.0);
            withInteger.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (5,) into shape (4,)");
            Action withNewAxis = () => A2()[np.array(1L), Slice.NewAxis] = np.arange(5.0);
            withNewAxis.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (5,) into shape (1,4)");

            var stored = A2();
            stored[np.array(1L)] = np.arange(4.0);
            Values(stored).Should().Equal(0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11);
            var two = A3();
            two[np.array(0L), np.array(1L)] = np.zeros(new Shape(1, 4));
            Values(two["0, 1"]).Should().Equal(0, 0, 0, 0);
        }
    }
}
