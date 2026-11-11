using System;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     <c>np.ndim(a)</c> / <c>np.shape(a)</c>, NumPy's module functions beside the <c>ndim</c> / <c>shape</c> properties:
    ///     an array, a view, a C# scalar (NumPy's <c>np.ndim(5)</c> is 0, <c>np.shape(5)</c> is <c>()</c>) and a C# array
    ///     literal (NumPy's list). Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class NdimShapeFunctionTests
    {
        /// <summary>Arrays and views: the rank and the lengths; a scalar literal is 0-d with an empty shape.</summary>
        [TestMethod]
        public void NdimShape_ArraysViewsAndScalars_AsNumPy()
        {
            var a = np.zeros(new Shape(2, 3, 4));
            np.ndim(a).Should().Be(3);
            np.shape(a).Should().Equal(2L, 3L, 4L);
            np.shape(np.arange(5)["::2"]).Should().Equal(3L);
            np.ndim(5).Should().Be(0);
            np.shape(5).Should().BeEmpty();
            np.ndim(new[,] { { 1, 2 }, { 3, 4 } }).Should().Be(2);
            np.shape(new[,] { { 1, 2 }, { 3, 4 } }).Should().Equal(2L, 2L);
        }

        /// <summary>np.shape returns a copy (NumPy's tuple is immutable): writing into it never reshapes the array.</summary>
        [TestMethod]
        public void Shape_ReturnsACopy()
        {
            var a = np.zeros(new Shape(2, 3));
            var s = np.shape(a);
            s[0] = 100;
            a.shape.Should().Equal(2L, 3L);
        }

        /// <summary>A null array is refused, as <see cref="np.size"/> refuses it (NumSharp has no object dtype for NumPy's
        ///     <c>np.ndim(None)</c>).</summary>
        [TestMethod]
        public void NdimShape_Null_Refused()
        {
            Action ndim = () => np.ndim(null);
            ndim.Should().Throw<ArgumentNullException>();
            Action shape = () => np.shape(null);
            shape.Should().Throw<ArgumentNullException>();
        }
    }
}
