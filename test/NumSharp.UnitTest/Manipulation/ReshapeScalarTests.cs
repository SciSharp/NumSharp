using System;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Tests for reshaping to scalar shape.
    /// </summary>
    public class ReshapeScalarTests
    {
        /// <summary>
        ///     BUG 73 FIX: reshape to scalar shape (Shape()) should work for single-element arrays.
        ///
        ///     NumPy:    np.array([42]).reshape(()) = array(42), shape=(), ndim=0
        ///     Fixed:    Works correctly
        ///     Was:      NullReferenceException
        /// </summary>
        [Test]
        public void Reshape_ToScalarShape_SingleElement_Works()
        {
            var a = np.array(new int[] { 42 }); // shape (1,)
            a.ndim.Should().Be(1);
            a.shape[0].Should().Be(1);

            var result = a.reshape(new Shape());

            result.ndim.Should().Be(0, "scalar shape has ndim=0");
            result.size.Should().Be(1, "scalar has size=1");
            result.GetInt32(0).Should().Be(42, "value should be preserved");
        }

        /// <summary>
        ///     Test reshaping from 2D single-element to scalar.
        /// </summary>
        [Test]
        public void Reshape_FromMultiDim_ToScalar_Works()
        {
            var a = np.array(new int[,] { { 99 } }); // shape (1, 1)
            a.ndim.Should().Be(2);

            var result = a.reshape(new Shape());

            result.ndim.Should().Be(0);
            result.GetInt32(0).Should().Be(99);
        }

        /// <summary>
        ///     Test that reshaping non-single-element array to scalar throws.
        /// </summary>
        [Test]
        public void Reshape_MultiElement_ToScalar_Throws()
        {
            var a = np.array(new int[] { 1, 2, 3 }); // size 3

            new Action(() => a.reshape(new Shape()))
                .Should().ThrowExactly<IncorrectShapeException>(
                    "Cannot reshape array of size 3 into scalar shape");
        }
    }
}
