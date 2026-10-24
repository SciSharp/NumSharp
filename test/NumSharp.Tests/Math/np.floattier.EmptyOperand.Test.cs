using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The binary float-tier ufuncs (arctan2, logaddexp, logaddexp2, nextafter, copysign, hypot, heaviside) with an EMPTY
    ///     operand: NumPy broadcasts the operands (its error for shapes that do not broadcast) and returns an empty array of
    ///     the loop dtype and the broadcast shape. NumSharp returned a clone of the empty operand (its own dtype and shape,
    ///     no broadcast check). Every expected value is NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class np_floattier_empty_operand_tests
    {
        /// <summary>The seven ufuncs, by name, as NumPy's two-operand calls.</summary>
        private static readonly (string Name, Func<NDArray, NDArray, NDArray> Call)[] Ufuncs =
        {
            ("arctan2", (a, b) => np.arctan2(a, b)),
            ("logaddexp", (a, b) => np.logaddexp(a, b)),
            ("logaddexp2", (a, b) => np.logaddexp2(a, b)),
            ("nextafter", (a, b) => np.nextafter(a, b)),
            ("copysign", (a, b) => np.copysign(a, b)),
            ("hypot", (a, b) => np.hypot(a, b)),
            ("heaviside", (a, b) => np.heaviside(a, b)),
        };

        /// <summary>
        ///     An empty operand gives NumPy's loop dtype and the broadcast shape: int8 operands give float16 (0, 3), a
        ///     (0, 1) int32 against a (5,) float32 gives float64 (0, 5), bool gives float16.
        /// </summary>
        [TestMethod]
        public void EmptyOperand_LoopDtypeAndBroadcastShape()
        {
            foreach (var (name, call) in Ufuncs)
            {
                using (var a = np.zeros(new Shape(0, 3), np.int8))
                using (var b = np.zeros(new Shape(0, 3), np.int8))
                using (var r = call(a, b))
                {
                    r.dtype.Should().Be(np.float16, name);
                    r.shape.Should().Equal(new long[] { 0, 3 }, name);
                }
                using (var a = np.zeros(new Shape(0, 1), np.int32))
                using (var b = np.zeros(new Shape(5), np.float32))
                using (var r = call(a, b))
                {
                    r.dtype.Should().Be(np.float64, name);
                    r.shape.Should().Equal(new long[] { 0, 5 }, name);
                }
                using (var a = np.zeros(new Shape(0), np.@bool))
                using (var b = np.zeros(new Shape(1), np.@bool))
                using (var r = call(a, b))
                {
                    r.dtype.Should().Be(np.float16, name);
                    r.shape.Should().Equal(new long[] { 0 }, name);
                }
            }
        }

        /// <summary><c>np.arctan2(0, empty int8)</c>: a C# literal (NumPy's weak Python int) leaves the loop at float16.</summary>
        [TestMethod]
        public void EmptyOperand_WeakLiteral_KeepsTheArrayTier()
        {
            using var e = np.zeros(new Shape(0, 3), np.int8);
            using var r = np.arctan2(0, e);
            r.dtype.Should().Be(np.float16);
            r.shape.Should().Equal(new long[] { 0, 3 });
        }

        /// <summary>
        ///     Shapes that do not broadcast raise NumPy's elementwise text even when an operand is empty (the empty one in
        ///     either position).
        /// </summary>
        [TestMethod]
        public void EmptyOperand_ShapesThatDoNotBroadcast_Raise()
        {
            foreach (var (name, call) in Ufuncs)
            {
                using (var a = np.zeros(new Shape(0, 3), np.float16))
                using (var b = np.zeros(new Shape(2), np.float16))
                {
                    Action act = () => call(a, b);
                    act.Should().Throw<IncorrectShapeException>(name)
                        .WithMessage("operands could not be broadcast together with shapes (0,3) (2,) ");
                }
                using (var a = np.zeros(new Shape(3), np.uint8))
                using (var b = np.zeros(new Shape(0), np.int8))
                {
                    Action act = () => call(a, b);
                    act.Should().Throw<IncorrectShapeException>(name)
                        .WithMessage("operands could not be broadcast together with shapes (3,) (0,) ");
                }
            }
        }
    }
}
