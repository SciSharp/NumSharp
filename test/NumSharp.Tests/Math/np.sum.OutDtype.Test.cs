using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The engine's add reduction (<see cref="TensorEngine.ReduceAdd"/>) into an <c>out=</c>: NumPy adds in the loop
    ///     dtype and casts the result into an out of another dtype. Every route used to write the LOOP dtype's bytes into
    ///     out's buffer as they were (an int64 sum read back as float64 denormals, a float64 sum written past a float32 out's
    ///     end) or to unbox the loop's value as out's type (InvalidCastException), and the flat float loops boxed every value
    ///     as a Complex (a switch expression's best common type), so even an out of the loop's own dtype failed. Every literal
    ///     was probed against NumPy 2.4.2 (<c>np.add.reduce(a, axis, dtype, out=o)</c>).
    /// </summary>
    [TestClass]
    public class np_sum_out_dtype_tests
    {
        /// <summary>An out of another dtype receives the loop's result cast into it, and is returned itself.</summary>
        [TestMethod]
        public void IntoAnOutOfAnotherDtype_CastsTheLoopsResult()
        {
            // int64 along axis 0 into float64: the column sums, not the int64 bits read as doubles.
            using (var a = np.arange(12L).reshape(3, 4))
            using (var o = np.zeros(new Shape(4), np.float64))
            {
                var r = a.TensorEngine.ReduceAdd(a, 0, false, null, o);
                ReferenceEquals(r, o).Should().BeTrue();
                o.ToArray<double>().Should().Equal(12.0, 15.0, 18.0, 21.0);
            }
            // float64 along axis 0 into float32: four float32 values, not four doubles written over a 16-byte buffer.
            using (var a = np.arange(12.0).reshape(3, 4))
            using (var o = np.zeros(new Shape(4), np.float32))
            {
                a.TensorEngine.ReduceAdd(a, 0, false, null, o);
                o.ToArray<float>().Should().Equal(12f, 15f, 18f, 21f);
            }
            // float64 flat into float32: the float64 sum rounded once (NumPy's 0x3F19999A).
            using (var a = np.array(new[] { 0.1, 0.2, 0.3 }))
            using (var o = np.zeros(Shape.Scalar, np.float32))
            {
                a.TensorEngine.ReduceAdd(a, null, false, null, o);
                BitConverter.SingleToUInt32Bits(o.GetSingle()).Should().Be(1058642330u);
            }
            // The bool loop (logical OR) into float64; float16 along an axis into float32; an empty int64 sum's identity.
            using (var a = np.array(new[] { true, false, true }))
            using (var o = np.zeros(Shape.Scalar, np.float64))
            {
                a.TensorEngine.ReduceAdd(a, null, false, np.@bool, o);
                o.GetDouble().Should().Be(1.0);
            }
            using (var a = np.full(new Shape(3, 4), (Half)0.5, np.float16))
            using (var o = np.zeros(new Shape(4), np.float32))
            {
                a.TensorEngine.ReduceAdd(a, 0, false, null, o);
                o.ToArray<float>().Should().Equal(1.5f, 1.5f, 1.5f, 1.5f);
            }
            using (var a = np.zeros(new Shape(0), np.int64))
            using (var o = np.full(Shape.Scalar, 7.0, np.float64))
            {
                a.TensorEngine.ReduceAdd(a, null, false, null, o);
                o.GetDouble().Should().Be(0.0);
            }
        }

        /// <summary>A flat float sum into an out of the loop's OWN dtype keeps its exact value (the boxing bug).</summary>
        [TestMethod]
        public void FlatFloatSum_IntoAnOutOfTheLoopsDtype_KeepsItsValue()
        {
            using (var a = np.array(new[] { 0.1, 0.2, 0.3 }))
            using (var o = np.zeros(Shape.Scalar, np.float64))
            {
                var r = a.TensorEngine.ReduceAdd(a, null, false, null, o);
                ReferenceEquals(r, o).Should().BeTrue();
                o.GetDouble().Should().Be(0.6000000000000001);
            }
            using (var a = np.array(new[] { 0.5f, 0.25f }))
            using (var o = np.zeros(Shape.Scalar, np.float32))
            {
                a.TensorEngine.ReduceAdd(a, null, false, null, o);
                o.GetSingle().Should().Be(0.75f);
            }
        }
    }
}
