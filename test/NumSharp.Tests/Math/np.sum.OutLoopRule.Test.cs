using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The engine's add reduction into an <c>out=</c> without <c>dtype=</c> (<see cref="TensorEngine.ReduceAdd"/>) is
    ///     NumPy's <c>add.reduce(a, axis, out=o)</c>: the reduce prologue checks <c>out</c> first (NumPy's texts), and the loop
    ///     is <c>promote_types(out.dtype, a.dtype)</c> without the small-integer widening a reduction applies without
    ///     <c>out</c> (<c>reducelike_promote_and_resolve</c>), the result stored with an unsafe cast. Every literal was probed
    ///     against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Before the fix the engine summed in NEP 50's accumulator whatever the out (an int8 sum into a bool out was the
    ///     truth of the int64 sum: True where NumPy's int8 sum wraps to 0 and stores False), broadcast its result into a
    ///     larger out, and raised its own texts for a misshapen or read-only out.
    /// </remarks>
    [TestClass]
    public class np_sum_out_loop_rule_tests
    {
        /// <summary>The loop is promote_types(out, a): it narrows, widens and changes kind with the out.</summary>
        [TestMethod]
        public void OutWithoutDtype_LoopIsThePromotionOfOutAndInput()
        {
            // int8 [100, 100, 56] into bool: the int8 loop wraps 256 to 0, stored as False.
            using (var a = np.array(new sbyte[] { 100, 100, 56 }))
            using (var o = np.zeros(Shape.Scalar, np.@bool))
            {
                var r = a.TensorEngine.ReduceAdd(a, null, false, null, o);
                ReferenceEquals(r, o).Should().BeTrue();
                o.GetBoolean().Should().BeFalse();
            }
            // float32 [16777216, 1, 1] into float64: summed in float64 (16777218), not in float32 (16777216).
            using (var a = np.array(new float[] { 16777216f, 1f, 1f }))
            using (var o = np.zeros(Shape.Scalar, np.float64))
            {
                a.TensorEngine.ReduceAdd(a, null, false, null, o);
                o.GetDouble().Should().Be(16777218.0);
            }
            // uint64 [2**63 + 1, 0] into int64: promote_types(int64, uint64) is float64, whose 9.223372036854776e18 casts to
            // int64 as INT64_MIN (NumPy's unsafe float -> int cast), not the wrapped 0x8000000000000001.
            using (var a = np.array(new ulong[] { (1UL << 63) + 1, 0 }))
            using (var o = np.zeros(Shape.Scalar, np.int64))
            {
                a.TensorEngine.ReduceAdd(a, null, false, null, o);
                o.GetInt64().Should().Be(long.MinValue);
            }
            // float16 [[2048, 1], [0.5, 0.25]] along axis 1, keepdims, into float64: summed in float64 (2049), not in
            // float16 (2048 + 1 rounds back to 2048).
            using (var a = np.array(new Half[,] { { (Half)2048, (Half)1 }, { (Half)0.5, (Half)0.25 } }))
            using (var o = np.zeros(new Shape(2, 1), np.float64))
            {
                a.TensorEngine.ReduceAdd(a, 1, true, null, o);
                o.ToArray<double>().Should().Equal(2049.0, 0.75);
            }
            // A dtype= still decides the loop: int8 into bool with dtype=int64 sums 256 and stores True.
            using (var a = np.array(new sbyte[] { 100, 100, 56 }))
            using (var o = np.zeros(Shape.Scalar, np.@bool))
            {
                a.TensorEngine.ReduceAdd(a, null, false, np.int64, o);
                o.GetBoolean().Should().BeTrue();
            }
        }

        /// <summary>The reduce prologue's checks of out, NumPy's order and texts.</summary>
        [TestMethod]
        public void OutChecks_AreTheReducePrologue()
        {
            using var a = np.arange(6.0).reshape(2, 3);

            using (var o = np.zeros(new Shape(2, 3)))
                new Action(() => a.TensorEngine.ReduceAdd(a, 0, false, null, o)).Should().Throw<ValueError>()
                    .WithMessage("output parameter for reduction operation add has the wrong number of dimensions: Found 2 but expected 1");

            using (var o = np.zeros(new Shape(4)))
                new Action(() => a.TensorEngine.ReduceAdd(a, 0, false, null, o)).Should().Throw<IncorrectShapeException>()
                    .WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (4,)->(4) (2,3) ");

            using (var o = np.zeros(new Shape(3)))
                new Action(() => a.TensorEngine.ReduceAdd(a, 0, true, null, o)).Should().Throw<ValueError>()
                    .WithMessage("output parameter for reduction operation add has the wrong number of dimensions: Found 1 but expected 2 (must match the operand's when keepdims=True)");

            using (var o = np.zeros(Shape.Scalar))
            {
                o.setflags(write: false);
                new Action(() => a.TensorEngine.ReduceAdd(a, null, false, null, o)).Should().Throw<ValueError>()
                    .WithMessage("output array is read-only");
            }

            // ufunc.reduce parses out= before the axis: a read-only out wins over an axis out of bounds.
            using (var o = np.zeros(new Shape(3)))
            {
                o.setflags(write: false);
                new Action(() => a.TensorEngine.ReduceAdd(a, 5, false, null, o)).Should().Throw<ValueError>()
                    .WithMessage("output array is read-only");
            }
        }
    }
}
