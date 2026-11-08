using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>_mean</c> / <c>_var</c> / <c>_std</c> with <c>out=</c> check their arguments in NumPy's statement order:
    ///     <c>_count_reduce_items</c> normalizes the axis first, <c>_var</c>'s mean and its <c>x = arr - arrmean</c> run next,
    ///     and only the LAST <c>umr_sum</c> parses <c>out</c> (its read-only check, its rank and shape). So with a read-only or
    ///     misshapen out, an axis out of bounds is the AxisError, a text input is its add.reduce refusal and a bool input in a
    ///     bool loop is the boolean-subtract TypeError. Every outcome was probed against NumPy 2.4.2, through the engine members
    ///     and through numpy.matrix's members (whose out= paths are those members).
    /// </summary>
    /// <remarks>
    ///     Before the fix the out's checks ran first (the reduce prologue numpy.matrix's members ran before the computation),
    ///     so every case below raised "output array is read-only" or a shape error instead.
    /// </remarks>
    [TestClass]
    public class np_mean_var_out_check_order_tests
    {
        /// <summary>A read-only (1, 3) matrix out.</summary>
        private static NDMatrix ReadOnlyRow()
        {
            var o = np.zeros(new Shape(1, 3));
            o.setflags(write: false);
            return np.asmatrix(o);
        }

        /// <summary>The axis comes before the out: matrix.mean / var / std(axis=5 or -3, out=read-only) is NumPy's AxisError.</summary>
        [TestMethod]
        public void MatrixMeanVarStd_AxisOutOfBounds_ComesBeforeAReadOnlyOut()
        {
            using var m = np.asmatrix(np.arange(6.0).reshape(2, 3));
            new Action(() => m.mean(5, null, ReadOnlyRow())).Should().Throw<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 2*");
            new Action(() => m.var(5, null, ReadOnlyRow())).Should().Throw<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 2*");
            new Action(() => m.std(5, null, ReadOnlyRow())).Should().Throw<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 2*");
            new Action(() => m.mean(-3, null, ReadOnlyRow())).Should().Throw<AxisError>()
                .WithMessage("axis -3 is out of bounds for array of dimension 2*");

            // matrix.sum is ufunc.reduce, which parses out= before the axis: the read-only out wins there.
            new Action(() => m.sum(5, null, ReadOnlyRow())).Should().Throw<ValueError>()
                .WithMessage("output array is read-only");
        }

        /// <summary>_var's x = arr - arrmean (two bools) raises before its last sum parses the out.</summary>
        [TestMethod]
        public void Var_BoolInputInABoolLoop_SubtractRefusalComesBeforeTheOut()
        {
            using var b = np.array(new[,] { { true, false, true }, { false, true, true } });
            using var ro = np.zeros(new Shape(3), np.@bool);
            ro.setflags(write: false);
            const string subtract = "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.";
            new Action(() => b.TensorEngine.ReduceVar(b, 0, false, null, np.@bool, ro)).Should().Throw<TypeError>().WithMessage(subtract);
            new Action(() => b.TensorEngine.ReduceStd(b, 0, false, null, np.@bool, ro)).Should().Throw<TypeError>().WithMessage(subtract);
            using var wrong = np.zeros(new Shape(4), np.@bool);
            new Action(() => b.TensorEngine.ReduceVar(b, 0, false, null, np.@bool, wrong)).Should().Throw<TypeError>().WithMessage(subtract);
        }

        /// <summary>_var's mean refuses a text input before its last sum parses the out (read-only or misshapen).</summary>
        [TestMethod]
        public void Var_TextInput_MeanRefusalComesBeforeTheOut()
        {
            using var u = np.array(new[,] { { "1", "2", "3" }, { "4", "5", "6" } });
            using var ro = np.zeros(new Shape(3));
            ro.setflags(write: false);
            new Action(() => u.TensorEngine.ReduceVar(u, 0, false, null, null, ro)).Should().Throw<TypeError>()
                .WithMessage("the resolved dtypes are not compatible with add.reduce*");
            using var wrong = np.zeros(new Shape(4));
            new Action(() => u.TensorEngine.ReduceVar(u, 0, false, null, null, wrong)).Should().Throw<TypeError>()
                .WithMessage("the resolved dtypes are not compatible with add.reduce*");

            // _mean's umr_sum parses out itself (the text routes' own order): a read-only out first.
            new Action(() => u.TensorEngine.ReduceMean(u, 0, false, null, ro)).Should().Throw<ValueError>()
                .WithMessage("output array is read-only");
        }

        /// <summary>With every earlier statement passing, the out's own checks keep NumPy's texts.</summary>
        [TestMethod]
        public void MeanVar_ValidAxis_OutChecksKeepNumPysTexts()
        {
            using var m = np.arange(6.0).reshape(2, 3);
            using var wrong = np.zeros(new Shape(4));
            new Action(() => m.TensorEngine.ReduceVar(m, 0, false, null, null, wrong)).Should().Throw<IncorrectShapeException>()
                .WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (4,)->(4) (2,3) ");
            new Action(() => m.TensorEngine.ReduceMean(m, 0, false, null, wrong)).Should().Throw<IncorrectShapeException>()
                .WithMessage("operands could not be broadcast together with remapped shapes [original->remapped]: (4,)->(4) (2,3) ");
            using var ro = np.zeros(new Shape(4));
            ro.setflags(write: false);
            new Action(() => m.TensorEngine.ReduceMean(m, 0, false, null, ro)).Should().Throw<ValueError>()
                .WithMessage("output array is read-only");
        }
    }
}
