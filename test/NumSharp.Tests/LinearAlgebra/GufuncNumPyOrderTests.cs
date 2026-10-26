using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     The product gufuncs (<c>np.matmul</c>, <c>np.vecdot</c>, <c>np.matvec</c>, <c>np.vecmat</c>) run their checks in
    ///     NumPy's call order: <c>ufunc_generic_fastcall</c>'s argument parse, then the loop resolution, then
    ///     <c>PyUFunc_GeneralizedFunctionInternal</c>'s gufunc checks. Every expected outcome was probed against NumPy 2.4.2;
    ///     the comments give the NumPy call.
    /// </summary>
    [TestClass]
    public class GufuncNumPyOrderTests
    {
        /// <summary>A read-only zero array of the given shape: NumPy's <c>z.setflags(write=False)</c>.</summary>
        private static NDArray ReadOnly(params long[] shape)
        {
            var z = np.zeros(new Shape(shape));
            z.setflags(write: false);
            return z;
        }

        [TestMethod]
        public void ReadOnlyOut_IsRefusedWhileTheArgumentsAreParsed()
        {
            // np.vecdot(np.ones((3,4)), np.ones((3,5)), out=ro) -> ValueError: output array is read-only (before the core
            // mismatch), and with operands that fit too (NumSharp delivered through np.copyto, whose text is
            // "assignment destination is read-only").
            Action mismatch = () => np.vecdot(np.ones(new Shape(3L, 4L)), np.ones(new Shape(3L, 5L)), ReadOnly(3));
            mismatch.Should().Throw<ValueError>().WithMessage("output array is read-only");
            Action fits = () => np.vecdot(np.ones(new Shape(3L, 4L)), np.ones(new Shape(3L, 4L)), ReadOnly(3));
            fits.Should().Throw<ValueError>().WithMessage("output array is read-only");
            // np.matvec(a34, np.ones(4), out=ro, axis=0) and np.vecmat(np.ones(3), a34, out=ro, keepdims=True): the read-only
            // out comes before the axis / keepdims refusals.
            var a34 = np.arange(12.0).reshape(3, 4);
            Action matvec = () => np.matvec(a34, np.ones(new Shape(4L)), ReadOnly(3), axis: 0);
            matvec.Should().Throw<ValueError>().WithMessage("output array is read-only");
            Action vecmat = () => np.vecmat(np.ones(new Shape(3L)), a34, ReadOnly(4), keepdims: true);
            vecmat.Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        [TestMethod]
        public void Matmul_ParsesDtypeThenOutThenCastingThenOrder()
        {
            var a34 = np.arange(12.0).reshape(3, 4);
            var a43 = np.arange(12.0).reshape(4, 3);
            // np.matmul(a34, a43, out=ro, casting='bogus') -> ValueError: output array is read-only
            Action roCasting = () => np.matmul(a34, a43, ReadOnly(3, 3), casting: "bogus");
            roCasting.Should().Throw<ValueError>().WithMessage("output array is read-only");
            // np.matmul(a34, a43, out=ro, order='Z') -> ValueError: output array is read-only
            Action roOrder = () => np.matmul(a34, a43, ReadOnly(3, 3), order: 'Z');
            roOrder.Should().Throw<ValueError>().WithMessage("output array is read-only");
            // np.matmul(a34, a43, dtype='U5', casting='bogus') -> the dtype= class TypeError
            Action dtypeCasting = () => np.matmul(a34, a43, dtype: np.dtype("U5"), casting: "bogus");
            dtypeCasting.Should().Throw<TypeError>().WithMessage(
                "The `dtype` and `signature` arguments to ufuncs only select the general DType and not details such as " +
                "the byte order or time unit. You can avoid this error by using the scalar types `np.float64` or the " +
                "dtype string notation.");
            // np.matmul(a34, a43, casting='bogus', order='Z') -> the casting ValueError (NumSharp checked the order first)
            Action castingOrder = () => np.matmul(a34, a43, casting: "bogus", order: 'Z');
            castingOrder.Should().Throw<ValueError>().WithMessage(
                "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'bogus')");
        }

        /// <summary>A 2x2 float64 operand [[1, 2], [3, 4]].</summary>
        private static NDArray F22 => np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        /// <summary>The 2x2 text [['1', '2'], ['3', '4']] in the given string dtype (S1, U5 or T).</summary>
        private static NDArray Text22(string dtype) =>
            np.array(new[,] { { "1", "2" }, { "3", "4" } }).astype(np.dtype(dtype));

        [TestMethod]
        public void InputCasts_AreCheckedUnderSameKind_BeforeTheGufuncChecks()
        {
            var c22 = F22.astype(np.complex128);
            // np.vecdot(c22, f22, dtype=np.float64) -> UFuncTypeError (NumSharp cast unsafely and computed)
            Action vecdot = () => np.vecdot(c22, F22, dtype: np.float64);
            vecdot.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'vecdot' input 0 from dtype('complex128') to dtype('float64') with casting rule 'same_kind'");
            // np.vecdot(c22, np.ones((2,3)), dtype=np.float64): the cast before the core mismatch
            Action beforeCore = () => np.vecdot(c22, np.ones(new Shape(2L, 3L)), dtype: np.float64);
            beforeCore.Should().Throw<ArgumentException>().WithMessage("Cannot cast ufunc 'vecdot' input 0 *");
            // np.matvec(c22, np.ones(2), dtype=np.float64, axis=0): the cast before the axis refusal
            Action beforeAxis = () => np.matvec(c22, np.ones(new Shape(2L)), axis: 0, dtype: np.float64);
            beforeAxis.Should().Throw<ArgumentException>().WithMessage("Cannot cast ufunc 'matvec' input 0 *");
            // np.matmul(i32, f8, casting='no') -> input 0 refused even without dtype=; 'safe' computes
            var i22 = F22.astype(np.int32);
            Action no = () => np.matmul(i22, F22, casting: "no");
            no.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'matmul' input 0 from dtype('int32') to dtype('float64') with casting rule 'no'");
            np.matmul(i22, F22, casting: "safe").Should().BeOfValues(7.0, 10.0, 15.0, 22.0).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void OutputCast_IsTheGufuncsCheck_BeforeTheGufuncChecks()
        {
            // np.vecdot(f22, f22, out=np.zeros(2, np.int8)) -> UFuncTypeError (NumSharp: np.copyto's "Cannot cast array
            // data ..." after computing)
            Action vecdot = () => np.vecdot(F22, F22, np.zeros(new Shape(2L), np.int8));
            vecdot.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'vecdot' output from dtype('float64') to dtype('int8') with casting rule 'same_kind'");
            // np.vecdot(f22, np.ones((2,3)), out=np.zeros(2, np.int8)): the cast before the core mismatch
            Action beforeCore = () => np.vecdot(F22, np.ones(new Shape(2L, 3L)), np.zeros(new Shape(2L), np.int8));
            beforeCore.Should().Throw<ArgumentException>().WithMessage("Cannot cast ufunc 'vecdot' output *");
            // np.vecdot(f22, f22, out=U5) -> ['5.0', '25.0'] (a number to fixed-width text is same_kind)
            var text = np.zeros(new Shape(2L), np.dtype("U5"));
            np.vecdot(F22, F22, text);
            text.ToString().Should().Be("['5.0' '25.0']");
            // np.matmul(f22, f22, out=U5, casting='safe') -> refused (not safe)
            Action safe = () => np.matmul(F22, F22, np.zeros(new Shape(2L, 2L), np.dtype("U5")), casting: "safe");
            safe.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'matmul' output from dtype('float64') to dtype('<U5') with casting rule 'safe'");
        }

        [TestMethod]
        public void FixedWidthText_UnderANumericRequest_IsACastNotAMissingLoop()
        {
            // np.matmul(s, s, dtype=np.float64) -> the input cast refusal (NumSharp: the no-loop text)
            Action s1 = () => np.matmul(Text22("S1"), Text22("S1"), dtype: np.float64);
            s1.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'matmul' input 0 from dtype('S1') to dtype('float64') with casting rule 'same_kind'");
            // np.matmul(f22, u, dtype=np.float64) -> input 1
            Action u1 = () => np.matmul(F22, Text22("U5"), dtype: np.float64);
            u1.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'matmul' input 1 from dtype('<U5') to dtype('float64') with casting rule 'same_kind'");
            // np.matmul(u, u, dtype=np.float64, casting='unsafe') -> [[7., 10.], [15., 22.]]: the loop converts the text
            np.matmul(Text22("U5"), Text22("U5"), dtype: np.float64, casting: "unsafe")
                .Should().BeOfValues(7.0, 10.0, 15.0, 22.0).And.BeShaped(2, 2);
            // np.matmul(t, f22, dtype=np.float64) -> no loop, naming the requested class (NumSharp said "-> None")
            Action t = () => np.matmul(Text22("T"), F22, dtype: np.float64);
            t.Should().Throw<TypeError>().WithMessage(
                "ufunc 'matmul' did not contain a loop with signature matching types (<class " +
                "'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Float64DType'>) -> <class 'numpy.dtypes.Float64DType'>");
        }
    }
}
