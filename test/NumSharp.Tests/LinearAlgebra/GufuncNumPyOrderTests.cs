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

        [TestMethod]
        public void OrderLetter_IsCaseInsensitive_AndANullCastingIsATypeError()
        {
            // NumPy's order converter takes one letter in either case: np.matmul(a, b, order='f') and
            // np.einsum('ij,jk->ik', a, b, order='f') are F-ordered, 'c' / 'a' / 'k' C-ordered (probed 2.4.2, no warning).
            // NumSharp refused the lowercase letters.
            var a = np.ones(new Shape(2L, 3L));
            var b = np.ones(new Shape(3L, 2L));
            foreach (char o in "fF")
            {
                var m = np.matmul(a, b, order: o);
                Assert.IsTrue(m.Shape.IsFContiguous && !m.Shape.IsContiguous, $"matmul order '{o}'");
                var e = np.einsum("ij,jk->ik", new[] { a, b }, order: o);
                Assert.IsTrue(e.Shape.IsFContiguous && !e.Shape.IsContiguous, $"einsum order '{o}'");
            }

            foreach (char o in "cakCAK")
            {
                Assert.IsTrue(np.matmul(a, b, order: o).Shape.IsContiguous, $"matmul order '{o}'");
                Assert.IsTrue(np.einsum("ij,jk->ik", new[] { a, b }, order: o).Shape.IsContiguous, $"einsum order '{o}'");
            }

            // Any other letter is the ValueError quoting it as given: order='z' -> (got 'z')
            Action matmulZ = () => np.matmul(a, b, order: 'z');
            matmulZ.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'z')");
            Action einsumZ = () => np.einsum("ij,jk->ik", new[] { a, b }, order: 'z');
            einsumZ.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'z')");

            // casting=None is a TypeError (NumSharp raised the name ValueError "(got '')"); a casting name is exact-case
            Action matmulNone = () => np.matmul(a, b, casting: null);
            matmulNone.Should().Throw<TypeError>().WithMessage("casting must be str, not NoneType");
            Action einsumNone = () => np.einsum("ij,jk->ik", new[] { a, b }, casting: null);
            einsumNone.Should().Throw<TypeError>().WithMessage("casting must be str, not NoneType");
            Action matmulUpper = () => np.matmul(a, b, casting: "SAFE");
            matmulUpper.Should().Throw<ValueError>().WithMessage(
                "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'SAFE')");
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

        [TestMethod]
        public void Matmul_CoreChecks_AreNumPysGufuncChecks()
        {
            // np.matmul(np.ones((3,4)), np.ones((3,4))) -> ValueError (NumSharp: IncorrectShapeException "matmul: Input
            // operand core dimension mismatch ...")
            Action core = () => np.matmul(np.ones(new Shape(3L, 4L)), np.ones(new Shape(3L, 4L)));
            core.Should().Throw<ValueError>().WithMessage(
                "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) " +
                "(size 3 is different from 4)");
            // np.matmul(np.array(2.0), np.ones(3)) -> ValueError (NumSharp: InvalidOperationException)
            Action zeroD = () => np.matmul(NDArray.Scalar(2.0), np.ones(new Shape(3L)));
            zeroD.Should().Throw<ValueError>().WithMessage(
                "matmul: Input operand 0 does not have enough dimensions (has 0, gufunc core with signature " +
                "(n?,k),(k,m?)->(n?,m?) requires 1)");
            // np.matmul(np.ones((3,4)), np.ones((4,5)), out=np.zeros((3,6))) -> the output's core mismatch
            Action outCore = () => np.matmul(np.ones(new Shape(3L, 4L)), np.ones(new Shape(4L, 5L)), np.zeros(new Shape(3L, 6L)));
            outCore.Should().Throw<ValueError>().WithMessage(
                "matmul: Output operand 0 has a mismatch in its core dimension 1, with gufunc signature " +
                "(n?,k),(k,m?)->(n?,m?) (size 6 is different from 5)");
            // np.matmul(np.ones((3,4)), np.ones((4,5)), out=np.zeros(())) -> a 0-d out drops n? and m? from every operand,
            // so input 1's k is read from its last axis: size 5 against 4
            Action outZeroD = () => np.matmul(np.ones(new Shape(3L, 4L)), np.ones(new Shape(4L, 5L)), np.zeros(Shape.Scalar));
            outZeroD.Should().Throw<ValueError>().WithMessage(
                "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) " +
                "(size 5 is different from 4)");
        }

        [TestMethod]
        public void MatvecVecmat_CoreChecks_AreNumPysGufuncChecks()
        {
            var a34 = np.arange(12.0).reshape(3, 4);
            // np.matvec(a34, np.ones(4), out=np.zeros(())) -> the output's rank
            Action outRank = () => np.matvec(a34, np.ones(new Shape(4L)), np.zeros(Shape.Scalar));
            outRank.Should().Throw<ValueError>().WithMessage(
                "matvec: Output operand 0 does not have enough dimensions (has 0, gufunc core with signature " +
                "(m,n),(n)->(m) requires 1)");
            // np.matvec(a34, np.ones(4), out=np.zeros(2)) -> the output's core size (NumSharp: np.copyto's broadcast error)
            Action outCore = () => np.matvec(a34, np.ones(new Shape(4L)), np.zeros(new Shape(2L)));
            outCore.Should().Throw<ValueError>().WithMessage(
                "matvec: Output operand 0 has a mismatch in its core dimension 0, with gufunc signature (m,n),(n)->(m) " +
                "(size 2 is different from 3)");
            // np.matvec(a34, np.ones(4), axis=0, keepdims=True) -> keepdims is refused before axis
            Action both = () => np.matvec(a34, np.ones(new Shape(4L)), axis: 0, keepdims: true);
            both.Should().Throw<TypeError>().WithMessage("matvec does not support keepdims: *");
            // np.matvec(a34, np.ones(4), axes=[(0,0),(0,),(0,)]) and np.vecmat(np.ones(3), a34, axes=[(0,),(1,1),(0,)])
            Action matvecRepeat = () => np.matvec(a34, np.ones(new Shape(4L)),
                axes: new[] { new[] { 0, 0 }, new[] { 0 }, new[] { 0 } });
            matvecRepeat.Should().Throw<ValueError>().WithMessage("axes item 0 has value 0 repeated");
            Action vecmatRepeat = () => np.vecmat(np.ones(new Shape(3L)), a34,
                axes: new[] { new[] { 0 }, new[] { 1, 1 }, new[] { 0 } });
            vecmatRepeat.Should().Throw<ValueError>().WithMessage("axes item 1 has value 1 repeated");
        }

        [TestMethod]
        public void LoopAxes_BroadcastAsNumPysIterator()
        {
            // np.matvec(np.ones((2,3,4)), np.ones((3,4))) -> the iterator's remapped-shapes text (NumSharp: "matmul: stacked
            // (batch) dimensions are not broadcastable: 2 vs 3.")
            Action matvec = () => np.matvec(np.ones(new Shape(2L, 3L, 4L)), np.ones(new Shape(3L, 4L)));
            matvec.Should().Throw<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,3,4)->(2,newaxis) " +
                "(3,4)->(3,newaxis)  and requested shape (3)");
            // np.vecdot(np.ones((2,3,4)), np.ones((3,2,4)))
            Action vecdot = () => np.vecdot(np.ones(new Shape(2L, 3L, 4L)), np.ones(new Shape(3L, 2L, 4L)));
            vecdot.Should().Throw<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,3,4)->(2,3) " +
                "(3,2,4)->(3,2)  and requested shape ()");
            // np.matmul(np.ones(4), np.ones((2,4,5)), out=np.zeros((3,5))): the out takes part (a 1-D x1 maps nothing)
            Action withOut = () => np.matmul(np.ones(new Shape(4L)), np.ones(new Shape(2L, 4L, 5L)), np.zeros(new Shape(3L, 5L)));
            withOut.Should().Throw<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (4,)->() " +
                "(2,4,5)->(2,newaxis) (3,5)->(3,5)  and requested shape (5)");
        }

        [TestMethod]
        public void Out_NeitherBroadcastsNorReduces()
        {
            var a = np.ones(new Shape(2L, 3L, 4L));
            var b = np.ones(new Shape(2L, 4L, 5L));
            // np.matmul(a, b, out=np.zeros((1,3,5))) -> NumPy lists the remapped shape from the last iterator axis
            Action one = () => np.matmul(a, b, np.zeros(new Shape(1L, 3L, 5L)));
            one.Should().Throw<ValueError>().WithMessage(
                "non-broadcastable output operand with shape (1,3,5) [remapped to (5,3,1)] doesn't match the broadcast " +
                "shape (2,3,5)");
            // np.matmul(a, b, out=np.zeros((3,5))) -> the out lacks the loop axis
            Action missing = () => np.matmul(a, b, np.zeros(new Shape(3L, 5L)));
            missing.Should().Throw<ValueError>().WithMessage(
                "output operand requires a reduction along dimension -1, but the reduction is not enabled. The dimension " +
                "size of 1 does not match the expected output shape.");
            // np.vecdot(np.ones((2,3)), np.ones((2,3)), out=np.zeros((1,2))) -> the out's extra leading axis takes copies
            var bigger = np.zeros(new Shape(1L, 2L));
            np.vecdot(np.ones(new Shape(2L, 3L)), np.ones(new Shape(2L, 3L)), bigger);
            bigger.Should().BeOfValues(3.0, 3.0).And.BeShaped(1, 2);
        }
    }
}
