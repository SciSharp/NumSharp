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
    }
}
