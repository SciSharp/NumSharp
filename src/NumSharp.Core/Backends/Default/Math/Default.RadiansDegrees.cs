// NumPy's np.radians / np.degrees ufuncs as DefaultEngine members: the same loops as deg2rad / rad2deg, refusing an
// operand under their own names.

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray Radians(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // radians is its own NumPy ufunc (not an alias object of deg2rad), so a refusal names it.
            Strings.NDStringUfuncs.ThrowIfUnary("radians", nd, UfuncLoopCode(dtype, "radians", nd, null, @out, where), @out, where);
            return Deg2Rad(nd, dtype, @out, where);
        }

        /// <inheritdoc />
        public override NDArray Degrees(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // degrees is its own NumPy ufunc (not an alias object of rad2deg), so a refusal names it.
            Strings.NDStringUfuncs.ThrowIfUnary("degrees", nd, UfuncLoopCode(dtype, "degrees", nd, null, @out, where), @out, where);
            return Rad2Deg(nd, dtype, @out, where);
        }
    }
}
