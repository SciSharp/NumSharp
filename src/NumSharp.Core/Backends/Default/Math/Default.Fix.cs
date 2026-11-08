// NumPy's np.fix (numpy/lib/_ufunclike_impl.py) as DefaultEngine's member: NumPy 2.4.2's body is the one statement
// `return nx.trunc(x, out=out)`. The facade is Math/np.fix.cs.

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        /// <remarks>np.trunc's member on x's engine (np.trunc's tail: no where=, no dtype=), so the result, the out= rules and
        ///     the error texts that name 'trunc' are trunc's own.</remarks>
        public override NDArray Fix(NDArray x, NDArray @out = null) => x.TensorEngine.Truncate(x, null, @out, null);
    }
}
