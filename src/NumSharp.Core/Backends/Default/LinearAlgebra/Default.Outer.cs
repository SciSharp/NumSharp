// NumPy's np.outer (numpy/_core/numeric.py) as DefaultEngine's member: its one statement, one multiply ufunc call over the
// two flattened inputs. The facade is LinearAlgebra/np.outer.cs; numpy.linalg's Array-API outer
// (DefaultLinalgEngine.ArrayApi.cs) calls it after its one-dimensional check.

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped /* the two flattened views are reclaimed; the product (or the caller's out) is yielded */]
        public override NDArray Outer(NDArray a, NDArray b, NDArray @out = null)
        {
            // multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out): np.expand_dims' body (the array object's) gives
            // the column and the row, and np.multiply's member on the column's engine broadcasts them into (a.size, b.size)
            // or into out, with the ufunc's out= rules (same_kind cast, never stretched, read-only refused).
            NDArray column = NDArray.expand_dims(a.ravel(), -1);
            NDArray row = NDArray.expand_dims(b.ravel(), 0);
            return column.TensorEngine.Multiply(column, row, null, @out, null);
        }
    }
}
