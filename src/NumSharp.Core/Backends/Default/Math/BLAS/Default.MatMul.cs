using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html</remarks>
        public override NDArray Matmul(NDArray lhs, NDArray rhs)
        {
            if (lhs.Shape.IsScalar || rhs.Shape.IsScalar)
                throw new InvalidOperationException("Matmul can't handle scalar multiplication, use `*` or `np.dot(..)` instead");

            //If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
            if (lhs.ndim == 1 && rhs.ndim == 2)
                throw new NotSupportedException("Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)");

            if (rhs.ndim == 1)
                rhs = np.expand_dims(rhs, 1);

            if (lhs.ndim == 2 || rhs.ndim == 2)
                return MultiplyMatrix(lhs, rhs);

            NDArray l = lhs;
            NDArray r = rhs;
            (l, r) = np.broadcast_arrays(l, r);
            var retShape = l.Shape.Clean();
            Console.WriteLine(retShape);
            Debug.Assert(l.shape[0] == r.shape[0]);
            var len = l.size;
            var ret = new NDArray(np._FindCommonArrayType(l.typecode, r.typecode), retShape);
            var iterShape = new Shape(retShape.dimensions.Take(retShape.dimensions.Length - 2).ToArray());
            var incr = new NDCoordinatesIncrementor(ref iterShape);
            var index = incr.Index;

            //TODO! we need to create IEnumeable<int> on NDCoordinatesIncrementor so we can do something like this:
            //TODO! Parallel.ForEach(incr, i => MultiplyMatrix(l[index], r[index], ret[index]));
            for (int i = 0; i < len; i++, incr.Next())
            {
                MultiplyMatrix(l[index], r[index], ret[index]);
            }

            return ret;
        }
    }
}
