using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cast(NDArray nd, Type dtype, bool copy) => Cast(nd, dtype.GetTypeCode(), copy);

        public override NDArray Cast(NDArray nd, NPTypeCode dtype, bool copy)
        {
            if (dtype == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(dtype));

            var engine = nd.TensorEngine;

            //incase its an empty array (the uninitialized-shape sentinel)
            if (nd.Shape.IsEmpty)
            {
                if (copy)
                    return new NDArray(dtype) { TensorEngine = engine };

                nd.Storage = new UnmanagedStorage(dtype) { Engine = engine };
                nd.TensorEngine = engine;
                return nd;
            }

            //incase it has a zero-size dimension (e.g. (1,0), (2,0,2)) — a real shape
            //carrying no elements. There is nothing to cast; just retype while preserving
            //the shape. (Shape.IsEmpty above only catches the uninitialized sentinel, so
            //this guard is required or the regular CastTo path below faults on length 0.)
            if (nd.size == 0)
            {
                var retyped = new NDArray(dtype, nd.Shape) { TensorEngine = engine };
                if (copy)
                    return retyped;

                nd.Storage = retyped.Storage;
                nd.TensorEngine = engine;
                return nd;
            }

            // same-dtype with copy=false is a no-op — NumPy's astype returns self when no cast and
            // no copy is requested (KEEPORDER leaves the existing layout untouched).
            if (nd.GetTypeCode == dtype && !copy)
                return nd;

            // Unified allocate-and-fill copy/cast core (KEEPORDER = NumPy astype order='K'), integrated
            // with NpyIter via NpyIter.CopyAs: same-dtype takes the SIMD copy (a single flat pass even
            // for F-contiguous / transposed sources), cross-dtype takes the IL cast kernels, and every
            // layout (contiguous / strided / broadcast / scalar) resolves to its best path. Replaces the
            // former scalar / (1,) / same-type-Clone / F-contig-special / CastCrossType branch maze.
            var result = NpyIter.CopyAs(dtype, nd, 'K', engine);
            if (copy)
                return result;

            nd.Storage = result.Storage;
            nd.TensorEngine = engine;
            return nd;
        }
    }
}
