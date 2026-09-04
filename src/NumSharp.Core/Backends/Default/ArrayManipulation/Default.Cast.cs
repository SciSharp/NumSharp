using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cast(NDArray nd, Type dtype, bool copy) => Cast(nd, dtype.GetTypeCode(), copy);

        // NPTypeCode is DefaultEngine's internal dtype currency — kept as a (non-abstract) helper
        // the whole engine calls directly; the TensorEngine abstraction exposes only the Type form.
        public NDArray Cast(NDArray nd, NPTypeCode dtype, bool copy)
        {
            if (dtype == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(dtype));

            var engine = nd.TensorEngine;

            // NumPy astype(copy=False) semantics (probed 2.4.2): copy=false only elides the
            // copy when no conversion is needed — `a.astype(a.dtype, copy=False) is a`. A dtype
            // change ALWAYS materializes a fresh array and NEVER touches the input. The former
            // behavior here swapped nd.Storage in-place on conversion, which leaked out of every
            // internal copy:false call site as operand corruption: np.allclose/np.isclose and
            // np.where silently flipped their arguments' dtype (float32 operands came back
            // float64). Do not reintroduce the storage swap.
            if (nd.GetTypeCode == dtype && !copy)
                return nd;

            //incase its an empty array (the uninitialized-shape sentinel)
            if (nd.Shape.IsEmpty)
                return new NDArray(dtype) { TensorEngine = engine };

            //incase it has a zero-size dimension (e.g. (1,0), (2,0,2)) — a real shape
            //carrying no elements. There is nothing to cast; just retype while preserving
            //the shape. (Shape.IsEmpty above only catches the uninitialized sentinel, so
            //this guard is required or the regular CastTo path below faults on length 0.)
            if (nd.size == 0)
                return new NDArray(dtype, nd.Shape) { TensorEngine = engine };

            // Unified allocate-and-fill copy/cast core (KEEPORDER = NumPy astype order='K'), integrated
            // with NDIter via NDIter.CopyAs: same-dtype takes the SIMD copy (a single flat pass even
            // for F-contiguous / transposed sources), cross-dtype takes the IL cast kernels, and every
            // layout (contiguous / strided / broadcast / scalar) resolves to its best path. Replaces the
            // former scalar / (1,) / same-type-Clone / F-contig-special / CastCrossType branch maze.
            return NDIter.CopyAs(dtype, nd, 'K', engine);
        }
    }
}
