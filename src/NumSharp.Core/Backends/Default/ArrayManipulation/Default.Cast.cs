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

            //incase its a scalar
            if (nd.Shape.IsScalar)
            {
                var ret = NDArray.Scalar(nd.GetAtIndex(0), dtype);
                ret.TensorEngine = engine;
                if (copy)
                    return ret;

                nd.Storage = ret.Storage;
                nd.TensorEngine = engine;
                return nd;
            }

            //incase its a (1,) shaped
            if (nd.Shape.size == 1 && nd.Shape.NDim == 1)
            {
                var ret = new NDArray(ArraySlice.Scalar(nd.GetAtIndex(0), dtype), Shape.Vector(1)) { TensorEngine = engine };
                if (copy)
                    return ret;

                nd.Storage = ret.Storage;
                nd.TensorEngine = engine;
                return nd;
            }

            //regular clone
            if (nd.GetTypeCode == dtype)
            {
                if (!copy)
                    return nd;
                //An F-contiguous (non-C) source must clone KEEPORDER (NumPy astype order='K').
                //The legacy Clone() produces a C-order buffer, so astype then runs a SECOND
                //copy('F') — two cache-hostile transposes (the bool|F 0.18x cliff). Route just this
                //case through CastCrossType: it allocates an F-order dst and NpyIter.Copy collapses
                //to TryCopySameType's identical-layout flat cpblk — one pass, no reorder. C-contig /
                //1-D / strided sources keep the lean direct Clone() (no CreateCopyState overhead).
                if (nd.Shape.NDim > 1 && nd.Shape.IsFContiguous && !nd.Shape.IsContiguous)
                    return CastCrossType(nd, dtype, engine);
                return clone();
            }
            else
            {
                //casting needed — SIMD copy-with-cast via NpyIter. The output layout mirrors
                //the source's contiguity (NumPy astype order='K'/KEEPORDER, methods.c:769): an
                //F-contiguous or transposed source casts in a single flat pass instead of the
                //cache-hostile reorder the legacy scalar CastTo loop incurred. NpyIter.Copy
                //already dispatches the same-type SIMD copy, the contiguous IL cast kernel, and
                //the strided/broadcast cast kernel, so every layout takes its best path.
                var result = CastCrossType(nd, dtype, engine);
                if (copy)
                    return result;

                nd.Storage = result.Storage;
                nd.TensorEngine = engine;
                return nd;
            }

            NDArray clone() => nd.Clone();
        }

        /// <summary>
        ///     Allocates a fresh array of <paramref name="dtype"/> whose memory order mirrors
        ///     the source's contiguity (KEEPORDER) and fills it from <paramref name="nd"/> via
        ///     <see cref="Iteration.NpyIter.Copy(NDArray, NDArray)"/>, which performs a
        ///     stride/broadcast-aware SIMD copy-with-cast.
        /// </summary>
        /// <remarks>
        ///     Mirroring contiguity is what lets a transposed (F-contiguous) source cast as a
        ///     flat both-F copy rather than a strided reorder. NumPy does the same: array_astype
        ///     defaults to NPY_KEEPORDER and allocates via PyArray_NewLikeArray
        ///     (numpy/_core/src/multiarray/methods.c). Strided / broadcast / negative-stride
        ///     sources are neither C- nor F-contiguous, so they land in a C-order output and take
        ///     NpyIter's stride-sorted cast kernel.
        /// </remarks>
        private static NDArray CastCrossType(NDArray nd, NPTypeCode dtype, TensorEngine engine)
        {
            // float->int and signed-narrow->UInt64 have no NumPy-faithful SIMD cast kernel yet
            // (DirectILKernelGenerator.DivergesFromNumpyCast declines them — the hardware
            // truncate/saturate emission diverges from NumPy's wrapping semantics). Routing them
            // through NpyIter.Copy would fall to its scalar strided cast, which is SLOWER than the
            // legacy contiguous Converts loop. Keep those families on the legacy path (correct,
            // and no slower than before) until the wrapping SIMD kernel lands.
            if (DirectILKernelGenerator.DivergesFromNumpyCast(nd.GetTypeCode, dtype))
            {
                var legacySrc = nd.Shape.IsSliced ? nd.Clone() : nd;
                return new NDArray(new UnmanagedStorage(ArraySlice.FromMemoryBlock(legacySrc.Array.CastTo(dtype), false), legacySrc.Shape)) { TensorEngine = engine };
            }

            // All other pairs: SIMD copy-with-cast into a KEEPORDER output (F-contiguous source
            // mirrors to F output, so a transpose casts as a flat both-F copy, not a reorder).
            var srcShape = nd.Shape;
            char order = (srcShape.IsFContiguous && !srcShape.IsContiguous) ? 'F' : 'C';
            var outShape = new Shape((long[])srcShape.dimensions.Clone(), order);
            var result = new NDArray(dtype, outShape, false) { TensorEngine = engine };
            NpyIter.Copy(result, nd);
            return result;
        }
    }
}
