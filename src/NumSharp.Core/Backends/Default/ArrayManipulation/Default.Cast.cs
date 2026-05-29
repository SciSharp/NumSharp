using System;
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
                //casting not needed
                return copy ? clone() : nd;
            }
            else
            {
                //casting needed
                if (copy)
                {
                    if (nd.Shape.IsSliced)
                        nd = clone();

                    return new NDArray(new UnmanagedStorage(ArraySlice.FromMemoryBlock(nd.Array.CastTo(dtype), false), nd.Shape)) { TensorEngine = engine };
                }
                else
                {
                    var storage = nd.Shape.IsSliced ? nd.Storage.Clone() : nd.Storage;
                    nd.Storage = new UnmanagedStorage(ArraySlice.FromMemoryBlock(storage.InternalArray.CastTo(dtype), false), storage.Shape) { Engine = engine };
                    nd.TensorEngine = engine;
                    return nd;
                }
            }

            NDArray clone() => nd.Clone();
        }
    }
}
