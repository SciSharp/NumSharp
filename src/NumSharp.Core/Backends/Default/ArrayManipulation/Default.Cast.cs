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

            //incase its an empty array
            if (nd.Shape.IsEmpty)
            {
                if (copy)
                    return new NDArray(dtype);

                nd.Storage = new UnmanagedStorage(dtype);
                return nd;
            }

            //incase its a scalar
            if (nd.Shape.IsScalar)
            {
                var ret = NDArray.Scalar(nd.GetAtIndex(0), dtype);
                if (copy)
                    return ret;

                nd.Storage = ret.Storage;
                return nd;
            }

            //incase its a (1,) shaped
            if (nd.Shape.size == 1 && nd.Shape.NDim == 1)
            {
                var ret = new NDArray(ArraySlice.Scalar(nd.GetAtIndex(0), dtype), Shape.Vector(1));
                if (copy)
                    return ret;

                nd.Storage = ret.Storage;
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

                    return new NDArray(new UnmanagedStorage(ArraySlice.FromMemoryBlock(nd.Array.CastTo(dtype), false), nd.Shape));
                }
                else
                {
                    var storage = nd.Shape.IsSliced ? nd.Storage.Clone() : nd.Storage;
                    nd.Storage = new UnmanagedStorage(ArraySlice.FromMemoryBlock(storage.InternalArray.CastTo(dtype), false), storage.Shape);
                    return nd;
                }
            }

            NDArray clone() => nd.Clone();
        }
    }
}
