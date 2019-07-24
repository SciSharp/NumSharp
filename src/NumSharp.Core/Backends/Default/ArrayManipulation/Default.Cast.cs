using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cast(NDArray nd, Type dtype, bool copy) => Cast(nd, dtype.GetTypeCode(), copy);

        public override NDArray Cast(NDArray nd, NPTypeCode dtype, bool copy)
        {
            if (dtype == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(dtype));

            NDArray clone() => new NDArray(nd.Storage.Clone());
            if (nd.Shape.IsEmpty)
            {
                if (copy)
                    return new NDArray(dtype);

                nd.Storage = new UnmanagedStorage(dtype);
                return nd;
            }

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

                    return new NDArray(new UnmanagedStorage(ArraySlice.FromMemoryBlock(nd.Array.Cast(dtype), false), nd.Shape));
                }
                else
                {
                    var storage = nd.Shape.IsSliced ? nd.Storage.Clone() : nd.Storage;
                    nd.Storage = new UnmanagedStorage(ArraySlice.FromMemoryBlock(storage.InternalArray.Cast(dtype), false), storage.Shape);
                    return nd;
                }
            }
        }
    }
}
