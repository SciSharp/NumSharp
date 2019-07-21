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
                    return new NDArray(new UnmanagedStorage(ArraySlice.FromMemoryBlock(nd.Array.Cast(dtype), false), nd.Shape));
                }
                else
                {
                    nd.Storage = new UnmanagedStorage(ArraySlice.FromMemoryBlock(nd.Array.Cast(dtype), false), nd.Shape);
                    return nd;
                }
            }
        }
    }
}
