using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Cast(NDArray nd, Type dtype, bool copy)
        {
            return null;
            //if (dtype == null)
            //{
            //    throw new ArgumentNullException(nameof(dtype));
            //}

            //NDArray clone()
            //{
            //    var copied = new NDArray(nd.dtype, nd.TensorEngine);
            //    copied.Storage.Allocate(ArrayConvert.To(nd.Array, dtype), nd.shape);

            //    return copied;
            //}

            //if (nd.dtype == dtype)
            //{
            //    //casting not needed
            //    return copy ? clone() : nd;
            //}
            //else
            //{
            //    //casting needed
            //    if (copy)
            //    {
            //        return clone();
            //    }

            //    //just re-set the data, conversion is handled inside.
            //    nd.Storage.ReplaceData(nd.Storage.GetData(), dtype);
            //    return nd;
            //}
        }

        public NDArray Cast(NDArray nd, NPTypeCode dtype, bool copy)
        {
            return null;
            //if (dtype == NPTypeCode.Empty)
            //{
            //    throw new ArgumentNullException(nameof(dtype));
            //}

            //NDArray clone()
            //{
            //    var copied = new NDArray(nd.dtype, nd.TensorEngine);
            //    copied.Storage.Allocate(ArrayConvert.To(nd.Array, dtype), nd.shape);

            //    return copied;
            //}

            //if (nd.GetTypeCode == dtype)
            //{
            //    //casting not needed
            //    return copy ? clone() : nd;
            //}
            //else
            //{
            //    //casting needed
            //    if (copy)
            //    {
            //        return clone();
            //    }

            //    //just re-set the data, conversion is handled inside.
            //    nd.Storage.ReplaceData(nd.Storage.GetData(), dtype);
            //    return nd;
            //}
        }
    }
}
