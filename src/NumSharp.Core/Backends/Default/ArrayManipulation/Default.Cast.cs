using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Backends
{
    public abstract partial class DefaultEngine
    {
        public NDArray Cast(NDArray nd, Type dtype, bool copy)
        {
            if (dtype == null)
            {
                throw new ArgumentNullException(nameof(dtype));
            }

            NDArray clone()
            {
                var copied = new NDArray(dtype);
                var shapePuffer = new Shape(nd.shape);

                //todo is there a need to allocate first? setdata should be sufficient.
                copied.Storage.Allocate(shapePuffer);

                copied.Storage.SetData(nd.Storage.CloneData());

                return copied;
            }

            if (nd.dtype == dtype)
            {
                //casting not needed
                return copy ? clone() : nd;
            }
            else
            {
                //casting needed
                if (copy)
                {
                    return clone();
                }

                //just re-set the data, conversion is handled inside.
                nd.Storage.SetData(nd.Storage.GetData(), dtype);
                return nd;
            }
        }
    }
}
