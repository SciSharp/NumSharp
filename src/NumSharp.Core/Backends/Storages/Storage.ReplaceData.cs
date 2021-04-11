using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
        public void ReplaceData(IArraySlice values, Type dtype)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(NDArray nd)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            //first try to convert to dtype only then we apply changes.
            _shape = nd.shape;
            _typecode = nd.GetTypeCode;
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{_dtype.Name} as a dtype is not supported.");

            //todo! what if nd is sliced

            SetInternalArray(nd.Shape.IsSliced ? nd.Storage.CloneData() : nd.Array);
        }

        public void ReplaceData(Array values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            SetInternalArray(_ChangeTypeOfArray(values, _dtype));

            if (_shape.IsEmpty)
                _shape = new Shape(values.Length);
        }

        public void ReplaceData(Array values, Type dtype)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(Array values, NPTypeCode typeCode)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(IArraySlice values)
        {
            throw new NotImplementedException();
        }
    }
}
