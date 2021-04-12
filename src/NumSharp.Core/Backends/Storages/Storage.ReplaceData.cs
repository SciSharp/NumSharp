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
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (dtype == null)
                throw new ArgumentNullException(nameof(dtype));

            var changedArray = _ChangeTypeOfArray(values, dtype);
            //first try to convert to dtype only then we apply changes.
            _typecode = _dtype.GetTypeCode();
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{dtype.Name} as a dtype is not supported.");
            SetInternalArray(changedArray);
        }

        public void ReplaceData(Array values, NPTypeCode typeCode)
            => ReplaceData(values, typeCode.AsType());

        public void ReplaceData(IArraySlice values)
        {
            SetInternalArray(values);

            if (_shape.IsEmpty)
                _shape = new Shape((int)values.Count); //TODO! when long index, remove cast int
        }
    }
}
