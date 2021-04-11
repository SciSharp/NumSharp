using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
        public ArraySlice<T> GetData<T>() where T : unmanaged
            => (ArraySlice<T>)_internalArray;

        public IArraySlice GetData()
        {
            return _internalArray;
        }

        public unsafe IStorage GetData(int* dims, int ndims)
        {
            throw new NotImplementedException();
        }

        public IStorage GetData(params int[] indices)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            indices = Shape.InferNegativeCoordinates(Shape.dimensions, indices);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                return Storage.CreateBroadcastedUnsafe(InternalArray.Slice(offset, shape.BroadcastInfo.OriginalShape.size), shape);
            }
            else if (this_shape.IsSliced)
            {
                // in this case we can not get a slice of contiguous memory, so we slice
                return GetView(indices.Select(Slice.Index).ToArray());
            }
            else
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                return Storage.Allocate(_internalArray.Slice(offset, shape.Size), shape);
            }
        }
    }
}
