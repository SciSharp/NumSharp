using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage
    {
        public void SetData(NDArray value, params int[] indices)
        {
            if (ReferenceEquals(value, null))
                throw new ArgumentNullException(nameof(value));

            var valueshape = value.Shape;
            bool valueIsScalary = valueshape.IsScalar || valueshape.NDim == 1 && valueshape.size == 1;

            //incase lhs or rhs are broadcasted or sliced (noncontagious)
            if (_shape.IsBroadcasted || _shape.IsSliced || valueshape.IsBroadcasted || valueshape.IsSliced)
            {
                MultiIterator.Assign(GetData(indices), value.Storage); //we use lhs stop because rhs is scalar which will fill all values of lhs
                return;
            }

            //by now value and this are contagious
            //////////////////////////////////////

            //incase it is 1 value assigned to all
            if (valueIsScalary && indices.Length != _shape.NDim)
            {
                GetData(indices).InternalArray.Fill(Converts.ChangeType(value.GetAtIndex(0), _typecode));
                //MultiIterator.Assign(GetData(indices), value.Storage); //we use lhs stop because rhs is scalar which will fill all values of lhs
                return;
            }

            //incase its a scalar to scalar assignment
            if (indices.Length == _shape.NDim)
            {
                if (!(valueIsScalary))
                    throw new IncorrectShapeException($"Can't SetData to a from a shape of {valueshape} to the target indices, these shapes can't be broadcasted together.");

                SetValue((ValueType)Converts.ChangeType(value.GetAtIndex(0), _typecode), (indices));
                return;
            }

            //regular case
            var (subShape, offset) = _shape.GetSubshape(indices);

            //if (!value.Storage.Shape.IsScalar && np.squeeze(subShape) != np.squeeze(value.Storage.Shape))
            //    throw new IncorrectShapeException($"Can't SetData to a from a shape of {value.Shape} to target shape {subShape}, the shape the coordinates point to mismatch the size of rhs (value)");

            if (subShape.size % valueshape.size != 0)
                throw new IncorrectShapeException($"Can't SetData to a from a shape of {valueshape} to target shape {subShape}, these shapes can't be broadcasted together.");

            //by now this ndarray is not broadcasted nor sliced
            unsafe
            {
                //ReSharper disable once RedundantCast
                //this must be a void* so it'll go through a typed switch.
                value.Storage.CastIfNecessary(_typecode).CopyTo((void*)((byte*)_address + _internalArray.ItemLength * offset));
            }
        }

        public void SetData(object value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetData(IArraySlice value, params int[] indices)
        {
            throw new NotImplementedException();
        }
    }
}
