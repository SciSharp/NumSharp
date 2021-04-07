using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public abstract class Storage : IStorage
    {
        public NPTypeCode DType { get; set; }
        public Shape Shape { get; set; }
        protected unsafe void* address;
        public unsafe void* Address
        {
            get => address;
            set => address = value;
        }

        public virtual void Allocate(Shape shape)
            => throw new NotImplementedException("");
        public NPTypeCode TypeCode => DType;
        public virtual TensorEngine Engine => new DefaultEngine();

        Type IStorage.DType => throw new NotImplementedException();

        public int DTypeSize => throw new NotImplementedException();

        public IArraySlice InternalArray => throw new NotImplementedException();

        public int Count { get; set; }



        public static Storage Allocate<T>(T x) where T : unmanaged
        {
            if (x is int int_x)
            {
                return new StorageOfInt32(int_x);
            }
            return null;
        }

        public static Storage Allocate<T>(T[] x, Shape? shape = null) where T : unmanaged
        {
            if (x is int[] int_x)
            {
                return new StorageOfInt32(int_x, shape);
            }
            return null;
        }

        public unsafe ReadOnlySpan<T> Read<T>()
        {
            return new ReadOnlySpan<T>(Address, Shape.Size);
        }

        public void Reshape(Shape shape)
        {
            throw new NotImplementedException();
        }

        public virtual IStorage Alias(Shape shape)
        {
            throw new NotImplementedException("");
        }

        public virtual IStorage Alias()
        {
            throw new NotImplementedException();
        }

        public void Allocate(IArraySlice values, Shape shape)
        {
            throw new NotImplementedException();
        }

        public IStorage Cast(Type type)
        {
            throw new NotImplementedException();
        }

        public ArraySlice<T> CloneData<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public bool GetBoolean(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public byte GetByte(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public int GetInt32(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public int GetInt64(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public float GetSingle(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public double GetDouble(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public ValueType GetAtIndex(int index)
        {
            throw new NotImplementedException();
        }

        public T GetAtIndex<T>(int index)
        {
            throw new NotImplementedException();
        }

        public ValueType GetValue<T>(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public ArraySlice<T> GetData<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(Array values)
        {
            throw new NotImplementedException();
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

        public void SetAtIndex(object value, int index)
        {
            throw new NotImplementedException();
        }

        public void SetBoolean(bool value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetByte(byte value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetInt32(int value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetInt64(long value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetSingle(float value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetDouble(double value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetData(IArraySlice value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetData(NDArray value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetValue(ValueType value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetValue(object value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public IStorage Cast(NPTypeCode typeCode)
        {
            throw new NotImplementedException();
        }

        public IStorage CastIfNecessary(NPTypeCode typeCode)
        {
            throw new NotImplementedException();
        }

        public IArraySlice CloneData()
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(IArraySlice values, Type dtype)
        {
            throw new NotImplementedException();
        }

        public void ReplaceData(NDArray nd)
        {
            throw new NotImplementedException();
        }

        public void SetData(object value, params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetValue<T>(T value, params int[] indices) where T : unmanaged
        {
            throw new NotImplementedException();
        }

        T IStorage.GetValue<T>(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public ValueType GetValue(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public IArraySlice GetData()
        {
            throw new NotImplementedException();
        }

        public NDArray GetData(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void Reshape(Shape shape, bool copy = false)
        {
            SetShapeUnsafe(Shape.Reshape(shape, copy));
        }

        public void CopyTo<T>(T[] array) where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public void CopyTo<T>(IMemoryBlock slice)
        {
            throw new NotImplementedException();
        }

        public unsafe void CopyTo<T>(void* address)
        {
            throw new NotImplementedException();
        }

        public void ExpandDimension(int axis)
        {
            throw new NotImplementedException();
        }

        public IStorage GetView(params Slice[] slices)
        {
            if (slices == null)
                throw new ArgumentNullException(nameof(slices));
            // deal with ellipsis and newaxis if any before continuing into GetViewInternal
            int ellipsis_count = 0;
            int newaxis_count = 0;
            foreach (var slice in slices)
            {
                if (slice.IsEllipsis)
                    ellipsis_count++;
                if (slice.IsNewAxis)
                    newaxis_count++;
            }

            // deal with ellipsis
            if (ellipsis_count > 1)
                throw new ArgumentException("IndexError: an index can only have a single ellipsis ('...')");
            else if (ellipsis_count == 1)
                slices = ExpandEllipsis(slices).ToArray();

            // deal with newaxis
            if (newaxis_count > 0)
            {
                IStorage view = this;
                for (var axis = 0; axis < slices.Length; axis++)
                {
                    var slice = slices[axis];
                    if (slice.IsNewAxis)
                    {
                        slices[axis] = Slice.All;
                        view = view.Alias(view.Shape.ExpandDimension(axis));
                    }
                }

                throw new NotImplementedException("");
                // return view.GetViewInternal(slices);
            }

            // slicing without newaxis
            return GetViewInternal(slices);
        }

        IStorage GetViewInternal(params Slice[] slices)
        {
            // NOTE: GetViewInternal can not deal with Slice.Ellipsis or Slice.NewAxis! 
            //handle memory slice if possible
            if (!Shape.IsSliced)
            {
                var indices = new int[slices.Length];
                for (var i = 0; i < slices.Length; i++)
                {
                    var inputSlice = slices[i];
                    if (!inputSlice.IsIndex)
                    {
                        //incase it is a trailing :, e.g. [2,2, :] in a shape (3,3,5,5) -> (5,5)
                        if (i == slices.Length - 1 && inputSlice == Slice.All)
                        {
                            Array.Resize(ref indices, indices.Length - 1);
                            goto _getdata;
                        }

                        goto _perform_slice;
                    }

                    indices[i] = inputSlice.Start.Value;
                }

_getdata:
                throw new NotImplementedException("");
                // return GetData(indices);
            }

//perform a regular slicing
_perform_slice:

// In case the slices selected are all ":"
// ReSharper disable once ConvertIfStatementToReturnStatement
            if (!Shape.IsRecursive && slices.All(s => Equals(Slice.All, s)))
                return Alias();

            //handle broadcasted shape
            if (Shape.IsBroadcasted)
                return Clone().Alias(Shape.Slice(slices));

            return Alias(Shape.Slice(slices));
        }

        IEnumerable<Slice> ExpandEllipsis(Slice[] slices)
        {
            // count dimensions without counting ellipsis or newaxis
            var count = 0;
            foreach (var slice in slices)
            {
                if (slice.IsNewAxis || slice.IsEllipsis)
                    continue;
                count++;
            }

            // expand 
            foreach (var slice in slices)
            {
                if (slice.IsEllipsis)
                {
                    for (int i = 0; i < Shape.NDim - count; i++)
                        yield return Slice.All;
                    continue;
                }

                yield return slice;
            }
        }

        public IStorage GetView(string slicing_notation)
        {
            throw new NotImplementedException();
        }

        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        public IStorage Clone()
        {
            throw new NotImplementedException();
        }

        public unsafe IStorage GetData(int* dims, int ndims)
        {
            throw new NotImplementedException();
        }

        IStorage IStorage.GetData(params int[] indices)
        {
            throw new NotImplementedException();
        }

        public void SetShapeUnsafe(Shape shape)
        {
            Shape = shape;
            Count = Shape.size;
        }
    }
}
