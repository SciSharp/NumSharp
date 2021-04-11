using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public abstract partial class Storage : IStorage
    {
        protected Type _dtype => _typecode.AsType();
        public Type DType => _dtype;

        protected Shape _shape;
        public Shape Shape { get => _shape; set => _shape = value; }
        public ref Shape ShapeReference => ref _shape;

        protected unsafe void* _address;
        public unsafe virtual void* Address
        {
            get => _address;
            set => _address = value;
        }

        protected NPTypeCode _typecode;
        public NPTypeCode TypeCode => _typecode;

        public TensorEngine Engine => BackendFactory.GetEngine(DType);

        public int DTypeSize
            => _typecode switch
            {
                NPTypeCode.String => IntPtr.Size,
                NPTypeCode.Boolean => sizeof(bool),
                _ => Marshal.SizeOf(_typecode.AsType())
            };

        protected IArraySlice _internalArray;
        public IArraySlice InternalArray
        {
            get => _internalArray;
            set => _internalArray = value;
        }

        public int Count { get; set; }

        public virtual void Allocate(Shape shape, Type dtype = null)
            => throw new NotImplementedException("");

        public static IStorage Allocate(object x, NPTypeCode? typeCode = null)
        {
            if(!typeCode.HasValue || x.GetType() == typeCode.Value.AsType())
            {
                return x switch
                {
                    bool bool_x => new StorageOfBoolean(bool_x),
                    byte byte_x => new StorageOfByte(byte_x),
                    int int_x => new StorageOfInt32(int_x),
                    long int_x => new StorageOfInt64(int_x),
                    float float_x => new StorageOfSingle(float_x),
                    double double_x => new StorageOfDouble(double_x),
                    _ => throw new NotImplementedException("")
                };
            }
            else
            {
                return typeCode switch
                {
                    NPTypeCode.Boolean => new StorageOfBoolean(Convert.ToBoolean(x)),
                    NPTypeCode.Byte => new StorageOfByte(Convert.ToByte(x)),
                    NPTypeCode.Int32 => new StorageOfInt32(Convert.ToInt32(x)),
                    NPTypeCode.Int64 => new StorageOfInt64(Convert.ToInt64(x)),
                    NPTypeCode.Single => new StorageOfSingle(Convert.ToSingle(x)),
                    NPTypeCode.Double => new StorageOfDouble(Convert.ToDouble(x)),
                    _ => throw new NotImplementedException("")
                };
            }
        }


        public static IStorage Allocate<T>(T x) where T : unmanaged
            => x switch
            {
                bool bool_x => new StorageOfBoolean(bool_x),
                byte byte_x => new StorageOfByte(byte_x),
                int int_x => new StorageOfInt32(int_x),
                long int_x => new StorageOfInt64(int_x),
                float float_x => new StorageOfSingle(float_x),
                double double_x => new StorageOfDouble(double_x),
                _ => throw new NotImplementedException("")
            };

        public static IStorage Allocate<T>(T[] x, Shape? shape = null) where T : unmanaged
            => x switch
            {
                bool[] bool_x => new StorageOfBoolean(bool_x, shape),
                byte[] byte_x => new StorageOfByte(byte_x),
                int[] int_x => new StorageOfInt32(int_x, shape),
                long[] int_x => new StorageOfInt64(int_x, shape),
                float[] float_x => new StorageOfSingle(float_x, shape),
                double[] double_x => new StorageOfDouble(double_x, shape),
                _ => throw new NotImplementedException("")
            };

        public static IStorage Allocate(IArraySlice x, Shape shape)
        {
            Storage storage = (Storage)BackendFactory.GetStorage(x.TypeCode);
            storage.Allocate(x, shape);
            return storage;
        }

        public static IStorage Allocate(ValueType x)
            => x switch
            {
                bool bool_x => new StorageOfBoolean(bool_x),
                byte byte_x => new StorageOfByte(byte_x),
                int int_x => new StorageOfInt32(int_x),
                long long_x => new StorageOfInt64(long_x),
                float float_x => new StorageOfSingle(float_x),
                double double_x => new StorageOfDouble(double_x),
                _ => throw new NotImplementedException("")
            };

        public unsafe void Allocate(IArraySlice values, Shape shape, bool copy = false)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Count != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            _shape = shape;
            _address = values.Address;
            _internalArray = values;
            Count = Convert.ToInt32(values.Count);
        }

        public static unsafe IStorage Allocate<T>(T[,] x) where T : unmanaged
        {
            var array = x.Cast<T>().ToArray();
            return Allocate(array, new Shape(Enumerable.Range(0, 2).Select(i => x.GetLength(i)).ToArray()));
        }

        public static IStorage CreateBroadcastedUnsafe(IArraySlice arraySlice, Shape shape)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));
            Storage storage = (Storage)BackendFactory.GetStorage(arraySlice.TypeCode);
            storage.Shape = shape;
            storage.SetInternalArray(arraySlice);
            return storage;
        }

        public static IStorage CreateBroadcastedUnsafe(IStorage storage, Shape shape)
            => CreateBroadcastedUnsafe(storage.InternalArray, shape);

        public unsafe ReadOnlySpan<T> Read<T>()
        {
            return new ReadOnlySpan<T>(Address, Shape.Size);
        }

        public void Reshape(Shape shape, bool copy = false)
        {
            Reshape(ref shape, copy);
        }

        public void Reshape(ref Shape shape, bool copy = false)
        {
            SetShapeUnsafe(Shape.Reshape(shape, copy));
        }

        public IStorage Alias()
        {
            return Alias(_shape);
        }

        public unsafe IStorage Alias(Shape shape)
        {
            var storage = BackendFactory.GetStorage(_typecode);
            storage.SetInternalArray(_internalArray);
            storage.Shape = shape;
            storage.Count = shape.size;
            return storage;
        }

        public IStorage Cast(Type type)
        {
            throw new NotImplementedException();
        }

        public ArraySlice<T> CloneData<T>() where T : unmanaged
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

        public void ExpandDimension(int axis)
        {
            _shape = _shape.ExpandDimension(axis);
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
            if (!_shape.IsSliced)
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
                return GetData(indices);
            }

//perform a regular slicing
_perform_slice:

// In case the slices selected are all ":"
// ReSharper disable once ConvertIfStatementToReturnStatement
            if (!_shape.IsRecursive && slices.All(s => Equals(Slice.All, s)))
                return Alias();

            //handle broadcasted shape
            if (_shape.IsBroadcasted)
                return Clone().Alias(_shape.Slice(slices));

            return Alias(_shape.Slice(slices));
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

        public T[] ToArray<T>() where T : unmanaged
            => Read<T>().ToArray();

        public virtual IStorage Clone()
        {
            return Storage.Allocate(_internalArray, _shape);
        }

        public void SetShapeUnsafe(Shape shape)
        {
            Shape = shape;
            Count = Shape.size;
        }

        public unsafe void SetInternalArray(IArraySlice array)
        {
            _internalArray = array;
            _address = _internalArray.Address;
            Count = Convert.ToInt32(array.Count);
        }

        /// <summary>
        ///     Replace internal storage array with given array.
        /// </summary>
        /// <param name="array">The array to set as internal storage</param>
        /// <exception cref="InvalidCastException">When type of <paramref name="array"/> does not match <see cref="DType"/> of this storage</exception>
        protected unsafe void SetInternalArray(Array array)
        {
            _internalArray = _typecode switch
            {
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                NPTypeCode.Boolean => ArraySlice.FromArray<bool>((bool[])array),
                NPTypeCode.Byte => ArraySlice.FromArray<byte>((byte[])array),
                NPTypeCode.Int32 => ArraySlice.FromArray<int>((int[])array),
                NPTypeCode.Int64 => ArraySlice.FromArray<long>((long[])array),
                NPTypeCode.Single => ArraySlice.FromArray<float>((float[])array),
                NPTypeCode.Double => ArraySlice.FromArray<double>((double[])array),
                _ => throw new NotSupportedException()
            };

            _address = _internalArray.Address;
            Count = Convert.ToInt32(array.Length);
        }

        /// <summary>
        ///     Changes the type of <paramref name="sourceArray"/> to <paramref name="to_dtype"/> if necessary.
        /// </summary>
        /// <param name="sourceArray">The array to change his type</param>
        /// <param name="to_dtype">The type to change to.</param>
        /// <remarks>If the return type is equal to source type, this method does not return a copy.</remarks>
        /// <returns>Returns <see cref="sourceArray"/> or new array with changed type to <see cref="to_dtype"/></returns>
        [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
        protected static Array _ChangeTypeOfArray(Array sourceArray, Type to_dtype)
        {
            if (to_dtype == sourceArray.GetType().GetElementType()) return sourceArray;
            return ArrayConvert.To(sourceArray, to_dtype);
        }
    }
}
