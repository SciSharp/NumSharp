using System;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Getters

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="IStorage.DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="IStorage.DType"/> is not <see cref="object"/></exception>
        public unsafe ValueType GetValue(params int[] indices)
        {
            switch (TypeCode)
            {
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return *((#2*)Address + _shape.GetOffset(indices));
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean: return *((bool*)Address + _shape.GetOffset(indices));
	            case NPTypeCode.Byte: return *((byte*)Address + _shape.GetOffset(indices));
	            case NPTypeCode.Int32: return *((int*)Address + _shape.GetOffset(indices));
	            case NPTypeCode.Int64: return *((long*)Address + _shape.GetOffset(indices));
	            case NPTypeCode.Single: return *((float*)Address + _shape.GetOffset(indices));
	            case NPTypeCode.Double: return *((double*)Address + _shape.GetOffset(indices));
	            default:
		            throw new NotSupportedException();
#endif
            }
        }

        public unsafe ValueType GetAtIndex(int index)
        {
#if _REGEN1
            switch (TypeCode)
            {
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return *((#2*)Address + _shape.TransformOffset(index));
	            %
	            default:
		            throw new NotSupportedException();
            }
#else
            switch (TypeCode)
            {
	            case NPTypeCode.Boolean: return *((bool*)Address + _shape.TransformOffset(index));
	            case NPTypeCode.Byte: return *((byte*)Address + _shape.TransformOffset(index));
	            case NPTypeCode.Int32: return *((int*)Address + _shape.TransformOffset(index));
	            case NPTypeCode.Int64: return *((long*)Address + _shape.TransformOffset(index));
	            case NPTypeCode.Single: return *((float*)Address + _shape.TransformOffset(index));
	            case NPTypeCode.Double: return *((double*)Address + _shape.TransformOffset(index));
	            default:
		            throw new NotSupportedException();
            }
#endif
        }

        [MethodImpl((MethodImplOptions)768)]
        public unsafe T GetAtIndex<T>(int index) where T : unmanaged => *((T*)Address + _shape.TransformOffset(index));

        /// <summary>
        ///     Gets a subshape based on given <paramref name="indices"/>.
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        /// <remarks>Does not copy, returns a <see cref="Slice"/> or a memory slice</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedStorage GetData(params int[] indices)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            indices = Shape.InferNegativeCoordinates(Shape.dimensions, indices);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                return UnmanagedStorage.CreateBroadcastedUnsafe(InternalArray.Slice(offset, shape.BroadcastInfo.OriginalShape.size), shape);
            }
            else if (this_shape.IsSliced)
            {
                // in this case we can not get a slice of contiguous memory, so we slice
                return GetView(indices.Select(Slice.Index).ToArray());
            }
            else
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                return new UnmanagedStorage(InternalArray.Slice(offset, shape.Size), shape);
            }
        }

        /// <summary>
        ///     Gets a subshape based on given <paramref name="indices"/>.
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        /// <remarks>Does not copy, returns a <see cref="Slice"/> or a memory slice</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public unsafe UnmanagedStorage GetData(int* dims, int ndims)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            Shape.InferNegativeCoordinates(Shape.dimensions, dims, ndims);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(dims, ndims);
                return UnmanagedStorage.CreateBroadcastedUnsafe(InternalArray.Slice(offset, shape.BroadcastInfo.OriginalShape.size), shape);
            }
            else if (this_shape.IsSliced)
            {
                // in this case we can not get a slice of contiguous memory, so we slice
                var slices = new Slice[ndims];
                for (int i = 0; i < ndims; i++)
                {
                    slices[i] = Slice.Index(*(dims + i));
                }

                return GetView(slices);
            }
            else
            {
                var (shape, offset) = this_shape.GetSubshape(dims, ndims);
                return new UnmanagedStorage(InternalArray.Slice(offset, shape.Size), shape);
            }
        }

        /// <summary>
        ///     Get reference to internal data storage and cast (also copies) elements to new dtype if necessary
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        /// <remarks>Copies if <typeparamref name="T"/> does not equal to <see cref="DType"/> or if Shape is sliced.</remarks>
        public ArraySlice<T> GetData<T>() where T : unmanaged
        {
            if (!typeof(T).IsValidNPType())
                throw new NotSupportedException($"Type {typeof(T).Name} is not a valid np.dtype");

            if (!Shape.IsContiguous || Shape.ModifiedStrides)
                return CloneData<T>();

            var internalArray = InternalArray;
            if (internalArray is ArraySlice<T> ret)
                return ret;

            return _ChangeTypeOfArray<T>(internalArray);
        }

        /// <summary>
        ///     Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indices">indices</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        /// <exception cref="NullReferenceException">When <typeparamref name="T"/> does not equal to <see cref="DType"/></exception>
        /// <remarks>If you provide less indices than there are dimensions, the rest are filled with 0.</remarks> //TODO! doc this in other similar methods
        public T GetValue<T>(params int[] indices) where T : unmanaged
        {
            unsafe
            {
                return *((T*)Address + _shape.GetOffset(indices));
            }
        }

        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IArraySlice GetData() => InternalArray;
#if _REGEN1
        #region Direct Getters
     
        %foreach supported_dtypes,supported_dtypes_lowercase%
        /// <summary>
        ///     Retrieves value of type <see cref="#2"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="#2"/></exception>
        public #2 Get#1(params int[] indices)
            => _array#1[_shape.GetOffset(indices)];

        %
        #endregion
#else

        #region Direct Getters
     
        /// <summary>
        ///     Retrieves value of type <see cref="bool"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="bool"/></exception>
        public bool GetBoolean(params int[] indices)
            => _arrayBoolean[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="byte"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        public byte GetByte(params int[] indices)
            => _arrayByte[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="int"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        public int GetInt32(params int[] indices)
            => _arrayInt32[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="long"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        public long GetInt64(params int[] indices)
            => _arrayInt64[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="float"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        public float GetSingle(params int[] indices)
            => _arraySingle[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="double"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        public double GetDouble(params int[] indices)
            => _arrayDouble[_shape.GetOffset(indices)];

        #endregion
#endif

        #endregion
    }
}
