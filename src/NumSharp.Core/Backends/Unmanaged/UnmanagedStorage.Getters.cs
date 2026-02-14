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
#if _REGEN
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1: return *((#2*)Address + _shape.GetOffset(indices));
	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return *((bool*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Byte: return *((byte*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Int16: return *((short*)Address + _shape.GetOffset(indices));
                case NPTypeCode.UInt16: return *((ushort*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Int32: return *((int*)Address + _shape.GetOffset(indices));
                case NPTypeCode.UInt32: return *((uint*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Int64: return *((long*)Address + _shape.GetOffset(indices));
                case NPTypeCode.UInt64: return *((ulong*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Char: return *((char*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Double: return *((double*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Single: return *((float*)Address + _shape.GetOffset(indices));
                case NPTypeCode.Decimal: return *((decimal*)Address + _shape.GetOffset(indices));
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        public unsafe ValueType GetAtIndex(int index)
        {
#if _REGEN
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
                case NPTypeCode.Int16: return *((short*)Address + _shape.TransformOffset(index));
                case NPTypeCode.UInt16: return *((ushort*)Address + _shape.TransformOffset(index));
                case NPTypeCode.Int32: return *((int*)Address + _shape.TransformOffset(index));
                case NPTypeCode.UInt32: return *((uint*)Address + _shape.TransformOffset(index));
                case NPTypeCode.Int64: return *((long*)Address + _shape.TransformOffset(index));
                case NPTypeCode.UInt64: return *((ulong*)Address + _shape.TransformOffset(index));
                case NPTypeCode.Char: return *((char*)Address + _shape.TransformOffset(index));
                case NPTypeCode.Double: return *((double*)Address + _shape.TransformOffset(index));
                case NPTypeCode.Single: return *((float*)Address + _shape.TransformOffset(index));
                case NPTypeCode.Decimal: return *((decimal*)Address + _shape.TransformOffset(index));
                default:
                    throw new NotSupportedException();
            }
#endif
        }

        [MethodImpl(OptimizeAndInline)]
        public unsafe T GetAtIndex<T>(int index) where T : unmanaged => *((T*)Address + _shape.TransformOffset(index));

        /// <summary>
        /// Gets a sub-array based on the given indices, returning a view that shares memory.
        /// </summary>
        /// <param name="indices">
        /// The indices specifying which dimensions to select. Negative indices are supported
        /// and are converted to positive indices relative to the dimension size.
        /// </param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> representing the sub-array. This is a view
        /// that shares memory with the original storage. The returned storage's
        /// <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The returned storage shares memory with this storage.
        /// Modifications through either storage affect the same underlying data.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner,
        /// ensuring all views in a chain point to the original storage, not intermediate views.
        /// </para>
        /// <para>
        /// <b>Code Paths:</b>
        /// <list type="bullet">
        ///   <item><b>Broadcasted shapes:</b> Creates a broadcasted view with base tracking</item>
        ///   <item><b>Non-contiguous shapes:</b> Delegates to <see cref="GetView(Slice[])"/> which uses <see cref="Alias(Shape)"/></item>
        ///   <item><b>Contiguous shapes:</b> Creates a direct memory slice with base tracking</item>
        /// </list>
        /// </para>
        /// </remarks>
        /// <seealso cref="GetView(Slice[])"/>
        /// <seealso cref="Alias()"/>
        [MethodImpl(OptimizeAndInline)]
        public UnmanagedStorage GetData(params int[] indices)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            indices = Shape.InferNegativeCoordinates(Shape.dimensions, indices);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                // NumPy-aligned: use bufferSize instead of BroadcastInfo.OriginalShape.size
                int sliceSize = shape.BufferSize > 0 ? shape.BufferSize : shape.size;
                var view = UnmanagedStorage.CreateBroadcastedUnsafe(InternalArray.Slice(offset, sliceSize), shape);
                view._baseStorage = _baseStorage ?? this;
                return view;
            }
            else if (!this_shape.IsContiguous)
            {
                // Non-contiguous shapes (stepped slices, transposed, etc.) cannot use
                // memory slicing. Create a view with indexed slices instead.
                return GetView(indices.Select(Slice.Index).ToArray());
            }
            else
            {
                // Contiguous shape: can take a direct memory slice.
                // GetSubshape computes the correct offset accounting for shape.offset.
                var (shape, offset) = this_shape.GetSubshape(indices);
                var view = new UnmanagedStorage(InternalArray.Slice(offset, shape.Size), shape);
                view._baseStorage = _baseStorage ?? this;
                return view;
            }
        }

        /// <summary>
        /// Gets a sub-array based on the given indices (pointer version), returning a view that shares memory.
        /// </summary>
        /// <param name="dims">Pointer to an array of dimension indices.</param>
        /// <param name="ndims">The number of indices in the array.</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> representing the sub-array. This is a view
        /// that shares memory with the original storage. The returned storage's
        /// <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The returned storage shares memory with this storage.
        /// Modifications through either storage affect the same underlying data.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner,
        /// ensuring all views in a chain point to the original storage, not intermediate views.
        /// </para>
        /// <para>
        /// <b>Code Paths:</b>
        /// <list type="bullet">
        ///   <item><b>Broadcasted shapes:</b> Creates a broadcasted view with base tracking</item>
        ///   <item><b>Non-contiguous shapes:</b> Delegates to <see cref="GetView(Slice[])"/> which uses <see cref="Alias(Shape)"/></item>
        ///   <item><b>Contiguous shapes:</b> Creates a direct memory slice with base tracking</item>
        /// </list>
        /// </para>
        /// </remarks>
        /// <seealso cref="GetData(int[])"/>
        [MethodImpl(OptimizeAndInline)]
        public unsafe UnmanagedStorage GetData(int* dims, int ndims)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            Shape.InferNegativeCoordinates(Shape.dimensions, dims, ndims);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(dims, ndims);
                // NumPy-aligned: use bufferSize instead of BroadcastInfo.OriginalShape.size
                int sliceSize = shape.BufferSize > 0 ? shape.BufferSize : shape.size;
                var view = UnmanagedStorage.CreateBroadcastedUnsafe(InternalArray.Slice(offset, sliceSize), shape);
                view._baseStorage = _baseStorage ?? this;
                return view;
            }
            else if (!this_shape.IsContiguous)
            {
                // Non-contiguous shapes (stepped slices, transposed, etc.) cannot use
                // memory slicing. Create a view with indexed slices instead.
                var slices = new Slice[ndims];
                for (int i = 0; i < ndims; i++)
                {
                    slices[i] = Slice.Index(*(dims + i));
                }

                return GetView(slices);
            }
            else
            {
                // Contiguous shape: can take a direct memory slice.
                // GetSubshape computes the correct offset accounting for shape.offset.
                var (shape, offset) = this_shape.GetSubshape(dims, ndims);
                var view = new UnmanagedStorage(InternalArray.Slice(offset, shape.Size), shape);
                view._baseStorage = _baseStorage ?? this;
                return view;
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

            if (!Shape.IsContiguous)
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
        [MethodImpl(Inline)]
        public IArraySlice GetData() => InternalArray;
#if _REGEN
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
        ///     Retrieves value of type <see cref="short"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        public short GetInt16(params int[] indices)
            => _arrayInt16[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        public ushort GetUInt16(params int[] indices)
            => _arrayUInt16[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="int"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        public int GetInt32(params int[] indices)
            => _arrayInt32[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        public uint GetUInt32(params int[] indices)
            => _arrayUInt32[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="long"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        public long GetInt64(params int[] indices)
            => _arrayInt64[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        public ulong GetUInt64(params int[] indices)
            => _arrayUInt64[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="char"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        public char GetChar(params int[] indices)
            => _arrayChar[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="double"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        public double GetDouble(params int[] indices)
            => _arrayDouble[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="float"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        public float GetSingle(params int[] indices)
            => _arraySingle[_shape.GetOffset(indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        public decimal GetDecimal(params int[] indices)
            => _arrayDecimal[_shape.GetOffset(indices)];

        #endregion
#endif

        #endregion
    }
}
