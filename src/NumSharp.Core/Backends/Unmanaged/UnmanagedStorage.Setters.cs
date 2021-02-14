using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Setters

        /// <summary>
        ///     Performs a set of index without calling <see cref="Shape.TransformOffset(int)"/>.
        /// </summary>
        public void SetAtIndexUnsafe(ValueType value, int index)
        {
            InternalArray.SetIndex(index, value);
        }

        /// <summary>
        ///     Performs a set of index without calling <see cref="Shape.TransformOffset(int)"/>.
        /// </summary>
        public void SetAtIndexUnsafe<T>(T value, int index) where T : unmanaged
        {
            unsafe
            {
                *((T*)Address + index) = value;
            }
        }

        public unsafe void SetAtIndex<T>(T value, int index) where T : unmanaged
        {
            *((T*)Address + _shape.TransformOffset(index)) = value;
        }

        public unsafe void SetAtIndex(object value, int index)
        {
            switch (_typecode)
            {
#if _REGEN1
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                    *((#2*)Address + _shape.TransformOffset(index)) = (#2) value;
                    return;
                %
                default:
                    throw new NotSupportedException();
#else

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.TransformOffset(index)) = (bool) value;
                    return;
                case NPTypeCode.Byte:
                    *((byte*)Address + _shape.TransformOffset(index)) = (byte) value;
                    return;
                case NPTypeCode.Int32:
                    *((int*)Address + _shape.TransformOffset(index)) = (int) value;
                    return;
                case NPTypeCode.Int64:
                    *((long*)Address + _shape.TransformOffset(index)) = (long) value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.TransformOffset(index)) = (float) value;
                    return;
                case NPTypeCode.Double:
                    *((double*)Address + _shape.TransformOffset(index)) = (double) value;
                    return;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Set a single value at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        public unsafe void SetValue<T>(T value, params int[] indices) where T : unmanaged
            => *((T*)Address + _shape.GetOffset(indices)) = value;

        /// <summary>
        ///     Set a single value at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        public unsafe void SetValue(object value, params int[] indices)
        {
            switch (_typecode)
            {
#if _REGEN1
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                    *((#2*)Address + _shape.GetOffset(indices)) = (#2) value;
                    return;
                %
                default:
                    throw new NotSupportedException();
#else

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.GetOffset(indices)) = (bool) value;
                    return;
                case NPTypeCode.Byte:
                    *((byte*)Address + _shape.GetOffset(indices)) = (byte) value;
                    return;
                case NPTypeCode.Int32:
                    *((int*)Address + _shape.GetOffset(indices)) = (int) value;
                    return;
                case NPTypeCode.Int64:
                    *((long*)Address + _shape.GetOffset(indices)) = (long) value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.GetOffset(indices)) = (float) value;
                    return;
                case NPTypeCode.Double:
                    *((double*)Address + _shape.GetOffset(indices)) = (double) value;
                    return;
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Set a single value at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        public void SetData(object value, params int[] indices)
        {
            switch (value)
            {
                case NDArray nd:
                    SetData(nd, indices);
                    return;
                case IArraySlice arr:
                    SetData(arr, indices);
                    return;
                case Array array:
                    SetData((NDArray)array, indices);
                    return;
                default:
                    //we assume this is a scalar.
                    SetValue(value, _shape.GetOffset(indices));
                    break;
            }
        }

        /// <summary>
        ///     Set a <see cref="NDArray"/> at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
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
                value.Storage.CastIfNecessary(_typecode).CopyTo((void*)(this.Address + (this.InternalArray.ItemLength * offset)));
            }
        }

        /// <summary>
        ///     Set a <see cref="IArraySlice"/> at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        public void SetData(IArraySlice value, params int[] indices)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            //casting is resolved inside
            var lhs = GetData(indices);

            if (lhs.Count % value.Count != 0)
                throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");

            if (this._shape.IsBroadcasted || _shape.IsSliced || lhs.Count != value.Count) //if broadcast required
            {
                MultiIterator.Assign(lhs, new UnmanagedStorage(value, value.Count == this.Count ? _shape.Clean(): Shape.Vector((int) value.Count))); //TODO! when long index, remove cast int
                return;
            }

            //by now this ndarray is not broadcasted nor sliced

            //this must be a void* so it'll go through a typed switch.
            (value.TypeCode == _typecode ? value : value.CastTo(_typecode))
                .CopyTo(lhs.InternalArray);
        }

        #region Typed Setters

#if _REGEN
	%foreach supported_dtypes,supported_dtypes_lowercase%
        /// <summary>
        ///     Sets a #2 at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set#1(#2 value, params int[] indices)         
        {
            unsafe {
                *((#2*)Address + _shape.GetOffset(indices)) = value;
            }
        }

    %
#else
        /// <summary>
        ///     Sets a bool at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetBoolean(bool value, params int[] indices)         
        {
            unsafe {
                *((bool*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a byte at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetByte(byte value, params int[] indices)         
        {
            unsafe {
                *((byte*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a int at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt32(int value, params int[] indices)         
        {
            unsafe {
                *((int*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a long at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt64(long value, params int[] indices)         
        {
            unsafe {
                *((long*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a float at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSingle(float value, params int[] indices)         
        {
            unsafe {
                *((float*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a double at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDouble(double value, params int[] indices)         
        {
            unsafe {
                *((double*)Address + _shape.GetOffset(indices)) = value;
            }
        }
#endif

        #endregion

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Copies values only if <paramref name="values"/> type does not match <see cref="DType"/> and doesn't change shape.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReplaceData(Array values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            SetInternalArray(_ChangeTypeOfArray(values, _dtype));

            if (_shape.IsEmpty)
                _shape = new Shape(values.Length);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        public void ReplaceData(IArraySlice values)
        {
            SetInternalArray(values);

            if (_shape.IsEmpty)
                _shape = new Shape((int)values.Count); //TODO! when long index, remove cast int
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        public void ReplaceData(IArraySlice values, Type dtype)
        {
            SetInternalArray(values);

            if (_shape.IsEmpty)
                _shape = new Shape((int) values.Count); //TODO! when long index, remove cast int
        }

        /// <summary>
        /// Set an Array to internal storage, cast it to new dtype and change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <remarks>Does not copy values unless cast in necessary and doesn't change shape.</remarks>
        public void ReplaceData(Array values, Type dtype)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (dtype == null)
                throw new ArgumentNullException(nameof(dtype));

            var changedArray = _ChangeTypeOfArray(values, dtype);
            //first try to convert to dtype only then we apply changes.
            _dtype = dtype;
            _typecode = _dtype.GetTypeCode();
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{dtype.Name} as a dtype is not supported.");
            SetInternalArray(changedArray);
        }

        /// <summary>
        ///     Set an Array to internal storage, cast it to new dtype and if necessary change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="typeCode"></param>
        /// <remarks>Does not copy values unless cast is necessary and doesn't change shape.</remarks>
        public void ReplaceData(Array values, NPTypeCode typeCode)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            var dtype = typeCode.AsType();
            var changedArray = _ChangeTypeOfArray(values, dtype);
            //first try to convert to dtype only then we apply changes.
            _dtype = dtype;
            _typecode = _dtype.GetTypeCode();
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{dtype.Name} as a dtype is not supported.");
            SetInternalArray(changedArray);
        }

        /// <summary>
        ///     Sets <see cref="nd"/> as the internal data storage and changes the internal storage data type to <see cref="nd"/> type.
        /// </summary>
        /// <param name="nd"></param>
        /// <remarks>Does not copy values and does change shape and dtype.</remarks>
        public void ReplaceData(NDArray nd)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            //first try to convert to dtype only then we apply changes.
            _shape = nd.shape;
            _dtype = nd.dtype;
            _typecode = nd.GetTypeCode;
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{_dtype.Name} as a dtype is not supported.");

            //todo! what if nd is sliced

            SetInternalArray(nd.Shape.IsSliced ? nd.Storage.CloneData() : nd.Array);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="shape">The shape to set in this storage. (without checking if shape matches storage)</param>
        /// <remarks>Copies values only if <paramref name="values"/> type does not match <see cref="DType"/> and doesn't change shape. Doesn't check if shape size matches.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReplaceData(Array values, Shape shape)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _shape = shape;
            SetInternalArray(_ChangeTypeOfArray(values, _dtype));
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="shape">The shape to set in this storage. (without checking if shape matches storage)</param>
        /// <remarks>Does not copy values and doesn't change shape. Doesn't check if shape size matches.</remarks>
        public void ReplaceData(IArraySlice values, Shape shape)
        {
            _shape = shape;
            SetInternalArray(values);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <param name="shape">The shape to set in this storage. (without checking if shape matches storage)</param>
        /// <remarks>Does not copy values and doesn't change shape. Doesn't check if shape size matches.</remarks>
        public void ReplaceData(IArraySlice values, Type dtype, Shape shape)
        {
            _shape = shape;
            SetInternalArray(values);
        }

        /// <summary>
        /// Set an Array to internal storage, cast it to new dtype and change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <param name="shape">The shape to set in this storage. (without checking if shape matches storage)</param>
        /// <remarks>Does not copy values unless cast in necessary and doesn't change shape. Doesn't check if shape size matches.</remarks>
        public void ReplaceData(Array values, Type dtype, Shape shape)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (dtype == null)
                throw new ArgumentNullException(nameof(dtype));

            var changedArray = _ChangeTypeOfArray(values, dtype);
            //first try to convert to dtype only then we apply changes.
            _dtype = dtype;
            _typecode = _dtype.GetTypeCode();
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{dtype.Name} as a dtype is not supported.");
            _shape = shape;
            SetInternalArray(changedArray);
        }

        /// <summary>
        ///     Set an Array to internal storage, cast it to new dtype and if necessary change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="typeCode"></param>
        /// <param name="shape">The shape to set in this storage. (without checking if shape matches storage)</param>
        /// <remarks>Does not copy values unless cast is necessary and doesn't change shape. Doesn't check if shape size matches.</remarks>
        public void ReplaceData(Array values, NPTypeCode typeCode, Shape shape)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            var dtype = typeCode.AsType();
            var changedArray = _ChangeTypeOfArray(values, dtype);
            //first try to convert to dtype only then we apply changes.
            _dtype = dtype;
            _typecode = _dtype.GetTypeCode();
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{dtype.Name} as a dtype is not supported.");
            _shape = shape;
            SetInternalArray(changedArray);
        }

        /// <summary>
        ///     Sets <see cref="nd"/> as the internal data storage and changes the internal storage data type to <see cref="nd"/> type.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="shape">The shape to set in this storage. (without checking if shape matches storage)</param>
        /// <remarks>Does not copy values and does change shape and dtype. Doesn't check if shape size matches.</remarks>
        public void ReplaceData(NDArray nd, Shape shape)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            //first try to convert to dtype only then we apply changes.
            _dtype = nd.dtype;
            _typecode = nd.GetTypeCode;
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{_dtype.Name} as a dtype is not supported.");

            _shape = shape;
            SetInternalArray(nd.Array);
        }

        #endregion
    }
}
