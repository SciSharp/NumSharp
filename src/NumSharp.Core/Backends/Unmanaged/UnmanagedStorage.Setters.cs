using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Setters

        /// <summary>
        ///     Throws if the underlying shape is not writeable (e.g., broadcast arrays).
        ///     NumPy raises: ValueError: assignment destination is read-only
        /// </summary>
        [MethodImpl(Inline)]
        private void ThrowIfNotWriteable()
        {
            NumSharpException.ThrowIfNotWriteable(_shape);
        }

        /// <summary>
        ///     Performs a set of index without calling <see cref="Shape.TransformOffset(long)"/>.
        /// </summary>
        public void SetAtIndexUnsafe(object value, long index)
        {
            InternalArray.SetIndex(index, value);
        }

        /// <summary>
        ///     Performs a set of index without calling <see cref="Shape.TransformOffset(long)"/>.
        /// </summary>
        public void SetAtIndexUnsafe<T>(T value, long index) where T : unmanaged
        {
            Debug.Assert(typeof(T) == _dtype, $"SetAtIndexUnsafe<{typeof(T).Name}> called on {_dtype.Name} array.");
            unsafe
            {
                *((T*)Address + index) = value;
            }
        }

        public unsafe void SetAtIndex<T>(T value, long index) where T : unmanaged
        {
            Debug.Assert(typeof(T) == _dtype, $"SetAtIndex<{typeof(T).Name}> called on {_dtype.Name} array.");
            ThrowIfNotWriteable();
            *((T*)Address + _shape.TransformOffset(index)) = value;
        }

        public unsafe void SetAtIndex(object value, long index)
        {
            ThrowIfNotWriteable();
            switch (_typecode)
            {
                // //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                // %foreach supported_dtypes,supported_dtypes_lowercase%
                // case NPTypeCode.#1:
                    // *((#2*)Address + _shape.TransformOffset(index)) = (#2) value;
                    // return;
                // %
                // default:
                    // throw new NotSupportedException();

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.TransformOffset(index)) = (bool)value;
                    return;
                case NPTypeCode.SByte:
                    *((sbyte*)Address + _shape.TransformOffset(index)) = (sbyte)value;
                    return;
                case NPTypeCode.Byte:
                    *((byte*)Address + _shape.TransformOffset(index)) = (byte)value;
                    return;
                case NPTypeCode.Int16:
                    *((short*)Address + _shape.TransformOffset(index)) = (short)value;
                    return;
                case NPTypeCode.UInt16:
                    *((ushort*)Address + _shape.TransformOffset(index)) = (ushort)value;
                    return;
                case NPTypeCode.Int32:
                    *((int*)Address + _shape.TransformOffset(index)) = (int)value;
                    return;
                case NPTypeCode.UInt32:
                    *((uint*)Address + _shape.TransformOffset(index)) = (uint)value;
                    return;
                case NPTypeCode.Int64:
                    *((long*)Address + _shape.TransformOffset(index)) = (long)value;
                    return;
                case NPTypeCode.UInt64:
                    *((ulong*)Address + _shape.TransformOffset(index)) = (ulong)value;
                    return;
                case NPTypeCode.Char:
                    *((char*)Address + _shape.TransformOffset(index)) = (char)value;
                    return;
                case NPTypeCode.Half:
                    *((Half*)Address + _shape.TransformOffset(index)) = (Half)value;
                    return;
                case NPTypeCode.Double:
                    *((double*)Address + _shape.TransformOffset(index)) = (double)value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.TransformOffset(index)) = (float)value;
                    return;
                case NPTypeCode.Decimal:
                    *((decimal*)Address + _shape.TransformOffset(index)) = (decimal)value;
                    return;
                case NPTypeCode.Complex:
                    *((System.Numerics.Complex*)Address + _shape.TransformOffset(index)) = (System.Numerics.Complex)value;
                    return;
                default:
                    throw new NotSupportedException();
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
        public unsafe void SetValue<T>(T value, int[] indices) where T : unmanaged
        {
            Debug.Assert(typeof(T) == _dtype, $"SetValue<{typeof(T).Name}> called on {_dtype.Name} array. Use matching type or non-generic SetValue for conversion.");
            ThrowIfNotWriteable();
            *((T*)Address + _shape.GetOffset(indices)) = value;
        }

        /// <summary>
        ///     Set a single value at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The coordinates (long version).</param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        public unsafe void SetValue<T>(T value, params long[] indices) where T : unmanaged
        {
            Debug.Assert(typeof(T) == _dtype, $"SetValue<{typeof(T).Name}> called on {_dtype.Name} array. Use matching type or non-generic SetValue for conversion.");
            ThrowIfNotWriteable();
            *((T*)Address + _shape.GetOffset(indices)) = value;
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
        public unsafe void SetValue(object value, int[] indices)
        {
            ThrowIfNotWriteable();
            switch (_typecode)
            {
                // //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                // %foreach supported_dtypes,supported_dtypes_lowercase%
                // case NPTypeCode.#1:
                    // *((#2*)Address + _shape.GetOffset(indices)) = (#2) value;
                    // return;
                // %
                // default:
                    // throw new NotSupportedException();

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.GetOffset(indices)) = (bool)value;
                    return;
                case NPTypeCode.SByte:
                    *((sbyte*)Address + _shape.GetOffset(indices)) = (sbyte)value;
                    return;
                case NPTypeCode.Byte:
                    *((byte*)Address + _shape.GetOffset(indices)) = (byte)value;
                    return;
                case NPTypeCode.Int16:
                    *((short*)Address + _shape.GetOffset(indices)) = (short)value;
                    return;
                case NPTypeCode.UInt16:
                    *((ushort*)Address + _shape.GetOffset(indices)) = (ushort)value;
                    return;
                case NPTypeCode.Int32:
                    *((int*)Address + _shape.GetOffset(indices)) = (int)value;
                    return;
                case NPTypeCode.UInt32:
                    *((uint*)Address + _shape.GetOffset(indices)) = (uint)value;
                    return;
                case NPTypeCode.Int64:
                    *((long*)Address + _shape.GetOffset(indices)) = (long)value;
                    return;
                case NPTypeCode.UInt64:
                    *((ulong*)Address + _shape.GetOffset(indices)) = (ulong)value;
                    return;
                case NPTypeCode.Char:
                    *((char*)Address + _shape.GetOffset(indices)) = (char)value;
                    return;
                case NPTypeCode.Half:
                    *((Half*)Address + _shape.GetOffset(indices)) = (Half)value;
                    return;
                case NPTypeCode.Double:
                    *((double*)Address + _shape.GetOffset(indices)) = (double)value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.GetOffset(indices)) = (float)value;
                    return;
                case NPTypeCode.Decimal:
                    *((decimal*)Address + _shape.GetOffset(indices)) = (decimal)value;
                    return;
                case NPTypeCode.Complex:
                    *((System.Numerics.Complex*)Address + _shape.GetOffset(indices)) = (System.Numerics.Complex)value;
                    return;
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Set a single value at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The coordinates (long version).</param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        public unsafe void SetValue(object value, params long[] indices)
        {
            ThrowIfNotWriteable();
            switch (_typecode)
            {
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.GetOffset(indices)) = (bool)value;
                    return;
                case NPTypeCode.SByte:
                    *((sbyte*)Address + _shape.GetOffset(indices)) = (sbyte)value;
                    return;
                case NPTypeCode.Byte:
                    *((byte*)Address + _shape.GetOffset(indices)) = (byte)value;
                    return;
                case NPTypeCode.Int16:
                    *((short*)Address + _shape.GetOffset(indices)) = (short)value;
                    return;
                case NPTypeCode.UInt16:
                    *((ushort*)Address + _shape.GetOffset(indices)) = (ushort)value;
                    return;
                case NPTypeCode.Int32:
                    *((int*)Address + _shape.GetOffset(indices)) = (int)value;
                    return;
                case NPTypeCode.UInt32:
                    *((uint*)Address + _shape.GetOffset(indices)) = (uint)value;
                    return;
                case NPTypeCode.Int64:
                    *((long*)Address + _shape.GetOffset(indices)) = (long)value;
                    return;
                case NPTypeCode.UInt64:
                    *((ulong*)Address + _shape.GetOffset(indices)) = (ulong)value;
                    return;
                case NPTypeCode.Char:
                    *((char*)Address + _shape.GetOffset(indices)) = (char)value;
                    return;
                case NPTypeCode.Half:
                    *((Half*)Address + _shape.GetOffset(indices)) = (Half)value;
                    return;
                case NPTypeCode.Double:
                    *((double*)Address + _shape.GetOffset(indices)) = (double)value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.GetOffset(indices)) = (float)value;
                    return;
                case NPTypeCode.Decimal:
                    *((decimal*)Address + _shape.GetOffset(indices)) = (decimal)value;
                    return;
                case NPTypeCode.Complex:
                    *((System.Numerics.Complex*)Address + _shape.GetOffset(indices)) = (System.Numerics.Complex)value;
                    return;
                default:
                    throw new NotSupportedException();
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
        public void SetData(object value, int[] indices)
        {
            ThrowIfNotWriteable();
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
                    SetAtIndex(value, _shape.GetOffset(indices));
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
        public void SetData(NDArray value, int[] indices)
        {
            ThrowIfNotWriteable();
            if (ReferenceEquals(value, null))
                throw new ArgumentNullException(nameof(value));

            // Wrap negative coordinates to [0, dim) and bounds-check each axis, exactly as the
            // getter's GetData(int[]) does. WITHOUT this, a negative scalar-index assignment
            // (b[(object)-1] = v / b[-1L] = v) wrote at buffer[-1] — an OUT-OF-BOUNDS WRITE that
            // both silently corrupted adjacent heap memory and left the array unchanged (NumPy
            // assigns the last element). InferNegativeCoordinates also raises NumPy's IndexError
            // for a genuinely out-of-range index.
            indices = Shape.InferNegativeCoordinates(_shape.dimensions, indices);

            var valueshape = value.Shape;
            bool valueIsScalary = valueshape.IsScalar || valueshape.NDim == 1 && valueshape.size == 1;

            //incase lhs or rhs are broadcasted or sliced (noncontagious)
            if (_shape.IsBroadcasted || _shape.IsSliced || valueshape.IsBroadcasted || valueshape.IsSliced)
            {
                NpyIter.Copy(GetData(indices), value.Storage); //we use lhs stop because rhs is scalar which will fill all values of lhs
                return;
            }

            //by now value and this are contagious
            //////////////////////////////////////
            
            //incase it is 1 value assigned to all
            if (valueIsScalary && indices.Length != _shape.NDim)
            {
                GetData(indices).InternalArray.Fill(Converts.ChangeType(value.GetAtIndex(0), _typecode));
                //NpyIter.Copy(GetData(indices), value.Storage); //we use lhs stop because rhs is scalar which will fill all values of lhs
                return;
            }

            //incase its a scalar to scalar assignment
            if (indices.Length == _shape.NDim)
            {
                // The coordinate consumes EVERY axis, so the target is a single element (shape ()).
                // NumPy requires a 0-d / scalar value there: a 1+-D array — even size 1, e.g.
                // a[3] = np.array([78]) or m[0,2] = np.array([94]) — raises "setting an array
                // element with a sequence.", it is NOT silently unwrapped to its first element.
                // (The looser `valueIsScalary`, which also accepts a (1,) array, is correct only
                // for the sub-array broadcast branch above, NOT for a single-element target.)
                if (!valueshape.IsScalar)
                    throw new ValueError("setting an array element with a sequence.");

                SetValue(Converts.ChangeType(value.GetAtIndex(0), _typecode), indices);
                return;
            }

            //regular case
            var (subShape, offset) = _shape.GetSubshape(indices);

            // Empty source: a valid no-op ONLY when the target region is ALSO empty (e.g. np.pad on
            // an array with an empty axis does `padded[originalSlice] = array` where both are size 0).
            // Assigning an empty array into a NON-empty region cannot broadcast -> NumPy ValueError.
            if (valueshape.size == 0)
            {
                if (subShape.size == 0)
                    return;
                string TupE(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
                throw new ValueError($"could not broadcast input array from shape {TupE(valueshape.dimensions)} into shape {TupE(subShape.dimensions)}");
            }

            //if (!value.Storage.Shape.IsScalar && np.squeeze(subShape) != np.squeeze(value.Storage.Shape))
            //    throw new IncorrectShapeException($"Can't SetData to a from a shape of {value.Shape} to target shape {subShape}, the shape the coordinates point to mismatch the size of rhs (value)");

            // NumPy assignment lets the SOURCE carry more dims than the target when the extra
            // LEADING dims are size 1 (they are squeezed away) — e.g. a (1,1,3,3) value into a
            // (1,3,3) region. Drop those leading ones toward the target rank before broadcasting.
            var vForBc = value;
            if (valueshape.NDim > subShape.NDim)
            {
                int drop = valueshape.NDim - subShape.NDim;
                bool leadingOnes = true;
                for (int i = 0; i < drop; i++)
                    if (valueshape.dimensions[i] != 1) { leadingOnes = false; break; }
                if (leadingOnes)
                {
                    var ndims = new long[subShape.NDim];
                    for (int i = 0; i < ndims.Length; i++)
                        ndims[i] = valueshape.dimensions[drop + i];
                    vForBc = value.reshape(ndims);
                }
                // else: a leading non-1 dim -> broadcast_to below raises -> ValueError.
            }

            // Fast path: the (squeezed) value exactly fills the region in C-order.
            if (vForBc.Shape.size == subShape.size && vForBc.Shape.dimensions.SequenceEqual(subShape.dimensions))
            {
                //by now this ndarray is not broadcasted nor sliced
                unsafe
                {
                    //ReSharper disable once RedundantCast
                    //this must be a void* so it'll go through a typed switch.
                    value.Storage.CastIfNecessary(_typecode).CopyTo((void*)(this.Address + (this.InternalArray.ItemLength * offset)));
                }
                return;
            }

            // Otherwise the value must BROADCAST to the target region (NumPy). The old check
            // only tested divisibility (subShape.size % valueshape.size), which let an
            // incompatible smaller value through and then copied it PARTIALLY — e.g.
            // a[0] = [1,2] into a (4,) row wrote [1,2,_,_]. NumPy raises a ValueError here.
            string Tup(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
            NDArray broadcasted;
            try
            {
                broadcasted = np.broadcast_to(vForBc, subShape);
            }
            catch (IncorrectShapeException)
            {
                throw new ValueError($"could not broadcast input array from shape {Tup(valueshape.dimensions)} into shape {Tup(subShape.dimensions)}");
            }

            // A valid broadcast that needs stretching (value smaller than the region, or a
            // different but compatible rank) -> copy through the iterator, which honours the
            // stride-0 broadcast dimensions instead of a flat partial memcpy.
            NpyIter.Copy(GetData(indices), broadcasted.Storage);
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
        public void SetData(IArraySlice value, int[] indices)
        {
            ThrowIfNotWriteable();
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            //casting is resolved inside
            var lhs = GetData(indices);

            if (lhs.Count % value.Count != 0)
                throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");

            if (this._shape.IsBroadcasted || _shape.IsSliced || lhs.Count != value.Count) //if broadcast required
            {
                NpyIter.Copy(lhs, new UnmanagedStorage(value, value.Count == this.Count ? _shape.Clean(): Shape.Vector(value.Count)));
                return;
            }

            //by now this ndarray is not broadcasted nor sliced

            //this must be a void* so it'll go through a typed switch.
            (value.TypeCode == _typecode ? value : CastSliceViaIterator(value, _typecode))
                .CopyTo(lhs.InternalArray);
        }

        /// <summary>
        ///     Set a value at given <see cref="indices"/> (long version).
        /// </summary>
        public void SetData(object value, params long[] indices)
        {
            ThrowIfNotWriteable();
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
                    SetAtIndex(value, _shape.GetOffset(indices));
                    break;
            }
        }

        /// <summary>
        ///     Set a <see cref="NDArray"/> at given <see cref="indices"/> (long version).
        /// </summary>
        public void SetData(NDArray value, params long[] indices)
        {
            // Delegate to the int[] overload so the SAME semantics apply: a scalar broadcasts
            // across the WHOLE selected sub-array (a[0] = v fills row 0, not just a[0,0]), a
            // smaller value broadcasts/tiles, and a value that cannot broadcast raises NumPy's
            // ValueError. This overload previously wrote only the FIRST element for a scalar and
            // linear-copied a larger/smaller value WITHOUT validation — reachable via the
            // object[] single-int setter (b[(object)0] = v) and the long[] coordinate shim.
            if (ReferenceEquals(value, null))
                throw new ArgumentNullException(nameof(value));
            var intIndices = new int[indices.Length];
            for (int i = 0; i < indices.Length; i++)
                intIndices[i] = checked((int)indices[i]);
            SetData(value, intIndices);
        }

        /// <summary>
        ///     Set a <see cref="IArraySlice"/> at given <see cref="indices"/> (long version).
        /// </summary>
        public void SetData(IArraySlice value, params long[] indices)
        {
            ThrowIfNotWriteable();
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            //casting is resolved inside
            var lhs = GetData(indices);

            if (lhs.Count % value.Count != 0)
                throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");

            if (this._shape.IsBroadcasted || _shape.IsSliced || lhs.Count != value.Count) //if broadcast required
            {
                NpyIter.Copy(lhs, new UnmanagedStorage(value, value.Count == this.Count ? _shape.Clean(): Shape.Vector(value.Count)));
                return;
            }

            //by now this ndarray is not broadcasted nor sliced

            //this must be a void* so it'll go through a typed switch.
            (value.TypeCode == _typecode ? value : CastSliceViaIterator(value, _typecode))
                .CopyTo(lhs.InternalArray);
        }

        #region Typed Setters

	// %foreach supported_dtypes,supported_dtypes_lowercase%
        // /// <summary>
        // ///     Sets a #2 at specific coordinates.
        // /// </summary>
        // /// <param name="value">The values to assign</param>
        // /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        // [MethodImpl(Inline)]
        // public void Set#1(#2 value, int[] indices)         
        // {
            // unsafe {
                // *((#2*)Address + _shape.GetOffset(indices)) = value;
            // }
        // }

    // %
        /// <summary>
        ///     Sets a bool at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetBoolean(bool value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((bool*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a byte at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetByte(byte value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((byte*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a short at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetInt16(short value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((short*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a ushort at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetUInt16(ushort value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((ushort*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a int at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetInt32(int value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((int*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a uint at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetUInt32(uint value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((uint*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a long at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetInt64(long value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((long*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a ulong at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetUInt64(ulong value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((ulong*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a char at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetChar(char value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((char*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a double at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetDouble(double value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((double*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a double at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetDouble(double value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((double*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a float at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetSingle(float value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((float*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a decimal at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetDecimal(decimal value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((decimal*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a sbyte at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetSByte(sbyte value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((sbyte*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a Half at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetHalf(Half value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((Half*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a Complex at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(Inline)]
        public void SetComplex(System.Numerics.Complex value, int[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((System.Numerics.Complex*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        #region Typed Setters (long[] overloads)

        /// <summary>
        ///     Sets a bool at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetBoolean(bool value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((bool*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a byte at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetByte(byte value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((byte*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a short at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetInt16(short value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((short*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a ushort at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetUInt16(ushort value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((ushort*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a int at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetInt32(int value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((int*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a uint at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetUInt32(uint value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((uint*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a long at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetInt64(long value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((long*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a ulong at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetUInt64(ulong value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((ulong*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a char at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetChar(char value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((char*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a float at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetSingle(float value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((float*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a decimal at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetDecimal(decimal value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((decimal*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a sbyte at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetSByte(sbyte value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((sbyte*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a Half at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetHalf(Half value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((Half*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        /// <summary>
        ///     Sets a Complex at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at (long version).</param>
        [MethodImpl(Inline)]
        public void SetComplex(System.Numerics.Complex value, params long[] indices)
        {
            ThrowIfNotWriteable();
            unsafe
            {
                *((System.Numerics.Complex*)Address + _shape.GetOffset(indices)) = value;
            }
        }

        #endregion

        #endregion

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Copies values only if <paramref name="values"/> type does not match <see cref="DType"/> and doesn't change shape.</remarks>
        [MethodImpl(Inline)]
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
                _shape = Shape.Vector(values.Count);
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
                _shape = Shape.Vector(values.Count);
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
        [MethodImpl(Inline)]
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
