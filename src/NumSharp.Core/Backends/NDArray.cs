/*
 * NumSharp
 * Copyright (C) 2018 Haiping Chen
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Apache License 2.0 as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the Apache License 2.0
 * along with this program.  If not, see <http://www.apache.org/licenses/LICENSE-2.0/>.
 */

//TODO! Complete all TODOs return(?:.*;|;)[\n\r]{1,2}[\t\s]+//

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Collections;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     An array object represents a multidimensional, homogeneous array of fixed-size items.<br></br>
    ///     An associated data-type object describes the format of each element in the array (its byte-order,<br></br>
    ///     how many bytes it occupies in memory, whether it is an integer, a floating point number, or something else, etc.)
    /// </summary>
    /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html</remarks>
    [DebuggerTypeProxy(nameof(NDArrayDebuggerProxy))]
    public partial class NDArray : ICloneable, IEnumerable
    {
        protected TensorEngine tensorEngine;

        #region Constructors

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        internal NDArray(UnmanagedStorage storage)
        {
            Storage = storage;
            tensorEngine = storage.Engine;
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        internal NDArray(UnmanagedStorage storage, Shape shape)
        {
            Storage = storage;
            storage.SetShapeUnsafe(shape);
            tensorEngine = storage.Engine;
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        internal NDArray(UnmanagedStorage storage, ref Shape shape)
        {
            Storage = new UnmanagedStorage(storage.InternalArray, shape);
            tensorEngine = storage.Engine;
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        internal NDArray(Type dtype, TensorEngine engine)
        {
            tensorEngine = engine;
            Storage = TensorEngine.GetStorage(dtype);
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="typeCode">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        internal NDArray(NPTypeCode typeCode, TensorEngine engine)
        {
            tensorEngine = engine;
            Storage = TensorEngine.GetStorage(typeCode);
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <remarks>This constructor does not call allocation/></remarks>
        public NDArray(Type dtype) : this(dtype, BackendFactory.GetEngine()) { }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="typeCode">Data type of elements</param>
        /// <remarks>This constructor does not call allocation/></remarks>
        public NDArray(NPTypeCode typeCode) : this(typeCode, BackendFactory.GetEngine()) { }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <param name="shape"></param>
        /// <param name="order"></param>
        /// <returns>Array with values</returns>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Array values, Shape shape = default, char order = 'C') : this(values.GetType().GetElementType())
        {
            if (order != 'C')
                shape.ChangeTensorLayout(order);
            Storage.Allocate(values, shape);
        }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <param name="shape"></param>
        /// <param name="order"></param>
        /// <returns>Array with values</returns>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(IArraySlice values, Shape shape = default, char order = 'C') : this(values.TypeCode)
        {
            if (order != 'C')
                shape.ChangeTensorLayout(order);
            Storage.Allocate(values, shape);
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Type dtype, Shape shape) : this(dtype, shape, true) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="dtype">Internal data type</param>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Type dtype, int size) : this(dtype, Shape.Vector(size), true) { }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(NPTypeCode dtype, Shape shape) : this(dtype, shape, true) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="dtype">Internal data type</param>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(NPTypeCode dtype, int size) : this(dtype, Shape.Vector(size), true) { }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Type dtype, Shape shape, bool fillZeros) : this(dtype)
        {
            Storage.Allocate(shape, dtype, fillZeros);
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(NPTypeCode dtype, Shape shape, bool fillZeros) : this(dtype)
        {
            Storage.Allocate(shape, dtype, fillZeros);
        }

        private NDArray(IArraySlice array, Shape shape) : this(array.TypeCode)
        {
            Storage.Allocate(array, shape);
        }

        #endregion

        /// <summary>
        /// Data type of NDArray
        /// </summary>
        public Type dtype => Storage.DType;

        /// <summary>
        ///     Gets the precomputed <see cref="NPTypeCode"/> of <see cref="dtype"/>.
        /// </summary>
        internal NPTypeCode GetTypeCode
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.TypeCode;
        }

        /// <summary>
        ///     Gets the address that this NDArray starts from.
        /// </summary>
        internal unsafe void* Address
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.Address;
        }

        /// <summary>
        /// Data length of every dimension
        /// </summary>
        public int[] shape => Storage.Shape.Dimensions;

        /// <summary>
        /// Dimension count
        /// </summary>
        public int ndim => Storage.Shape.NDim;

        /// <summary>
        /// Total of elements
        /// </summary>
        public int size => Storage.Shape.Size;

        public int dtypesize => Storage.DTypeSize;

        public char order => Storage.Shape.Order;

        public int[] strides => Storage.Shape.Strides;

        //TODO! when reshaping a slice is done - this should not clone in case of a sliced shape.
        public NDArray flat => Shape.IsSliced ? new NDArray(Storage.Clone(), Shape.Vector(size)) : new NDArray(Storage, Shape.Vector(size));

        internal Shape Shape
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.Shape;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Storage.Reshape(value);
        }

        /// <summary>
        /// The internal storage that stores data for this <see cref="NDArray"/>.
        /// </summary>
        internal UnmanagedStorage Storage;


        /// <summary>
        ///     The tensor engine that handles this <see cref="NDArray"/>.
        /// </summary>
        public TensorEngine TensorEngine
        {
            get => tensorEngine ?? Storage.Engine ?? BackendFactory.GetEngine();
            set => tensorEngine = (value ?? Storage.Engine ?? BackendFactory.GetEngine());
        }

        /// <summary>
        /// Shortcut for access internal elements
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public ArraySlice<T> Data<T>() where T : unmanaged
        {
            return Storage.GetData<T>();
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
            Storage.SetData(value, indices);
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
            Storage.SetData(value, indices);
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
            Storage.SetData(value, indices);
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
        public unsafe void SetData<T>(T value, params int[] indices) where T : unmanaged
        {
            Storage.SetData<T>(value, indices);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data storage and changes the internal storage data type to <see cref="dtype"/> and casts <see cref="values"/> if necessary.
        /// </summary>
        /// <param name="values">The values to set as internal data soruce</param>
        /// <param name="dtype">The type to change this storage to and the type to cast <see cref="values"/> if necessary.</param>
        /// <remarks>Does not copy values unless cast is necessary.</remarks>
        // ReSharper disable once ParameterHidesMember
        public void ReplaceData(Array values, Type dtype)
        {
            Storage.ReplaceData(values, dtype);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data storage and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values.</remarks>
        public void ReplaceData(Array values)
        {
            Storage.ReplaceData(values);
        }

        /// <summary>
        ///     Sets <see cref="nd"/> as the internal data storage and changes the internal storage data type to <see cref="nd"/> type.
        /// </summary>
        /// <param name="nd"></param>
        /// <remarks>Does not copy values and does change shape and dtype.</remarks>
        public void ReplaceData(NDArray nd)
        {
            Storage.ReplaceData(nd);
        }

        /// <summary>
        ///     Set an Array to internal storage, cast it to new dtype and if necessary change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="typeCode"></param>
        /// <remarks>Does not copy values unless cast is necessary and doesn't change shape.</remarks>
        public void ReplaceData(Array values, NPTypeCode typeCode)
        {
            Storage.ReplaceData(values, typeCode);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        public void ReplaceData(IArraySlice values, Type dtype)
        {
            Storage.ReplaceData(values, dtype);
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        public void ReplaceData(IArraySlice values)
        {
            Storage.ReplaceData(values);
        }

        /// <summary>
        ///     Get: Gets internal storage array by calling <see cref="IStorage.GetData"/><br></br>
        ///     Set: Replace internal storage by calling <see cref="IStorage.ReplaceData(System.Array)"/>
        /// </summary>
        /// <remarks>Setting does not replace internal storage array.</remarks>
        internal IArraySlice Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.InternalArray;
        }

        public ArraySlice<T> CloneData<T>() where T : unmanaged => Storage.CloneData<T>();

        public IArraySlice CloneData() => Storage.CloneData();

        /// <summary>
        ///     Copy of the array, cast to a specified type.
        /// </summary>
        /// <param name="dtype">The dtype to cast this array.</param>
        /// <param name="copy">By default, astype always returns a newly allocated array. If this is set to false, the input internal array is replaced instead of returning a new NDArray with the casted data.</param>
        /// <returns>An <see cref="NDArray"/> of given <paramref name="dtype"/>.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public NDArray astype(Type dtype, bool copy = true) => TensorEngine.Cast(this, dtype, copy);

        /// <summary>
        ///     Copy of the array, cast to a specified type.
        /// </summary>
        /// <param name="dtype">The dtype to cast this array.</param>
        /// <param name="copy">By default, astype always returns a newly allocated array. If this is set to false, the input internal array is replaced instead of returning a new NDArray with the casted data.</param>
        /// <returns>An <see cref="NDArray"/> of given <paramref name="dtype"/>.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html</remarks>
        public NDArray astype(NPTypeCode typeCode, bool copy = true) => TensorEngine.Cast(this, typeCode, copy);

        /// <summary>
        /// Clone the whole NDArray
        /// internal storage is also cloned into 2nd memory area
        /// </summary>
        /// <returns>Cloned NDArray</returns>
        object ICloneable.Clone() => Clone();

        /// <summary>
        /// Clone the whole NDArray
        /// internal storage is also cloned into 2nd memory area
        /// </summary>
        /// <returns>Cloned NDArray</returns>
        public NDArray Clone() => new NDArray(this.Storage.Clone()) {tensorEngine = TensorEngine};

        public IEnumerator GetEnumerator()
        {
            if (Array == null || Shape.IsEmpty || Shape.size == 0)
                return _empty().GetEnumerator();

#if _REGEN
            #region Compute
		switch (GetTypeCode)
		{
			%foreach supported_currently_supported,supported_currently_supported_lowercase%
			case NPTypeCode.#1: return new NDIterator<#2>(this, false).GetEnumerator();
			%
			default:
				throw new NotSupportedException();
		}
            #endregion
#else

            #region Compute

            switch (GetTypeCode)
            {
                case NPTypeCode.Boolean: return new NDIterator<bool>(this, false).GetEnumerator();
                case NPTypeCode.Byte: return new NDIterator<byte>(this, false).GetEnumerator();
                case NPTypeCode.Int16: return new NDIterator<short>(this, false).GetEnumerator();
                case NPTypeCode.UInt16: return new NDIterator<ushort>(this, false).GetEnumerator();
                case NPTypeCode.Int32: return new NDIterator<int>(this, false).GetEnumerator();
                case NPTypeCode.UInt32: return new NDIterator<uint>(this, false).GetEnumerator();
                case NPTypeCode.Int64: return new NDIterator<long>(this, false).GetEnumerator();
                case NPTypeCode.UInt64: return new NDIterator<ulong>(this, false).GetEnumerator();
                case NPTypeCode.Char: return new NDIterator<char>(this, false).GetEnumerator();
                case NPTypeCode.Double: return new NDIterator<double>(this, false).GetEnumerator();
                case NPTypeCode.Single: return new NDIterator<float>(this, false).GetEnumerator();
                case NPTypeCode.Decimal: return new NDIterator<decimal>(this, false).GetEnumerator();
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif

            IEnumerable _empty()
            {
                yield break;
            }
        }

        /// <summary>
        /// New view of array with the same data.
        /// </summary>
        /// <returns></returns>
        public NDArray view(Type dtype = null)
        {
            if (dtype != null && dtype != this.dtype)
                Storage.ReplaceData(Array, dtype);

            var nd = new NDArray(Array, shape);

            return nd;
        }

        #region Getters

        /// <summary>
        ///     Gets the internal storage and converts it to <typeparamref name="T"/> if necessary.
        /// </summary>
        /// <typeparam name="T">The returned type.</typeparam>
        /// <returns>An array of type <typeparamref name="T"/></returns>
        public ArraySlice<T> GetData<T>() where T : unmanaged => Storage.GetData<T>();

        /// <summary>
        ///     Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        public IArraySlice GetData() => Storage.GetData();

        /// <summary>
        ///     Retrieves value of type <see cref="bool"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="bool"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool GetBoolean(params int[] indices) => Storage.GetBoolean(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="byte"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte GetByte(params int[] indices) => Storage.GetByte(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="char"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public char GetChar(params int[] indices) => Storage.GetChar(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public decimal GetDecimal(params int[] indices) => Storage.GetDecimal(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="double"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetDouble(params int[] indices) => Storage.GetDouble(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="short"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public short GetInt16(params int[] indices) => Storage.GetInt16(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="int"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetInt32(params int[] indices) => Storage.GetInt32(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="long"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetInt64(params int[] indices) => Storage.GetInt64(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="float"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetSingle(params int[] indices) => Storage.GetSingle(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ushort GetUInt16(params int[] indices) => Storage.GetUInt16(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint GetUInt32(params int[] indices) => Storage.GetUInt32(indices);

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong GetUInt64(params int[] indices) => Storage.GetUInt64(indices);

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="object"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public object GetValue(params int[] indices) => Storage.GetValue(indices);

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="object"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T GetValue<T>(params int[] indices) where T : unmanaged => Storage.GetData<T>(indices);

        /// <summary>
        ///     Retrieves value of 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public object GetAtIndex(int index) => Storage.GetAtIndex(index);

        /// <summary>
        ///     Retrieves value of 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T GetAtIndex<T>(int index) where T : unmanaged => Storage.GetAtIndex<T>(index);

        /// <summary>
        ///     Retrieves value of 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetAtIndex(object obj, int index) => Storage.SetAtIndex(obj, index);

        /// <summary>
        ///     Retrieves value of 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetAtIndex<T>(T value, int index) where T : unmanaged => Storage.SetAtIndex(value, index);


        //TODO! add SetInt32 and such methods!
#if _REGEN
	%foreach supported_currently_supported,supported_currently_supported_lowercase%
	    /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="object"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set#1(#2 value, params int[] indices) => Storage.Set#1(indices);

    %
#else
#endif

        #endregion

        private class NDArrayDebuggerProxy
        {
            private readonly NDArray NDArray;

            /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
            public NDArrayDebuggerProxy(NDArray ndArray)
            {
                NDArray = ndArray;
            }

            /// <summary>Returns a string that represents the current object.</summary>
            /// <returns>A string that represents the current object.</returns>
            public override string ToString()
            {
                //TODO! make a truncuted to string.
                return NDArray.ToString(false);
            }
        }
    }
}
