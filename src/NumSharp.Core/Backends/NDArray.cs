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
using System.Collections;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
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
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray : IIndex, ICloneable, IEnumerable
    {
        protected TensorEngine tensorEngine;

        #region Constructors

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        public NDArray(UnmanagedStorage storage)
        {
            Storage = storage;
            tensorEngine = storage.Engine;
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="shape">The shape to set for this NDArray, does not perform checks.</param>
        /// <remarks>Doesn't copy. Does not perform checks for <paramref name="shape"/>.</remarks>
        protected internal NDArray(UnmanagedStorage storage, Shape shape)
        {
            Storage = storage.Alias(ref shape);
            tensorEngine = storage.Engine;
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="shape">The shape to set for this NDArray, does not perform checks.</param>
        /// <remarks>Doesn't copy. Does not perform checks for <paramref name="shape"/>.</remarks>
        protected internal NDArray(UnmanagedStorage storage, ref Shape shape)
        {
            Storage = storage.Alias(ref shape);
            tensorEngine = storage.Engine;
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        protected internal NDArray(Type dtype, TensorEngine engine)
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
        protected internal NDArray(NPTypeCode typeCode, TensorEngine engine)
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

            if (shape.IsEmpty)
                shape = Shape.ExtractShape(values);

            Storage.Allocate(values.ResolveRank() != 1 ? ArraySlice.FromArray(Arrays.Flatten(values), false) : ArraySlice.FromArray(values, false), shape);
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

            if (shape.IsEmpty)
                shape = Shape.Vector((int) values.Count); //TODO! when long index, remove cast int

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
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="dtype">Internal data type</param>
        /// <param name="size">The size as a single dimension shape</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Type dtype, int size, bool fillZeros) : this(dtype, Shape.Vector(size), fillZeros) { }

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
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="dtype">Internal data type</param>
        /// <param name="size">The size as a single dimension shape</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(NPTypeCode dtype, int size, bool fillZeros) : this(dtype, Shape.Vector(size), true) { }

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
        ///     The dtype of this array.
        /// </summary>
        public Type dtype => Storage.DType;

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of this array.
        /// </summary>
        public NPTypeCode typecode => Storage.TypeCode;

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
        protected internal unsafe void* Address
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.Address;
        }

        /// <summary>
        ///     Data length of every dimension
        /// </summary>
        public int[] shape
        {
            get => Storage.Shape.Dimensions;
            set => Storage.Reshape(value);
        }

        /// <summary>
        ///     Dimension count
        /// </summary>
        public int ndim => Storage.Shape.NDim;

        /// <summary>
        ///     Total of elements
        /// </summary>
        public int size => Storage.Shape.Size;

        public int dtypesize => Storage.DTypeSize;

        public char order => Storage.Shape.Order;

        public int[] strides => Storage.Shape.Strides;

        /// <summary>
        ///     A 1-D iterator over the array.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flat.html</remarks>
        public NDArray flat
        {
            get
            {
                if (ndim == 1 || Shape.IsScalar) //because it is already flat, there is no need to clone even if it is already sliced.
                    return new NDArray(Storage);
                return this.reshape(size);
            }
        }

        /// <summary>
        ///     The transposed array. <br></br>
        ///     Same as self.transpose().
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html</remarks>
        public NDArray T
        {
            get
            {
                return transpose();
            }
        }

        /// <summary>
        ///     The shape representing this <see cref="NDArray"/>.
        /// </summary>
        public Shape Shape
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.Shape;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Storage.Reshape(value);
        }

        /// <summary>
        /// The internal storage that stores data for this <see cref="NDArray"/>.
        /// </summary>
        protected internal UnmanagedStorage Storage;

        /// <summary>
        ///     The tensor engine that handles this <see cref="NDArray"/>.
        /// </summary>
        public TensorEngine TensorEngine
        {
            [DebuggerStepThrough] get => tensorEngine ?? Storage.Engine ?? BackendFactory.GetEngine();
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
        ///     Get: Gets internal storage array by calling <see cref="IStorage.GetData"/><br></br>
        ///     Set: Replace internal storage by calling <see cref="IStorage.ReplaceData(System.Array)"/>
        /// </summary>
        /// <remarks>Setting does not replace internal storage array.</remarks>
        protected internal IArraySlice Array
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

#if _REGEN1
            #region Compute
		    switch (GetTypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
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
			    case NPTypeCode.Int32: return new NDIterator<int>(this, false).GetEnumerator();
			    case NPTypeCode.Int64: return new NDIterator<long>(this, false).GetEnumerator();
			    case NPTypeCode.Single: return new NDIterator<float>(this, false).GetEnumerator();
			    case NPTypeCode.Double: return new NDIterator<double>(this, false).GetEnumerator();
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
        ///     New view of array with the same data.
        /// </summary>
        /// <param name="dtype">
        ///     Data-type descriptor of the returned view, e.g., float32 or int16. The default, None, results in the view having the same data-type as a.
        ///     This argument can also be specified as an ndarray sub-class, which then specifies the type of the returned object (this is equivalent to setting the type parameter).
        /// </param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html</remarks>
        public NDArray view(Type dtype = null)
        {
            //TODO! this shouldnt be a cast in case dtype != null, it should be an unsafe reinterpret (see remarks).
            return dtype == null || dtype == this.dtype ? new NDArray(Storage.Alias()) : new NDArray(Storage.Cast(dtype));
        }

        /// <summary>
        ///     New view of array with the same data.
        /// </summary>
        /// <param name="dtype">
        ///     Data-type descriptor of the returned view, e.g., float32 or int16. The default, None, results in the view having the same data-type as a.
        ///     This argument can also be specified as an ndarray sub-class, which then specifies the type of the returned object (this is equivalent to setting the type parameter).
        /// </param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html</remarks>
        public NDArray<T> view<T>() where T : unmanaged 
            => view(typeof(T)).AsGeneric<T>();

        #region Getters

        /// <summary>
        ///     Get all NDArray slices at that specific dimension.
        /// </summary>
        /// <param name="axis">Zero-based dimension index on which axis and forward of it to select data., e.g. dimensions=1, shape is (2,2,3,3), returned shape = 4 times of (3,3)</param>
        /// <remarks>Does not perform copy.</remarks>
        /// <example>
        /// <code>
        ///     var nd = np.arange(27).reshape(3,1,3,3);<br></br>
        ///     var ret = nd.GetNDArrays(1);<br></br>
        ///     Assert.IsTrue(ret.All(n=>n.Shape == new Shape(3,3));<br></br>
        ///     Assert.IsTrue(ret.Length == 3);<br></br><br></br>
        ///     var nd = np.arange(27).reshape(3,1,3,3);<br></br>
        ///     
        ///     var ret = nd.GetNDArrays(0);<br></br>
        ///     Assert.IsTrue(ret.All(n=>n.Shape == new Shape(1,3,3));<br></br>
        ///     Assert.IsTrue(ret.Length == 3);<br></br>
        /// </code>
        /// </example>
        [SuppressMessage("ReSharper", "LoopCanBeConvertedToQuery")]
        public NDArray[] GetNDArrays(int axis = 0)
        {
            axis += 1; //axis is 0-based, we need 1 based (aka count)
            if (axis <= 0 || axis > Shape.dimensions.Length)
                throw new ArgumentOutOfRangeException(nameof(axis));

            //get all the dimensions involved till the axis
            var dims = Storage.Shape.dimensions;
            int[] selectDimensions = new int[axis];
            for (int i = 0; i < axis; i++)
                selectDimensions[i] = dims[i];

            //compute len
            int len = 1;
            foreach (var i in selectDimensions)
                len = len * i;

            var ret = new NDArray[len];
            var iter = new NDCoordinatesIncrementor(selectDimensions);
            var index = iter.Index; //heap the pointer to that array.
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = new NDArray(Storage.GetData(index));
                iter.Next();
            }

            return ret;
        }

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
        ///     Gets a NDArray at given <paramref name="indices"/>.
        /// </summary>
        /// <param name="indices">The coordinates to the wanted value</param>
        /// <remarks>Does not copy, returns a memory slice - this is similar to this[int[]]</remarks>
        public NDArray GetData(params int[] indices) => new NDArray(Storage.GetData(indices)) {tensorEngine = this.tensorEngine};

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
        ///     Retrieves value of type <see cref="double"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetDouble(params int[] indices) => Storage.GetDouble(indices);

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
        ///     Retrieves value of unspecified type (will figure using <see cref="DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="object"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ValueType GetValue(params int[] indices) => Storage.GetValue(indices);

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="object"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T GetValue<T>(params int[] indices) where T : unmanaged => Storage.GetValue<T>(indices);

        /// <summary>
        ///     Retrieves value of 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ValueType GetAtIndex(int index) => Storage.GetAtIndex(index);

        /// <summary>
        ///     Retrieves value of 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T GetAtIndex<T>(int index) where T : unmanaged => Storage.GetAtIndex<T>(index);

        #endregion

        #region Setters

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
        ///      Set a <see cref="NDArray"/>, <see cref="IArraySlice"/>, <see cref="Array"/> or a scalar value at given <see cref="indices"/>.
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
        public void SetValue(ValueType value, params int[] indices)
        {
            Storage.SetValue(value, indices);
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
        public void SetValue(object value, params int[] indices)
        {
            Storage.SetValue(value, indices);
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
        public void SetValue<T>(T value, params int[] indices) where T : unmanaged
        {
            Storage.SetValue<T>(value, indices);
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
        ///     Retrieves value at given linear (offset) <paramref name="index"/>.
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

#if _REGEN1
	%foreach supported_dtypes,supported_dtypes_lowercase%
        /// <summary>
        ///     Sets a #2 at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Set#1(#2 value, params int[] indices) => Storage.Set#1(value, indices);

    %
#else
        /// <summary>
        ///     Sets a bool at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetBoolean(bool value, params int[] indices) => Storage.SetBoolean(value, indices);

        /// <summary>
        ///     Sets a byte at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetByte(byte value, params int[] indices) => Storage.SetByte(value, indices);

        /// <summary>
        ///     Sets a int at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt32(int value, params int[] indices) => Storage.SetInt32(value, indices);

        /// <summary>
        ///     Sets a long at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt64(long value, params int[] indices) => Storage.SetInt64(value, indices);

        /// <summary>
        ///     Sets a float at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSingle(float value, params int[] indices) => Storage.SetSingle(value, indices);

        /// <summary>
        ///     Sets a double at specific coordinates.
        /// </summary>
        /// <param name="value">The values to assign</param>
        /// <param name="indices">The coordinates to set <paramref name="value"/> at.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDouble(double value, params int[] indices) => Storage.SetDouble(value, indices);
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
