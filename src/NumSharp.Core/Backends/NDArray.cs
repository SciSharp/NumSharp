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

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
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
    public partial class NDArray : ICloneable, IEnumerable
    {

        protected NDArray(IStorage storage)
        {
            Storage = storage;
            TensorEngine = storage.Engine;
        }


        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        internal NDArray(Type dtype, ITensorEngine engine)
        {
            TensorEngine = engine;
            Storage = TensorEngine.GetStorage(dtype);
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="typeCode">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        internal NDArray(NPTypeCode typeCode, ITensorEngine engine)
        {
            TensorEngine = engine;
            Storage = TensorEngine.GetStorage(typeCode);
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <remarks>This constructor does not call allocation/></remarks>
        public NDArray(Type dtype) : this(dtype, BackendFactory.GetEngine())
        { }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="typeCode">Data type of elements</param>
        /// <remarks>This constructor does not call allocation/></remarks>
        public NDArray(NPTypeCode typeCode) : this(typeCode, BackendFactory.GetEngine())
        { }

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
            if (shape.IsEmpty)
                shape = new Shape(values.Length);

            shape.ChangeTensorLayout(order);
            Storage.Allocate(shape);
            Storage.ReplaceData(values);
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Type dtype, Shape shape) : this(dtype)
        {
            Storage.Allocate(shape);
        }

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
        /// Data length of every dimension
        /// </summary>
        public int[] shape => Storage.Shape.Dimensions;

        public int len => Storage.Shape.NDim == 0 ? 1 : Storage.Shape.Dimensions[0];

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

        public Slice slice => Storage.Slice;

        public int[] strides => Storage.Shape.Strides;

        /// <summary>
        /// The internal storage that stores data for this <see cref="NDArray"/>.
        /// </summary>

        internal IStorage Storage
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set;
        }

        public ITensorEngine TensorEngine { get; set; }

        /// <summary>
        /// Shortcut for access internal elements
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] Data<T>()
        {
            if (slice is null)
                return Storage.GetData<T>();
            else if (Storage.SupportsSpan)
                return Storage.View<T>().ToArray();
            else
                return Storage.GetData<T>();
        }

        public T Data<T>(params int[] indice) => Storage.GetData<T>(indice);

        public void SetData<T>(T value, params int[] indice) => Storage.SetData(value, indice);

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
        ///     Sets <see cref="values"/> as the internal data storage and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values.</remarks>
        public void ReplaceData(NDArray values)
        {
            Storage.ReplaceData(values);
        }

        /// <summary>
        ///     Gets the internal storage and converts it to <typeparamref name="T"/> if necessary.
        /// </summary>
        /// <typeparam name="T">The returned type.</typeparam>
        /// <returns>An array of type <typeparamref name="T"/></returns>
        public T[] GetData<T>()
        {
            return Storage.GetData<T>();
        }

        /// <summary>
        ///     Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        public Array GetData()
        {
            return Storage.GetData();
        }

        public int GetIndexInShape(params int[] select) => Storage.Shape.GetIndexInShape(slice, select);

        /// <summary>
        ///     Get: Gets internal storage array by calling <see cref="IStorage.GetData"/><br></br>
        ///     Set: Replace internal storage by calling <see cref="IStorage.ReplaceData(System.Array)"/>
        /// </summary>
        /// <remarks>Setting does not replace internal storage array.</remarks>
        internal Array Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Storage.GetData();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Storage.ReplaceData(value);
            }
        }

        public T[] CloneData<T>() => Storage.CloneData() as T[];

        public T Max<T>() => Data<T>().Max();

        // NumPy Signature: ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
        public NDArray astype(Type dtype, bool copy = false)
        {
            return BackendFactory.GetEngine().Cast(this, dtype, copy);
        }

        /// <summary>
        /// Clone the whole NDArray
        /// internal storage is also cloned into 2nd memory area
        /// </summary>
        /// <returns>Cloned NDArray</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        /// Clone the whole NDArray
        /// internal storage is also cloned into 2nd memory area
        /// </summary>
        /// <returns>Cloned NDArray</returns>
        NDArray Clone()
        {
            var puffer = new NDArray(this.dtype);
            var shapePuffer = new Shape(this.shape);
            puffer.Storage.Allocate(shapePuffer);
            puffer.Storage.ReplaceData(this.Storage.CloneData());

            return puffer;
        }

        public IEnumerator GetEnumerator()
        {
            switch (GetTypeCode)
            {
#if _REGEN
#else
#endif
            }


            switch (dtype.Name)
            {
                case "Int32": //todo! handle cases up to 16 dimensions and then generate using regen.
                {
                    switch (ndim)
                    {
                        case 0:
                            yield return Storage.GetInt32(0);
                            break;
                        case 1:
                            var arr = Data<int>();
                            for (int i = 0; i < size; i++)
                                yield return arr[i];
                            break;
                        case 2:
                            var l = shape[0];
                            for (int i = 0; i < l; i++)
                                yield return this[i];
                            break;
                        default:
                            throw new NotImplementedException();
                    }

                    break;
                }
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
        ///     Retrieves value of type <see cref="bool"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="bool"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool GetBoolean(params int[] indices)
        {
            return Storage.GetBoolean(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="byte"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte GetByte(params int[] indices)
        {
            return Storage.GetByte(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="char"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public char GetChar(params int[] indices)
        {
            return Storage.GetChar(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="Complex"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="Complex"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Complex GetComplex(params int[] indices)
        {
            return Storage.GetComplex(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public decimal GetDecimal(params int[] indices)
        {
            return Storage.GetDecimal(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="double"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetDouble(params int[] indices)
        {
            return Storage.GetDouble(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="short"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public short GetInt16(params int[] indices)
        {
            return Storage.GetInt16(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="int"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetInt32(params int[] indices)
        {
            return Storage.GetInt32(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="long"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetInt64(params int[] indices)
        {
            return Storage.GetInt64(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="NDArray"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="NDArray"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NDArray GetNDArray(params int[] indices)
        {
            return Storage.GetNDArray(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="float"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetSingle(params int[] indices)
        {
            return Storage.GetSingle(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="string"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="string"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string GetString(params int[] indices)
        {
            return Storage.GetString(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ushort GetUInt16(params int[] indices)
        {
            return Storage.GetUInt16(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint GetUInt32(params int[] indices)
        {
            return Storage.GetUInt32(indices);
        }

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/>.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong GetUInt64(params int[] indices)
        {
            return Storage.GetUInt64(indices);
        }

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="object"/></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public object GetValue(params int[] indices)
        {
            return Storage.GetValue(indices);
        }

        #endregion

        public override int GetHashCode() //todo! This cant be computed just using ndim and size.. NDArrays with different content will return true.
        {
            unchecked
            {
                var result = 1337;
                result = (result * 397) ^ this.ndim;
                result = (result * 397) ^ this.size;
                return result;
            }
        }
    }
}
