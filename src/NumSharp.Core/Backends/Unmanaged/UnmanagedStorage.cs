using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

// ReSharper disable once CheckNamespace
namespace NumSharp.Backends
{
    /// <summary>
    ///     Serves as a typed storage for an array.
    /// </summary>
    /// <remarks>
    ///     Responsible for :<br></br>
    ///      - store data type, elements, Shape<br></br>
    ///      - offers methods for accessing elements depending on shape<br></br>
    ///      - offers methods for casting elements<br></br>
    ///      - offers methods for change tensor order<br></br>
    ///      - GetData always return reference object to the true storage<br></br>
    ///      - GetData{T} and SetData{T} change dtype and cast storage<br></br>
    ///      - CloneData always create a clone of storage and return this as reference object<br></br>
    ///      - CloneData{T} clone storage and cast this clone <br></br>
    /// </remarks>
    public partial class UnmanagedStorage : ICloneable
    {
#if _REGEN1
        %foreach supported_dtypes,supported_dtypes_lowercase%
        protected ArraySlice<#2> _array#1;
        %
#else
        protected ArraySlice<bool> _arrayBoolean;
        protected ArraySlice<byte> _arrayByte;
        protected ArraySlice<char> _arrayChar;
        protected ArraySlice<int> _arrayInt32;
        protected ArraySlice<long> _arrayInt64;
        protected ArraySlice<float> _arraySingle;
        protected ArraySlice<double> _arrayDouble;
#endif
        public IArraySlice InternalArray;
        public unsafe byte* Address;
        public int Count;

        protected Type _dtype;
        protected NPTypeCode _typecode;
        protected Shape _shape;

        /// <summary>
        ///     The data type of internal storage array.
        /// </summary>
        /// <value>numpys equal dtype</value>
        /// <remarks>Has to be compliant with <see cref="NPTypeCode"/>.</remarks>
        public Type DType => _dtype;

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of <see cref="IStorage.DType"/>.
        /// </summary>
        public NPTypeCode TypeCode => _typecode;

        /// <summary>
        ///     The size in bytes of a single value of <see cref="DType"/>
        /// </summary>
        /// <remarks>Computed by <see cref="Marshal.SizeOf(object)"/></remarks>
        public int DTypeSize
            => _typecode switch
            {
                NPTypeCode.String => IntPtr.Size,
                NPTypeCode.Boolean => sizeof(bool),
                _ => Marshal.SizeOf(_dtype)
            };

        /// <summary>
        ///     The shape representing the data in this storage.
        /// </summary>
        public Shape Shape
        {
            get
            {
                return _shape;
            }
            set
            {
                this.Reshape(ref value);
            }
        }

        /// <summary>
        ///     The shape representing the data in this storage.
        /// </summary>
        public ref Shape ShapeReference => ref _shape;

        /// <summary>
        ///     Spans <see cref="Address"/> &lt;-&gt; <see cref="Count"/>
        /// </summary>
        /// <remarks>This ignores completely slicing.</remarks>
        public Span<T> AsSpan<T>()
        {
            if (!_shape.IsContiguous)
                throw new InvalidOperationException("Unable to span a non-contiguous storage.");

            unsafe
            {
                return new Span<T>(Address, Count);
            }
        }

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        public TensorEngine Engine { get; protected internal set; }

        public static UnmanagedStorage Scalar<T>(T value) where T : unmanaged => new UnmanagedStorage(ArraySlice.Scalar<T>(value));

        public static UnmanagedStorage Scalar(object value) => new UnmanagedStorage(ArraySlice.Scalar(value));

        public static UnmanagedStorage Scalar(object value, NPTypeCode typeCode) => new UnmanagedStorage(ArraySlice.Scalar(value, typeCode));

        /// <summary>
        ///     Wraps given <paramref name="arraySlice"/> in <see cref="UnmanagedStorage"/> with a broadcasted shape.
        /// </summary>
        /// <param name="arraySlice">The slice to wrap </param>
        /// <param name="shape">The shape to represent this storage, can be a broadcast.</param>
        /// <remarks>Named unsafe because there it does not perform a check if the shape is valid for this storage size.</remarks>
        public static UnmanagedStorage CreateBroadcastedUnsafe(IArraySlice arraySlice, Shape shape)
        {
            var ret = new UnmanagedStorage();
            ret._Allocate(shape, arraySlice);
            return ret;
        }

        /// <summary>
        ///     Wraps given <paramref name="storage"/> in <see cref="UnmanagedStorage"/> with a broadcasted shape.
        /// </summary>
        /// <param name="storage">The storage to take <see cref="InternalArray"/> from.</param>
        /// <param name="shape">The shape to represent this storage, can be a broadcast.</param>
        /// <remarks>Named unsafe because there it does not perform a check if the shape is valid for this storage size.</remarks>
        public static UnmanagedStorage CreateBroadcastedUnsafe(UnmanagedStorage storage, Shape shape)
        {
            var ret = new UnmanagedStorage();
            ret._Allocate(shape, storage.InternalArray);
            return ret;
        }


        private UnmanagedStorage() { }

        /// <summary>
        ///     Scalar constructor
        /// </summary>
        private unsafe UnmanagedStorage(IArraySlice values)
        {
            _shape = Shape.Scalar;
            _dtype = (_typecode = values.TypeCode).AsType();
            Address = (byte*)values.Address;
            Count = 1;
            SetInternalArray(values);
        }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="dtype"/>.
        /// </summary>
        /// <param name="dtype">The type of this storage</param>
        /// <remarks>Usually <see cref="Allocate(NumSharp.Shape,System.Type)"/> is called after this constructor.</remarks>
        public UnmanagedStorage(Type dtype)
        {
            _dtype = dtype ?? throw new ArgumentNullException(nameof(dtype));
            _typecode = dtype.GetTypeCode();
        }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="typeCode"/>.
        /// </summary>
        /// <param name="typeCode">The type of this storage</param>
        /// <remarks>Usually <see cref="Allocate(NumSharp.Shape,System.Type)"/> is called after this constructor.</remarks>
        public UnmanagedStorage(NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            _dtype = typeCode.AsType();
            _typecode = typeCode;
        }

        private UnmanagedStorage(object value)
        {
            _Allocate(Shape.Scalar, ArraySlice.Scalar(value));
        }

        /// <summary>
        ///     Wraps given <paramref name="arraySlice"/> in <see cref="UnmanagedStorage"/>.
        /// </summary>
        /// <param name="arraySlice">The slice to wrap </param>
        public UnmanagedStorage(IArraySlice arraySlice, Shape shape)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (shape.size != arraySlice.Count)
                throw new IncorrectShapeException($"Given shape size ({shape.size}) does not match the size of the given storage size ({Count})");

            _Allocate(shape, arraySlice);
        }


#if _REGEN1
        %foreach supported_dtypes,supported_dtypes_lowercase%
        public UnmanagedStorage(#2 scalar)
        {            
            _dtype = typeof(#1);
            _typecode = InfoOf<#2>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _array#1 = ArraySlice.Scalar<#2>(scalar);
            unsafe
            {
                Address = (byte*)_array#1.Address;
                Count = _array#1.Count;
            }
        }

        %
#else
        public UnmanagedStorage(bool scalar)
        {            
            _dtype = typeof(Boolean);
            _typecode = InfoOf<bool>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayBoolean = ArraySlice.Scalar<bool>(scalar);
            unsafe
            {
                Address = (byte*)_arrayBoolean.Address;
                Count = _arrayBoolean.Count;
            }
        }

        public UnmanagedStorage(byte scalar)
        {            
            _dtype = typeof(Byte);
            _typecode = InfoOf<byte>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayByte = ArraySlice.Scalar<byte>(scalar);
            unsafe
            {
                Address = (byte*)_arrayByte.Address;
                Count = _arrayByte.Count;
            }
        }

        public UnmanagedStorage(int scalar)
        {            
            _dtype = typeof(Int32);
            _typecode = InfoOf<int>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayInt32 = ArraySlice.Scalar<int>(scalar);
            unsafe
            {
                Address = (byte*)_arrayInt32.Address;
                Count = _arrayInt32.Count;
            }
        }

        public UnmanagedStorage(long scalar)
        {            
            _dtype = typeof(Int64);
            _typecode = InfoOf<long>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayInt64 = ArraySlice.Scalar<long>(scalar);
            unsafe
            {
                Address = (byte*)_arrayInt64.Address;
                Count = _arrayInt64.Count;
            }
        }

        public UnmanagedStorage(float scalar)
        {            
            _dtype = typeof(Single);
            _typecode = InfoOf<float>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arraySingle = ArraySlice.Scalar<float>(scalar);
            unsafe
            {
                Address = (byte*)_arraySingle.Address;
                Count = _arraySingle.Count;
            }
        }

        public UnmanagedStorage(double scalar)
        {            
            _dtype = typeof(Double);
            _typecode = InfoOf<double>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayDouble = ArraySlice.Scalar<double>(scalar);
            unsafe
            {
                Address = (byte*)_arrayDouble.Address;
                Count = _arrayDouble.Count;
            }
        }
#endif
#if _REGEN
        %foreach supported_dtypes,supported_dtypes_lowercase%
        public UnmanagedStorage(#1[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(#1);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _array#1 = new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromArray(values));
            unsafe
            {
                Address = (byte*)_array#1.Address;
                Count = values.Length;
            }
        }
        %
#else
        public UnmanagedStorage(Boolean[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Boolean);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayBoolean = new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayBoolean.Address;
                Count = values.Length;
            }
        }
        public UnmanagedStorage(Byte[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Byte);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayByte = new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayByte.Address;
                Count = values.Length;
            }
        }
        public UnmanagedStorage(Int32[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Int32);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayInt32 = new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayInt32.Address;
                Count = values.Length;
            }
        }
        public UnmanagedStorage(Int64[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Int64);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayInt64 = new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayInt64.Address;
                Count = values.Length;
            }
        }
        public UnmanagedStorage(Single[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Single);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arraySingle = new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arraySingle.Address;
                Count = values.Length;
            }
        }
        public UnmanagedStorage(Double[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Double);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayDouble = new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayDouble.Address;
                Count = values.Length;
            }
        }
#endif

        #region Switched Accessing

        /// <summary>
        ///     Replace internal storage array with given array.
        /// </summary>
        /// <param name="array">The array to set as internal storage</param>
        /// <exception cref="InvalidCastException">When type of <paramref name="array"/> does not match <see cref="DType"/> of this storage</exception>
        protected unsafe void SetInternalArray(Array array)
        {
            switch (_typecode)
            {
#if _REGEN1
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                {
                    InternalArray = _array#1 = ArraySlice.FromArray<#2>((#2[])array);
                    Address = (byte*) _array#1.Address;
                    Count = _array#1.Count;
                    break;
                }
                %
                default:
                    throw new NotSupportedException();
#else
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                {
                    InternalArray = _arrayBoolean = ArraySlice.FromArray<bool>((bool[])array);
                    Address = (byte*) _arrayBoolean.Address;
                    Count = _arrayBoolean.Count;
                    break;
                }
                case NPTypeCode.Byte:
                {
                    InternalArray = _arrayByte = ArraySlice.FromArray<byte>((byte[])array);
                    Address = (byte*) _arrayByte.Address;
                    Count = _arrayByte.Count;
                    break;
                }
                case NPTypeCode.Int32:
                {
                    InternalArray = _arrayInt32 = ArraySlice.FromArray<int>((int[])array);
                    Address = (byte*) _arrayInt32.Address;
                    Count = _arrayInt32.Count;
                    break;
                }
                case NPTypeCode.Int64:
                {
                    InternalArray = _arrayInt64 = ArraySlice.FromArray<long>((long[])array);
                    Address = (byte*) _arrayInt64.Address;
                    Count = _arrayInt64.Count;
                    break;
                }
                case NPTypeCode.Single:
                {
                    InternalArray = _arraySingle = ArraySlice.FromArray<float>((float[])array);
                    Address = (byte*) _arraySingle.Address;
                    Count = _arraySingle.Count;
                    break;
                }
                case NPTypeCode.Double:
                {
                    InternalArray = _arrayDouble = ArraySlice.FromArray<double>((double[])array);
                    Address = (byte*) _arrayDouble.Address;
                    Count = _arrayDouble.Count;
                    break;
                }
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Replace internal storage array with given array.
        /// </summary>
        /// <param name="array">The array to set as internal storage</param>
        /// <exception cref="InvalidCastException">When type of <paramref name="array"/> does not match <see cref="DType"/> of this storage</exception>
        protected unsafe void SetInternalArray(IArraySlice array)
        {
            switch (_typecode)
            {
#if _REGEN1
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                {
                    InternalArray = _array#1 = (ArraySlice<#2>)array;
                    Address = (byte*) _array#1.Address;
                    Count = _array#1.Count;
                    break;
                }
                %
                default:
                    throw new NotSupportedException();
#else
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                {
                    InternalArray = _arrayBoolean = (ArraySlice<bool>)array;
                    Address = (byte*) _arrayBoolean.Address;
                    Count = _arrayBoolean.Count;
                    break;
                }
                case NPTypeCode.Byte:
                {
                    InternalArray = _arrayByte = (ArraySlice<byte>)array;
                    Address = _arrayByte.Address;
                    Count = _arrayByte.Count;
                    break;
                }
                case NPTypeCode.Char:
                {
                    InternalArray = _arrayChar = (ArraySlice<char>)array;
                    Address = (byte*)_arrayChar.Address;
                    Count = _arrayChar.Count;
                    break;
                }
                case NPTypeCode.Int32:
                {
                    InternalArray = _arrayInt32 = (ArraySlice<int>)array;
                    Address = (byte*) _arrayInt32.Address;
                    Count = _arrayInt32.Count;
                    break;
                }
                case NPTypeCode.Int64:
                {
                    InternalArray = _arrayInt64 = (ArraySlice<long>)array;
                    Address = (byte*) _arrayInt64.Address;
                    Count = _arrayInt64.Count;
                    break;
                }
                case NPTypeCode.Single:
                {
                    InternalArray = _arraySingle = (ArraySlice<float>)array;
                    Address = (byte*) _arraySingle.Address;
                    Count = _arraySingle.Count;
                    break;
                }
                case NPTypeCode.Double:
                {
                    InternalArray = _arrayDouble = (ArraySlice<double>)array;
                    Address = (byte*) _arrayDouble.Address;
                    Count = _arrayDouble.Count;
                    break;
                }
                default:
                    throw new NotSupportedException();
#endif
            }
        }

        #endregion

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

        /// <summary>
        ///     Changes the type of <paramref name="sourceArray"/> to <paramref name="to_dtype"/> if necessary.
        /// </summary>
        /// <param name="sourceArray">The array to change his type</param>
        /// <remarks>If the return type is equal to source type, this method does not return a copy.</remarks>
        /// <returns>Returns <see cref="sourceArray"/> or new array with changed type to <see cref="to_dtype"/></returns>
        [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
        protected static ArraySlice<TOut> _ChangeTypeOfArray<TOut>(IArraySlice sourceArray) where TOut : unmanaged
        {
            if (typeof(TOut) == sourceArray.GetType().GetElementType()) return (ArraySlice<TOut>)sourceArray;
            return (ArraySlice<TOut>)sourceArray.CastTo<TOut>();
        }

        #region Allocation

        protected void _Allocate(Shape shape, IArraySlice values)
        {
            //if (shape.IsSliced)
            //{
            //    values = values.Clone();
            //    shape = Shape.Clean();
            //}

            _shape = shape;
            _typecode = values.TypeCode;

            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{values.TypeCode} as a dtype is not supported.");

            _dtype = _typecode.AsType();
            SetInternalArray(values);
            Count = shape.size;
        }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, Type dtype = null)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            _Allocate(shape, ArraySlice.Allocate(dtype ?? DType, shape.size, true));
        }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, Type dtype, bool fillZeros)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            _Allocate(shape, ArraySlice.Allocate(dtype ?? DType, shape.size, fillZeros));
        }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, NPTypeCode dtype, bool fillZeros)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (dtype == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(dtype));

            _Allocate(shape, ArraySlice.Allocate(dtype, shape.size, fillZeros));
        }

        /// <summary>
        ///     Allocate <paramref name="array"/> into memory.
        /// </summary>
        /// <param name="array">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="array"/></remarks>
        public void Allocate(Array array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (array.Length == 0)
                throw new ArgumentException("values can't be an empty array", nameof(array));

            var slice = ArraySlice.FromArray(array);
            _Allocate(Shape.ExtractShape(array), slice);
        }

        /// <summary>
        ///     Assign this <see cref="ArraySlice{T}"/> as the internal array storage and assign <see cref="shape"/> to it.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="copy">Should perform a copy of <paramref name="values"/></param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate<T>(ArraySlice<T> values, Shape shape, bool copy = false) where T : unmanaged
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Count != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            _Allocate(shape, copy ? values.Clone() : values);
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="copy">Should perform a copy of <paramref name="values"/></param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate(IArraySlice values, Shape shape, bool copy = false)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Count != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            _Allocate(shape, (IArraySlice)(copy ? values.Clone() : values));
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        /// <param name="shape">The shape of given array</param>
        public void Allocate(Array values, Shape shape)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Length != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            Type elementType = values.GetType();
            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            _Allocate(shape, ArraySlice.FromArray(values));
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate<T>(T[] values) where T : unmanaged
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (values.Length == 0)
                throw new ArgumentException("values can't be an empty array", nameof(values));

            Shape shape;
            if (values.Rank > 1)
            {
                int[] dim = new int[values.Rank];
                for (int idx = 0; idx < dim.Length; idx++)
                    dim[idx] = values.GetLength(idx);
                shape = new Shape(dim);
            }
            else
            {
                shape = new Shape(values.Length);
            }

            Type elementType = values.GetType();
            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            _Allocate(shape, ArraySlice.FromArray(values));
        }

        #endregion


        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        public void CopyTo(IntPtr ptr)
        {
            unsafe
            {
                CopyTo(ptr.ToPointer());
            }
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        /// <param name="address">The address to copy to.</param>
        public unsafe void CopyTo(void* address)
        {
#if _REGEN1
            #region Compute

		    switch (TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1:
			    {
				    CopyTo<#2>((#2*)address);
                    break;
			    }

			    %
			    default:
				    throw new NotSupportedException();
		    }

            #endregion
#else

            #region Compute

		    switch (TypeCode)
		    {
			    case NPTypeCode.Boolean:
			    {
				    CopyTo<bool>((bool*)address);
                    break;
			    }

			    case NPTypeCode.Byte:
			    {
				    CopyTo<byte>((byte*)address);
                    break;
			    }

			    case NPTypeCode.Int32:
			    {
				    CopyTo<int>((int*)address);
                    break;
			    }

			    case NPTypeCode.Int64:
			    {
				    CopyTo<long>((long*)address);
                    break;
			    }

			    case NPTypeCode.Single:
			    {
				    CopyTo<float>((float*)address);
                    break;
			    }

			    case NPTypeCode.Double:
			    {
				    CopyTo<double>((double*)address);
                    break;
			    }

			    default:
				    throw new NotSupportedException();
		    }

            #endregion
#endif
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The block to copy to.</param>
        public unsafe void CopyTo(IMemoryBlock block)
        {
            if (block.TypeCode != _typecode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > block.Count)
                throw new ArgumentOutOfRangeException(nameof(block), $"Unable to copy from this storage to given memory block because this storage count is larger than the given memory block's length.");

#if _REGEN1
            #region Compute

		    switch (TypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1:
			    {
				    CopyTo<#2>((#2*)block.Address);
                    break;
			    }

			    %
			    default:
				    throw new NotSupportedException();
		    }

            #endregion
#else

            #region Compute

		    switch (TypeCode)
		    {
			    case NPTypeCode.Boolean:
			    {
				    CopyTo<bool>((bool*)block.Address);
                    break;
			    }

			    case NPTypeCode.Byte:
			    {
				    CopyTo<byte>((byte*)block.Address);
                    break;
			    }

			    case NPTypeCode.Int32:
			    {
				    CopyTo<int>((int*)block.Address);
                    break;
			    }

			    case NPTypeCode.Int64:
			    {
				    CopyTo<long>((long*)block.Address);
                    break;
			    }

			    case NPTypeCode.Single:
			    {
				    CopyTo<float>((float*)block.Address);
                    break;
			    }

			    case NPTypeCode.Double:
			    {
				    CopyTo<double>((double*)block.Address);
                    break;
			    }

			    default:
				    throw new NotSupportedException();
		    }

            #endregion
#endif
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The block to copy to.</param>
        public unsafe void CopyTo<T>(IMemoryBlock<T> block) where T : unmanaged
        {
            if (block.TypeCode != _typecode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > block.Count)
                throw new ArgumentOutOfRangeException(nameof(block), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            CopyTo<T>(block.Address);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        /// <param name="address">The address to copy to.</param>
        public unsafe void CopyTo<T>(T* address) where T : unmanaged
        {
            if (address == (T*)0)
                throw new ArgumentNullException(nameof(address));

            if (typeof(T) != _dtype)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (!Shape.IsContiguous || Shape.ModifiedStrides)
            {
                var dst = ArraySlice.Wrap<T>(address, Count);
                MultiIterator.Assign(new UnmanagedStorage(dst, Shape.Clean()), this);
                return;
            }

            var bytesCount = Count * InfoOf<T>.Size;
            Buffer.MemoryCopy(Address, address, bytesCount, bytesCount);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given array.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        public unsafe void CopyTo<T>(T[] array) where T : unmanaged
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (typeof(T) != _dtype)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > array.Length)
                throw new ArgumentOutOfRangeException(nameof(array), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            fixed (T* dst = array)
            {
                var bytesCount = Count * InfoOf<T>.Size;
                Buffer.MemoryCopy(Address, dst, bytesCount, bytesCount);
            }
        }

        [MethodImpl((MethodImplOptions)512)]
        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            if (typeof(T).GetTypeCode() != InternalArray.TypeCode)
                throw new ArrayTypeMismatchException($"The given type argument '{typeof(T).Name}' doesn't match the type of the internal data '{InternalArray.TypeCode}'");

            var src = (T*)Address;
            var ret = new T[Shape.Size];

            if (Shape.IsContiguous)
            {
                fixed (T* dst = ret)
                {
                    var len = sizeof(T) * ret.Length;
                    Buffer.MemoryCopy(src, dst, len, len);
                }
            }
            else
            {
                var incr = new NDCoordinatesIncrementor(Shape.dimensions);
                int[] current = incr.Index;
                Func<int[], int> getOffset = Shape.GetOffset;
                int i = 0;

                do ret[i++] = src[getOffset(current)];
                while (incr.Next() != null);
            }

            return ret;
        }
    }
}
