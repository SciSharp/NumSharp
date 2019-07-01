using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

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
    public class TypedArrayStorage : IStorage
    {
#if _REGEN
        %foreach supported_dtypes
        protected #1[] _array#1;
#else
        protected NDArray[] _arrayNDArray;
        protected Complex[] _arrayComplex;
        protected Boolean[] _arrayBoolean;
        protected Byte[] _arrayByte;
        protected Int16[] _arrayInt16;
        protected UInt16[] _arrayUInt16;
        protected Int32[] _arrayInt32;
        protected UInt32[] _arrayUInt32;
        protected Int64[] _arrayInt64;
        protected UInt64[] _arrayUInt64;
        protected Char[] _arrayChar;
        protected Double[] _arrayDouble;
        protected Single[] _arraySingle;
        protected Decimal[] _arrayDecimal;
        protected String[] _arrayString;
#endif

        protected Type _DType;
        protected NPTypeCode _typecode;
        protected Shape _Shape;
        protected Slice _slice; //todo! Unused? theres a similar property below with get and set

        /// <summary>
        /// Flag current storage order
        /// This flag will be different from Shape.Order when order is changed 
        /// </summary>
        private string order = "C";

        /// <summary>
        ///     Does this instance support spanning?
        /// </summary>
        public bool SupportsSpan => true;

        /// <summary>
        ///     The data type of internal storage array.
        /// </summary>
        /// <value>numpys equal dtype</value>
        /// <remarks>Has to be compliant with <see cref="NPTypeCode"/>.</remarks>
        public Type DType => _DType;

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of <see cref="IStorage.DType"/>.
        /// </summary>
        public NPTypeCode TypeCode => _typecode;

        /// <summary>
        ///     The size in bytes of a single value of <see cref="DType"/>
        /// </summary>
        /// <remarks>Computed by <see cref="Marshal.SizeOf(object)"/></remarks>
        public int DTypeSize
        {
            get
            {
                if (_typecode == NPTypeCode.NDArray || _typecode == NPTypeCode.String)
                {
                    return IntPtr.Size;
                }

                return Marshal.SizeOf(_DType);
            }
        }

        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>numpys equal shape</value>
        public Shape Shape => _Shape;

        /// <summary>
        ///     The current slice this <see cref="IStorage"/> instance currently represent.
        /// </summary>
        public Slice Slice { get; set; } //todo! shouldn't it be read-only?

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        public ITensorEngine Engine { get; internal set; }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="dtype"/>.
        /// </summary>
        /// <param name="dtype">The type of this storage</param>
        public TypedArrayStorage(Type dtype)
        {
            _DType = dtype ?? throw new ArgumentNullException(nameof(dtype));
            _typecode = dtype.GetTypeCode();
            _Shape = new Shape(0);
        }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="typeCode"/>.
        /// </summary>
        /// <param name="typeCode">The type of this storage</param>
        public TypedArrayStorage(NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty) throw new ArgumentNullException(nameof(typeCode));
            _DType = typeCode.AsType();
            _typecode = typeCode;
            _Shape = new Shape(0);
        }

        //todo! create scalar constuctors?

#if _REGEN
        %foreach supported_dtypes%
        public TypedArrayStorage(#1[] values)
        {            
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(#1);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _array#1 = values;
        }
        %
#else
        public TypedArrayStorage(NDArray[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(NDArray);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayNDArray = values;
        }

        public TypedArrayStorage(Complex[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Complex);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayComplex = values;
        }

        public TypedArrayStorage(Boolean[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Boolean);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayBoolean = values;
        }

        public TypedArrayStorage(Byte[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Byte);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayByte = values;
        }

        public TypedArrayStorage(Int16[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Int16);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayInt16 = values;
        }

        public TypedArrayStorage(UInt16[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(UInt16);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayUInt16 = values;
        }

        public TypedArrayStorage(Int32[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Int32);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayInt32 = values;
        }

        public TypedArrayStorage(UInt32[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(UInt32);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayUInt32 = values;
        }

        public TypedArrayStorage(Int64[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Int64);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayInt64 = values;
        }

        public TypedArrayStorage(UInt64[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(UInt64);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayUInt64 = values;
        }

        public TypedArrayStorage(Char[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Char);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayChar = values;
        }

        public TypedArrayStorage(Double[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Double);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayDouble = values;
        }

        public TypedArrayStorage(Single[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Single);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arraySingle = values;
        }

        public TypedArrayStorage(Decimal[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(Decimal);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayDecimal = values;
        }

        public TypedArrayStorage(String[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            _DType = typeof(String);
            _typecode = _DType.GetTypeCode();
            _Shape = new Shape(values.Length);
            _arrayString = values;
        }
#endif

        #region Switched Accessing

        /// <summary>
        ///     Replace internal storage array with given array.
        /// </summary>
        /// <param name="array">The array to set as internal storage</param>
        /// <exception cref="InvalidCastException">When type of <paramref name="array"/> does not match <see cref="DType"/> of this storage</exception>
        protected void SetInternalArray(Array array)
        {
            switch (_typecode)
            {
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                {
                    _array#1 = (#2[]) array;
                    break;
                }
                %
                default:
                    throw new NotImplementedException();
#else
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.NDArray:
                {
                    _arrayNDArray = (NDArray[])array;
                    break;
                }

                case NPTypeCode.Complex:
                {
                    _arrayComplex = (Complex[])array;
                    break;
                }

                case NPTypeCode.Boolean:
                {
                    _arrayBoolean = (bool[])array;
                    break;
                }

                case NPTypeCode.Byte:
                {
                    _arrayByte = (byte[])array;
                    break;
                }

                case NPTypeCode.Int16:
                {
                    _arrayInt16 = (short[])array;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    _arrayUInt16 = (ushort[])array;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    _arrayInt32 = (int[])array;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    _arrayUInt32 = (uint[])array;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    _arrayInt64 = (long[])array;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    _arrayUInt64 = (ulong[])array;
                    break;
                }

                case NPTypeCode.Char:
                {
                    _arrayChar = (char[])array;
                    break;
                }

                case NPTypeCode.Double:
                {
                    _arrayDouble = (double[])array;
                    break;
                }

                case NPTypeCode.Single:
                {
                    _arraySingle = (float[])array;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    _arrayDecimal = (decimal[])array;
                    break;
                }

                case NPTypeCode.String:
                {
                    _arrayString = (string[])array;
                    break;
                }

                default:
                    throw new NotImplementedException();
#endif
            }
        }

        /// <summary>
        ///     Gets a single value from current 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected object GetInternalValue(int index)
        {
            switch (_typecode)
            {
#if _REGEN
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                {
                    return _array#1[index];
                    
                }
                %
                default:
                    throw new NotImplementedException();
#else

                case NPTypeCode.NDArray:
                {
                    return _arrayNDArray[index];
                }

                case NPTypeCode.Complex:
                {
                    return _arrayComplex[index];
                }

                case NPTypeCode.Boolean:
                {
                    return _arrayBoolean[index];
                }

                case NPTypeCode.Byte:
                {
                    return _arrayByte[index];
                }

                case NPTypeCode.Int16:
                {
                    return _arrayInt16[index];
                }

                case NPTypeCode.UInt16:
                {
                    return _arrayUInt16[index];
                }

                case NPTypeCode.Int32:
                {
                    return _arrayInt32[index];
                }

                case NPTypeCode.UInt32:
                {
                    return _arrayUInt32[index];
                }

                case NPTypeCode.Int64:
                {
                    return _arrayInt64[index];
                }

                case NPTypeCode.UInt64:
                {
                    return _arrayUInt64[index];
                }

                case NPTypeCode.Char:
                {
                    return _arrayChar[index];
                }

                case NPTypeCode.Double:
                {
                    return _arrayDouble[index];
                }

                case NPTypeCode.Single:
                {
                    return _arraySingle[index];
                }

                case NPTypeCode.Decimal:
                {
                    return _arrayDecimal[index];
                }

                case NPTypeCode.String:
                {
                    return _arrayString[index];
                }

                default:
                    throw new NotImplementedException();
#endif
            }
        }

        /// <summary>
        ///     Set value in current active array, if types do not match the <see cref="Convert"/> is used.
        /// </summary>
        /// <param name="index">The index to set <paramref name="value"/> at</param>
        /// <param name="value">The value to set</param>
        protected void SetOrConvertInternalValue(object value, int index)
        {
            switch (_typecode)
            {
#if _REGEN
                //Based on benchmark `ArrayAssignmentUnspecifiedType`

                %foreach supported_primitives,supported_primitives_lowercase%
                case NPTypeCode.#1:
                {
                    _array#1[index] = Convert.To#1(value);
                    break;
                    
                }
                %
#else
                //Based on benchmark `ArrayAssignmentUnspecifiedType`

                case NPTypeCode.Boolean:
                {
                    _arrayBoolean[index] = Convert.ToBoolean(value);
                    break;
                }

                case NPTypeCode.Byte:
                {
                    _arrayByte[index] = Convert.ToByte(value);
                    break;
                }

                case NPTypeCode.Int16:
                {
                    _arrayInt16[index] = Convert.ToInt16(value);
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    _arrayUInt16[index] = Convert.ToUInt16(value);
                    break;
                }

                case NPTypeCode.Int32:
                {
                    _arrayInt32[index] = Convert.ToInt32(value);
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    _arrayUInt32[index] = Convert.ToUInt32(value);
                    break;
                }

                case NPTypeCode.Int64:
                {
                    _arrayInt64[index] = Convert.ToInt64(value);
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    _arrayUInt64[index] = Convert.ToUInt64(value);
                    break;
                }

                case NPTypeCode.Char:
                {
                    _arrayChar[index] = Convert.ToChar(value);
                    break;
                }

                case NPTypeCode.Double:
                {
                    _arrayDouble[index] = Convert.ToDouble(value);
                    break;
                }

                case NPTypeCode.Single:
                {
                    _arraySingle[index] = Convert.ToSingle(value);
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    _arrayDecimal[index] = Convert.ToDecimal(value);
                    break;
                }

                case NPTypeCode.String:
                {
                    _arrayString[index] = Convert.ToString(value);
                    break;
                }
#endif
                case NPTypeCode.NDArray:
                {
                    _arrayNDArray[index] = (NDArray)value; //try explicit casting
                    break;
                }

                case NPTypeCode.Complex:
                {
                    _arrayComplex[index] = value is Complex c ? c : new Complex(Convert.ToDouble(value), 0d);
                    break;
                }

                default:
                    throw new NotImplementedException();
            }
        }

        /// <summary>
        ///     Set value in current active array, if types do not match then <see cref="InvalidCastException"/> will be thrown.
        /// </summary>
        /// <param name="index">The index to set <paramref name="value"/> at</param>
        /// <param name="value">The value to set</param>
        /// <exception cref="InvalidCastException">When <see cref="value"/>'s type does not equal to <see cref="DType"/>.</exception>
        protected void SetInternalValueUnsafe(object value, int index)
        {
            switch (_typecode)
            {
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_primitives,supported_primitives_lowercase%
                case NPTypeCode.#1:
                {
                    _array#1[index] = (#1) value;
                    break;
                    
                }
                %
#else

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                {
                    _arrayBoolean[index] = (Boolean)value;
                    break;
                }

                case NPTypeCode.Byte:
                {
                    _arrayByte[index] = (Byte)value;
                    break;
                }

                case NPTypeCode.Int16:
                {
                    _arrayInt16[index] = (Int16)value;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    _arrayUInt16[index] = (UInt16)value;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    _arrayInt32[index] = (Int32)value;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    _arrayUInt32[index] = (UInt32)value;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    _arrayInt64[index] = (Int64)value;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    _arrayUInt64[index] = (UInt64)value;
                    break;
                }

                case NPTypeCode.Char:
                {
                    _arrayChar[index] = (Char)value;
                    break;
                }

                case NPTypeCode.Double:
                {
                    _arrayDouble[index] = (Double)value;
                    break;
                }

                case NPTypeCode.Single:
                {
                    _arraySingle[index] = (Single)value;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    _arrayDecimal[index] = (Decimal)value;
                    break;
                }

                case NPTypeCode.String:
                {
                    _arrayString[index] = (String)value;
                    break;
                }
#endif
                case NPTypeCode.NDArray:
                {
                    _arrayNDArray[index] = (NDArray)value; //try explicit casting
                    break;
                }

                case NPTypeCode.Complex:
                {
                    _arrayComplex[index] = (Complex)value;
                    break;
                }

                default:
                    throw new NotImplementedException();
            }
        }

        /// <summary>
        ///     Gets the internal array based on <see cref="type"/>.
        /// </summary>
        /// <returns>Will return null if <see cref="type"/> != <see cref="DType"/></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected Array GetInternalArray(Type type)
        {
            return GetInternalArray(type.GetTypeCode());
        }

        /// <summary>
        ///     Gets the internal array based on <see cref="DType"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected Array GetInternalArray()
        {
            return GetInternalArray(_typecode);
        }

        /// <summary>
        ///     Gets the internal array based on <see cref="type"/>.
        /// </summary>
        /// <returns>Will return null if <see cref="typeCode"/> != <see cref="DType"/></returns>
        protected Array GetInternalArray(NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if _REGEN
                %foreach supported_dtypes%
                case NPTypeCode.#1:
                {
                    return _array#1;
                }
                %
                default:
                    throw new NotImplementedException();
#else

                case NPTypeCode.NDArray:
                {
                    return _arrayNDArray;
                }

                case NPTypeCode.Complex:
                {
                    return _arrayComplex;
                }

                case NPTypeCode.Boolean:
                {
                    return _arrayBoolean;
                }

                case NPTypeCode.Byte:
                {
                    return _arrayByte;
                }

                case NPTypeCode.Int16:
                {
                    return _arrayInt16;
                }

                case NPTypeCode.UInt16:
                {
                    return _arrayUInt16;
                }

                case NPTypeCode.Int32:
                {
                    return _arrayInt32;
                }

                case NPTypeCode.UInt32:
                {
                    return _arrayUInt32;
                }

                case NPTypeCode.Int64:
                {
                    return _arrayInt64;
                }

                case NPTypeCode.UInt64:
                {
                    return _arrayUInt64;
                }

                case NPTypeCode.Char:
                {
                    return _arrayChar;
                }

                case NPTypeCode.Double:
                {
                    return _arrayDouble;
                }

                case NPTypeCode.Single:
                {
                    return _arraySingle;
                }

                case NPTypeCode.Decimal:
                {
                    return _arrayDecimal;
                }

                case NPTypeCode.String:
                {
                    return _arrayString;
                }

                default:
                    throw new NotImplementedException();
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

        #region Allocation

        protected void _Allocate(Shape shape, Type dtype, Array values)
        {
            _Shape = shape;

            if (dtype != null)
            {
                _DType = dtype;
                _typecode = _DType.GetTypeCode();
                if (_typecode == NPTypeCode.Empty)
                    throw new NotSupportedException($"{dtype.Name} as a dtype is not supported.");
            }

            SetInternalArray(values);
        }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, Type dtype = null)
        {
            dtype = dtype ?? DType;
            _Allocate(shape, dtype, Arrays.Create(dtype, new int[] {shape.Size}));
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate(Array values)
        {
            if (values == null)
            {
                throw new ArgumentNullException(nameof(values));
            }

            Shape shape;
            //get lengths incase it is multi-dimensional
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

            _Allocate(shape, elementType, values);
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
            {
                throw new ArgumentNullException(nameof(values));
            }

            Type elementType = values.GetType();
            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            _Allocate(shape, elementType, values);
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate<T>(T[] values)
        {
            if (values == null)
            {
                throw new ArgumentNullException(nameof(values));
            }

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

            _Allocate(shape, elementType, values);
        }

        #endregion

        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Array GetData()
        {
            #region Commented Out

            /*
             if(order != Shape.Order)
            {
                if (Shape.Order == "F")
                {
                    var s = new Shape(Shape.Dimensions);
                    s.ChangeTensorLayout("C");

                    switch (Type.GetTypeCode(DType))
                    {
                        case TypeCode.Int32:
                            {
                                var array = new int[Shape.Size];
                                switch (Shape.NDim)
                                {
                                    case 2:
                                        for (int row = 0; row < Shape[0]; row++)
                                            for (int col = 0; col < Shape[1]; col++)
                                                array[s.GetIndexInShape(row, col)] = _arrayInt32[Shape.GetIndexInShape(row, col)];
                                        _arrayInt32 = array;
                                        break;
                                    default:
                                        throw new NotImplementedException("Array GetData() Order F Changed.");
                                }
                            }
                            break;
                        case TypeCode.Single:
                            {
                                var array = new float[Shape.Size];
                                switch (Shape.NDim)
                                {
                                    case 2:
                                        for (int row = 0; row < Shape[0]; row++)
                                            for (int col = 0; col < Shape[1]; col++)
                                                array[s.GetIndexInShape(row, col)] = _arraySingle[Shape.GetIndexInShape(row, col)];
                                        _arraySingle = array;
                                        break;
                                    default:
                                        throw new NotImplementedException("Array GetData() Order F Changed.");
                                }
                            }
                            break;
                    }

                    Shape.ChangeTensorLayout("C");
                    order = Shape.Order;
                }
                else if (Shape.Order == "C")
                {
                    throw new NotImplementedException("Array GetData() Order C Changed.");
                }

                
            }*/

            #endregion

            return GetInternalArray(DType);
        }

        /// <summary>
        ///     Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        public Array CloneData()
        {
            return ArrayConvert.Clone(GetInternalArray());
        }

        /// <summary>
        ///     Get reference to internal data storage and cast (also copies) elements to new dtype if necessary
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        public T[] GetData<T>()
        {
            if (!typeof(T).IsValidNPType())
            {
                throw new NotSupportedException($"Type {typeof(T).Name} is not a valid np.dtype");
            }

            //if (typeof(T).Name != _DType.Name)
            //{
            //    Console.WriteLine($"Warning: GetData {typeof(T).Name} is not {_DType.Name} of storage.");
            //}

            var internalArray = GetInternalArray();
            if (internalArray is T[] ret)
                return ret;

            return (T[])_ChangeTypeOfArray(internalArray, typeof(T));
        }

        /// <summary>
        ///     Attempts to cast internal storage to an array of type <typeparamref name="T"/> and returns the result.
        /// </summary>
        /// <typeparam name="T">The type that is expected.</typeparam>
        public T[] AsArray<T>()
        {
            if (!typeof(T).IsValidNPType())
            {
                throw new NotSupportedException($"Type {typeof(T).Name} is not a valid np.dtype");
            }

            return (T[])GetInternalArray(DType);
        }

        /// <summary>
        ///     Get all elements from cloned storage as T[] and cast dtype
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage as T[]</returns>
        public T[] CloneData<T>()
        {
            // This will perform a conversion and/or copy.
            return ArrayConvert.To<T>(GetInternalArray(DType));
        }

        /// <summary>
        ///     Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indices">indices</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        /// <exception cref="NullReferenceException">When <typeparamref name="T"/> does not equal to <see cref="DType"/></exception>
        public T GetData<T>(params int[] indices)
        {
            return AsArray<T>()[Shape.GetIndexInShape(Slice, indices)];
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
            //todo! This is seriously performance heavy for setting a single value to a specific index.

            if (value == null)
                throw new ArgumentNullException(nameof(value));

            if (indices.Length == 0)
                throw new ArgumentException("indices cannot be an empty collection.", nameof(indices));


            if (value is NDArray nd)
            {
                if (np.isscalar(nd))
                {
                    SetInternalValueUnsafe(nd.Array.GetValue(0), _Shape.GetIndexInShape(Slice, indices));
                    return;
                }

                var targetIndex = _Shape.GetIndexInShape(Slice, indices);

                int offset = 0;
                if (Shape.NDim == 1)
                    offset = targetIndex + (Slice?.Start ?? 0);
                else
                    offset = targetIndex + (Slice?.Start ?? 0) * Shape.Strides[0];
                if (offset != 0)
                {
                    var from = nd.Array;
                    Array.Copy(from, 0, GetInternalArray(DType), offset, from.Length);
                    return;
                }

                value = nd.Array;
                //fall back down
            }

            SetOrConvertInternalValue(value, _Shape.GetIndexInShape(Slice, indices));
        }

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

            SetInternalArray(_ChangeTypeOfArray(values, values.GetType().GetElementType()));
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
            _DType = dtype;
            _typecode = _DType.GetTypeCode();
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
            _DType = dtype;
            _typecode = _DType.GetTypeCode();
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
            _Shape = nd.shape;
            _DType = nd.dtype;
            _typecode = _DType.GetTypeCode(); 
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{_DType.Name} as a dtype is not supported.");
            SetInternalArray(nd.Array);
        }

        public void Reshape(params int[] dimensions)
        {
            //todo! Shouldnt there be some verification regarding if shape is valid for this type or now?
            _Shape = new Shape(dimensions);
        }

        public Span<T> View<T>(Slice slice = null)
        {
            if (slice is null)
                slice = Slice;

            if (slice is null)
            {
                return GetData<T>().AsSpan();
            }
            else
            {
                var shape = Shape.GetShape(Shape.Dimensions, axis: 0);
                var offset = Shape.GetSize(shape);
                return GetData<T>().AsSpan(slice.Start.Value * offset, slice.Length.Value * offset);
            }
        }

        public Span<T> GetSpanData<T>(Slice slice, params int[] indice)
        {
            int stride = Shape.NDim == 0 ? 1 : Shape.Strides[indice.Length - 1];
            int idx = Shape.GetIndexInShape(Slice, indice);
            int offset = idx + (Slice is null ? 0 : Slice.Start.Value) * Shape.Strides[0];

            return GetData<T>().AsSpan(offset, stride);
        }

        public TypedArrayStorage Clone()
        {
            var puffer = (TypedArrayStorage)Engine.GetStorage(_DType);
            puffer.Allocate(_Shape.Clone()); //allocate is necessary if non-C# memory storage is used.
            puffer.ReplaceData((Array)GetData().Clone()); //todo! check if theres a faster way to clone.

            return puffer;
        }

        object ICloneable.Clone()
        {
            return Clone();
        }

        #region Getters

#if _REGEN
        %foreach supported_dtypes,supported_dtypes_lowercase%
        /// <summary>
        ///     Retrieves value of type <see cref="#2"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="#2"/></exception>
        public #2 Get#1(params int[] indices)
            => _array#1[Shape.GetIndexInShape(Slice, indices)];

        %
#else

        /// <summary>
        ///     Retrieves value of type <see cref="NDArray"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="NDArray"/></exception>
        public NDArray GetNDArray(params int[] indices)
            => _arrayNDArray[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="Complex"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="Complex"/></exception>
        public Complex GetComplex(params int[] indices)
            => _arrayComplex[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="bool"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="bool"/></exception>
        public bool GetBoolean(params int[] indices)
            => _arrayBoolean[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="byte"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        public byte GetByte(params int[] indices)
            => _arrayByte[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="short"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        public short GetInt16(params int[] indices)
            => _arrayInt16[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        public ushort GetUInt16(params int[] indices)
            => _arrayUInt16[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="int"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        public int GetInt32(params int[] indices)
            => _arrayInt32[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        public uint GetUInt32(params int[] indices)
            => _arrayUInt32[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="long"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        public long GetInt64(params int[] indices)
            => _arrayInt64[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        public ulong GetUInt64(params int[] indices)
            => _arrayUInt64[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="char"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        public char GetChar(params int[] indices)
            => _arrayChar[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="double"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        public double GetDouble(params int[] indices)
            => _arrayDouble[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="float"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        public float GetSingle(params int[] indices)
            => _arraySingle[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        public decimal GetDecimal(params int[] indices)
            => _arrayDecimal[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="string"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="string"/></exception>
        public string GetString(params int[] indices)
            => _arrayString[Shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="IStorage.DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="IStorage.DType"/> is not <see cref="object"/></exception>
        public object GetValue(params int[] indices)
        {
            return GetInternalValue(Shape.GetIndexInShape(Slice, indices));
        }

#endif

        #endregion
    }
}
