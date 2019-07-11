using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;
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
    public partial class UnmanagedStorage : IStorage
    {
#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
        protected ArraySlice<#2> _array#1;
#else
        protected ArraySlice<byte> _arrayByte;
        protected ArraySlice<short> _arrayInt16;
        protected ArraySlice<ushort> _arrayUInt16;
        protected ArraySlice<int> _arrayInt32;
        protected ArraySlice<uint> _arrayUInt32;
        protected ArraySlice<long> _arrayInt64;
        protected ArraySlice<ulong> _arrayUInt64;
        protected ArraySlice<char> _arrayChar;
        protected ArraySlice<double> _arrayDouble;
        protected ArraySlice<float> _arraySingle;
        protected ArraySlice<decimal> _arrayDecimal;
#endif
        public IArraySlice InternalArray;
        public unsafe byte* Address;
        public int Count;

        protected Type _dtype;
        protected NPTypeCode _typecode;

        protected Shape _shape;
        //protected Slice _slice; //todo! Unused? theres a similar property below with get and set

        /// <summary>
        ///     Does this instance support spanning?
        /// </summary>
        public bool SupportsSpan => true; //TODO! now that we always support spanning, I think we should remove it.

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
        {
            get
            {
                if (_typecode == NPTypeCode.NDArray || _typecode == NPTypeCode.String)
                {
                    return IntPtr.Size;
                }

                return Marshal.SizeOf(_dtype);
            }
        }

        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>numpys equal shape</value>
        public Shape Shape => _shape;

        public Span<T> AsSpan<T>()
        {
            unsafe
            {
                return new Span<T>(Address, Count);
            }
        }

        /// <summary>
        ///     The current slice this <see cref="IStorage"/> instance currently represent.
        /// </summary>
        public Slice Slice { get; set; } //todo! shouldn't it be read-only?

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        public TensorEngine Engine { get; internal set; }


        public static UnmanagedStorage Scalar<T>(T value) where T : unmanaged
        {
            return new UnmanagedStorage(ArraySlice.Scalar(value));
        }

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
            _shape = new Shape(0);
        }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="dtype"/>.
        /// </summary>
        /// <param name="dtype">The type of this storage</param>
        public UnmanagedStorage(object value)
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

            _Allocate(shape, arraySlice);
        }


        //todo! create scalar constuctors?
#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
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

        public UnmanagedStorage(short scalar)
        {
            _dtype = typeof(Int16);
            _typecode = InfoOf<short>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayInt16 = ArraySlice.Scalar<short>(scalar);
            unsafe
            {
                Address = (byte*)_arrayInt16.Address;
                Count = _arrayInt16.Count;
            }
        }

        public UnmanagedStorage(ushort scalar)
        {
            _dtype = typeof(UInt16);
            _typecode = InfoOf<ushort>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayUInt16 = ArraySlice.Scalar<ushort>(scalar);
            unsafe
            {
                Address = (byte*)_arrayUInt16.Address;
                Count = _arrayUInt16.Count;
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

        public UnmanagedStorage(uint scalar)
        {
            _dtype = typeof(UInt32);
            _typecode = InfoOf<uint>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayUInt32 = ArraySlice.Scalar<uint>(scalar);
            unsafe
            {
                Address = (byte*)_arrayUInt32.Address;
                Count = _arrayUInt32.Count;
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

        public UnmanagedStorage(ulong scalar)
        {
            _dtype = typeof(UInt64);
            _typecode = InfoOf<ulong>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayUInt64 = ArraySlice.Scalar<ulong>(scalar);
            unsafe
            {
                Address = (byte*)_arrayUInt64.Address;
                Count = _arrayUInt64.Count;
            }
        }

        public UnmanagedStorage(char scalar)
        {
            _dtype = typeof(Char);
            _typecode = InfoOf<char>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayChar = ArraySlice.Scalar<char>(scalar);
            unsafe
            {
                Address = (byte*)_arrayChar.Address;
                Count = _arrayChar.Count;
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

        public UnmanagedStorage(decimal scalar)
        {
            _dtype = typeof(Decimal);
            _typecode = InfoOf<decimal>.NPTypeCode;
            _shape = Shape.Scalar;
            InternalArray = _arrayDecimal = ArraySlice.Scalar<decimal>(scalar);
            unsafe
            {
                Address = (byte*)_arrayDecimal.Address;
                Count = _arrayDecimal.Count;
            }
        }
#endif
#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
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

        public UnmanagedStorage(Int16[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Int16);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayInt16 = new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayInt16.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(UInt16[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(UInt16);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayUInt16 = new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayUInt16.Address;
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

        public UnmanagedStorage(UInt32[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(UInt32);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayUInt32 = new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayUInt32.Address;
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

        public UnmanagedStorage(UInt64[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(UInt64);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayUInt64 = new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayUInt64.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Char[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Char);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayChar = new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayChar.Address;
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

        public UnmanagedStorage(Decimal[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Decimal);
            _typecode = _dtype.GetTypeCode();
            _shape = new Shape(values.Length);
            InternalArray = _arrayDecimal = new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromArray(values));
            unsafe
            {
                Address = (byte*)_arrayDecimal.Address;
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
        protected void SetInternalArray(Array array)
        {
            switch (_typecode)
            {
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    InternalArray = _array#1 = ArraySlice.FromArray<#2>((#2[])array);
                    break;
                }
                %
                default:
                    throw new NotSupportedException();
#else
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Byte:
                {
                    InternalArray = _arrayByte = ArraySlice.FromArray<byte>((byte[])array);
                    break;
                }

                case NPTypeCode.Int16:
                {
                    InternalArray = _arrayInt16 = ArraySlice.FromArray<short>((short[])array);
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    InternalArray = _arrayUInt16 = ArraySlice.FromArray<ushort>((ushort[])array);
                    break;
                }

                case NPTypeCode.Int32:
                {
                    InternalArray = _arrayInt32 = ArraySlice.FromArray<int>((int[])array);
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    InternalArray = _arrayUInt32 = ArraySlice.FromArray<uint>((uint[])array);
                    break;
                }

                case NPTypeCode.Int64:
                {
                    InternalArray = _arrayInt64 = ArraySlice.FromArray<long>((long[])array);
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    InternalArray = _arrayUInt64 = ArraySlice.FromArray<ulong>((ulong[])array);
                    break;
                }

                case NPTypeCode.Char:
                {
                    InternalArray = _arrayChar = ArraySlice.FromArray<char>((char[])array);
                    break;
                }

                case NPTypeCode.Double:
                {
                    InternalArray = _arrayDouble = ArraySlice.FromArray<double>((double[])array);
                    break;
                }

                case NPTypeCode.Single:
                {
                    InternalArray = _arraySingle = ArraySlice.FromArray<float>((float[])array);
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    InternalArray = _arrayDecimal = ArraySlice.FromArray<decimal>((decimal[])array);
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
        protected void SetInternalArray(IArraySlice array)
        {
            switch (_typecode)
            {
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    InternalArray = _array#1 = (ArraySlice<#2>)array;
                    break;
                }
                %
                default:
                    throw new NotSupportedException();
#else
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Byte:
                {
                    InternalArray = _arrayByte = (ArraySlice<byte>)array;
                    break;
                }

                case NPTypeCode.Int16:
                {
                    InternalArray = _arrayInt16 = (ArraySlice<short>)array;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    InternalArray = _arrayUInt16 = (ArraySlice<ushort>)array;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    InternalArray = _arrayInt32 = (ArraySlice<int>)array;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    InternalArray = _arrayUInt32 = (ArraySlice<uint>)array;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    InternalArray = _arrayInt64 = (ArraySlice<long>)array;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    InternalArray = _arrayUInt64 = (ArraySlice<ulong>)array;
                    break;
                }

                case NPTypeCode.Char:
                {
                    InternalArray = _arrayChar = (ArraySlice<char>)array;
                    break;
                }

                case NPTypeCode.Double:
                {
                    InternalArray = _arrayDouble = (ArraySlice<double>)array;
                    break;
                }

                case NPTypeCode.Single:
                {
                    InternalArray = _arraySingle = (ArraySlice<float>)array;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    InternalArray = _arrayDecimal = (ArraySlice<decimal>)array;
                    break;
                }

                default:
                    throw new NotSupportedException();
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
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    return _array#1[index];
                }
                %
                default:
                    throw new NotSupportedException();
#else
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

                default:
                    throw new NotSupportedException();
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

                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    _array#1[index] = Convert.To#1(value);
                    break;
                    
                }
                %
#else
                //Based on benchmark `ArrayAssignmentUnspecifiedType`

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
#endif

                default:
                    throw new NotImplementedException();
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
        /// <param name="to_dtype">The type to change to.</param>
        /// <remarks>If the return type is equal to source type, this method does not return a copy.</remarks>
        /// <returns>Returns <see cref="sourceArray"/> or new array with changed type to <see cref="to_dtype"/></returns>
        [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
        protected static IArraySlice _ChangeTypeOfArray<TOut>(IArraySlice sourceArray) where TOut : unmanaged
        {
            //TODO! unit test this.
            if (typeof(TOut) == sourceArray.GetType().GetElementType()) return sourceArray;
            switch (InfoOf<TOut>.NPTypeCode)
            {
#if _REGEN
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    return new ArraySlice<TOut>(UnmanagedArray.Cast<#1, TOut>(sourceArray.MemoryBlock));
                }
                %
                default:
                    throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Byte, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Int16:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Int16, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.UInt16:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<UInt16, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Int32:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Int32, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.UInt32:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<UInt32, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Int64:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Int64, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.UInt64:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<UInt64, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Char:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Char, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Double:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Double, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Single:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Single, TOut>(sourceArray.MemoryBlock));
                }

                case NPTypeCode.Decimal:
                {
                    return new ArraySlice<TOut>(UnmanagedMemoryBlock.Cast<Decimal, TOut>(sourceArray.MemoryBlock));
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }

        #region Allocation

        protected void _Allocate(Shape shape, IArraySlice values)
        {
            _shape = shape;
            _typecode = values.TypeCode;

            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{values.TypeCode} as a dtype is not supported.");

            _dtype = _typecode.AsType();
            Count = values.Count;
            unsafe
            {
                Address = (byte*)values.Address;
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
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            _Allocate(shape, ArraySlice.FromArray(Arrays.Create(dtype ?? DType, new int[] {shape.Size})));
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate(Array values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (values.Length == 0)
                throw new ArgumentException("values can't be an empty array", nameof(values));

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
            var slice = ArraySlice.FromArray(values);
            _Allocate(shape, slice);
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
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IArraySlice GetData()
        {
            return InternalArray;
        }

        /// <summary>
        ///     Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        public IArraySlice CloneData()
        {
            return InternalArray.Clone();
        }


        /// <summary>
        ///     Attempts to cast internal storage to an array of type <typeparamref name="T"/> and returns the result, therefore results can be null.
        /// </summary>
        /// <typeparam name="T">The type that is expected.</typeparam>
        public ArraySlice<T> AsArray<T>() where T : unmanaged
        {
            //if (!typeof(T).IsValidNPType())
            return (ArraySlice<T>)InternalArray;
        }

        /// <summary>
        ///     Get all elements from cloned storage as T[] and cast dtype
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage as <see cref="ArraySlice{T}"/></returns>
        public ArraySlice<T> CloneData<T>() where T : unmanaged
        {
            return (ArraySlice<T>)InternalArray.Clone();
        }

        /// <summary>
        ///     Get reference to internal data storage and cast (also copies) elements to new dtype if necessary
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        public ArraySlice<T> GetData<T>() where T : unmanaged
        {
            if (!typeof(T).IsValidNPType())
            {
                throw new NotSupportedException($"Type {typeof(T).Name} is not a valid np.dtype");
            }

            //if (typeof(T).Name != _DType.Name)
            //{
            //    Console.WriteLine($"Warning: GetData {typeof(T).Name} is not {_DType.Name} of storage.");
            //}

            var internalArray = InternalArray;
            if (internalArray is ArraySlice<T> ret)
                return ret;

            return (ArraySlice<T>)_ChangeTypeOfArray<T>(internalArray);
        }

        /// <summary>
        ///     Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indices">indices</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        /// <exception cref="NullReferenceException">When <typeparamref name="T"/> does not equal to <see cref="DType"/></exception>
        public T GetData<T>(params int[] indices) where T : unmanaged
        {
            unsafe
            {
                return *((T*)Address + _shape.GetIndexInShape(Slice, indices));
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
        public unsafe void SetData<T>(T value, params int[] indices) where T : unmanaged
        {
            *((T*)this.Address + _shape.GetIndexInShape(indices)) = value;
        }

        public unsafe void SetIndex<T>(T value, int index) where T : unmanaged
        {
            *((T*)this.Address + index) = value;
        }

        public unsafe void SetIndex(object value, int index)
        {
            switch (_typecode)
            {
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    *((#2*)Address + index) = (#2) value;
                    break;
                    
                }
                %
                default:
                    throw new NotSupportedException();
#else

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Byte:
                    {
                        *((byte*)Address + index) = (byte)value;
                        break;
                    }

                case NPTypeCode.Int16:
                    {
                        *((short*)Address + index) = (short)value;
                        break;
                    }

                case NPTypeCode.UInt16:
                    {
                        *((ushort*)Address + index) = (ushort)value;
                        break;
                    }

                case NPTypeCode.Int32:
                    {
                        *((int*)Address + index) = (int)value;
                        break;
                    }

                case NPTypeCode.UInt32:
                    {
                        *((uint*)Address + index) = (uint)value;
                        break;
                    }

                case NPTypeCode.Int64:
                    {
                        *((long*)Address + index) = (long)value;
                        break;
                    }

                case NPTypeCode.UInt64:
                    {
                        *((ulong*)Address + index) = (ulong)value;
                        break;
                    }

                case NPTypeCode.Char:
                    {
                        *((char*)Address + index) = (char)value;
                        break;
                    }

                case NPTypeCode.Double:
                    {
                        *((double*)Address + index) = (double)value;
                        break;
                    }

                case NPTypeCode.Single:
                    {
                        *((float*)Address + index) = (float)value;
                        break;
                    }

                case NPTypeCode.Decimal:
                    {
                        *((decimal*)Address + index) = (decimal)value;
                        break;
                    }

                default:
                    throw new NotSupportedException();
#endif
            }

        }

        [MethodImpl((MethodImplOptions)768)]
        public unsafe T GetIndex<T>(int index) where T : unmanaged
        {
            return *((T*)Address + index);
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
            if (value == null)
                throw new ArgumentNullException(nameof(value));

            unsafe
            {
                var (subShape, offset) = _shape.GetSubshape(indices);
                if (subShape != value.Storage.Shape)
                    throw new IncorrectShapeException();
                var step = value.dtypesize;
                var len = step * value.Storage.Count;
                Buffer.MemoryCopy(value.Storage.Address, Address + offset * step, len, len);
                //TODO! TEST!
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

            unsafe
            {
                var (subShape, offset) = _shape.GetSubshape(indices); //TODO! We dont need subshape unless it is debug mode.
                if (subShape.size != value.Count)
                    throw new IncorrectShapeException();

                var step = value.ItemLength;
                var len = step * value.Count;
                Buffer.MemoryCopy(value.Address, Address + offset * step, len, len);
                //TODO! TEST!
            }
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
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        public void ReplaceData(IArraySlice values)
        {
            SetInternalArray(values);
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
            _typecode = _dtype.GetTypeCode();
            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{_dtype.Name} as a dtype is not supported.");
            SetInternalArray(nd.Array);
        }

        public void Reshape(params int[] dimensions)
        {
            var newShape = new Shape(dimensions);
            if (newShape.size != _shape.size)
                throw new IncorrectShapeException();

            _shape = newShape;
        }

        public void Reshape(Shape shape)
        {
            if (shape.size != _shape.size)
                throw new IncorrectShapeException();

            _shape = shape;
        }

        public ArraySlice<T> View<T>(Slice slice = null) where T : unmanaged //TODO! this should return UnmanagedStorage
        {
            if (slice is null)
                slice = Slice;

            if (slice is null)
            {
                return GetData<T>();
            }
            else
            {
                var shape = Shape.GetShape(_shape.Dimensions, axis: 0);
                var offset = Shape.GetSize(shape);
                return GetData<T>().Slice(slice.Start.Value * offset, slice.Length.Value * offset);
            }
        }

        public ArraySlice<T> GetSpanData<T>(Slice slice, params int[] indice) where T : unmanaged
        {
            int stride = _shape.NDim == 0 ? 1 : _shape.Strides[indice.Length - 1];
            int idx = _shape.GetIndexInShape(Slice, indice);
            int offset = idx + (Slice is null ? 0 : Slice.Start.Value) * _shape.Strides[0];

            return GetData<T>().Slice(offset, stride);
        }

        public UnmanagedStorage Clone()
        {
            var puffer = Engine.GetStorage(_dtype);
            puffer.Allocate(InternalArray.Clone(), _shape);
            //TODO there should be a Clone function in Engine
            return puffer;
        }

        object ICloneable.Clone()
        {
            return Clone();
        }

        #region Getters

#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
        /// <summary>
        ///     Retrieves value of type <see cref="#2"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="#2"/></exception>
        public #2 Get#1(params int[] indices)
            => _array#1[_shape.GetIndexInShape(Slice, indices)];

        %
#else

        /// <summary>
        ///     Retrieves value of type <see cref="byte"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        public byte GetByte(params int[] indices)
            => _arrayByte[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="short"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        public short GetInt16(params int[] indices)
            => _arrayInt16[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        public ushort GetUInt16(params int[] indices)
            => _arrayUInt16[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="int"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        public int GetInt32(params int[] indices)
            => _arrayInt32[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        public uint GetUInt32(params int[] indices)
            => _arrayUInt32[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="long"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        public long GetInt64(params int[] indices)
            => _arrayInt64[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        public ulong GetUInt64(params int[] indices)
            => _arrayUInt64[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="char"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        public char GetChar(params int[] indices)
            => _arrayChar[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="double"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        public double GetDouble(params int[] indices)
            => _arrayDouble[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="float"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        public float GetSingle(params int[] indices)
            => _arraySingle[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/> from internal storage.
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        public decimal GetDecimal(params int[] indices)
            => _arrayDecimal[_shape.GetIndexInShape(Slice, indices)];

        /// <summary>
        ///     Retrieves value of type <see cref="string"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="string"/></exception>
        public string GetString(params int[] indices)
        {
            throw new NotImplementedException();
        }

#endif

        #endregion


        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="IStorage.DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="IStorage.DType"/> is not <see cref="object"/></exception>
        public unsafe object GetValue(params int[] indices)
        {
            switch (TypeCode)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	            {
		            return *((#2*)Address + _shape.GetIndexInShape(Slice, indices));
	            }

	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                {
                    return *((byte*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Int16:
                {
                    return *((short*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.UInt16:
                {
                    return *((ushort*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Int32:
                {
                    return *((int*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.UInt32:
                {
                    return *((uint*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Int64:
                {
                    return *((long*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.UInt64:
                {
                    return *((ulong*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Char:
                {
                    return *((char*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Double:
                {
                    return *((double*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Single:
                {
                    return *((float*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                case NPTypeCode.Decimal:
                {
                    return *((decimal*)Address + _shape.GetIndexInShape(Slice, indices));
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }

        //TODO! THESE:
        public unsafe object GetIndex(int index)
        {
            switch (TypeCode)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	            {
		            return *((#2*)Address + index);
	            }

	            %
	            default:
		            throw new NotSupportedException();
#else
                case NPTypeCode.Byte:
                {
                    return *((byte*)Address + index);
                }

                case NPTypeCode.Int16:
                {
                    return *((short*)Address + index);
                }

                case NPTypeCode.UInt16:
                {
                    return *((ushort*)Address + index);
                }

                case NPTypeCode.Int32:
                {
                    return *((int*)Address + index);
                }

                case NPTypeCode.UInt32:
                {
                    return *((uint*)Address + index);
                }

                case NPTypeCode.Int64:
                {
                    return *((long*)Address + index);
                }

                case NPTypeCode.UInt64:
                {
                    return *((ulong*)Address + index);
                }

                case NPTypeCode.Char:
                {
                    return *((char*)Address + index);
                }

                case NPTypeCode.Double:
                {
                    return *((double*)Address + index);
                }

                case NPTypeCode.Single:
                {
                    return *((float*)Address + index);
                }

                case NPTypeCode.Decimal:
                {
                    return *((decimal*)Address + index);
                }

                default:
                    throw new NotSupportedException();
#endif
            }
        }

        public object this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => GetValue(indices);
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => SetIndex(value, _shape.GetIndexInShape(Slice, indices));
        }
    }
}
