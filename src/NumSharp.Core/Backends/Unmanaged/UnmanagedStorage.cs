using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
    public partial class UnmanagedStorage : IStorage, ICloneable
    {
#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
        protected ArraySlice<#2> _array#1;
#else
        protected ArraySlice<bool> _arrayBoolean;
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

        /// <summary>
        ///     Spans <see cref="Address"/> &lt;-&gt; <see cref="Count"/>
        /// </summary>
        /// <remarks>This ignores completely slicing.</remarks>
        public Span<T> AsSpan<T>()
        {
            if (_shape.IsSliced)
                throw new InvalidOperationException("Unable to span a sliced storage.");

            unsafe
            {
                return new Span<T>(Address, Count);
            }
        }

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        public TensorEngine Engine { get; internal set; }

        public static UnmanagedStorage Scalar<T>(T value) where T : unmanaged => new UnmanagedStorage(ArraySlice.Scalar<T>(value));

        public static UnmanagedStorage Scalar(object value) => new UnmanagedStorage(ArraySlice.Scalar(value));

        #region Aliasing

        /// <summary>
        ///     Creates an alias to this UnmanagedStorage.
        /// </summary>
        public UnmanagedStorage Alias()
        {
            var r = new UnmanagedStorage();
            r._shape = _shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = _shape.size; //incase shape is sliced
            return r;
        }

        /// <summary>
        ///     Creates an alias to this UnmanagedStorage with a specific shape.
        /// </summary>
        /// <remarks>Doesn't check if Shape matches the internal storage.</remarks>
        public UnmanagedStorage Alias(Shape shape)
        {
            var r = new UnmanagedStorage();
            r._shape = shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = shape.size; //incase shape is sliced
            return r;
        }

        /// <summary>
        ///     Creates an alias to this UnmanagedStorage with a specific shape.
        /// </summary>
        /// <remarks>Doesn't check if Shape matches the internal storage.</remarks>
        public UnmanagedStorage Alias(ref Shape shape)
        {
            var r = new UnmanagedStorage();
            r._shape = shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = shape.size; //incase shape is sliced
            return r;
        }

        #endregion

        #region Casting

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <typeparam name="T">The dtype to convert to</typeparam>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast<T>() where T : unmanaged
        {
            if (Shape.IsEmpty)
                return new UnmanagedStorage(typeof(T));

            if (_dtype == typeof(T))
                return Clone();

            //this also handles slices
            return new UnmanagedStorage((ArraySlice<T>)InternalArray.CastTo<T>(), Shape.Clone(true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="typeCode">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast(NPTypeCode typeCode)
        {
            if (Shape.IsEmpty)
                return new UnmanagedStorage(typeCode);

            if (_typecode == typeCode)
                return Clone();

            //this also handles slices
            return new UnmanagedStorage((IArraySlice)InternalArray.CastTo(typeCode), Shape.Clone(true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="dtype">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast(Type dtype)
        {
            return Cast(dtype.GetTypeCode());
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <typeparam name="T">The dtype to convert to</typeparam>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <typeparamref name="T"/></remarks>
        public UnmanagedStorage CastIfNecessary<T>() where T : unmanaged
        {
            if (Shape.IsEmpty || _dtype == typeof(T))
                return this;

            //this also handles slices
            return new UnmanagedStorage((ArraySlice<T>)InternalArray.CastTo<T>(), Shape.Clone(true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="typeCode">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <paramref name="typeCode"/></remarks>
        public UnmanagedStorage CastIfNecessary(NPTypeCode typeCode)
        {
            if (Shape.IsEmpty || _typecode == typeCode)
                return this;

            //this also handles slices
            return new UnmanagedStorage((IArraySlice)InternalArray.CastTo(typeCode), Shape.Clone(true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="dtype">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <paramref name="typeCode"/></remarks>
        public UnmanagedStorage CastIfNecessary(Type dtype)
        {
            return CastIfNecessary(dtype.GetTypeCode());
        }

        #endregion

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
        protected unsafe void SetInternalArray(Array array)
        {
            switch (_typecode)
            {
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
                    Address = (byte*)_arrayBoolean.Address;
                    Count = _arrayBoolean.Count;
                    break;
                }

                case NPTypeCode.Byte:
                {
                    InternalArray = _arrayByte = ArraySlice.FromArray<byte>((byte[])array);
                    Address = (byte*)_arrayByte.Address;
                    Count = _arrayByte.Count;
                    break;
                }

                case NPTypeCode.Int16:
                {
                    InternalArray = _arrayInt16 = ArraySlice.FromArray<short>((short[])array);
                    Address = (byte*)_arrayInt16.Address;
                    Count = _arrayInt16.Count;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    InternalArray = _arrayUInt16 = ArraySlice.FromArray<ushort>((ushort[])array);
                    Address = (byte*)_arrayUInt16.Address;
                    Count = _arrayUInt16.Count;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    InternalArray = _arrayInt32 = ArraySlice.FromArray<int>((int[])array);
                    Address = (byte*)_arrayInt32.Address;
                    Count = _arrayInt32.Count;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    InternalArray = _arrayUInt32 = ArraySlice.FromArray<uint>((uint[])array);
                    Address = (byte*)_arrayUInt32.Address;
                    Count = _arrayUInt32.Count;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    InternalArray = _arrayInt64 = ArraySlice.FromArray<long>((long[])array);
                    Address = (byte*)_arrayInt64.Address;
                    Count = _arrayInt64.Count;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    InternalArray = _arrayUInt64 = ArraySlice.FromArray<ulong>((ulong[])array);
                    Address = (byte*)_arrayUInt64.Address;
                    Count = _arrayUInt64.Count;
                    break;
                }

                case NPTypeCode.Char:
                {
                    InternalArray = _arrayChar = ArraySlice.FromArray<char>((char[])array);
                    Address = (byte*)_arrayChar.Address;
                    Count = _arrayChar.Count;
                    break;
                }

                case NPTypeCode.Double:
                {
                    InternalArray = _arrayDouble = ArraySlice.FromArray<double>((double[])array);
                    Address = (byte*)_arrayDouble.Address;
                    Count = _arrayDouble.Count;
                    break;
                }

                case NPTypeCode.Single:
                {
                    InternalArray = _arraySingle = ArraySlice.FromArray<float>((float[])array);
                    Address = (byte*)_arraySingle.Address;
                    Count = _arraySingle.Count;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    InternalArray = _arrayDecimal = ArraySlice.FromArray<decimal>((decimal[])array);
                    Address = (byte*)_arrayDecimal.Address;
                    Count = _arrayDecimal.Count;
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
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
                    Address = (byte*)_arrayBoolean.Address;
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

                case NPTypeCode.Int16:
                {
                    InternalArray = _arrayInt16 = (ArraySlice<short>)array;
                    Address = (byte*)_arrayInt16.Address;
                    Count = _arrayInt16.Count;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    InternalArray = _arrayUInt16 = (ArraySlice<ushort>)array;
                    Address = (byte*)_arrayUInt16.Address;
                    Count = _arrayUInt16.Count;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    InternalArray = _arrayInt32 = (ArraySlice<int>)array;
                    Address = (byte*)_arrayInt32.Address;
                    Count = _arrayInt32.Count;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    InternalArray = _arrayUInt32 = (ArraySlice<uint>)array;
                    Address = (byte*)_arrayUInt32.Address;
                    Count = _arrayUInt32.Count;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    InternalArray = _arrayInt64 = (ArraySlice<long>)array;
                    Address = (byte*)_arrayInt64.Address;
                    Count = _arrayInt64.Count;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    InternalArray = _arrayUInt64 = (ArraySlice<ulong>)array;
                    Address = (byte*)_arrayUInt64.Address;
                    Count = _arrayUInt64.Count;
                    break;
                }

                case NPTypeCode.Char:
                {
                    InternalArray = _arrayChar = (ArraySlice<char>)array;
                    Address = (byte*)_arrayChar.Address;
                    Count = _arrayChar.Count;
                    break;
                }

                case NPTypeCode.Double:
                {
                    InternalArray = _arrayDouble = (ArraySlice<double>)array;
                    Address = (byte*)_arrayDouble.Address;
                    Count = _arrayDouble.Count;
                    break;
                }

                case NPTypeCode.Single:
                {
                    InternalArray = _arraySingle = (ArraySlice<float>)array;
                    Address = (byte*)_arraySingle.Address;
                    Count = _arraySingle.Count;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    InternalArray = _arrayDecimal = (ArraySlice<decimal>)array;
                    Address = (byte*)_arrayDecimal.Address;
                    Count = _arrayDecimal.Count;
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
            if (shape.IsSliced)
            {
                values = values.Clone();
                shape = new Shape((int[])shape.dimensions.Clone());
            }

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

        #region Cloning

        /// <summary>
        ///     Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        public IArraySlice CloneData()
        {
            //Incase shape is not sliced, we can copy the internal buffer.
            if (!Shape.IsSliced)
                return InternalArray.Clone();

            //Linear copy of all the sliced items.

            var ret = ArraySlice.Allocate(InternalArray.TypeCode, Shape.size, false);
            MultiIterator.Assign(new UnmanagedStorage(ret, Shape.Clean()), this);

            return ret;
        }

        /// <summary>
        ///     Get all elements from cloned storage as <see cref="ArraySlice{T}"/> and cast if necessary.
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage and casted (if necessary) as <see cref="ArraySlice{T}"/></returns>
        public ArraySlice<T> CloneData<T>() where T : unmanaged
        {
            var cloned = CloneData();
            if (cloned.TypeCode != InfoOf<T>.NPTypeCode)
                return (ArraySlice<T>)cloned.CastTo<T>();

            return (ArraySlice<T>)cloned;
        }

        /// <summary>
        ///     Perform a complete copy of this <see cref="UnmanagedStorage"/> and <see cref="InternalArray"/>.
        /// </summary>
        /// <remarks>If shape is sliced, discards any slicing properties but copies only the sliced data</remarks>
        public UnmanagedStorage Clone() => new UnmanagedStorage(CloneData(), _shape.Clone(true, true));

        object ICloneable.Clone() => Clone();

        #endregion

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
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                    *((#2*)Address + _shape.TransformOffset(index)) = (#2) value;
                    return;
                %
                default:
                    throw new NotSupportedException();
#else

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.TransformOffset(index)) = (bool)value;
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
                case NPTypeCode.Double:
                    *((double*)Address + _shape.TransformOffset(index)) = (double)value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.TransformOffset(index)) = (float)value;
                    return;
                case NPTypeCode.Decimal:
                    *((decimal*)Address + _shape.TransformOffset(index)) = (decimal)value;
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
#if _REGEN
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                %foreach supported_currently_supported,supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                    *((#2*)Address + _shape.GetOffset(indices)) = (#2) value;
                    return;
                %
                default:
                    throw new NotSupportedException();
#else

                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                    *((bool*)Address + _shape.GetOffset(indices)) = (bool)value;
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
                case NPTypeCode.Double:
                    *((double*)Address + _shape.GetOffset(indices)) = (double)value;
                    return;
                case NPTypeCode.Single:
                    *((float*)Address + _shape.GetOffset(indices)) = (float)value;
                    return;
                case NPTypeCode.Decimal:
                    *((decimal*)Address + _shape.GetOffset(indices)) = (decimal)value;
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
            if (value is NDArray nd)
            {
                SetData(nd, indices);
                return;
            }

            if (value is IArraySlice arr)
            {
                SetData(arr, indices);
                return;
            }

            if (value is Array array)
            {
                SetData((NDArray)array, indices);
                return;
            }

            //we assume this is a scalar.
            SetValue(value, _shape.GetOffset(indices));
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

            var (subShape, offset) = _shape.GetSubshape(indices);

            //is it a scalar to scalar
            if (value.Shape.IsScalar && subShape.IsScalar)
            {
                SetAtIndexUnsafe((ValueType)Convert.ChangeType(value.GetAtIndex(0), _dtype), offset);
                return;
            }

            //if (!value.Storage.Shape.IsScalar && np.squeeze(subShape) != np.squeeze(value.Storage.Shape))
            //    throw new IncorrectShapeException($"Can't SetData to a from a shape of {value.Shape} to target shape {subShape}, the shape the coordinates point to mismatch the size of rhs (value)");

            if (subShape.size % value.size != 0)
                throw new IncorrectShapeException($"Can't SetData to a from a shape of {value.Shape} to target shape {subShape}, these shapes can't be broadcasted together.");

            //if dtypes doesn't match, cast is performed inside.
            MultiIterator.Assign(GetData(indices), value.Storage); //we use lhs stop because rhs is scalar which will fill all values of lhs
            //TODO! there are cases where we can perform Buffer.MemoryCopy, when data is continous
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

            //if (value.TypeCode != TypeCode)
            //    value = (IArraySlice)value.CastTo(TypeCode);

            MultiIterator.Assign(GetData(indices), new UnmanagedStorage(value, Shape.Vector(value.Count)));
        }

        #region Typed Setters

#if _REGEN
	%foreach supported_currently_supported,supported_currently_supported_lowercase%
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetByte(byte value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt16(short value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetUInt16(ushort value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt32(int value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetUInt32(uint value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetInt64(long value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetUInt64(ulong value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetChar(char value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDouble(double value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetSingle(float value, params int[] indices)
        {
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDecimal(decimal value, params int[] indices)
        {
            unsafe
            {
                *((decimal*)Address + _shape.GetOffset(indices)) = value;
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
                _shape = new Shape(values.Count);
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
                _shape = new Shape(values.Count);
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

        #region Shaping

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        public void Reshape(params int[] dimensions)
        {
            var newShape = new Shape(dimensions);
            if (newShape.size != _shape.size)
                throw new IncorrectShapeException($"Given shape size ({newShape.size}) does not match the size of the existing storage size ({_shape.size})");

            _shape = newShape;
            Count = _shape.size;
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        public void Reshape(Shape shape)
        {
            if (shape.size != _shape.size)
                throw new IncorrectShapeException($"Given shape size ({shape.size}) does not match the size of the existing storage size ({_shape.size})");

            _shape = shape;
            Count = _shape.size;
        }

        /// <summary>
        ///     Set the shape of this storage without checking if sizes match.
        /// </summary>
        /// <remarks>Used during broadcasting</remarks>
        internal void SetShapeUnsafe(Shape shape)
        {
            _shape = shape;
            Count = _shape.size;
        }

        #region Slicing

        public UnmanagedStorage GetView(string slicing_notation) => GetView(Slice.ParseSlices(slicing_notation));

        public UnmanagedStorage GetView(params Slice[] slices) => Alias(_shape.Slice(slices));

        #endregion

        #endregion

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
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
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
            if (Shape.IsSliced)
                return GetView(indices.Select(Slice.Index).ToArray());

            var (shape, offset) = _shape.GetSubshape(indices);
            return new UnmanagedStorage(InternalArray.Slice(offset, shape.Size), shape);
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

            //TODO! this should clone based on the slice!

            if (Shape.IsSliced)
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
        public IArraySlice GetData()
        {
            return InternalArray;
        }

        //TODO! because this is not typed, I think it should be removed:
        //public object this[params int[] indices]
        //{
        //    [MethodImpl(MethodImplOptions.AggressiveInlining)]
        //    get => GetValue(indices);
        //    [MethodImpl(MethodImplOptions.AggressiveInlining)]
        //    set => SetAtIndex(value, _shape.GetOffset(indices));
        //}

#if _REGEN
#region Direct Getters
     
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
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

        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            if (typeof(T).GetTypeCode() != InternalArray.TypeCode)
                throw new ArrayTypeMismatchException($"The given type argument '{typeof(T).Name}' doesn't match the type of the internal data '{InternalArray.TypeCode}'");
            var addr = (T*)Address;
            var ret = new T[Shape.Size];
            var incr = new NDCoordinatesIncrementor(Shape.dimensions);
            int[] current = incr.Index;
            int i = 0;
            do
            {
                ret[i++] = (*(addr + Shape.GetOffset(current)));
            } while (incr.Next() != null);

            return ret;
        }
    }
}
