using System;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     Reads the element at a storage offset as <typeparamref name="T"/>: one load when <typeparamref name="T"/> is
        ///     the dtype's own type, NumPy's cast of the stored value otherwise (the cast <c>astype</c> runs) — never a
        ///     reinterpretation of the bytes.
        /// </summary>
        /// <typeparam name="T">The type the caller wants the element as.</typeparam>
        /// <param name="offset">The element's offset in the buffer (already resolved and checked by the caller).</param>
        /// <returns>The element, converted when the dtype differs.</returns>
        /// <exception cref="NotSupportedException">The storage has no element type (an empty storage).</exception>
        /// <remarks>
        ///     The generic and typed getters used to load <c>sizeof(T)</c> bytes at the element's address whatever the
        ///     dtype, so <c>GetValue&lt;double&gt;(1)</c> of a float32 array read elements 1 and 2 as one double and the last
        ///     element read past the buffer (parity QA-12, Release only: the type test was a <c>Debug.Assert</c>). The
        ///     matching case stays a single load behind one <see cref="Type"/> reference compare; the converting case is
        ///     a separate method so the common path stays small enough to inline.
        /// </remarks>
        [MethodImpl(Inline)]
        internal unsafe T ReadAs<T>(long offset) where T : unmanaged
        {
            if (typeof(T) == _dtype)
                return *((T*)Address + offset);
            return ReadConverted<T>(offset);
        }

        /// <summary>
        ///     Writes <paramref name="value"/> into the element at a storage offset: one store when <typeparamref name="T"/>
        ///     is the dtype's own type, NumPy's cast of the value to the dtype otherwise (NumPy's setitem of a scalar of
        ///     another type) — never <c>sizeof(T)</c> raw bytes over a narrower element.
        /// </summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="value">The value to store.</param>
        /// <param name="offset">The element's offset in the buffer (already resolved and checked by the caller).</param>
        /// <exception cref="NotSupportedException">The storage has no element type (an empty storage).</exception>
        /// <remarks>
        ///     A double written into a float32 array used to overwrite two elements, and into its last element four bytes
        ///     past the buffer (parity QA-12). The cast is <see cref="Converts.ChangeType{TIn, TOut}(TIn)"/>, the
        ///     NumPy-aligned conversion <c>astype</c> uses (truncation and wrap for float → integer, NaN → <c>True</c> for
        ///     bool, the real part of a complex value). A text storage stores the value's <c>str()</c> through its setitem.
        /// </remarks>
        [MethodImpl(Inline)]
        internal unsafe void WriteAs<T>(T value, long offset) where T : unmanaged
        {
            if (typeof(T) == _dtype)
            {
                *((T*)Address + offset) = value;
                return;
            }

            WriteConverted(value, offset);
        }

        /// <summary>
        ///     The converting half of <see cref="ReadAs{T}"/>: loads the element at the dtype it is stored as and casts it.
        /// </summary>
        /// <typeparam name="T">The type the caller wants the element as.</typeparam>
        /// <param name="offset">The element's offset in the buffer.</param>
        /// <returns>The converted element.</returns>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private unsafe T ReadConverted<T>(long offset) where T : unmanaged
        {
            switch (_typecode)
            {
                case NPTypeCode.Boolean: return Converts.ChangeType<bool, T>(*((bool*)Address + offset));
                case NPTypeCode.SByte: return Converts.ChangeType<sbyte, T>(*((sbyte*)Address + offset));
                case NPTypeCode.Byte: return Converts.ChangeType<byte, T>(*((byte*)Address + offset));
                case NPTypeCode.Int16: return Converts.ChangeType<short, T>(*((short*)Address + offset));
                case NPTypeCode.UInt16: return Converts.ChangeType<ushort, T>(*((ushort*)Address + offset));
                case NPTypeCode.Int32: return Converts.ChangeType<int, T>(*((int*)Address + offset));
                case NPTypeCode.UInt32: return Converts.ChangeType<uint, T>(*((uint*)Address + offset));
                case NPTypeCode.Int64: return Converts.ChangeType<long, T>(*((long*)Address + offset));
                case NPTypeCode.UInt64: return Converts.ChangeType<ulong, T>(*((ulong*)Address + offset));
                case NPTypeCode.Char: return Converts.ChangeType<char, T>(*((char*)Address + offset));
                // HalfAt builds the value from its 16-bit pattern (the .NET 8 tier-0 boxing over-read, see HalfAt).
                case NPTypeCode.Half: return Converts.ChangeType<Half, T>(HalfAt((ushort*)Address + offset));
                case NPTypeCode.Double: return Converts.ChangeType<double, T>(*((double*)Address + offset));
                case NPTypeCode.Single: return Converts.ChangeType<float, T>(*((float*)Address + offset));
                case NPTypeCode.Decimal: return Converts.ChangeType<decimal, T>(*((decimal*)Address + offset));
                case NPTypeCode.Complex: return Converts.ChangeType<System.Numerics.Complex, T>(*((System.Numerics.Complex*)Address + offset));
                // A text element decodes through its slice (string / byte[] / the StringDType NA object); the object path
                // parses it the way NumPy's float(np.str_('1.5')) does.
                case NPTypeCode.String: return Converts.ChangeType<T>(InternalArray.GetIndex(offset));
                default: throw new NotSupportedException($"Unable to read an element of a storage with type {_typecode}.");
            }
        }

        /// <summary>
        ///     The converting half of <see cref="WriteAs{T}"/>: casts the value to the dtype and stores it.
        /// </summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="value">The value to store.</param>
        /// <param name="offset">The element's offset in the buffer.</param>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private unsafe void WriteConverted<T>(T value, long offset) where T : unmanaged
        {
            switch (_typecode)
            {
                case NPTypeCode.Boolean: *((bool*)Address + offset) = Converts.ChangeType<T, bool>(value); return;
                case NPTypeCode.SByte: *((sbyte*)Address + offset) = Converts.ChangeType<T, sbyte>(value); return;
                case NPTypeCode.Byte: *((byte*)Address + offset) = Converts.ChangeType<T, byte>(value); return;
                case NPTypeCode.Int16: *((short*)Address + offset) = Converts.ChangeType<T, short>(value); return;
                case NPTypeCode.UInt16: *((ushort*)Address + offset) = Converts.ChangeType<T, ushort>(value); return;
                case NPTypeCode.Int32: *((int*)Address + offset) = Converts.ChangeType<T, int>(value); return;
                case NPTypeCode.UInt32: *((uint*)Address + offset) = Converts.ChangeType<T, uint>(value); return;
                case NPTypeCode.Int64: *((long*)Address + offset) = Converts.ChangeType<T, long>(value); return;
                case NPTypeCode.UInt64: *((ulong*)Address + offset) = Converts.ChangeType<T, ulong>(value); return;
                case NPTypeCode.Char: *((char*)Address + offset) = Converts.ChangeType<T, char>(value); return;
                case NPTypeCode.Half: *((Half*)Address + offset) = Converts.ChangeType<T, Half>(value); return;
                case NPTypeCode.Double: *((double*)Address + offset) = Converts.ChangeType<T, double>(value); return;
                case NPTypeCode.Single: *((float*)Address + offset) = Converts.ChangeType<T, float>(value); return;
                case NPTypeCode.Decimal: *((decimal*)Address + offset) = Converts.ChangeType<T, decimal>(value); return;
                case NPTypeCode.Complex: *((System.Numerics.Complex*)Address + offset) = Converts.ChangeType<T, System.Numerics.Complex>(value); return;
                // NumPy's STRING_setitem / UNICODE_setitem / stringdtype_setitem store the value's str() in the cell.
                case NPTypeCode.String: InternalArray.SetIndex(offset, value); return;
                default: throw new NotSupportedException($"Unable to write an element of a storage with type {_typecode}.");
            }
        }
    }
}
