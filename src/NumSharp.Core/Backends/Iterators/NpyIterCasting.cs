using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Type casting utilities for NpyIter.
    /// Validates casting rules and performs type conversions.
    /// </summary>
    public static unsafe class NpyIterCasting
    {
        /// <summary>
        /// Check if casting from srcType to dstType is allowed under the given casting rule.
        /// </summary>
        public static bool CanCast(NPTypeCode srcType, NPTypeCode dstType, NPY_CASTING casting)
        {
            if (srcType == dstType)
                return true;

            switch (casting)
            {
                case NPY_CASTING.NPY_NO_CASTING:
                    // Only same type allowed
                    return false;

                case NPY_CASTING.NPY_EQUIV_CASTING:
                    // Only byte order changes (not applicable in .NET)
                    return false;

                case NPY_CASTING.NPY_SAFE_CASTING:
                    return IsSafeCast(srcType, dstType);

                case NPY_CASTING.NPY_SAME_KIND_CASTING:
                    return IsSameKindCast(srcType, dstType);

                case NPY_CASTING.NPY_UNSAFE_CASTING:
                    // Any cast allowed
                    return true;

                default:
                    return false;
            }
        }

        /// <summary>
        /// Check if casting is "safe" (no loss of precision).
        /// Safe casts: smaller int -> larger int, any int -> float64, float32 -> float64,
        /// any non-complex numeric -> complex, half -> single/double.
        /// </summary>
        private static bool IsSafeCast(NPTypeCode srcType, NPTypeCode dstType)
        {
            // Same type is always safe
            if (srcType == dstType)
                return true;

            // Complex absorbs everything except: complex -> non-complex is never safe.
            if (IsComplex(srcType) && !IsComplex(dstType))
                return false;
            if (IsComplex(dstType))
            {
                // Every real type casts safely into complex128. NumPy treats real→complex128
                // as safe across the board — can_cast(int64, complex128, 'safe') is True —
                // consistent with its treatment of int64→float64 itself as safe. The precision
                // loss above 2^53 mirrors int64→float64 exactly and is accepted by NumPy.
                return true;
            }

            int srcSize = InfoOf.GetSize(srcType);
            int dstSize = InfoOf.GetSize(dstType);

            // Get type categories
            bool srcIsFloat = IsFloatingPoint(srcType);
            bool dstIsFloat = IsFloatingPoint(dstType);
            bool srcIsSigned = IsSignedInteger(srcType);
            bool dstIsSigned = IsSignedInteger(dstType);
            bool srcIsUnsigned = IsUnsignedInteger(srcType);
            bool dstIsUnsigned = IsUnsignedInteger(dstType);

            // Float to int is never safe
            if (srcIsFloat && !dstIsFloat)
                return false;

            // Half (float16) widens safely to Single (float32) and Double (float64).
            if (srcType == NPTypeCode.Half)
                return dstType == NPTypeCode.Single || dstType == NPTypeCode.Double || dstType == NPTypeCode.Decimal;

            // Casting INTO Half (float16): its 11-bit mantissa exactly represents integers
            // only up to ±2048, so just bool and the 8-bit ints widen safely (NumPy:
            // can_cast(uint8, float16, 'safe') is True). int16/uint16 and wider, plus any
            // float→Half narrowing, lose precision and are not safe.
            if (dstType == NPTypeCode.Half)
                return srcType == NPTypeCode.Boolean
                    || srcType == NPTypeCode.Byte
                    || srcType == NPTypeCode.SByte;

            // Larger to smaller is never safe
            if (srcSize > dstSize && !dstIsFloat)
                return false;

            // Float32 to float64 is safe
            if (srcType == NPTypeCode.Single && dstType == NPTypeCode.Double)
                return true;

            // Float64 to float32 is NOT safe (loss of precision)
            if (srcType == NPTypeCode.Double && dstType == NPTypeCode.Single)
                return false;

            // Int to float64 is safe (all ints fit in float64)
            if ((srcIsSigned || srcIsUnsigned) && dstType == NPTypeCode.Double)
                return true;

            // Int to float32 is safe for small ints
            if ((srcIsSigned || srcIsUnsigned) && dstType == NPTypeCode.Single && srcSize <= 2)
                return true;

            // Signed to unsigned is never safe (negatives can't be represented).
            if (srcIsSigned && dstIsUnsigned)
                return false;

            // Unsigned to signed is safe only when the signed type is strictly wider, so the
            // entire unsigned range fits: uint8→int16/int32/int64, uint16→int32/int64,
            // uint32→int64. (NumPy: can_cast(uint8, int16, 'safe') is True.)
            if (srcIsUnsigned && dstIsSigned)
                return srcSize < dstSize;

            // Same signedness, smaller to larger is safe
            if ((srcIsSigned && dstIsSigned) || (srcIsUnsigned && dstIsUnsigned))
                return srcSize <= dstSize;

            // For boolean
            if (srcType == NPTypeCode.Boolean)
                return true;  // Bool can safely convert to any numeric

            return false;
        }

        /// <summary>
        /// Check if casting is "same_kind" — NumPy's NPY_SAME_KIND_CASTING. This is a
        /// strict superset of <see cref="IsSafeCast"/> plus the looser within-kind casts,
        /// matching numpy <c>can_cast(.., 'same_kind')</c> exactly across the type matrix:
        /// <list type="bullet">
        ///   <item>every safe cast (int→float64, bool→numeric, float32→float64, …);</item>
        ///   <item>float → float, including narrowing (float64→float32, →float16);</item>
        ///   <item>int → int for every signedness pair EXCEPT signed → unsigned
        ///         (unsigned→signed and same-sign narrowing are allowed, e.g. int64→int32,
        ///         uint16→int8; but int32→uint32 is not);</item>
        ///   <item>int → float, even when lossy (int64→float32);</item>
        ///   <item>real (int or float) → complex.</item>
        /// </list>
        /// Notably NOT same_kind: float → int, int/float → bool, signed → unsigned,
        /// complex → real (all match numpy's '.' entries).
        /// </summary>
        private static bool IsSameKindCast(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType == dstType)
                return true;

            // same_kind is a superset of safe — fixes int→float (e.g. int32→float64),
            // which is safe yet cross-kind and was previously rejected.
            if (IsSafeCast(srcType, dstType))
                return true;

            bool srcFloat = IsFloatingPoint(srcType);
            bool dstFloat = IsFloatingPoint(dstType);
            bool srcSigned = IsSignedInteger(srcType);
            bool srcInt = srcSigned || IsUnsignedInteger(srcType);
            bool dstInt = IsSignedInteger(dstType) || IsUnsignedInteger(dstType);

            // float → float (narrowing within the floating kind, e.g. float64→float32).
            if (srcFloat && dstFloat)
                return true;

            // int → int: every signedness pair EXCEPT signed → unsigned.
            if (srcInt && dstInt)
                return !(srcSigned && IsUnsignedInteger(dstType));

            // int → float, even when lossy (e.g. int64 → float32).
            if (srcInt && dstFloat)
                return true;

            // real (int or float) → complex is same_kind (numpy classes every real→complex
            // as at least same_kind; the safe ones are already short-circuited above, so
            // this adds the int64/uint64→complex pair the conservative safe rule withholds).
            if ((srcInt || srcFloat) && IsComplex(dstType))
                return true;

            return false;
        }

        private static bool IsFloatingPoint(NPTypeCode type)
        {
            return type == NPTypeCode.Half || type == NPTypeCode.Single ||
                   type == NPTypeCode.Double || type == NPTypeCode.Decimal;
        }

        private static bool IsSignedInteger(NPTypeCode type)
        {
            return type == NPTypeCode.SByte || type == NPTypeCode.Int16 ||
                   type == NPTypeCode.Int32 || type == NPTypeCode.Int64;
        }

        private static bool IsUnsignedInteger(NPTypeCode type)
        {
            return type == NPTypeCode.Byte || type == NPTypeCode.UInt16 ||
                   type == NPTypeCode.UInt32 || type == NPTypeCode.UInt64 || type == NPTypeCode.Char;
        }

        private static bool IsComplex(NPTypeCode type) => type == NPTypeCode.Complex;

        /// <summary>
        /// Validate all operand casts in an iterator state.
        /// Throws InvalidCastException if any cast is not allowed.
        /// Also packs combined transfer flags into the top 8 bits of state.ItFlags
        /// per NumPy nditer_constr.c:3542.
        /// </summary>
        public static void ValidateCasts(ref NpyIterState state, NPY_CASTING casting)
        {
            NpyArrayMethodFlags combinedFlags = NpyArrayMethodFlags.None;
            bool anyCast = false;

            for (int op = 0; op < state.NOp; op++)
            {
                var srcType = state.GetOpSrcDType(op);
                var dstType = state.GetOpDType(op);

                if (srcType != dstType)
                {
                    if (!CanCast(srcType, dstType, casting))
                    {
                        throw new InvalidCastException(
                            $"Iterator operand {op} dtype could not be cast from {srcType.AsNumpyDtypeName()} " +
                            $"to {dstType.AsNumpyDtypeName()} according to the rule '{GetCastingName(casting)}'");
                    }

                    anyCast = true;
                    combinedFlags |= ComputeCastTransferFlags(srcType, dstType);
                }
                else
                {
                    // Same-type copies also have transfer characteristics
                    combinedFlags |= NpyArrayMethodFlags.SUPPORTS_UNALIGNED |
                                     NpyArrayMethodFlags.NO_FLOATINGPOINT_ERRORS |
                                     NpyArrayMethodFlags.IS_REORDERABLE;
                }
            }

            // Pack into top 8 bits of ItFlags (NumPy parity: nditer_constr.c:3542)
            if (anyCast || state.NOp > 0)
            {
                uint packed = ((uint)combinedFlags & 0xFFu) << NpyIterConstants.TRANSFERFLAGS_SHIFT;
                state.ItFlags = (state.ItFlags & ~NpyIterConstants.TRANSFERFLAGS_MASK) | packed;
            }
        }

        /// <summary>
        /// Compute the NpyArrayMethodFlags that characterize a single cast transfer.
        /// In .NET:
        /// - REQUIRES_PYAPI is never set (no Python).
        /// - SUPPORTS_UNALIGNED is always set (raw byte-pointer loops).
        /// - NO_FLOATINGPOINT_ERRORS is always set (.NET casts truncate silently).
        /// - IS_REORDERABLE is set for numeric↔numeric casts (element-wise, commutative).
        /// </summary>
        private static NpyArrayMethodFlags ComputeCastTransferFlags(NPTypeCode srcType, NPTypeCode dstType)
        {
            var flags = NpyArrayMethodFlags.SUPPORTS_UNALIGNED |
                        NpyArrayMethodFlags.NO_FLOATINGPOINT_ERRORS |
                        NpyArrayMethodFlags.IS_REORDERABLE;
            return flags;
        }

        private static string GetCastingName(NPY_CASTING casting)
        {
            return casting switch
            {
                NPY_CASTING.NPY_NO_CASTING => "no",
                NPY_CASTING.NPY_EQUIV_CASTING => "equiv",
                NPY_CASTING.NPY_SAFE_CASTING => "safe",
                NPY_CASTING.NPY_SAME_KIND_CASTING => "same_kind",
                NPY_CASTING.NPY_UNSAFE_CASTING => "unsafe",
                _ => "unknown"
            };
        }

        /// <summary>
        /// Find common dtype for all operands (for COMMON_DTYPE flag).
        /// Returns the dtype that all operands can be safely promoted to.
        /// </summary>
        public static NPTypeCode FindCommonDtype(NDArray[] operands, int nop)
        {
            if (nop == 0)
                return NPTypeCode.Double;

            NPTypeCode result = operands[0].typecode;

            for (int i = 1; i < nop; i++)
            {
                result = PromoteTypes(result, operands[i].typecode);
            }

            return result;
        }

        /// <summary>
        /// Promote two types to a common type. Internal so iterator construction
        /// can resolve the common dtype of VIRTUAL operands across null slots
        /// (FindCommonDtype assumes a dense non-null operand array).
        /// </summary>
        internal static NPTypeCode PromoteTypes(NPTypeCode a, NPTypeCode b)
        {
            if (a == b)
                return a;

            // Complex absorbs everything (highest kind).
            if (IsComplex(a) || IsComplex(b))
                return NPTypeCode.Complex;

            // Float always wins over int
            if (IsFloatingPoint(a) && !IsFloatingPoint(b))
                return a;
            if (IsFloatingPoint(b) && !IsFloatingPoint(a))
                return b;

            // Both float - use larger
            if (IsFloatingPoint(a) && IsFloatingPoint(b))
            {
                int sizeA = InfoOf.GetSize(a);
                int sizeB = InfoOf.GetSize(b);
                return sizeA >= sizeB ? a : b;
            }

            // Both int - complex promotion rules
            bool aIsSigned = IsSignedInteger(a);
            bool bIsSigned = IsSignedInteger(b);
            int sizeA2 = InfoOf.GetSize(a);
            int sizeB2 = InfoOf.GetSize(b);

            if (aIsSigned == bIsSigned)
            {
                // Same signedness - use larger
                return sizeA2 >= sizeB2 ? a : b;
            }

            // Mixed signedness - promote to signed of larger size or double size
            int maxSize = Math.Max(sizeA2, sizeB2);
            if (aIsSigned)
            {
                // a is signed, b is unsigned
                if (sizeA2 > sizeB2) return a;  // Signed is larger
                // Need next larger signed
                return maxSize switch
                {
                    1 => NPTypeCode.Int16,
                    2 => NPTypeCode.Int32,
                    4 => NPTypeCode.Int64,
                    _ => NPTypeCode.Double  // Fallback
                };
            }
            else
            {
                // b is signed, a is unsigned
                if (sizeB2 > sizeA2) return b;
                return maxSize switch
                {
                    1 => NPTypeCode.Int16,
                    2 => NPTypeCode.Int32,
                    4 => NPTypeCode.Int64,
                    _ => NPTypeCode.Double
                };
            }
        }

        // =========================================================================
        // Type Conversion Functions
        // =========================================================================

        /// <summary>
        /// Convert a single value from srcType to dstType.
        /// </summary>
        /// <remarks>
        /// Complex needs special handling on either end because a double intermediate
        /// would drop the imaginary component. Real -> Complex sets imaginary=0; Complex
        /// -> Real takes the real part (matching NumPy's ComplexWarning truncation).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void ConvertValue(void* src, void* dst, NPTypeCode srcType, NPTypeCode dstType)
        {
            // Fast path: same type
            if (srcType == dstType)
            {
                int size = InfoOf.GetSize(srcType);
                Buffer.MemoryCopy(src, dst, size, size);
                return;
            }

            // Complex pathways — go through a Complex intermediate to preserve both
            // real and imaginary components when both endpoints are Complex, and to
            // drop imaginary cleanly on Complex -> real cast (NumPy ComplexWarning).
            if (srcType == NPTypeCode.Complex)
            {
                Complex c = *(Complex*)src;
                if (dstType == NPTypeCode.Complex)
                {
                    *(Complex*)dst = c;
                    return;
                }
                // Complex -> bool is truthy when EITHER part is non-zero (NumPy: bool(z) == (z != 0)).
                // Every other Complex -> real/int target takes the real part (NumPy ComplexWarning).
                if (dstType == NPTypeCode.Boolean)
                {
                    *(bool*)dst = c.Real != 0.0 || c.Imaginary != 0.0;
                    return;
                }
                WriteFromDouble(dst, c.Real, dstType);
                return;
            }
            if (dstType == NPTypeCode.Complex)
            {
                double real = ReadAsDouble(src, srcType);
                *(Complex*)dst = new Complex(real, 0.0);
                return;
            }

            // Read the source through a LOSSLESS intermediate. The previous code read every
            // source as double, which (a) dropped the low bits of Int64/UInt64 and (b) wrote
            // integer destinations with C# saturating float->int / int->int casts. Both diverge
            // from NumPy, which uses MODULAR WRAPPING for integer narrowing and C truncation
            // (int.MinValue sentinel on NaN/overflow) for float->int. Routing through the
            // Converts.* table — proven bit-exact with NumPy across the full 15x15 cast matrix —
            // restores parity.
            switch (srcType)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                    WriteFromDouble(dst, ReadAsDouble(src, srcType), dstType);
                    return;
                case NPTypeCode.UInt64:
                    WriteFromUInt64(dst, *(ulong*)src, dstType);
                    return;
                case NPTypeCode.Decimal:
                    WriteFromDecimal(dst, *(decimal*)src, dstType);
                    return;
                default:
                    // Boolean/Byte/SByte/Int16/UInt16/Int32/UInt32/Int64/Char all fit in long.
                    WriteFromInt64(dst, ReadAsInt64(src, srcType), dstType);
                    return;
            }
        }

        /// <summary>Read an integer-category source (everything except UInt64/Decimal/floats/Complex) losslessly as long.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long ReadAsInt64(void* ptr, NPTypeCode type)
            => type switch
            {
                NPTypeCode.Boolean => *(bool*)ptr ? 1L : 0L,
                NPTypeCode.Byte => *(byte*)ptr,
                NPTypeCode.SByte => *(sbyte*)ptr,
                NPTypeCode.Int16 => *(short*)ptr,
                NPTypeCode.UInt16 => *(ushort*)ptr,
                NPTypeCode.Int32 => *(int*)ptr,
                NPTypeCode.UInt32 => *(uint*)ptr,
                NPTypeCode.Int64 => *(long*)ptr,
                NPTypeCode.Char => *(char*)ptr,
                _ => throw new NotSupportedException($"ReadAsInt64: unsupported source {type}")
            };

        /// <summary>
        /// Read any numeric type (except Complex) as double.
        /// Complex must be handled by the caller — going through double would silently
        /// drop the imaginary component.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static double ReadAsDouble(void* ptr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => *(bool*)ptr ? 1.0 : 0.0,
                NPTypeCode.Byte => *(byte*)ptr,
                NPTypeCode.SByte => *(sbyte*)ptr,
                NPTypeCode.Int16 => *(short*)ptr,
                NPTypeCode.UInt16 => *(ushort*)ptr,
                NPTypeCode.Int32 => *(int*)ptr,
                NPTypeCode.UInt32 => *(uint*)ptr,
                NPTypeCode.Int64 => *(long*)ptr,
                NPTypeCode.UInt64 => *(ulong*)ptr,
                NPTypeCode.Half => (double)*(Half*)ptr,
                NPTypeCode.Single => *(float*)ptr,
                NPTypeCode.Double => *(double*)ptr,
                NPTypeCode.Decimal => (double)*(decimal*)ptr,
                NPTypeCode.Char => *(char*)ptr,
                _ => throw new NotSupportedException($"Unsupported type: {type}")
            };
        }

        /// <summary>Integer source (fits in long): integer-&gt;integer wraps (NumPy modular); integer-&gt;float/decimal is a plain conversion.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void WriteFromInt64(void* ptr, long v, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean: *(bool*)ptr = v != 0; break;
                case NPTypeCode.Byte: *(byte*)ptr = unchecked((byte)v); break;
                case NPTypeCode.SByte: *(sbyte*)ptr = unchecked((sbyte)v); break;
                case NPTypeCode.Int16: *(short*)ptr = unchecked((short)v); break;
                case NPTypeCode.UInt16: *(ushort*)ptr = unchecked((ushort)v); break;
                case NPTypeCode.Int32: *(int*)ptr = unchecked((int)v); break;
                case NPTypeCode.UInt32: *(uint*)ptr = unchecked((uint)v); break;
                case NPTypeCode.Int64: *(long*)ptr = v; break;
                case NPTypeCode.UInt64: *(ulong*)ptr = unchecked((ulong)v); break;
                case NPTypeCode.Char: *(char*)ptr = unchecked((char)v); break;
                case NPTypeCode.Half: *(Half*)ptr = (Half)(double)v; break;
                case NPTypeCode.Single: *(float*)ptr = v; break;
                case NPTypeCode.Double: *(double*)ptr = v; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = v; break;
                default: throw new NotSupportedException($"Unsupported type: {type}");
            }
        }

        /// <summary>UInt64 source: same wrapping rules, but the value can exceed long range.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void WriteFromUInt64(void* ptr, ulong v, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean: *(bool*)ptr = v != 0; break;
                case NPTypeCode.Byte: *(byte*)ptr = unchecked((byte)v); break;
                case NPTypeCode.SByte: *(sbyte*)ptr = unchecked((sbyte)v); break;
                case NPTypeCode.Int16: *(short*)ptr = unchecked((short)v); break;
                case NPTypeCode.UInt16: *(ushort*)ptr = unchecked((ushort)v); break;
                case NPTypeCode.Int32: *(int*)ptr = unchecked((int)v); break;
                case NPTypeCode.UInt32: *(uint*)ptr = unchecked((uint)v); break;
                case NPTypeCode.Int64: *(long*)ptr = unchecked((long)v); break;
                case NPTypeCode.UInt64: *(ulong*)ptr = v; break;
                case NPTypeCode.Char: *(char*)ptr = unchecked((char)v); break;
                case NPTypeCode.Half: *(Half*)ptr = (Half)(double)v; break;
                case NPTypeCode.Single: *(float*)ptr = v; break;
                case NPTypeCode.Double: *(double*)ptr = v; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = v; break;
                default: throw new NotSupportedException($"Unsupported type: {type}");
            }
        }

        /// <summary>
        /// Float source (and Complex-&gt;real): NumPy float-&gt;int truncates toward zero with an
        /// int.MinValue/overflow sentinel — delegated to Converts.* for exact parity; float-&gt;float
        /// is plain rounding.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void WriteFromDouble(void* ptr, double value, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean: *(bool*)ptr = value != 0; break;
                case NPTypeCode.Byte: *(byte*)ptr = Converts.ToByte(value); break;
                case NPTypeCode.SByte: *(sbyte*)ptr = Converts.ToSByte(value); break;
                case NPTypeCode.Int16: *(short*)ptr = Converts.ToInt16(value); break;
                case NPTypeCode.UInt16: *(ushort*)ptr = Converts.ToUInt16(value); break;
                case NPTypeCode.Int32: *(int*)ptr = Converts.ToInt32(value); break;
                case NPTypeCode.UInt32: *(uint*)ptr = Converts.ToUInt32(value); break;
                case NPTypeCode.Int64: *(long*)ptr = Converts.ToInt64(value); break;
                case NPTypeCode.UInt64: *(ulong*)ptr = Converts.ToUInt64(value); break;
                case NPTypeCode.Char: *(char*)ptr = Converts.ToChar(value); break;
                case NPTypeCode.Half: *(Half*)ptr = (Half)value; break;
                case NPTypeCode.Single: *(float*)ptr = (float)value; break;
                case NPTypeCode.Double: *(double*)ptr = value; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = Converts.ToDecimal(value); break;
                default: throw new NotSupportedException($"Unsupported type: {type}");
            }
        }

        /// <summary>Decimal source: integer destinations truncate via Converts (NumPy parity); float/decimal are plain.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void WriteFromDecimal(void* ptr, decimal value, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean: *(bool*)ptr = value != 0; break;
                case NPTypeCode.Byte: *(byte*)ptr = Converts.ToByte(value); break;
                case NPTypeCode.SByte: *(sbyte*)ptr = Converts.ToSByte(value); break;
                case NPTypeCode.Int16: *(short*)ptr = Converts.ToInt16(value); break;
                case NPTypeCode.UInt16: *(ushort*)ptr = Converts.ToUInt16(value); break;
                case NPTypeCode.Int32: *(int*)ptr = Converts.ToInt32(value); break;
                case NPTypeCode.UInt32: *(uint*)ptr = Converts.ToUInt32(value); break;
                case NPTypeCode.Int64: *(long*)ptr = Converts.ToInt64(value); break;
                case NPTypeCode.UInt64: *(ulong*)ptr = Converts.ToUInt64(value); break;
                case NPTypeCode.Char: *(char*)ptr = Converts.ToChar(value); break;
                case NPTypeCode.Half: *(Half*)ptr = (Half)(double)value; break;
                case NPTypeCode.Single: *(float*)ptr = (float)value; break;
                case NPTypeCode.Double: *(double*)ptr = (double)value; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = value; break;
                default: throw new NotSupportedException($"Unsupported type: {type}");
            }
        }

        /// <summary>
        /// Copy array data with type conversion.
        /// </summary>
        public static void CopyWithCast(
            void* src, long srcStride, NPTypeCode srcType,
            void* dst, long dstStride, NPTypeCode dstType,
            long count)
        {
            int srcElemSize = InfoOf.GetSize(srcType);
            int dstElemSize = InfoOf.GetSize(dstType);

            byte* srcPtr = (byte*)src;
            byte* dstPtr = (byte*)dst;

            for (long i = 0; i < count; i++)
            {
                ConvertValue(srcPtr, dstPtr, srcType, dstType);
                srcPtr += srcStride * srcElemSize;
                dstPtr += dstStride * dstElemSize;
            }
        }

        /// <summary>
        /// Copy strided data to contiguous buffer with type conversion.
        /// </summary>
        public static void CopyStridedToContiguousWithCast(
            void* src, long* strides, NPTypeCode srcType,
            void* dst, NPTypeCode dstType,
            long* shape, int ndim, long count)
        {
            int srcElemSize = InfoOf.GetSize(srcType);
            int dstElemSize = InfoOf.GetSize(dstType);

            byte* srcBase = (byte*)src;
            byte* dstPtr = (byte*)dst;

            var coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            for (long i = 0; i < count; i++)
            {
                // Calculate source offset
                long srcOffset = 0;
                for (int d = 0; d < ndim; d++)
                    srcOffset += coords[d] * strides[d];

                ConvertValue(srcBase + srcOffset * srcElemSize, dstPtr, srcType, dstType);
                dstPtr += dstElemSize;

                // Advance coordinates
                for (int d = ndim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        /// <summary>
        /// Copy contiguous buffer to strided data with type conversion.
        /// </summary>
        public static void CopyContiguousToStridedWithCast(
            void* src, NPTypeCode srcType,
            void* dst, long* strides, NPTypeCode dstType,
            long* shape, int ndim, long count)
        {
            int srcElemSize = InfoOf.GetSize(srcType);
            int dstElemSize = InfoOf.GetSize(dstType);

            byte* srcPtr = (byte*)src;
            byte* dstBase = (byte*)dst;

            var coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            for (long i = 0; i < count; i++)
            {
                // Calculate destination offset
                long dstOffset = 0;
                for (int d = 0; d < ndim; d++)
                    dstOffset += coords[d] * strides[d];

                ConvertValue(srcPtr, dstBase + dstOffset * dstElemSize, srcType, dstType);
                srcPtr += srcElemSize;

                // Advance coordinates
                for (int d = ndim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        /// <summary>
        /// Copy strided source to strided destination with type conversion.
        /// Handles broadcast on the source via stride=0 dimensions and arbitrary
        /// destination strides. Strides are in element counts (not bytes); element
        /// size multiplication happens internally via <see cref="InfoOf.GetSize"/>.
        ///
        /// Primitive (non-Complex/Decimal) pairs resolve a typed
        /// <see cref="Converts.FindConverter{TIn,TOut}"/> delegate ONCE and run a typed
        /// inner loop — no per-element type dispatch. Complex/Decimal endpoints keep the
        /// scalar <see cref="ConvertValue"/> path: it encodes their exact NumPy semantics
        /// (complex real-part drop / truthy-bool) and their FindConverter converters can box;
        /// both are 16-byte so the per-element scalar cost amortizes either way.
        /// </summary>
        public static void CopyStridedToStridedWithCast(
            void* src, long* srcStrides, NPTypeCode srcType,
            void* dst, long* dstStrides, NPTypeCode dstType,
            long* shape, int ndim, long count)
        {
            if (count == 0)
                return;

            if (IsComplexOrDecimal(srcType) || IsComplexOrDecimal(dstType))
                CastStridedScalar(src, srcStrides, srcType, dst, dstStrides, dstType, shape, ndim, count);
            else
                CastStridedTypedDst(src, srcStrides, srcType, dst, dstStrides, dstType, shape, ndim, count);
        }

        private static bool IsComplexOrDecimal(NPTypeCode t)
            => t == NPTypeCode.Complex || t == NPTypeCode.Decimal;

        // ---- Typed fast path (primitive↔primitive). Two nested switches resolve TOut then
        //      TIn (13 arms each), reaching CastStridedTyped<TIn,TOut> which hoists the
        //      converter out of the loop. -------------------------------------------------
        private static void CastStridedTypedDst(
            void* src, long* ss, NPTypeCode srcType,
            void* dst, long* ds, NPTypeCode dstType,
            long* shape, int ndim, long count)
        {
            switch (dstType)
            {
                case NPTypeCode.Boolean: CastStridedTypedSrc<bool>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Byte: CastStridedTypedSrc<byte>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.SByte: CastStridedTypedSrc<sbyte>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Int16: CastStridedTypedSrc<short>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.UInt16: CastStridedTypedSrc<ushort>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Int32: CastStridedTypedSrc<int>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.UInt32: CastStridedTypedSrc<uint>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Int64: CastStridedTypedSrc<long>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.UInt64: CastStridedTypedSrc<ulong>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Char: CastStridedTypedSrc<char>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Half: CastStridedTypedSrc<Half>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Single: CastStridedTypedSrc<float>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Double: CastStridedTypedSrc<double>(src, ss, srcType, dst, ds, shape, ndim, count); break;
                default: CastStridedScalar(src, ss, srcType, dst, ds, dstType, shape, ndim, count); break;
            }
        }

        private static void CastStridedTypedSrc<TOut>(
            void* src, long* ss, NPTypeCode srcType,
            void* dst, long* ds, long* shape, int ndim, long count) where TOut : unmanaged
        {
            switch (srcType)
            {
                case NPTypeCode.Boolean: CastStridedTyped<bool, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Byte: CastStridedTyped<byte, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.SByte: CastStridedTyped<sbyte, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Int16: CastStridedTyped<short, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.UInt16: CastStridedTyped<ushort, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Int32: CastStridedTyped<int, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.UInt32: CastStridedTyped<uint, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Int64: CastStridedTyped<long, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.UInt64: CastStridedTyped<ulong, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Char: CastStridedTyped<char, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Half: CastStridedTyped<Half, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Single: CastStridedTyped<float, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                case NPTypeCode.Double: CastStridedTyped<double, TOut>(src, ss, dst, ds, shape, ndim, count); break;
                default: throw new NotSupportedException($"primitive cast source {srcType}");
            }
        }

        /// <summary>
        /// Typed strided cast. <see cref="Converts.FindConverter{TIn,TOut}"/> is resolved ONCE
        /// (a non-boxing <c>Converts.{Src}To{Dst}</c> method-group delegate for every primitive
        /// pair) and applied with the same incremental-coord + tight-inner-run addressing as the
        /// scalar path — but with zero per-element type dispatch.
        /// </summary>
        private static void CastStridedTyped<TIn, TOut>(
            void* src, long* srcStrides, void* dst, long* dstStrides,
            long* shape, int ndim, long count) where TIn : unmanaged where TOut : unmanaged
        {
            // FindConverter resolves a non-boxing Converts.{Src}To{Dst} method-group delegate
            // ONCE; it stays uniform across all primitive pairs (0.65-0.71x NumPy). Converts.
            // ChangeType<TIn,TOut> called per element is faster for bool/char (~0.86x) but
            // FALLS TO A BOXING DEFAULT for Half inputs (~0.23x — worse than the scalar path),
            // so the resolved delegate is the safe, landmine-free choice.
            var convert = Converts.FindConverter<TIn, TOut>();
            var srcBase = (TIn*)src;
            var dstBase = (TOut*)dst;

            if (ndim == 0)
            {
                dstBase[0] = convert(srcBase[0]);
                return;
            }

            int last = ndim - 1;
            long inner = shape[last];
            long srcInner = srcStrides[last]; // element strides; TIn*/TOut* index handles sizing
            long dstInner = dstStrides[last];

            long outerCount = count / inner;
            long* coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            long srcRunOff = 0, dstRunOff = 0;
            for (long r = 0; r < outerCount; r++)
            {
                long s = srcRunOff, dd = dstRunOff;
                for (long i = 0; i < inner; i++)
                {
                    dstBase[dd] = convert(srcBase[s]);
                    s += srcInner;
                    dd += dstInner;
                }

                for (int d = last - 1; d >= 0; d--)
                {
                    srcRunOff += srcStrides[d];
                    dstRunOff += dstStrides[d];
                    if (++coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                    srcRunOff -= srcStrides[d] * shape[d];
                    dstRunOff -= dstStrides[d] * shape[d];
                }
            }
        }

        /// <summary>
        /// Scalar strided cast via <see cref="ConvertValue"/> — incremental-coord + tight-inner-run
        /// addressing (no per-element Σ coords·strides recompute), conversion kept verbatim. Used
        /// for Complex/Decimal endpoints (exact NumPy semantics, box-free) and as the safety
        /// fallback for any unmapped dst.
        /// </summary>
        private static void CastStridedScalar(
            void* src, long* srcStrides, NPTypeCode srcType,
            void* dst, long* dstStrides, NPTypeCode dstType,
            long* shape, int ndim, long count)
        {
            int srcElemSize = InfoOf.GetSize(srcType);
            int dstElemSize = InfoOf.GetSize(dstType);

            byte* srcBase = (byte*)src;
            byte* dstBase = (byte*)dst;

            if (ndim == 0)
            {
                ConvertValue(srcBase, dstBase, srcType, dstType);
                return;
            }

            int last = ndim - 1;
            long inner = shape[last];
            long srcInnerB = srcStrides[last] * srcElemSize; // inner byte step (may be 0/neg)
            long dstInnerB = dstStrides[last] * dstElemSize;

            long outerCount = count / inner; // == product(shape[0..last-1])
            long* coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            long srcRunOff = 0, dstRunOff = 0;
            for (long r = 0; r < outerCount; r++)
            {
                byte* sp = srcBase + srcRunOff * srcElemSize;
                byte* dp = dstBase + dstRunOff * dstElemSize;
                for (long i = 0; i < inner; i++)
                {
                    ConvertValue(sp, dp, srcType, dstType);
                    sp += srcInnerB;
                    dp += dstInnerB;
                }

                for (int d = last - 1; d >= 0; d--)
                {
                    srcRunOff += srcStrides[d];
                    dstRunOff += dstStrides[d];
                    if (++coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                    srcRunOff -= srcStrides[d] * shape[d];
                    dstRunOff -= dstStrides[d] * shape[d];
                }
            }
        }
    }
}
