using System;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Type casting utilities for NpyIter.
    /// Validates casting rules and performs type conversions.
    /// </summary>
    internal static unsafe class NpyIterCasting
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
        /// Safe casts: smaller int -> larger int, any int -> float64, float32 -> float64
        /// </summary>
        private static bool IsSafeCast(NPTypeCode srcType, NPTypeCode dstType)
        {
            // Same type is always safe
            if (srcType == dstType)
                return true;

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

            // Signed to unsigned is not safe
            if (srcIsSigned && dstIsUnsigned)
                return false;

            // Unsigned to signed requires larger type
            if (srcIsUnsigned && dstIsSigned && srcSize >= dstSize)
                return false;

            // Same signedness, smaller to larger is safe
            if ((srcIsSigned && dstIsSigned) || (srcIsUnsigned && dstIsUnsigned))
                return srcSize <= dstSize;

            // For boolean
            if (srcType == NPTypeCode.Boolean)
                return true;  // Bool can safely convert to any numeric

            return false;
        }

        /// <summary>
        /// Check if casting is "same kind" (both integers, or both floats).
        /// </summary>
        private static bool IsSameKindCast(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (srcType == dstType)
                return true;

            bool srcIsFloat = IsFloatingPoint(srcType);
            bool dstIsFloat = IsFloatingPoint(dstType);
            bool srcIsInt = IsSignedInteger(srcType) || IsUnsignedInteger(srcType);
            bool dstIsInt = IsSignedInteger(dstType) || IsUnsignedInteger(dstType);

            // Same kind = both floats or both integers
            if (srcIsFloat && dstIsFloat)
                return true;
            if (srcIsInt && dstIsInt)
                return true;

            // Boolean is compatible with integers
            if (srcType == NPTypeCode.Boolean && dstIsInt)
                return true;
            if (srcIsInt && dstType == NPTypeCode.Boolean)
                return true;

            return false;
        }

        private static bool IsFloatingPoint(NPTypeCode type)
        {
            return type == NPTypeCode.Single || type == NPTypeCode.Double || type == NPTypeCode.Decimal;
        }

        private static bool IsSignedInteger(NPTypeCode type)
        {
            return type == NPTypeCode.Int16 || type == NPTypeCode.Int32 || type == NPTypeCode.Int64;
        }

        private static bool IsUnsignedInteger(NPTypeCode type)
        {
            return type == NPTypeCode.Byte || type == NPTypeCode.UInt16 ||
                   type == NPTypeCode.UInt32 || type == NPTypeCode.UInt64 || type == NPTypeCode.Char;
        }

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
        /// Promote two types to a common type.
        /// </summary>
        private static NPTypeCode PromoteTypes(NPTypeCode a, NPTypeCode b)
        {
            if (a == b)
                return a;

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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ConvertValue(void* src, void* dst, NPTypeCode srcType, NPTypeCode dstType)
        {
            // Fast path: same type
            if (srcType == dstType)
            {
                int size = InfoOf.GetSize(srcType);
                Buffer.MemoryCopy(src, dst, size, size);
                return;
            }

            // Read source value as double (intermediate)
            double value = ReadAsDouble(src, srcType);

            // Write to destination
            WriteFromDouble(dst, value, dstType);
        }

        /// <summary>
        /// Read any numeric type as double.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ReadAsDouble(void* ptr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => *(bool*)ptr ? 1.0 : 0.0,
                NPTypeCode.Byte => *(byte*)ptr,
                NPTypeCode.Int16 => *(short*)ptr,
                NPTypeCode.UInt16 => *(ushort*)ptr,
                NPTypeCode.Int32 => *(int*)ptr,
                NPTypeCode.UInt32 => *(uint*)ptr,
                NPTypeCode.Int64 => *(long*)ptr,
                NPTypeCode.UInt64 => *(ulong*)ptr,
                NPTypeCode.Single => *(float*)ptr,
                NPTypeCode.Double => *(double*)ptr,
                NPTypeCode.Decimal => (double)*(decimal*)ptr,
                NPTypeCode.Char => *(char*)ptr,
                _ => throw new NotSupportedException($"Unsupported type: {type}")
            };
        }

        /// <summary>
        /// Write double value to any numeric type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteFromDouble(void* ptr, double value, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean: *(bool*)ptr = value != 0; break;
                case NPTypeCode.Byte: *(byte*)ptr = (byte)value; break;
                case NPTypeCode.Int16: *(short*)ptr = (short)value; break;
                case NPTypeCode.UInt16: *(ushort*)ptr = (ushort)value; break;
                case NPTypeCode.Int32: *(int*)ptr = (int)value; break;
                case NPTypeCode.UInt32: *(uint*)ptr = (uint)value; break;
                case NPTypeCode.Int64: *(long*)ptr = (long)value; break;
                case NPTypeCode.UInt64: *(ulong*)ptr = (ulong)value; break;
                case NPTypeCode.Single: *(float*)ptr = (float)value; break;
                case NPTypeCode.Double: *(double*)ptr = value; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = (decimal)value; break;
                case NPTypeCode.Char: *(char*)ptr = (char)value; break;
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
        /// </summary>
        public static void CopyStridedToStridedWithCast(
            void* src, long* srcStrides, NPTypeCode srcType,
            void* dst, long* dstStrides, NPTypeCode dstType,
            long* shape, int ndim, long count)
        {
            int srcElemSize = InfoOf.GetSize(srcType);
            int dstElemSize = InfoOf.GetSize(dstType);

            byte* srcBase = (byte*)src;
            byte* dstBase = (byte*)dst;

            var coords = stackalloc long[Math.Max(1, ndim)];
            for (int d = 0; d < ndim; d++)
                coords[d] = 0;

            for (long i = 0; i < count; i++)
            {
                long srcOffset = 0;
                long dstOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    srcOffset += coords[d] * srcStrides[d];
                    dstOffset += coords[d] * dstStrides[d];
                }

                ConvertValue(
                    srcBase + srcOffset * srcElemSize,
                    dstBase + dstOffset * dstElemSize,
                    srcType, dstType);

                // Advance coordinates (innermost-first for C-order traversal).
                for (int d = ndim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }
    }
}
