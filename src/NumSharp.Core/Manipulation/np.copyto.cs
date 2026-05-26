using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Copies values from one array to another, broadcasting as necessary.
        /// </summary>
        /// <param name="dst">The array into which values are copied.</param>
        /// <param name="src">The array from which values are copied.</param>
        /// <param name="casting">Controls what kind of data casting may occur when copying. Default <c>"same_kind"</c>.
        ///     Allowed values: <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>, <c>"same_kind"</c>, <c>"unsafe"</c>.</param>
        /// <param name="where">Optional boolean mask broadcast to <paramref name="dst"/>'s shape. Elements of
        ///     <paramref name="src"/> are only written to <paramref name="dst"/> where the mask is <c>true</c>.
        ///     <c>null</c> (default) is equivalent to <c>where=True</c> — every element is copied.</param>
        /// <exception cref="ArgumentException">If <paramref name="dst"/> is read-only, or <paramref name="casting"/>
        ///     is not a recognised casting name, or <paramref name="where"/> is not a boolean array.</exception>
        /// <exception cref="InvalidCastException">If casting from <paramref name="src"/>'s dtype to
        ///     <paramref name="dst"/>'s dtype is not allowed under the chosen rule (NumPy raises
        ///     <c>TypeError</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.copyto.html</remarks>
        public static void copyto(NDArray dst, NDArray src, string casting = "same_kind", NDArray @where = null)
        {
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));

            if (src is null)
                throw new ArgumentNullException(nameof(src));

            // NumPy raises ValueError on write to a read-only destination — the closest .NET
            // analogue is ArgumentException so callers can catch the canonical "value error" type.
            if (!dst.Shape.IsWriteable)
                throw new ArgumentException("assignment destination is read-only", nameof(dst));

            // NumPy raises ValueError for unrecognised casting names.
            NPY_CASTING castingRule = ParseCastingName(casting);

            // NumPy raises TypeError when the cast violates the rule.
            if (!NpyIterCasting.CanCast(src.GetTypeCode, dst.GetTypeCode, castingRule))
            {
                throw new InvalidCastException(
                    $"Cannot cast array data from dtype('{src.GetTypeCode.AsNumpyDtypeName()}') " +
                    $"to dtype('{dst.GetTypeCode.AsNumpyDtypeName()}') " +
                    $"according to the rule '{CastingRuleName(castingRule)}'");
            }

            if (@where is null)
            {
                NpyIter.Copy(dst, src);
                return;
            }

            if (@where.GetTypeCode != NPTypeCode.Boolean)
                throw new ArgumentException(
                    $"where must be a boolean array, got dtype('{@where.GetTypeCode.AsNumpyDtypeName()}')",
                    nameof(@where));

            // 0-d scalar mask short-circuit (matches NumPy array_assign_array.c:433-446).
            // Skips broadcasting+per-element iteration when the mask is unambiguous:
            //   where=False → no-op, where=True → fall through to the unmasked fast path.
            if (@where.Shape.IsScalar || @where.size == 1)
            {
                bool value = ReadScalarBool(@where);
                if (!value)
                    return;
                NpyIter.Copy(dst, src);
                return;
            }

            CopyWithMask(dst, src, @where);
        }

        private static unsafe bool ReadScalarBool(NDArray array)
        {
            byte* basePtr = array.Storage.Address + array.Shape.offset * InfoOf.GetSize(NPTypeCode.Boolean);
            return *(bool*)basePtr;
        }

        private static NPY_CASTING ParseCastingName(string casting)
        {
            if (casting is null)
                throw new ArgumentNullException(nameof(casting));

            switch (casting)
            {
                case "no": return NPY_CASTING.NPY_NO_CASTING;
                case "equiv": return NPY_CASTING.NPY_EQUIV_CASTING;
                case "safe": return NPY_CASTING.NPY_SAFE_CASTING;
                case "same_kind": return NPY_CASTING.NPY_SAME_KIND_CASTING;
                case "unsafe": return NPY_CASTING.NPY_UNSAFE_CASTING;
                default:
                    throw new ArgumentException(
                        $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')",
                        nameof(casting));
            }
        }

        private static string CastingRuleName(NPY_CASTING casting)
        {
            switch (casting)
            {
                case NPY_CASTING.NPY_NO_CASTING: return "no";
                case NPY_CASTING.NPY_EQUIV_CASTING: return "equiv";
                case NPY_CASTING.NPY_SAFE_CASTING: return "safe";
                case NPY_CASTING.NPY_SAME_KIND_CASTING: return "same_kind";
                case NPY_CASTING.NPY_UNSAFE_CASTING: return "unsafe";
                default: return "unknown";
            }
        }

        /// <summary>
        ///     Conditional copy: writes <paramref name="src"/> into <paramref name="dst"/> only at positions
        ///     where the broadcast <paramref name="where"/> mask is <c>true</c>. <paramref name="src"/> and
        ///     <paramref name="where"/> are both broadcast to <paramref name="dst"/>'s shape; broadcast
        ///     incompatibility surfaces as an <see cref="IncorrectShapeException"/> from <see cref="broadcast_to"/>.
        ///     Iterates in C-order to match NumPy's element traversal.
        /// </summary>
        private static unsafe void CopyWithMask(NDArray dst, NDArray src, NDArray @where)
        {
            // Broadcast src + mask shapes against dst — validates compatibility and yields
            // shape-matched views with stride=0 along stretched dimensions.
            Shape srcShape = np.broadcast_to(src.Shape, dst.Shape);
            Shape maskShape = np.broadcast_to(@where.Shape, dst.Shape);
            Shape dstShape = dst.Shape;

            long size = dstShape.size;
            if (size == 0)
                return;

            int ndim = dstShape.NDim;
            NPTypeCode srcType = src.GetTypeCode;
            NPTypeCode dstType = dst.GetTypeCode;
            int srcElemSize = InfoOf.GetSize(srcType);
            int dstElemSize = InfoOf.GetSize(dstType);
            int maskElemSize = InfoOf.GetSize(NPTypeCode.Boolean);

            byte* srcBase = src.Storage.Address + srcShape.offset * srcElemSize;
            byte* dstBase = dst.Storage.Address + dstShape.offset * dstElemSize;
            byte* maskBase = @where.Storage.Address + maskShape.offset * maskElemSize;

            // 0-d scalar destination: single conditional element.
            if (ndim == 0)
            {
                if (*(bool*)maskBase)
                    NpyIterCasting.ConvertValue(srcBase, dstBase, srcType, dstType);
                return;
            }

            long* shape = stackalloc long[ndim];
            long* srcStrides = stackalloc long[ndim];
            long* dstStrides = stackalloc long[ndim];
            long* maskStrides = stackalloc long[ndim];

            for (int d = 0; d < ndim; d++)
            {
                shape[d] = dstShape.dimensions[d];
                srcStrides[d] = srcShape.strides[d];
                dstStrides[d] = dstShape.strides[d];
                maskStrides[d] = maskShape.strides[d];
            }

            // IL fast path: SIMD masked-cast kernel (ConditionalSelect for 1:1 lane strategies;
            // scalar inner loop with mask gate + inline conversion for widen/narrow strategies).
            // Both paths use incremental coord advance — no mod/div per element.
            var maskedKernel = NumSharp.Backends.Kernels.DirectILKernelGenerator
                .TryGetMaskedCastKernel(srcType, dstType);
            if (maskedKernel != null)
            {
                maskedKernel(srcBase, dstBase, maskBase, srcStrides, dstStrides, maskStrides, shape, ndim);
                return;
            }

            // Scalar fallback for unsupported types (Decimal/Complex/Half/Char/Boolean involved).
            long* coords = stackalloc long[ndim];
            for (int d = 0; d < ndim; d++) coords[d] = 0;

            for (long i = 0; i < size; i++)
            {
                long srcOffset = 0;
                long dstOffset = 0;
                long maskOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    srcOffset += coords[d] * srcStrides[d];
                    dstOffset += coords[d] * dstStrides[d];
                    maskOffset += coords[d] * maskStrides[d];
                }

                if (*(bool*)(maskBase + maskOffset * maskElemSize))
                {
                    NpyIterCasting.ConvertValue(
                        srcBase + srcOffset * srcElemSize,
                        dstBase + dstOffset * dstElemSize,
                        srcType, dstType);
                }

                // Advance C-order (innermost dimension changes fastest).
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
