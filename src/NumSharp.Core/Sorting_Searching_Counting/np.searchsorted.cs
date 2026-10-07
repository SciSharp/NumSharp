using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <exception cref="ArgumentException"><paramref name="side"/> is not <c>"left"</c> / <c>"right"</c> (NumPy's two
        ///     ValueError texts).</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is not 1-D, a sorter of uint64, or a sorter whose size differs
        ///     from <paramref name="a"/>'s.</exception>
        /// <exception cref="TypeError">A sorter that is not 1-D (<c>could not parse sorter argument</c>) or not integer
        ///     (<c>sorter must only contain integers</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text <paramref name="a"/> with elements: NumPy searches
        ///     it as text (the computational phase).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static long searchsorted(NDArray a, int v, string side = "left", NDArray sorter = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, sorter);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy's checks run first, and an empty haystack answers without a comparison.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                using var key = NDArray.Scalar(v);
                using var answer = SearchSortedTextAnswer(a, key, side, sorter);
                return answer.GetInt64();
            }
            ValidateSearchSorted(a, side, sorter);
            // NumPy searches in result_type(a, v), NEVER casts the key down to a's dtype
            // (see the NDArray overload below). A C# int literal is a strong int32 scalar.
            NPTypeCode common = promote_types(a.typecode, NPTypeCode.Int32);
            NDArray aP = a.typecode == common ? a : a.astype(common);
            return SearchSortedScalar(aP, Converts.ChangeType(v, common), side == "left", sorter);
        }

        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <exception cref="ArgumentException">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="ValueError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="TypeError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        public static long searchsorted(NDArray a, double v, string side = "left", NDArray sorter = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, sorter);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy's checks run first, and an empty haystack answers without a comparison.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                using var key = NDArray.Scalar(v);
                using var answer = SearchSortedTextAnswer(a, key, side, sorter);
                return answer.GetInt64();
            }
            ValidateSearchSorted(a, side, sorter);
            // A C# double is a strong float64 scalar; NumPy searches a float32 array for a
            // Python float in float64 too (probed 2.4.2), so promote_types(a, Double) matches.
            NPTypeCode common = promote_types(a.typecode, NPTypeCode.Double);
            NDArray aP = a.typecode == common ? a : a.astype(common);
            return SearchSortedScalar(aP, Converts.ChangeType(v, common), side == "left", sorter);
        }

        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Find the indices into a sorted array <paramref name="a"/> such that, if the corresponding elements
        /// in <paramref name="v"/> were inserted before the indices, the order of <paramref name="a"/> would be preserved.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Values to insert into <paramref name="a"/>. May be a scalar or any shape.</param>
        /// <param name="side">If "left" (default), the index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Array of insertion points with the same shape as <paramref name="v"/>, or a scalar if <paramref name="v"/> is a scalar.</returns>
        /// <exception cref="ArgumentException">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="ValueError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="TypeError">See <see cref="searchsorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="UnicodeDecodeError">A text search converts bytes that are not ASCII to the common <c>U</c> dtype.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text in <paramref name="a"/> or <paramref name="v"/> with
        ///     something to compare: NumPy searches as text (the computational phase). No key, or an empty haystack, answers
        ///     without a comparison.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        [NDScoped]
        public static NDArray searchsorted(NDArray a, NDArray v, string side = "left", NDArray sorter = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, v, sorter);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy's checks run first, and no key / an empty haystack answers without a comparison.
            if (Backends.Strings.NDStringUfuncs.IsString(a) || Backends.Strings.NDStringUfuncs.IsString(v))
                return SearchSortedTextAnswer(a, v, side, sorter);
            ValidateSearchSorted(a, side, sorter);
            bool leftSide = side == "left";

            // NumPy (PyArray_SearchSorted) promotes BOTH the sorted array `a` and the keys `v`
            // to their common dtype (result_type) and searches there. It must never cast the
            // keys DOWN to a's dtype — that would truncate/wrap them, e.g.
            //   searchsorted([1,2,3,4] int, 2.5) == 2, not floor(2.5) giving index 1;
            //   searchsorted([1,2,3,4] int8, 300) == 4, not (int8)300 giving a wrapped key.
            NPTypeCode common = promote_types(a.typecode, v.typecode);
            NDArray aP = a.typecode == common ? a : a.astype(common);

            if (v.Shape.IsScalar)
            {
                object scalar = Converts.ChangeType(v.Storage.GetAtIndex(0), common);
                long idx = SearchSortedScalar(aP, scalar, leftSide, sorter);
                return NDArray.Scalar(idx);
            }

            Shape outShape = Shape.Adopt(v.Shape.dimensions);

            if (v.size == 0)
                return new NDArray(NPTypeCode.Int64, outShape, false);

            // Promote v to the common dtype and force contiguous (NumPy: PyArray_CARRAY_RO).
            NDArray vTyped = EnsureContiguousOfType(v, common);
            NDArray sorterTyped = (sorter is null) ? null : EnsureContiguousInt64(sorter);
            NDArray output = new NDArray(NPTypeCode.Int64, outShape, false);

            int elemSize = DirectILKernelGenerator.GetTypeSize(common);
            unsafe
            {
                bool contigA = aP.Shape.IsContiguous;
                var kernel = DirectILKernelGenerator.GetSearchSortedKernel(common, leftSide, sorter is not null, contigA);
                long arrStride = contigA ? elemSize : aP.Shape.strides[0] * elemSize;
                void* arrPtr = (void*)((byte*)aP.Storage.Address + aP.Shape.offset * elemSize);
                void* keyPtr = (void*)vTyped.Address;
                void* sorterPtr = sorterTyped is null ? null : (void*)sorterTyped.Address;
                long* retPtr = (long*)output.Address;
                kernel(arrPtr, aP.size, arrStride, keyPtr, vTyped.size, sorterPtr, retPtr);
            }

            return output;
        }

        private static long SearchSortedScalar(NDArray a, object scalarValue, bool leftSide, NDArray sorter)
        {
            NDArray sorterTyped = (sorter is null) ? null : EnsureContiguousInt64(sorter);
            int elemSize = DirectILKernelGenerator.GetTypeSize(a.typecode);
            long result;
            unsafe
            {
                // 16-byte buffer fits all 15 dtypes (largest = decimal/complex at 16 bytes).
                byte* keyBuf = stackalloc byte[16];
                WriteScalar(keyBuf, scalarValue, a.typecode);
                long* retBuf = stackalloc long[1];
                bool contigA = a.Shape.IsContiguous;
                var kernel = DirectILKernelGenerator.GetSearchSortedKernel(a.typecode, leftSide, sorter is not null, contigA);
                long arrStride = contigA ? elemSize : a.Shape.strides[0] * elemSize;
                void* arrPtr = (void*)((byte*)a.Storage.Address + a.Shape.offset * elemSize);
                void* sorterPtr = sorterTyped is null ? null : (void*)sorterTyped.Address;
                kernel(arrPtr, a.size, arrStride, keyBuf, 1, sorterPtr, retBuf);
                result = retBuf[0];
            }
            return result;
        }

        private static unsafe void WriteScalar(byte* dest, object value, NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Boolean: *(bool*)dest = (bool)value; break;
                case NPTypeCode.SByte:   *(sbyte*)dest = (sbyte)value; break;
                case NPTypeCode.Byte:    *(byte*)dest = (byte)value; break;
                case NPTypeCode.Int16:   *(short*)dest = (short)value; break;
                case NPTypeCode.UInt16:  *(ushort*)dest = (ushort)value; break;
                case NPTypeCode.Int32:   *(int*)dest = (int)value; break;
                case NPTypeCode.UInt32:  *(uint*)dest = (uint)value; break;
                case NPTypeCode.Int64:   *(long*)dest = (long)value; break;
                case NPTypeCode.UInt64:  *(ulong*)dest = (ulong)value; break;
                case NPTypeCode.Char:    *(char*)dest = (char)value; break;
                case NPTypeCode.Half:    *(Half*)dest = (Half)value; break;
                case NPTypeCode.Single:  *(float*)dest = (float)value; break;
                case NPTypeCode.Double:  *(double*)dest = (double)value; break;
                case NPTypeCode.Decimal: *(decimal*)dest = (decimal)value; break;
                case NPTypeCode.Complex: *(System.Numerics.Complex*)dest = (System.Numerics.Complex)value; break;
                default: throw new NotSupportedException($"WriteScalar: type {typeCode} not supported");
            }
        }

        private static NDArray EnsureContiguousOfType(NDArray src, NPTypeCode target)
        {
            if (src.typecode == target && src.Shape.IsContiguous)
                return src;
            var typed = src.typecode == target ? src : src.astype(target, copy: true);
            return typed.Shape.IsContiguous ? typed : typed.copy();
        }

        private static NDArray EnsureContiguousInt64(NDArray sorter)
        {
            if (sorter.typecode == NPTypeCode.Int64 && sorter.Shape.IsContiguous)
                return sorter;
            var typed = sorter.typecode == NPTypeCode.Int64 ? sorter : sorter.astype(NPTypeCode.Int64, copy: true);
            return typed.Shape.IsContiguous ? typed : typed.copy();
        }

        /// <summary>
        ///     NumPy's <c>searchsorted</c> argument checks, in its order: the side (the argument parser), the haystack's rank
        ///     (<c>PyArray_CheckFromAny(a, dtype, 1, 1)</c>), then the sorter (<c>PyArray_SearchSorted</c>).
        /// </summary>
        /// <param name="a">The haystack.</param>
        /// <param name="side">The side name.</param>
        /// <param name="sorter">The sorter, or null.</param>
        /// <exception cref="ArgumentException">An unknown side (<c>search side must be 'left' or 'right' (got 'x')</c>), or a
        ///     case-insensitive / partial one (<c>search side must be one of 'left' or 'right'</c>).</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is not 1-D (<c>object too deep for desired array</c> /
        ///     <c>object of too small depth for desired array</c>), a uint64 sorter (<c>could not parse sorter argument</c>,
        ///     its 'safe' cast to intp fails), or a sorter whose size differs (<c>sorter.size must equal a.size</c>).</exception>
        /// <exception cref="TypeError">A sorter that is not 1-D (<c>could not parse sorter argument</c>) or not integer —
        ///     bool, float, text — (<c>sorter must only contain integers</c>).</exception>
        /// <remarks>A float sorter used to be truncated, a 0-d haystack searched as one element, and a 2-D sorter reported
        ///     with NumSharp's own text.</remarks>
        private static void ValidateSearchSorted(NDArray a, string side, NDArray sorter)
        {
            ValidateSearchSide(side);

            if (a.ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (a.ndim < 1)
                throw new ValueError("object of too small depth for desired array");

            ValidateSearchSorter(sorter, a.size);
        }

        /// <summary>
        ///     NumPy's <c>searchside_parser</c>: exactly <c>"left"</c> or <c>"right"</c>. A string whose first character is
        ///     <c>l</c> / <c>L</c> / <c>r</c> / <c>R</c> but is not the exact name gets the finalized deprecation's text; any
        ///     other string, the general one.
        /// </summary>
        /// <param name="side">The side name.</param>
        /// <exception cref="ArgumentException">The side is not accepted (NumPy's ValueError texts).</exception>
        private static void ValidateSearchSide(string side)
        {
            if (side == "left" || side == "right")
                return;
            if (!string.IsNullOrEmpty(side) && side[0] is 'l' or 'L' or 'r' or 'R')
                throw new ArgumentException("search side must be one of 'left' or 'right'", nameof(side));
            throw new ArgumentException($"search side must be 'left' or 'right' (got '{side}')", nameof(side));
        }

        /// <summary>
        ///     The sorter half of <c>PyArray_SearchSorted</c>: 1-D (else <c>could not parse sorter argument</c>, a TypeError —
        ///     the rank error is replaced), integer (bool and floats are not), castable 'safely' to intp (uint64 is not), and
        ///     as long as the haystack.
        /// </summary>
        /// <param name="sorter">The sorter, or null (no checks).</param>
        /// <param name="haystackSize">The haystack's size.</param>
        /// <exception cref="TypeError">Not 1-D, or not integer.</exception>
        /// <exception cref="ValueError">uint64, or a size that differs from the haystack's.</exception>
        private static void ValidateSearchSorter(NDArray sorter, long haystackSize)
        {
            if (sorter is null)
                return;
            if (sorter.ndim != 1)
                throw new TypeError("could not parse sorter argument");
            if (!IsIntegerIndexDtype(sorter.typecode))
                throw new TypeError("sorter must only contain integers");
            if (sorter.typecode == NPTypeCode.UInt64)
                throw new ValueError("could not parse sorter argument");
            if (sorter.size != haystackSize)
                throw new ValueError("sorter.size must equal a.size");
        }

        /// <summary>
        ///     <c>np.searchsorted</c> when the haystack or the keys are text — NumPy's <c>PyArray_SearchSorted</c> up to its
        ///     binary search: the side, the common descriptor (<c>PyArray_DescrFromObject(v, a.dtype)</c>: a number with text
        ///     gives text), the keys converted to it (bytes that are not ASCII fail to become <c>U</c>; formatting a number
        ///     cannot fail), the haystack's rank and conversion, the sorter. Then, with no key the result is an empty intp
        ///     array of the keys' shape, and with an empty haystack every key's insertion point is 0 — no comparison either
        ///     way; otherwise the search compares text (the computational phase).
        /// </summary>
        /// <param name="a">The haystack.</param>
        /// <param name="v">The keys (a 0-d array for a scalar key).</param>
        /// <param name="side">The side name.</param>
        /// <param name="sorter">The sorter, or null.</param>
        /// <returns>The int64 result (0-d for a 0-d key) when no comparison is needed.</returns>
        /// <exception cref="ArgumentException">An unknown side.</exception>
        /// <exception cref="DTypePromotionError">The keys and the haystack have no common dtype.</exception>
        /// <exception cref="UnicodeDecodeError">Keys or haystack of bytes that are not ASCII converted to a <c>U</c> common dtype.</exception>
        /// <exception cref="ValueError">The haystack is not 1-D, or a sorter error (see <see cref="ValidateSearchSorter"/>).</exception>
        /// <exception cref="TypeError">A sorter error (see <see cref="ValidateSearchSorter"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The search compares text.</exception>
        private static NDArray SearchSortedTextAnswer(NDArray a, NDArray v, string side, NDArray sorter)
        {
            ValidateSearchSide(side);
            var common = DTypePromotion.PromoteTypes(v.dtype, a.dtype);

            // The keys, then the haystack, converted to the common descriptor: only S -> U can fail (a non-ASCII byte), so
            // that conversion is run and discarded; widening and formatting numbers cannot fail.
            ThrowIfSearchConversionFails(v, common);
            if (a.ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (a.ndim < 1)
                throw new ValueError("object of too small depth for desired array");
            ThrowIfSearchConversionFails(a, common);
            ValidateSearchSorter(sorter, a.size);

            // Nothing compared: an empty result, or every key's insertion point 0 in an empty haystack.
            if (v.size == 0 || a.size == 0)
                return new NDArray(NPTypeCode.Int64, v.ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])v.Shape.dimensions.Clone()), true);
            throw Backends.Strings.NDStringUfuncs.NextPhase("searchsorted", Backends.Strings.NDStringUfuncs.IsString(a) ? a : v);
        }

        /// <summary>
        ///     Runs (and discards) the one conversion of a text search that can fail: bytes to the common <c>U</c> dtype, which
        ///     NumPy's <c>PyArray_CheckFromAny</c> performs on every element (raising at the first byte that is not ASCII).
        /// </summary>
        /// <param name="x">The keys or the haystack.</param>
        /// <param name="common">The common descriptor.</param>
        /// <exception cref="UnicodeDecodeError">A byte that is not ASCII.</exception>
        private static void ThrowIfSearchConversionFails(NDArray x, DType common)
        {
            if (x.typecode == NPTypeCode.String && common.Meta == DTypeRegistry.Str && !x.Storage.StringSlice.IsUnicode && x.size > 0)
                x.astype(common, true, 'K', "unsafe").Dispose();
        }
    }
}
