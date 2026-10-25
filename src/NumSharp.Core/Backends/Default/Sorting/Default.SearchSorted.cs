using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_SearchSorted</c> for a Python int key: a StringDType haystack searches Python objects (no
        ///     common descriptor), a fixed-width text haystack searches text (the key formatted into the common string dtype),
        ///     and a numeric haystack searches in <c>promote_types(a, int32)</c> with the key on the stack (no 0-d array: this
        ///     route is about 15x cheaper per call than the array member). Moved from the public <c>np.searchsorted</c> by the
        ///     np-layer dtype plan's P2, statement for statement; the public function keeps the matrix boundary's plain region.
        /// </remarks>
        public override long SearchSorted(NDArray a, int v, string side = "left", NDArray sorter = null)
        {
            // StringDType (tested first: IsString refuses it): a Python int has no common descriptor with it, so NumPy
            // searches Python objects (a str against an int raises its TypeError; an empty haystack compares nothing).
            if (Strings.NDStringUfuncs.IsVString(a))
            {
                using var key = NDArray.Scalar(v);
                using var answer = SearchSortedVStringAnswer(a, key, side, sorter);
                return answer.GetInt64();
            }
            // Text: NumPy's checks run first, and an empty haystack answers without a comparison.
            if (Strings.NDStringUfuncs.IsString(a))
            {
                using var key = NDArray.Scalar(v);
                using var answer = SearchSortedTextAnswer(a, key, side, sorter);
                return answer.GetInt64();
            }
            ValidateSearchSorted(a, side, sorter);
            // NumPy searches in result_type(a, v), NEVER casts the key down to a's dtype
            // (see the NDArray overload below). A C# int literal is a strong int32 scalar.
            NPTypeCode common = SearchSortedCommonType(a.typecode, NPTypeCode.Int32);
            NDArray aP = a.typecode == common ? a : a.astype(common);
            return SearchSortedScalar(aP, Converts.ChangeType(v, common), side == "left", sorter);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_SearchSorted</c> for a Python float key: the routes of the int member, the numeric search in
        ///     <c>promote_types(a, float64)</c> (NumPy searches a float32 haystack for a Python float in float64 too, probed
        ///     2.4.2). Moved from the public <c>np.searchsorted</c> by the np-layer dtype plan's P2.
        /// </remarks>
        public override long SearchSorted(NDArray a, double v, string side = "left", NDArray sorter = null)
        {
            // StringDType: a Python float has no common descriptor with it — NumPy searches Python objects.
            if (Strings.NDStringUfuncs.IsVString(a))
            {
                using var key = NDArray.Scalar(v);
                using var answer = SearchSortedVStringAnswer(a, key, side, sorter);
                return answer.GetInt64();
            }
            // Text: NumPy's checks run first, and an empty haystack answers without a comparison.
            if (Strings.NDStringUfuncs.IsString(a))
            {
                using var key = NDArray.Scalar(v);
                using var answer = SearchSortedTextAnswer(a, key, side, sorter);
                return answer.GetInt64();
            }
            ValidateSearchSorted(a, side, sorter);
            // A C# double is a strong float64 scalar; NumPy searches a float32 array for a
            // Python float in float64 too (probed 2.4.2), so promote_types(a, Double) matches.
            NPTypeCode common = SearchSortedCommonType(a.typecode, NPTypeCode.Double);
            NDArray aP = a.typecode == common ? a : a.astype(common);
            return SearchSortedScalar(aP, Converts.ChangeType(v, common), side == "left", sorter);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_SearchSorted</c>: the StringDType and fixed-width text routes first (each runs NumPy's checks
        ///     in NumPy's order), then the numeric search — both operands promoted to their common dtype (the keys are never
        ///     cast DOWN to the haystack's), a 0-d key searched from the stack, and the typed IL kernel over the haystack's
        ///     stride with the keys and the sorter read where their views start. Moved from the public <c>np.searchsorted</c>
        ///     by the np-layer dtype plan's P2, statement for statement; the public function keeps the matrix boundary's plain
        ///     region.
        /// </remarks>
        [NDScoped] // reclaims the promoted haystack, the contiguous keys and the int64 sorter (the result is returned)
        public override NDArray SearchSorted(NDArray a, NDArray v, string side = "left", NDArray sorter = null)
        {
            // StringDType (tested first: IsString refuses it): NumPy's descriptor resolution, conversions and binary search
            // over the StringDType compare, or over Python objects when the operands have no common StringDType.
            if (Strings.NDStringUfuncs.IsVString(a) || Strings.NDStringUfuncs.IsVString(v))
                return SearchSortedVStringAnswer(a, v, side, sorter);
            // Text: NumPy's checks run first, and no key / an empty haystack answers without a comparison.
            if (Strings.NDStringUfuncs.IsString(a) || Strings.NDStringUfuncs.IsString(v))
                return SearchSortedTextAnswer(a, v, side, sorter);
            ValidateSearchSorted(a, side, sorter);
            bool leftSide = side == "left";

            // NumPy (PyArray_SearchSorted) promotes BOTH the sorted array `a` and the keys `v`
            // to their common dtype (result_type) and searches there. It must never cast the
            // keys DOWN to a's dtype — that would truncate/wrap them, e.g.
            //   searchsorted([1,2,3,4] int, 2.5) == 2, not floor(2.5) giving index 1;
            //   searchsorted([1,2,3,4] int8, 300) == 4, not (int8)300 giving a wrapped key.
            NPTypeCode common = SearchSortedCommonType(a.typecode, v.typecode);
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
                // The keys and the sorter may be the caller's own arrays (already of the dtype and C-contiguous): a view at
                // a non-zero offset (an np.split child) starts Shape.offset elements into its storage, which Address does
                // not include.
                void* keyPtr = (void*)((byte*)vTyped.Address + vTyped.Shape.offset * elemSize);
                void* sorterPtr = sorterTyped is null ? null : (void*)((long*)sorterTyped.Address + sorterTyped.Shape.offset);
                long* retPtr = (long*)output.Address;
                // The kernel stops at the first sorter entry outside the haystack it reaches (argbinsearch's -1).
                if (kernel(arrPtr, aP.size, arrStride, keyPtr, vTyped.size, sorterPtr, retPtr) < 0)
                    throw new ValueError("Sorter index out of range.");
            }

            return output;
        }

        /// <summary>
        ///     The common dtype of a numeric search, <c>np.promote_types(a, v)</c> over type codes, computed here instead of
        ///     called: the same two statements (an identical pair is itself, any other pair the NEP 42 engine's common class),
        ///     so the engine makes no call up into the np.* layer.
        /// </summary>
        /// <param name="a">The haystack's type code.</param>
        /// <param name="v">The key's type code (int32 for a C# int key, float64 for a C# double one).</param>
        /// <returns>The type code both operands are searched in.</returns>
        /// <exception cref="DTypePromotionError">No common dtype exists (not reached by the numeric route: every pair of
        ///     storage-backed numeric types promotes).</exception>
        private static NPTypeCode SearchSortedCommonType(NPTypeCode a, NPTypeCode v)
            => a == v ? a : DTypePromotion.PromoteTypes(DType.From(a), DType.From(v)).GetTypeCode();

        /// <summary>
        ///     One key searched in a numeric haystack already of the common dtype: the key written to a stack buffer of that
        ///     dtype and the typed IL kernel run once.
        /// </summary>
        /// <param name="a">The haystack, of the common dtype (any 1-D layout: the kernel walks its stride).</param>
        /// <param name="scalarValue">The key, already converted to <paramref name="a"/>'s dtype (a boxed value of its CLR
        ///     type).</param>
        /// <param name="leftSide">True for <c>side='left'</c>.</param>
        /// <param name="sorter">The validated sorter (any integer dtype the checks accept), or null.</param>
        /// <returns>The insertion point.</returns>
        /// <exception cref="ValueError">A sorter index outside the haystack that the bisection reaches (<c>Sorter index out
        ///     of range.</c>, NumPy's argbinsearch check).</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> has a dtype with no stack representation (not
        ///     reached: the text routes run before).</exception>
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
                // A sorter view at a non-zero offset starts Shape.offset elements into its storage (see the array form).
                void* sorterPtr = sorterTyped is null ? null : (void*)((long*)sorterTyped.Address + sorterTyped.Shape.offset);
                // The kernel stops at the first sorter entry outside the haystack it reaches (argbinsearch's -1).
                if (kernel(arrPtr, a.size, arrStride, keyBuf, 1, sorterPtr, retBuf) < 0)
                    throw new ValueError("Sorter index out of range.");
                result = retBuf[0];
            }
            return result;
        }

        /// <summary>Writes a boxed scalar of <paramref name="typeCode"/>'s CLR type to <paramref name="dest"/>.</summary>
        /// <param name="dest">A buffer of at least the dtype's size (16 bytes covers every dtype).</param>
        /// <param name="value">The value, boxed as exactly the dtype's CLR type (an unboxing cast of another type throws).</param>
        /// <param name="typeCode">The dtype to write.</param>
        /// <exception cref="InvalidCastException"><paramref name="value"/> is not boxed as the dtype's CLR type.</exception>
        /// <exception cref="NotSupportedException">A dtype with no fixed-size scalar (text).</exception>
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

        /// <summary>
        ///     The keys as the kernel reads them: <paramref name="src"/> itself when it already is a C-contiguous array of
        ///     <paramref name="target"/> (a view at an offset included — the caller adds <c>Shape.offset</c>), else a
        ///     C-contiguous converted copy (NumPy's <c>PyArray_CARRAY_RO</c> conversion).
        /// </summary>
        /// <param name="src">The keys.</param>
        /// <param name="target">The common dtype.</param>
        /// <returns><paramref name="src"/> or a new C-contiguous array of <paramref name="target"/>.</returns>
        private static NDArray EnsureContiguousOfType(NDArray src, NPTypeCode target)
        {
            if (src.typecode == target && src.Shape.IsContiguous)
                return src;
            var typed = src.typecode == target ? src : src.astype(target, copy: true);
            return typed.Shape.IsContiguous ? typed : typed.copy();
        }

        /// <summary>
        ///     The sorter as the kernel reads it: <paramref name="sorter"/> itself when it already is a C-contiguous int64
        ///     array (a view at an offset included — the caller adds <c>Shape.offset</c>), else a C-contiguous int64 copy
        ///     (NumPy converts the sorter to intp).
        /// </summary>
        /// <param name="sorter">The validated sorter (an integer dtype the checks accept).</param>
        /// <returns><paramref name="sorter"/> or a new C-contiguous int64 array.</returns>
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
            if (!IsSearchSorterIntegerDtype(sorter.typecode))
                throw new TypeError("sorter must only contain integers");
            if (sorter.typecode == NPTypeCode.UInt64)
                throw new ValueError("could not parse sorter argument");
            if (sorter.size != haystackSize)
                throw new ValueError("sorter.size must equal a.size");
        }

        /// <summary>
        ///     Whether a sorter of this dtype passes NumPy's <c>issubdtype(sorter.dtype, np.integer)</c> test: the eight
        ///     integer types and NumSharp's char (NumPy's uint16, which the oracle stands it for); bool is not an integer
        ///     there.
        /// </summary>
        /// <param name="typeCode">The sorter's type code.</param>
        /// <returns>True for an integer sorter.</returns>
        /// <remarks>The same set as <c>np.insert</c> / <c>np.delete</c>'s index test (<c>IsIntegerIndexDtype</c> in the np
        ///     layer), written here so the engine does not reach into the facade's private helpers.</remarks>
        private static bool IsSearchSorterIntegerDtype(NPTypeCode typeCode)
            => typeCode.IsInteger() || typeCode == NPTypeCode.Char;

        /// <summary>
        ///     <c>np.searchsorted</c> when the haystack or the keys are text — NumPy's <c>PyArray_SearchSorted</c>: the side,
        ///     the common descriptor (<c>PyArray_DescrFromObject(v, a.dtype)</c>: a number with text gives text), the keys
        ///     converted to it (bytes that are not ASCII fail to become <c>U</c>; formatting a number cannot fail), the
        ///     haystack's rank and conversion, the sorter. Then, with no key the result is an empty intp array of the keys'
        ///     shape, and with an empty haystack every key's insertion point is 0; otherwise NumPy's generic binary search
        ///     runs over the text (<see cref="Sorting.TextSearch"/>: its exact comparisons, monotonic carry and
        ///     sorter checks).
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
        /// <exception cref="ValueError">A sorter index outside the haystack that the search reaches (<c>Sorter index out of
        ///     range.</c>).</exception>
        [NDScoped]
        private static NDArray SearchSortedTextAnswer(NDArray a, NDArray v, string side, NDArray sorter)
        {
            ValidateSearchSide(side);
            var common = DTypePromotion.PromoteTypes(v.dtype, a.dtype);

            // The keys, then the haystack, converted to the common descriptor in NumPy's order (PyArray_CheckFromAny on the
            // keys, then on the haystack with its rank checked first): only S -> U can fail (a non-ASCII byte), and the
            // conversions are kept for the search. An operand of the common kind is read as it is, whatever its width (the
            // comparison pads the narrower string with zeros, the cast's padding).
            NDArray keys = ConvertForTextSearch(v, common);
            if (a.ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (a.ndim < 1)
                throw new ValueError("object of too small depth for desired array");
            NDArray haystack = ConvertForTextSearch(a, common);
            ValidateSearchSorter(sorter, a.size);

            // Nothing compared: an empty result, or every key's insertion point 0 in an empty haystack.
            if (v.size == 0 || a.size == 0)
                return new NDArray(NPTypeCode.Int64, v.ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])v.Shape.dimensions.Clone()), true);

            NDArray sorterTyped = sorter is null ? null : EnsureContiguousInt64(sorter);
            return Sorting.TextSearch.SearchSorted(haystack, keys, side == "left", sorterTyped);
        }

        /// <summary>
        ///     <c>np.searchsorted</c> when the haystack or the keys are <c>StringDType</c> — NumPy's <c>PyArray_SearchSorted</c>:
        ///     the side; the descriptor <c>PyArray_DescrFromObject(v, a.dtype)</c>, which is
        ///     <c>PyArray_PromoteTypes(v.dtype, a.dtype)</c> with the KEYS first (StringDType with StringDType of a compatible
        ///     NA object, or with U, promotes; anything else fails to promote and gives object); the keys converted to it,
        ///     then the haystack (its rank checked first) — a U ARRAY refuses the 'safe' cast into StringDType, a Python str
        ///     key is packed; the sorter. Then NumPy's generic binary search over the StringDType compare
        ///     (<see cref="Sorting.VStringSearch.SearchSorted"/>) or over Python objects
        ///     (<see cref="Sorting.VStringSearch.SearchSortedObjects"/>), which runs even for an empty haystack: the
        ///     first key is compared with itself, and each key with the one before.
        /// </summary>
        /// <remarks>
        ///     The promoted descriptor holds the present NA object and the FIRST operand's <c>coerce</c> flag (NumPy's
        ///     <c>common_instance</c> reads <c>dtype1-&gt;coerce</c> twice), and a U operand counts as the default
        ///     <c>StringDType()</c> (its cast's default descriptor). Only error texts can show it: a U-array key into a
        ///     <c>StringDType(coerce=False)</c> haystack fails naming <c>StringDType()</c>, not the haystack's descriptor.
        /// </remarks>
        /// <param name="a">The haystack.</param>
        /// <param name="v">The keys (a 0-d array for a scalar key; a <see cref="NDArray.WasPythonStr"/> array is a Python
        ///     str).</param>
        /// <param name="side">The side name.</param>
        /// <param name="sorter">The sorter, or null.</param>
        /// <returns>The int64 insertion points, of the keys' shape (0-d for a 0-d key).</returns>
        /// <exception cref="ArgumentException">An unknown side.</exception>
        /// <exception cref="TypeError">A U array converted to StringDType (<c>Cannot cast array data from dtype('&lt;U1') to
        ///     StringDType() according to the rule 'safe'</c>; <c>scalar</c> for a 0-d one), a sorter error, or the first
        ///     comparison Python refuses on the object path (<c>'&lt;' not supported between instances of 'str' and
        ///     'int'</c>).</exception>
        /// <exception cref="ValueError">The haystack is not 1-D, a sorter error, a sorter index outside the haystack that the
        ///     search reaches, or a comparison that touched a None-like NA.</exception>
        [NDScoped]
        private static NDArray SearchSortedVStringAnswer(NDArray a, NDArray v, string side, NDArray sorter)
        {
            ValidateSearchSide(side);
            bool vT = Strings.NDStringUfuncs.IsVString(v), aT = Strings.NDStringUfuncs.IsVString(a);
            // The common descriptor (null: promotion failed, the object path). NumPy's handle_promotion calls
            // PyArray_PromoteTypes(keys descriptor, haystack descriptor) — the keys FIRST — and turns a failure (two
            // different NA objects, or StringDType with a non-U class) into the object dtype instead of raising.
            DType common = null;
            if (vT && aT)
            {
                if (VStringParameters.CompatibleNA(v.dtype.VString, a.dtype.VString, out _, out _))
                    common = DTypePromotion.PromoteTypes(v.dtype, a.dtype);
            }
            else if ((vT && a.typecode == NPTypeCode.String && a.dtype.kind == 'U')
                     || (aT && v.typecode == NPTypeCode.String && v.dtype.kind == 'U'))
                common = DTypePromotion.PromoteTypes(v.dtype, a.dtype);

            // The keys to the descriptor: only a U array can fail ('same_kind' into StringDType); a Python str is packed.
            if (common is not null)
                ThrowIfUnsafeTextCast(v, common, allowPythonStr: true);
            if (a.ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (a.ndim < 1)
                throw new ValueError("object of too small depth for desired array");
            if (common is not null)
                ThrowIfUnsafeTextCast(a, common, allowPythonStr: false);
            ValidateSearchSorter(sorter, a.size);
            NDArray sorterTyped = sorter is null ? null : EnsureContiguousInt64(sorter);
            bool left = side == "left";
            if (common is null)
                return Sorting.VStringSearch.SearchSortedObjects(a, v, left, sorterTyped);
            // A Python str key becomes a StringDType key of the common descriptor: NumPy packs the str itself (setitem), so
            // its trailing NULs are kept, where the 0-d U array standing for it drops them.
            NDArray keys = vT ? v : NDArray.PackForVStringStore(v, common);
            return Sorting.VStringSearch.SearchSorted(a, keys, left, sorterTyped, common);
        }

        /// <summary>
        ///     NumPy's 'safe' conversion check of a search operand into the StringDType descriptor: a <c>U</c> array cannot be
        ///     cast safely (U → StringDType is 'same_kind'); a StringDType operand can (the common descriptor only adds the
        ///     other's NA object).
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="common">The StringDType descriptor.</param>
        /// <param name="allowPythonStr">Whether a <see cref="NDArray.WasPythonStr"/> operand (a Python str, which NumPy packs
        ///     instead of casting) is accepted.</param>
        /// <exception cref="TypeError">A U array (<c>Cannot cast array data from dtype('&lt;U1') to StringDType() according
        ///     to the rule 'safe'</c>; <c>scalar</c> for a 0-d array).</exception>
        private static void ThrowIfUnsafeTextCast(NDArray x, DType common, bool allowPythonStr)
        {
            if (Strings.NDStringUfuncs.IsVString(x))
                return;
            if (allowPythonStr && x.WasPythonStr)
                return;
            throw new TypeError(
                $"Cannot cast {(x.ndim == 0 ? "scalar" : "array data")} from {x.dtype.ToString(true)} to {common.ToString(true)} " +
                "according to the rule 'safe'");
        }

        /// <summary>
        ///     One operand of a text search in the common descriptor's kind: the operand itself when it is already of that
        ///     kind (any width), else its conversion (a number formatted, bytes decoded) — the array the search reads.
        /// </summary>
        /// <param name="x">The keys or the haystack.</param>
        /// <param name="common">The common descriptor.</param>
        /// <returns><paramref name="x"/> or a new array of <paramref name="common"/>.</returns>
        /// <exception cref="UnicodeDecodeError">Bytes that are not ASCII converted to a <c>U</c> common descriptor.</exception>
        private static NDArray ConvertForTextSearch(NDArray x, DType common)
            => x.dtype.Meta == common.Meta ? x : x.astype(common, true, 'K', "unsafe");
    }
}
