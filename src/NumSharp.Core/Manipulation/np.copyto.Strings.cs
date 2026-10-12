using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     <c>np.copyto(dst, src)</c> with a Python <c>str</c> source: NumPy's <c>PyArray_FromAny(src)</c> makes it a 0-d
        ///     Unicode array of its own length (<c>'abcdef'</c> is <c>&lt;U6</c>), which is then assigned like any 0-d source —
        ///     truncated into a shorter <c>U</c> destination, ASCII-encoded into an <c>S</c> one (under a casting rule that
        ///     allows <c>U → S</c>: <c>'unsafe'</c>).
        /// </summary>
        /// <param name="dst">The array into which the text is copied.</param>
        /// <param name="src">The text (not null).</param>
        /// <param name="casting">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <param name="where">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> or <paramref name="src"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidCastException">The cast from <c>&lt;U{len}</c> to the destination's dtype is not allowed:
        ///     <c>Cannot cast scalar from dtype('&lt;U6') to dtype('S3') according to the rule 'same_kind'</c>.</exception>
        /// <exception cref="UnicodeEncodeError">The text is not ASCII and the destination is <c>S</c>.</exception>
        /// <exception cref="OverflowException">The destination is an integer dtype and the text is an integer outside it
        ///     (NumPy's LP64 OverflowError texts).</exception>
        /// <exception cref="SystemError">The text is one unit above U+10FFFF (only reachable through a raw view).</exception>
        /// <exception cref="ValueError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>; also a numeric destination
        ///     the text does not parse into (<c>invalid literal for int() with base 10: np.str_('x')</c> — NumPy parses the 0-d
        ///     array the text became, so it quotes the NumPy scalar).</exception>
        /// <exception cref="TypeError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="ArgumentException">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <remarks>
        ///     A C# <see cref="string"/> would otherwise reach <see cref="copyto(NDArray, NDArray, string, NDArray)"/> through
        ///     NumSharp's implicit <c>string → NDArray</c> conversion, which still reads <c>"[1, 2]"</c>-shaped text as a numeric
        ///     matrix (a legacy NumSharp shorthand); this overload keeps every text a text, as NumPy does.
        ///     <para><see cref="NDAnchoredAttribute"/>: the destination and the mask stay reachable while the copy runs, as on
        ///     <see cref="copyto(NDArray, NDArray, string, NDArray)"/> (the text's own array is the <c>using</c> local).</para>
        /// </remarks>
        [NDAnchored]
        public static void copyto(NDArray dst, string src, string casting = "same_kind", NDArray @where = null)
        {
            NPY_CASTING rule = CopytoCastingConverter(casting);
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            using var value = NDStringArrays.FromScalar(src, null);
            DefaultEngine.ValidateWhereMask(@where);
            CopytoAssignArray(dst, value, @where, rule);
        }

        /// <summary>
        ///     <c>PyArray_AssignArray</c> / <c>PyArray_AssignRawScalar</c> when either operand is a fixed-width string array:
        ///     NumPy's checks in NumPy's order with DESCRIPTOR-level cast rules (the width is part of the dtype: <c>&lt;U5</c> to
        ///     <c>&lt;U3</c> is <c>same_kind</c>, <c>U → S</c> is <c>unsafe</c>), then the string copy route.
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="src">The source (any rank; a 0-d source is NumPy's raw scalar assignment).</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="casting">The casting level.</param>
        /// <exception cref="ValueError">A read-only destination, or a source / mask that does not broadcast.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/> (NumPy's TypeError
        ///     text, "array data" or "scalar").</exception>
        /// <exception cref="UnicodeEncodeError">A selected <c>U</c> element is not ASCII and the destination is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">A selected <c>S</c> element is not ASCII and the destination is <c>U</c>.</exception>
        /// <exception cref="SystemError">A selected <c>U</c> element holds a unit above <c>U+10FFFF</c> and the destination is <c>S</c>.</exception>
        /// <exception cref="OverflowException">An allowed text → integer cast meets a value outside the destination (see
        ///     <see cref="CopytoTextNumber"/>).</exception>
        /// <remarks>
        ///     <para><b>Which elements are converted.</b> NumPy's masked cast converts only the elements the mask selects, so a
        ///     non-ASCII element under a false mask is never encoded (no error); its unmasked cast — and the temporary copy of
        ///     an overlapping source — converts every element. A 0-d source is converted ONCE before the copy when the
        ///     destination has more than one element (an error then precedes the mask's broadcast check and ignores the mask),
        ///     and inside the loop for a one-element destination (a false mask skips it).</para>
        ///     <para><b>Kernels.</b> The unmasked copy is the string copy route of
        ///     <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/> (unit views, SIMD copy / cast kernels). The masked
        ///     copy is the same masked-assign walker as the numeric dtypes, copying each selected element lane by lane
        ///     (<see cref="Backends.Kernels.DirectILKernelGenerator.MaskedAssignAnyWidth"/>); a source of another width or
        ///     class is first staged in the destination's dtype, one element per distinct source element.</para>
        /// </remarks>
        private static unsafe void CopytoAssignStrings(NDArray dst, NDArray src, NDArray @where, NPY_CASTING casting)
        {
            // A StringDType operand has its own route: its cells move only through the allocators (np.copyto.VString.cs).
            if (src.Storage.IsVString || dst.Storage.IsVString)
            {
                CopytoAssignVStrings(dst, src, @where, casting);
                return;
            }
            bool scalar = src.ndim == 0;
            DType from = src.dtype, to = dst.dtype;

            // AssignArray: a redundant self-copy (the same data, dtype, shape and strides) returns before every check.
            if (!scalar && CopytoData(src) == CopytoData(dst) && from.Equals(to) && SameLayout(src.Shape, dst.Shape))
                return;

            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            if (!DTypeCasting.CanCastTypeTo(from, to, casting))
                throw NDStringCasting.InvalidCast(from, to, casting, scalar);
            // A cast the rule allows between text and numbers formats / parses values: the computational phase — reached
            // only after NumPy's shape checks, and only for elements NumPy converts.
            if (!src.Storage.IsString || !dst.Storage.IsString)
            {
                CopytoTextNumber(dst, src, @where, from, to);
                return;
            }

            if (scalar)
            {
                CopytoAssignStringScalar(dst, src, @where);
                return;
            }

            // An overlapping source is copied first unless dst is 1-D with both runs in the same direction (NumPy's rule).
            int dnd = dst.ndim;
            bool overlapNeedsCopy =
                ((dnd == 1 && dst.Shape.strides[0] * src.Shape.strides[src.ndim - 1] < 0) || dnd > 1)
                && NDMemOverlap.SolveMayShareMemory(src, dst, 0) != MemOverlap.No;

            long[] srcStrides = CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides,
                "input array", stripLeadingUnitAxes: true);
            UnmanagedStorage source = src.ndim > dnd ? StrippedSourceView(src, dnd) : src.Storage;
            bool crossClass = from.Meta != to.Meta;

            // NumPy's temporary copy of an overlapping source is an UNMASKED cast of every element, made before the mask is
            // read — so its conversion error precedes the 0-d mask and the mask's broadcast check.
            if (overlapNeedsCopy && crossClass && dst.size > 0)
                NDStringCopy.ThrowIfCannotConvert(dst.Storage, source, null);

            // "optimization: scalar boolean mask"
            if (@where is not null && @where.ndim == 0)
            {
                if (!*(bool*)CopytoData(@where))
                    return;
                @where = null;
            }

            if (@where is null)
            {
                NDIter.Copy(dst.Storage, source);
                return;
            }

            long[] maskStrides = CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                "where mask", stripLeadingUnitAxes: false);
            if (dst.size == 0)
                return;

            // Same class, same width, no overlap to copy first: the masked walk reads the source in place (a 1-D
            // same-direction overlap is walked in NumPy's raw order, reading its own writes as NumPy does).
            if (!overlapNeedsCopy && !crossClass && from.itemsize == to.itemsize)
            {
                CopytoMaskedRaw(dst, CopytoData(src), srcStrides, CopytoData(@where), maskStrides);
                return;
            }

            // The masked cast converts only the selected elements: check exactly those before anything is written.
            if (crossClass && !overlapNeedsCopy)
                NDStringCopy.ThrowIfCannotConvert(dst.Storage, source, @where);

            using var stage = CopytoStageStrings(dst, source, srcStrides);
            CopytoMaskedRaw(dst, CopytoData(stage), srcStrides, CopytoData(@where), maskStrides);
        }

        /// <summary>
        ///     <c>PyArray_AssignArray</c> / <c>PyArray_AssignRawScalar</c> between text and numbers once the destination's
        ///     writeability and the cast rule passed: NumPy's remaining checks and conversions in its order. A number is
        ///     written as its scalar's <c>str()</c>, a text parsed by CPython's <c>int()</c> / <c>float()</c> /
        ///     <c>complex()</c> (<see cref="NDTextConversion"/>), and only the elements NumPy converts are converted.
        /// </summary>
        /// <param name="dst">The destination (writeable).</param>
        /// <param name="src">The source (string or numeric; the other kind than <paramref name="dst"/>).</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="from">The source descriptor.</param>
        /// <param name="to">The destination descriptor.</param>
        /// <exception cref="ValueError">The source or the mask does not broadcast to <paramref name="dst"/>
        ///     (<c>could not broadcast input array from shape (2,) into shape (3,)</c>, <c>… where mask …</c>), or a converted
        ///     text is not a number of the destination's kind (NumPy's <c>int()</c> / <c>float()</c> / <c>complex()</c>
        ///     texts).</exception>
        /// <exception cref="OverflowException">A converted integer text does not fit the destination.</exception>
        /// <exception cref="SystemError">A converted <c>U</c> element is one unit above U+10FFFF.</exception>
        /// <exception cref="UnicodeDecodeError">A converted <c>S</c> element is not UTF-8 and the destination is
        ///     complex.</exception>
        /// <remarks>
        ///     <para><b>Which elements are converted, and when.</b> A 0-d source is converted ONCE, before the mask is read,
        ///     when the destination has more than one element (NumPy's <c>PyArray_CastRawArrays</c> into a scalar buffer — a
        ///     parse error then ignores the mask); a one-element destination converts it inside the masked loop (a false mask
        ///     converts nothing). An overlapping source is first converted whole into a temporary of the destination's dtype
        ///     (NumPy's <c>PyArray_NewLikeArray</c> + unmasked assignment). Otherwise the masked transfer converts only the
        ///     selected runs (<c>_strided_masked_wrapper_transfer_function</c>), so an unselected bad text never raises.</para>
        ///     <para><b>Partial writes.</b> A failing parse leaves the destination as NumPy leaves it: every 128-element block
        ///     before the failing one written, the failing block and everything after untouched.</para>
        /// </remarks>
        private static unsafe void CopytoTextNumber(NDArray dst, NDArray src, NDArray @where, DType from, DType to)
        {
            bool parses = src.Storage.IsString;
            // Text into bool is a string's truth value (NumPy's string-to-bool cast), computed by the truth route.
            if (parses && dst.typecode == NPTypeCode.Boolean)
            {
                CopytoTextTruth(dst, src, @where);
                return;
            }
            if (src.ndim == 0)
            {
                // AssignRawScalar: a destination of more than one element gets the scalar converted ONCE, before the mask;
                // the converted 0-d value is then assigned like a same-dtype scalar (mask and all).
                if (dst.size > 1)
                {
                    using var converted = CopytoConvertWhole(src, to, Shape.NewScalar());
                    np.copyto(dst, converted, "unsafe", @where);
                    return;
                }
                if (@where is not null)
                    CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                        "where mask", stripLeadingUnitAxes: false);
                // Nothing written (or, for one element, nothing converted inside the masked loop) when nothing is selected.
                if (dst.size == 0 || (@where is not null && !np.any(@where)))
                    return;
                NDTextConversion.Convert(dst.Storage, src.Storage);
                return;
            }

            // An overlapping source (two views of one buffer read as text and as numbers) is copied first by an UNMASKED
            // conversion of every element into a temporary of the destination's dtype — after its own broadcast check, so
            // the broadcast error still comes first.
            int dnd = dst.ndim;
            bool overlapNeedsCopy =
                ((dnd == 1 && dst.Shape.strides[0] * src.Shape.strides[src.ndim - 1] < 0) || dnd > 1)
                && NDMemOverlap.SolveMayShareMemory(src, dst, 0) != MemOverlap.No;
            CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array",
                stripLeadingUnitAxes: true);
            UnmanagedStorage source = src.ndim > dnd ? StrippedSourceView(src, dnd) : src.Storage;
            if (overlapNeedsCopy && dst.size > 0)
            {
                using var temp = CopytoConvertWhole(new NDArray(source), to, dst.Shape.KeepOrder());
                np.copyto(dst, temp, "unsafe", @where);
                return;
            }

            // "optimization: scalar boolean mask"
            if (@where is not null && @where.ndim == 0)
            {
                if (!*(bool*)CopytoData(@where))
                    return;
                @where = null;
            }
            if (@where is not null)
                CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask",
                    stripLeadingUnitAxes: false);
            if (dst.size == 0 || (@where is not null && !np.any(@where)))
                return;
            NDTextConversion.Convert(dst.Storage, source, @where?.Storage);
        }

        /// <summary>
        ///     Converts a whole source into a fresh array of <paramref name="to"/> laid out as <paramref name="layout"/> — the
        ///     scalar buffer of NumPy's <c>PyArray_AssignRawScalar</c> (a 0-d layout) and the temporary of an overlapping
        ///     source (the destination's dimensions in KEEPORDER layout).
        /// </summary>
        /// <param name="src">The source (text or numbers), broadcastable to <paramref name="layout"/>.</param>
        /// <param name="to">The destination's dtype (the temporary's).</param>
        /// <param name="layout">The temporary's shape (a fresh dense layout).</param>
        /// <returns>The converted array (the caller disposes it).</returns>
        /// <exception cref="ValueError">A text is not a number of <paramref name="to"/>'s kind.</exception>
        /// <exception cref="OverflowException">An integer text does not fit <paramref name="to"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> element is one unit above U+10FFFF.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> element converted to complex is not UTF-8.</exception>
        private static NDArray CopytoConvertWhole(NDArray src, DType to, Shape layout)
        {
            var temp = NDArray.ForOverwrite(to, layout);
            try
            {
                if (temp.size > 0)
                    NDTextConversion.Convert(temp.Storage, src.Storage);
                return temp;
            }
            catch
            {
                temp.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     <see cref="CopytoTextNumber"/> for a bool destination: NumPy's string-to-bool cast assigns each string's truth
        ///     value (non-empty is True), in NumPy's order — the scalar / overlap conversions that run before the mask, the
        ///     broadcast checks, then the masked assignment, which converts only the elements the mask selects.
        /// </summary>
        /// <param name="dst">The bool destination (writeable; the cast rule allowed it).</param>
        /// <param name="src">The text source.</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <exception cref="ValueError">The source or the mask does not broadcast to <paramref name="dst"/>.</exception>
        /// <exception cref="SystemError">A converted <c>U</c> element holds a unit above U+10FFFF (NumPy's
        ///     <c>invalid maximum character passed to PyUnicode_New</c>); an element the mask leaves out is not converted, so
        ///     it never raises.</exception>
        private static unsafe void CopytoTextTruth(NDArray dst, NDArray src, NDArray @where)
        {
            NDArray truth = null;
            try
            {
                if (src.ndim == 0)
                {
                    // AssignRawScalar: a destination of more than one element gets the scalar converted ONCE, before the mask.
                    if (dst.size > 1)
                        truth = NDStringTruth.Truth(src, NDStringTruth.Flavor.Cast);
                    if (@where is not null)
                        CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                            "where mask", stripLeadingUnitAxes: false);
                    if (dst.size == 0 || (@where is not null && !np.any(@where)))
                        return;
                    // One element: converted inside the masked loop, i.e. only now that it is selected.
                    truth ??= NDStringTruth.Truth(src, NDStringTruth.Flavor.Cast);
                    np.copyto(dst, truth, "unsafe", @where);
                    return;
                }

                // An overlapping source (one buffer read as text and as bool) is copied first by an UNMASKED conversion of
                // every element — after its own broadcast check, so the broadcast error still comes first.
                int dnd = dst.ndim;
                bool overlapNeedsCopy =
                    ((dnd == 1 && dst.Shape.strides[0] * src.Shape.strides[src.ndim - 1] < 0) || dnd > 1)
                    && NDMemOverlap.SolveMayShareMemory(src, dst, 0) != MemOverlap.No;
                CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array",
                    stripLeadingUnitAxes: true);
                if (overlapNeedsCopy && dst.size > 0)
                    truth = NDStringTruth.Truth(src, NDStringTruth.Flavor.Cast);
                if (@where is not null && @where.ndim == 0)
                {
                    // "optimization: scalar boolean mask"
                    if (!*(bool*)CopytoData(@where))
                        return;
                    @where = null;
                }
                if (@where is not null)
                    CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides, "where mask",
                        stripLeadingUnitAxes: false);
                if (dst.size == 0 || (@where is not null && !np.any(@where)))
                    return;

                // The masked transfer converts only the selected elements: a bad code point raises only when the mask
                // selects (a destination slot fed by) its element. Unmasked, every element is converted.
                truth ??= @where is null
                    ? NDStringTruth.Truth(src, NDStringTruth.Flavor.Cast)
                    : MaskedTruth(dst, src, @where);
                np.copyto(dst, truth, "unsafe", @where);
            }
            finally
            {
                truth?.Dispose();
            }
        }

        /// <summary>
        ///     The truth array of a text source assigned under a mask: the nonzero route (never raises), after checking that
        ///     no element the mask selects is a one-unit <c>U</c> string above U+10FFFF — the elements NumPy's masked cast
        ///     converts, and the only strings CPython refuses to build (a longer string's maximum character is capped).
        /// </summary>
        /// <param name="dst">The destination (the broadcast target of the source and the mask).</param>
        /// <param name="src">The text source.</param>
        /// <param name="where">The mask (validated, broadcastable to <paramref name="dst"/>).</param>
        /// <returns>The source's truth array (the caller owns it).</returns>
        /// <exception cref="SystemError">A selected element is one unit above U+10FFFF (its first unit above it, every later
        ///     unit zero).</exception>
        private static NDArray MaskedTruth(NDArray dst, NDArray src, NDArray @where)
        {
            if (src.dtype.kind == 'U' && src.dtype.itemsize > 0)
            {
                using var units = NDStringArrays.UnitArray(src);                     // src.shape + (chars,), uint32
                long chars = units.Shape.dimensions[units.ndim - 1];
                // Per element: the first unit above U+10FFFF and (for a wider element) every later unit zero.
                using var first = units["..., 0"];
                using var high = first > 0x10FFFFu;
                using var later = chars > 1 ? units["..., 1:"] : null;
                using var laterNonzero = later is null ? null : np.any(later, axis: -1);
                using var laterEmpty = laterNonzero is null ? null : np.logical_not(laterNonzero);
                using var bad = laterEmpty is null ? high.view() : np.logical_and(high, laterEmpty);
                // copyto strips the source's leading extent-1 axes beyond the destination's rank (the broadcast check
                // already passed, so every extra leading axis has extent 1); the rest broadcasts as the assignment does.
                var badDims = bad.shape;
                using var badRanked = badDims.Length > dst.ndim
                    ? bad.reshape(badDims[(badDims.Length - dst.ndim)..])
                    : bad.view();
                // A selected destination slot whose (broadcast) source element is bad means NumPy converts that element.
                using var badView = np.broadcast_to(badRanked, dst.Shape);
                using var selectedBad = np.logical_and(badView, @where);
                if (np.any(selectedBad))
                    throw new SystemError(NDStringTruth.InvalidCodePointMessage);
            }
            return NDStringTruth.Truth(src, NDStringTruth.Flavor.Nonzero);
        }

        /// <summary>
        ///     <c>PyArray_AssignRawScalar</c> for a 0-d string source into a string destination (writeability and the cast
        ///     rule already checked): converted ONCE into the destination's dtype when the destination holds more than one
        ///     element (before the mask is looked at), else inside the loop — only when the one element is selected — then
        ///     written to every selected element.
        /// </summary>
        /// <param name="dst">The destination string array (writeable).</param>
        /// <param name="src">The 0-d string source.</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <exception cref="ValueError">The mask does not broadcast to <paramref name="dst"/>.</exception>
        /// <exception cref="UnicodeEncodeError">The <c>U</c> value is not ASCII and the destination is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">The <c>S</c> value is not ASCII and the destination is <c>U</c>.</exception>
        /// <exception cref="SystemError">The <c>U</c> value holds a unit above <c>U+10FFFF</c> and the destination is <c>S</c>.</exception>
        private static unsafe void CopytoAssignStringScalar(NDArray dst, NDArray src, NDArray @where)
        {
            long size = dst.size;
            // PyArray_EquivTypes: NumSharp stores every string natively, so equivalence is the class and the width.
            bool equivalent = src.dtype.Meta == dst.dtype.Meta && src.dtype.itemsize == dst.dtype.itemsize;
            NDArray converted = null;
            try
            {
                // "Make a copy of the src data if it's a different dtype than 'dst' ... and the destination we're copying to
                // has more than one element."
                if (!equivalent && size > 1)
                    converted = ConvertStringScalar(src, dst.dtype);

                long[] maskStrides = null;
                if (@where is not null)
                    maskStrides = CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                        "where mask", stripLeadingUnitAxes: false);

                if (size == 0)
                    return;

                if (!equivalent && converted is null)
                {
                    // One element: NumPy casts inside the (masked) loop, so an unselected element is never converted.
                    if (@where is not null && !*(bool*)CopytoData(@where))
                        return;
                    converted = ConvertStringScalar(src, dst.dtype);
                }

                var value = converted ?? src;
                var zero = new long[dst.ndim];
                if (@where is null)
                {
                    byte selected = 1;
                    CopytoMaskedRaw(dst, CopytoData(value), zero, &selected, zero);
                    return;
                }

                CopytoMaskedRaw(dst, CopytoData(value), zero, CopytoData(@where), maskStrides);
            }
            finally
            {
                converted?.Dispose();
            }
        }

        /// <summary>
        ///     A 0-d string value converted into <paramref name="dtype"/> through the string copy route (truncated or
        ///     NUL-padded; <c>S ↔ U</c> ASCII-checked).
        /// </summary>
        /// <param name="value">The 0-d string array.</param>
        /// <param name="dtype">The destination's (sized, native) string descriptor.</param>
        /// <returns>A new 0-d array of <paramref name="dtype"/> the caller disposes.</returns>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> value is not ASCII and <paramref name="dtype"/> is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> value is not ASCII and <paramref name="dtype"/> is <c>U</c>.</exception>
        /// <exception cref="SystemError">A <c>U</c> value holds a unit above <c>U+10FFFF</c> and <paramref name="dtype"/> is <c>S</c>.</exception>
        private static NDArray ConvertStringScalar(NDArray value, DType dtype)
        {
            var converted = new NDArray(dtype, Shape.NewScalar(), false);
            try
            {
                NDIter.Copy(converted.Storage, value.Storage);
                return converted;
            }
            catch
            {
                converted.Dispose(); // a failed conversion returns its buffer now instead of at finalization
                throw;
            }
        }

        /// <summary>
        ///     The source of a masked string copy staged in the destination's dtype: one element per DISTINCT source element
        ///     (an axis along which the source does not move keeps one), C-ordered, converted / truncated / NUL-padded without
        ///     the ASCII check (the caller has checked every element the mask selects; an unselected one is never written).
        ///     Rewrites <paramref name="srcStrides"/> to the stage's element strides on the destination's axes.
        /// </summary>
        /// <param name="dst">The destination (its dtype and dimensions).</param>
        /// <param name="source">The source storage, at most the destination's rank.</param>
        /// <param name="srcStrides">In: the source's element strides broadcast to the destination's axes (0 where it does not
        ///     move). Out: the stage's element strides on the same axes (still 0 there).</param>
        /// <returns>The stage, an array the caller disposes.</returns>
        private static NDArray CopytoStageStrings(NDArray dst, UnmanagedStorage source, long[] srcStrides)
        {
            int nd = dst.ndim;
            long[] dstDims = dst.Shape.dimensions;
            var dims = new long[nd];
            for (int d = 0; d < nd; d++)
                dims[d] = srcStrides[d] == 0 ? 1 : dstDims[d];

            // The source seen through the destination's axes, each distinct element once.
            var aligned = source.Alias(new Shape(dims, (long[])srcStrides.Clone(), source.Shape.offset, source.Shape.bufferSize));

            var stage = NDArray.ForOverwrite(dst.dtype, new Shape(dims)); // CopyUnchecked writes every element
            try
            {
                NDStringCopy.CopyUnchecked(stage.Storage, aligned);
            }
            catch
            {
                stage.Dispose();
                throw;
            }

            long dense = 1;
            for (int d = nd - 1; d >= 0; d--)
            {
                srcStrides[d] = dims[d] == 1 ? 0 : dense;
                dense *= dims[d];
            }
            return stage;
        }
    }
}
