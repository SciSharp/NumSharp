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
        /// <exception cref="StringComputationNotSupportedException">The destination is numeric, the rule allows the cast and
        ///     NumPy parses the text (a destination of more than one element, or one element the mask selects): parsing text
        ///     into numbers is the computational phase of the string dtypes.</exception>
        /// <exception cref="ValueError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="TypeError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="ArgumentException">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <remarks>
        ///     A C# <see cref="string"/> would otherwise reach <see cref="copyto(NDArray, NDArray, string, NDArray)"/> through
        ///     NumSharp's implicit <c>string → NDArray</c> conversion, which still reads <c>"[1, 2]"</c>-shaped text as a numeric
        ///     matrix (a legacy NumSharp shorthand); this overload keeps every text a text, as NumPy does.
        /// </remarks>
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
        /// <exception cref="StringComputationNotSupportedException">An allowed cast between text and numbers that NumPy would
        ///     run on an element (the computational phase) — see <see cref="CopytoTextNumber"/>; an empty destination or a mask
        ///     that selects nothing returns.</exception>
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
        ///     writeability and the cast rule passed: NumPy's remaining checks run in its order, and the assignment is refused
        ///     (the computational phase) only where NumPy's conversion has an observable effect — a PARSE of text into a
        ///     number it runs (which can fail), or a WRITE of formatted text (formatting a number cannot fail, so an unwritten
        ///     one is unobservable). An empty destination, or a mask that selects nothing, therefore returns, as in NumPy.
        /// </summary>
        /// <param name="dst">The destination (writeable).</param>
        /// <param name="src">The source (string or numeric; the other kind than <paramref name="dst"/>).</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="from">The source descriptor.</param>
        /// <param name="to">The destination descriptor.</param>
        /// <exception cref="ValueError">The source or the mask does not broadcast to <paramref name="dst"/>
        ///     (<c>could not broadcast input array from shape (2,) into shape (3,)</c>, <c>… where mask …</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy parses a text element — a 0-d text source into a
        ///     destination of more than one element (parsed once, before the mask is read), an overlapping text source NumPy
        ///     copies first — or writes a converted element the mask selects.</exception>
        /// <remarks>
        ///     NumPy's masked transfer converts only the selected runs (<c>_strided_masked_wrapper_transfer_function</c>), so a
        ///     mask with no true element converts nothing; a mask that broadcasts to a non-empty destination maps every one of
        ///     its elements to at least one destination element, so <c>any(where)</c> decides.
        /// </remarks>
        private static unsafe void CopytoTextNumber(NDArray dst, NDArray src, NDArray @where, DType from, DType to)
        {
            // Text into numbers parses (can fail, so every parse NumPy runs is the computational phase); numbers into text
            // format (cannot fail, so only a write matters).
            bool parses = src.Storage.IsString;
            if (src.ndim == 0)
            {
                // AssignRawScalar: a destination of more than one element gets the scalar converted ONCE, before the mask.
                if (dst.size > 1 && parses)
                    throw NDStringCasting.TextConversion(from, to);
                if (@where is not null)
                    CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                        "where mask", stripLeadingUnitAxes: false);
                // Nothing written (or, for one element, nothing converted inside the masked loop) when nothing is selected.
                if (dst.size == 0 || (@where is not null && !np.any(@where)))
                    return;
                throw NDStringCasting.TextConversion(from, to);
            }

            // An overlapping source (two views of one buffer read as text and as numbers) is copied first by an UNMASKED
            // conversion of every element — after its own broadcast check, so the broadcast error still comes first.
            int dnd = dst.ndim;
            bool overlapNeedsCopy =
                ((dnd == 1 && dst.Shape.strides[0] * src.Shape.strides[src.ndim - 1] < 0) || dnd > 1)
                && NDMemOverlap.SolveMayShareMemory(src, dst, 0) != MemOverlap.No;
            CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides, "input array",
                stripLeadingUnitAxes: true);
            if (overlapNeedsCopy && dst.size > 0 && parses)
                throw NDStringCasting.TextConversion(from, to);

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
            throw NDStringCasting.TextConversion(from, to);
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
