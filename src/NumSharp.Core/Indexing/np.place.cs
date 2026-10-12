using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Change elements of <paramref name="arr"/> based on a boolean mask.
        ///     Where <paramref name="mask"/> is true (walked in C-order), the
        ///     next value from <paramref name="vals"/> (cycling) is written into
        ///     <paramref name="arr"/>. In-place.
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">
        ///     Boolean mask. Must have the same total <em>size</em> as
        ///     <paramref name="arr"/> — NumPy allows shape mismatch as long as
        ///     element counts match.
        /// </param>
        /// <param name="vals">
        ///     Values to write, converted to <paramref name="arr"/>'s dtype and cycled to fill the True positions. An
        ///     ARRAY of values must convert under NumPy's <c>'safe'</c> rule (a 0-d value — NumSharp's stand-in for a
        ///     Python scalar — is converted unconditionally). Must be non-empty when at least one mask entry is True.
        /// </param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError"><paramref name="arr"/> is read-only (checked first, as NumPy does), or
        ///     <paramref name="mask"/>'s size differs from <paramref name="arr"/>'s (<c>place: mask and data must be the
        ///     same size</c>), or <paramref name="vals"/> is empty while some mask entry is True (<c>Cannot insert from an
        ///     empty array!</c>).</exception>
        /// <exception cref="InvalidCastException">An array <paramref name="vals"/> does not cast 'safely' to
        ///     <paramref name="arr"/>'s dtype (NumPy's TypeError: <c>Cannot cast array data from dtype('float64') to
        ///     dtype('int64') according to the rule 'safe'</c>) — even when <paramref name="arr"/> is empty.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts); numbers into a text array are written as their <c>str()</c>.</exception>
        /// <exception cref="SystemError">A <c>U</c> mask holds a unit above U+10FFFF: NumPy's string-to-bool cast of the mask
        ///     raises it, before the size check. (A text mask is otherwise read by its truth values: a non-empty string is
        ///     True.)</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.place.html
        ///     <para><b>Order (NumPy's <c>arr_place</c>).</b> The target's writeability; the mask's conversion to bool and its
        ///     size; the values' conversion (<c>PyArray_FromAny(vals, arr.dtype, CARRAY)</c>: the 'safe' rule for an array,
        ///     every value converted — so a non-ASCII byte value raises even under an all-false mask); then the empty-values
        ///     check and the loop. A NUMBER formatted into a text array cannot fail, so it is refused only where it is
        ///     written; a TEXT mask is refused where its truth is read (the empty-values check, or the loop).</para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the contiguous scratch of a non-contiguous target and the
        ///     bool/dtype-converted copies of <paramref name="mask"/> and <paramref name="vals"/> are released on
        ///     return, exceptions included. A mask or values array used as-is is the caller's (built before the scope
        ///     opened), so the scope never disposes it.</para>
        /// </remarks>
        public static void place(NDArray arr, NDArray mask, NDArray vals)
            => PlaceCore(arr, mask, vals, sequence: false);

        /// <summary>
        ///     <see cref="place(NDArray, NDArray, NDArray)"/> with the values given as a C# array — NumPy's Python LIST of
        ///     values: <c>np.place(a, mask, ['P', 'QQ', 'RRRR'])</c> is <c>np.place(a, mask, new[] { "P", "QQ", "RRRR" })</c>.
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="arr"/>'s size.</param>
        /// <param name="vals">
        ///     The values, cycled over the True positions. NumPy converts a list STRAIGHT INTO <paramref name="arr"/>'s dtype
        ///     (<c>PyArray_FromAny(list, arr.dtype)</c>, as <c>np.array(vals, dtype=arr.dtype)</c> does), so there is no
        ///     <c>'safe'</c> rule: <c>'RRRR'</c> into a <c>&lt;U3</c> array truncates to <c>'RRR'</c>, <c>1.9</c> into an int
        ///     array truncates to 1 — where an ndarray of the same values raises the cast TypeError.
        /// </param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">The checks of <see cref="place(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array.</exception>
        /// <remarks>
        ///     C# binds a typed array argument (<c>double[]</c>, <c>string[]</c>) here rather than to the
        ///     <see cref="NDArray"/> overload: an identity/reference conversion beats NumSharp's implicit
        ///     <c>Array → NDArray</c> conversion. Pass <c>np.array(...)</c> to get NumPy's ndarray semantics.
        /// </remarks>
        [NDScoped]
        public static void place(NDArray arr, NDArray mask, Array vals)
        {
            if (vals is null) throw new ArgumentNullException(nameof(vals));
            // The list's own dtype is discovered first (np.array), then converted into arr's dtype by PlaceCore after the
            // size check, as NumPy's conversion follows its mask checks. The discovered array is released with the scope.
            PlaceCore(arr, mask, np.array(vals), sequence: true);
        }

        /// <summary>
        ///     The body of both <c>np.place</c> forms (NumPy's <c>arr_place</c>, in its order).
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="arr"/>'s size.</param>
        /// <param name="vals">The values (an ndarray, or the array discovered from a list).</param>
        /// <param name="sequence">True when <paramref name="vals"/> came from a list: NumPy converts it into the target's
        ///     dtype without the <c>'safe'</c> rule.</param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">See <see cref="place(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="InvalidCastException">An ndarray of values that does not cast 'safely'.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">See <see cref="place(NDArray, NDArray, NDArray)"/>.</exception>
        [NDScoped]
        private static void PlaceCore(NDArray arr, NDArray mask, NDArray vals, bool sequence)
        {
            using var plain = NDMatrixBoundary.PlainFor(arr, mask, vals);   // in place: the helpers below see a matrix as a plain array
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (mask is null) throw new ArgumentNullException(nameof(mask));
            if (vals is null) throw new ArgumentNullException(nameof(vals));

            // Read-only target reported first (NumPy uses WRITEBACKIFCOPY; probed 2.4.2:
            // place(ro, all-false-mask, v) still raises read-only). The contiguous kernel
            // (ExecutePlace) writes arr.Storage.Address directly; the non-contiguous path is
            // already guarded via np.copyto.
            NumSharpException.ThrowIfNotWriteable(arr.Shape, "WRITEBACKIFCOPY base");

            // PyArray_FROM_OTF(mask, NPY_BOOL, CARRAY | FORCECAST) comes next, before the size check: a text mask becomes
            // its truth values here (NumPy's string-to-bool cast; a U unit above U+10FFFF raises SystemError before any
            // later error). The fresh truth array is reclaimed with the scope.
            if (mask.Storage.IsVString)
                mask = mask.astype(NPTypeCode.Boolean, copy: true, order: 'C'); // StringDType's bool cast (NA rules)
            else if (mask.typecode == NPTypeCode.String)
                mask = Backends.Strings.NDStringTruth.Truth(mask, Backends.Strings.NDStringTruth.Flavor.Cast, 'C');

            if (mask.size != arr.size)
                throw new ValueError("place: mask and data must be the same size");

            // NumPy converts the values before it looks at the mask's contents — the casting rule (a dtype check, so an
            // empty target raises too) and every value's conversion.
            NDArray converted = PlaceValuesLikeNumPy(vals, arr, sequence);

            // Empty arr ⇒ empty mask ⇒ no-op.
            if (arr.size == 0)
                return;

            // NumPy WRITEBACKIFCOPY semantics for non-contig views — see np.put for
            // the same pattern. ascontiguousarray clones the view into a contig
            // scratch; the kernel writes into the scratch; np.copyto pushes the
            // changes back through the view's strides to the parent storage.
            if (!arr.Shape.IsContiguous)
            {
                // The recursive call opens its own nested scope; the scratch itself is released by this one. It re-runs
                // the checks on the already converted values (the same dtype: nothing left to fail but the mask's).
                var scratch = np.ascontiguousarray(arr);
                PlaceCore(scratch, mask, converted ?? vals, sequence);
                np.copyto(arr, scratch, casting: "unsafe");
                return;
            }

            // Cast mask to contig bool (a text mask already is: its truth array, made above); cast vals to contig arr.dtype. A converted copy is built inside the
            // scope and released with it; an operand used as-is is the caller's and is never disposed.
            NDArray maskCast;
            if (mask.GetTypeCode == NPTypeCode.Boolean && mask.Shape.IsContiguous)
            {
                maskCast = mask;
            }
            else
            {
                // C order: the kernel walks the mask linearly beside the target's C-order positions, and astype's
                // default 'K' keeps an F-order mask F-ordered (an int8 F-order [[1, 1], [0, 0]] selected positions 0 and
                // 2 where NumPy selects 0 and 1).
                maskCast = mask.GetTypeCode == NPTypeCode.Boolean
                    ? np.ascontiguousarray(mask)
                    : mask.astype(NPTypeCode.Boolean, copy: true, order: 'C');
            }

            if (vals.size == 0 && AnyTrueMask(maskCast))
                throw new ValueError("Cannot insert from an empty array!");

            if (vals.size == 0)
                return;   // mask all-false and empty vals — NumPy treats as no-op

            NDArray valsCast = converted ?? (vals.Shape.IsContiguous ? vals : np.ascontiguousarray(vals));
            if (arr.Storage.IsVString)
            {
                // A StringDType target: the same place runs on marks (the values' element numbers, cycled over the True
                // positions) and the marked elements receive the strings. (NumPy 2.4.2 crashes the process here.)
                var mk = maskCast;
                Backends.Strings.NDVStringMove.ScatterVia(arr, valsCast, (marks, index) => PlaceCore(marks, mk, index, false));
                return;
            }
            ExecutePlace(arr, maskCast, valsCast);
        }

        /// <summary>
        ///     The value conversion of <see cref="place(NDArray, NDArray, NDArray)"/> and
        ///     <see cref="putmask(NDArray, NDArray, NDArray)"/> — NumPy's <c>PyArray_FromAny(values, target.dtype, 0, 0,
        ///     NPY_ARRAY_CARRAY)</c>: an ARRAY of values must convert under the <c>'safe'</c> rule (a dtype check, made
        ///     whatever the sizes); a C# literal's 0-d array (<see cref="NDArray.WasPythonLiteral"/>, NumPy's Python scalar)
        ///     is PACKED with PyArray_Pack's range checks; any other 0-d value — NumSharp's stand-in for a NumPy scalar,
        ///     which NumPy also converts without the rule — and a list (a C# array) are converted unconditionally; then
        ///     every value is converted.
        /// </summary>
        /// <param name="values">The caller's values.</param>
        /// <param name="target">The array written into (its dtype is the conversion's target).</param>
        /// <param name="sequence">True when <paramref name="values"/> was discovered from a list: no <c>'safe'</c> rule, and
        ///     a list of strings parsed into a numeric target quotes each Python str (<c>'x'</c>) in its error.</param>
        /// <returns>The converted values (C-contiguous, the target's dtype), or null when <paramref name="values"/> already
        ///     holds the target's element type.</returns>
        /// <exception cref="InvalidCastException">An array of values does not cast 'safely' (NumPy's TypeError text).</exception>
        /// <exception cref="OverflowException">A C# literal out of the target's integer range (NumPy's OverflowError).</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer target.</exception>
        /// <exception cref="TypeError">A complex literal into a real target.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric target (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts).</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> target.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> target.</exception>
        /// <remarks>An ARRAY of text values never casts safely to a number, and numbers cast safely to text only when the
        ///     text is wide enough to hold any of them (<c>int64 → &lt;U21</c>): both come from the rule check.</remarks>
        private static NDArray PlaceValuesLikeNumPy(NDArray values, NDArray target, bool sequence)
        {
            // A C# literal (NDArray.WasPythonLiteral / WasPythonStr) is NumPy's Python scalar: PyArray_FromAny(values, dtype,
            // CARRAY) packs it with PyArray_Pack's range checks (np.putmask(int8, m, 1000) / np.place(int8, m, 1000) raise
            // "Python integer 1000 out of bounds for int8"; a str is parsed, its error quoting the str) instead of applying
            // an array's 'safe' rule or cast. A literal of the target's own dtype is exact and is used as is.
            if ((values.WasPythonLiteral || values.WasPythonStr) && target.typecode != NPTypeCode.String)
            {
                var packed = NDArray.PackForStore(values, target.typecode, pyArrayPackRoute: false);
                return ReferenceEquals(packed, values) ? null : packed;
            }

            // PyArray_FromArray (an ndarray) checks the rule; a list is built in the target's dtype directly.
            if (!sequence && values.ndim != 0
                && !DTypeCasting.CanCastTypeTo(values.dtype, target.dtype, NPY_CASTING.NPY_SAFE_CASTING))
                throw Backends.Strings.NDStringCasting.InvalidCast(values.dtype, target.dtype, NPY_CASTING.NPY_SAFE_CASTING, scalar: false);

            if (values.HasSameElementType(target))
                return null;

            // Into StringDType (or from it): the StringDType casts (text, numbers' str(), the NA rules), in C order.
            if (target.Storage.IsVString || values.Storage.IsVString)
                return values.astype(target.dtype, copy: true, order: 'C');

            bool textValues = values.typecode == NPTypeCode.String, textTarget = target.typecode == NPTypeCode.String;
            if (textValues != textTarget)
            {
                // Text into numbers (a 0-d text array, or a list of strings — an ndarray of text failed the 'safe' rule
                // above) is parsed, numbers into text formatted, every value now, before the mask is read.
                var result = NDArray.ForOverwrite(target.dtype, Shape.Adopt((long[])values.Shape.dimensions.Clone()));
                try
                {
                    if (result.size > 0)
                        Backends.Strings.NDTextConversion.Convert(result.Storage, values.Storage,
                            sequence ? Backends.Strings.TextReprStyle.PythonObject : Backends.Strings.TextReprStyle.ArrayElement);
                    return result;
                }
                catch
                {
                    result.Dispose();
                    throw;
                }
            }
            // Number to number (unsafe for a scalar, the rule already checked for an array), or text to text (truncated /
            // NUL-padded, S <-> U ASCII-checked): every value converted now, as NumPy does before reading the mask. In C
            // order: the place / putmask kernels read the values linearly (PyArray_FromAny's CARRAY), and astype's default
            // 'K' kept an F-order input F-ordered (np.place's int32 F-order values were written in memory order).
            return values.astype(target.dtype, copy: true, order: 'C');
        }

        private static unsafe bool AnyTrueMask(NDArray maskBool)
        {
            // Walk the contig bool buffer linearly. mask.size > 0 by the time we get here.
            byte* p = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            long n = maskBool.size;
            for (long i = 0; i < n; i++)
                if (p[i] != 0) return true;
            return false;
        }

        private static unsafe void ExecutePlace(NDArray arr, NDArray maskBool, NDArray valsCast)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(arr.dtypesize);
            var kernel = DirectILKernelGenerator.GetPlaceKernel(copyKind);
            if (kernel == null)
                throw new NotSupportedException("np.place: IL kernel unavailable");

            byte* dstPtr = (byte*)arr.Storage.Address + arr.Shape.offset * arr.dtypesize;
            byte* maskPtr = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            kernel(dstPtr, maskPtr, arr.size, valsPtr, valsCast.size, arr.dtypesize);
        }
    }
}
