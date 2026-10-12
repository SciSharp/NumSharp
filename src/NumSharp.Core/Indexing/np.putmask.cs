using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Change elements of <paramref name="a"/> based on a boolean mask and
        ///     input values. Where <paramref name="mask"/> is true (walked in
        ///     C-order), writes <c>a.flat[i] = values.flat[i % values.size]</c>.
        ///     In-place.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">
        ///     Boolean mask. Must have the same total <em>size</em> as
        ///     <paramref name="a"/> — NumPy allows shape mismatch as long as element
        ///     counts match. Non-bool masks are cast to bool (<c>!= 0</c>; NaN/inf → true).
        /// </param>
        /// <param name="values">
        ///     Values to write, cast to <paramref name="a"/>'s dtype. Cycled by
        ///     <em>position</em>: the value written at flat index <c>i</c> is
        ///     <c>values.flat[i % values.size]</c> (the cursor advances on every
        ///     element, unlike <see cref="place"/> which advances only on True
        ///     positions). An empty <paramref name="values"/> is a no-op — NumPy
        ///     returns without touching <paramref name="a"/> (unlike
        ///     <see cref="place"/>, which raises).
        /// </param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.putmask.html
        ///     <para>
        ///     Differs from <see cref="place"/> in two probed ways (NumPy 2.4.2):
        ///     the values cursor advances by position (every element) rather than per
        ///     True, and an empty <paramref name="values"/> is a silent no-op rather
        ///     than a <c>ValueError</c>.
        ///     </para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the scratch copy (a non-contiguous target, or one that may overlap
        ///     <paramref name="values"/>/<paramref name="mask"/>) and the bool/dtype-converted copies of
        ///     <paramref name="mask"/> and <paramref name="values"/> are released on return, exceptions included. A
        ///     mask or values array used as-is is the caller's (built before the scope opened), so the scope never
        ///     disposes it.</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is read-only (checked first, as NumPy does), or
        ///     <paramref name="mask"/>'s size differs from <paramref name="a"/>'s (<c>putmask: mask and data must be the
        ///     same size</c>).</exception>
        /// <exception cref="InvalidCastException">An array <paramref name="values"/> does not cast 'safely' to
        ///     <paramref name="a"/>'s dtype (NumPy's TypeError: <c>Cannot cast array data from dtype('float64') to
        ///     dtype('int32') according to the rule 'safe'</c>) — even when it is empty. A 0-d value (NumSharp's stand-in
        ///     for a Python scalar) is converted unconditionally.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts); numbers into a text array are written as their <c>str()</c>.</exception>
        /// <exception cref="SystemError">A <c>U</c> mask holds a unit above U+10FFFF: NumPy's string-to-bool cast of the mask
        ///     raises it, before the size check. (A text mask is otherwise read by its truth values: a non-empty string is
        ///     True.)</exception>
        public static void putmask(NDArray a, NDArray mask, NDArray values)
            => PutMaskCore(a, mask, values, sequence: false);

        /// <summary>
        ///     <see cref="putmask(NDArray, NDArray, NDArray)"/> with the values given as a C# array — NumPy's Python LIST of
        ///     values: <c>np.putmask(a, mask, [1.9, 2.9])</c> is <c>np.putmask(a, mask, new[] { 1.9, 2.9 })</c>.
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="a"/>'s size.</param>
        /// <param name="values">
        ///     The values, cycled by position. NumPy converts a list STRAIGHT INTO <paramref name="a"/>'s dtype
        ///     (<c>PyArray_FromAny(list, a.dtype)</c>), so there is no <c>'safe'</c> rule: <c>[1.9, 2.9]</c> into an int32
        ///     array truncates — where <c>np.array([1.9, 2.9])</c> raises the cast TypeError.
        /// </param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">The checks of <see cref="putmask(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric array.</exception>
        /// <remarks>C# binds a typed array argument here rather than to the <see cref="NDArray"/> overload (a reference
        ///     conversion beats NumSharp's implicit <c>Array → NDArray</c> one): pass <c>np.array(...)</c> for NumPy's ndarray
        ///     semantics.</remarks>
        [NDScoped]
        public static void putmask(NDArray a, NDArray mask, Array values)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));
            // The list's own dtype is discovered first, then converted into a's dtype by PutMaskCore after the size check.
            PutMaskCore(a, mask, np.array(values), sequence: true);
        }

        /// <summary>
        ///     The body of both <c>np.putmask</c> forms (NumPy's <c>PyArray_PutMask</c>, in its order).
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="a"/>'s size.</param>
        /// <param name="values">The values (an ndarray, or the array discovered from a list).</param>
        /// <param name="sequence">True when <paramref name="values"/> came from a list: no <c>'safe'</c> rule.</param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ValueError">See <see cref="putmask(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="InvalidCastException">An ndarray of values that does not cast 'safely'.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">See <see cref="putmask(NDArray, NDArray, NDArray)"/>.</exception>
        [NDScoped]
        private static void PutMaskCore(NDArray a, NDArray mask, NDArray values, bool sequence)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, mask, values);   // in place: the helpers below see a matrix as a plain array
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (mask is null) throw new ArgumentNullException(nameof(mask));
            if (values is null) throw new ArgumentNullException(nameof(values));

            // Read-only target reported FIRST — NumPy checks FailUnlessWriteable right after the
            // is-an-array check, before the mask-size test and before the nv==0 no-op (probed 2.4.2:
            // putmask(ro, all-false-mask, v) and putmask(ro, mask, []) both raise read-only). The
            // contiguous kernel writes a.Storage.Address directly; the non-contiguous path is guarded
            // via np.copyto below.
            NumSharpException.ThrowIfNotWriteable(a.Shape, "putmask: output array");

            // PyArray_FROM_OTF(mask, NPY_BOOL, CARRAY | FORCECAST) comes next, before the size check: a text mask becomes
            // its truth values here (NumPy's string-to-bool cast; a U unit above U+10FFFF raises SystemError before any
            // later error). The fresh truth array is reclaimed with the scope and never overlaps `a`.
            if (mask.Storage.IsVString)
                mask = mask.astype(NPTypeCode.Boolean, copy: true, order: 'C'); // StringDType's bool cast (NA rules)
            else if (mask.typecode == NPTypeCode.String)
                mask = Backends.Strings.NDStringTruth.Truth(mask, Backends.Strings.NDStringTruth.Flavor.Cast, 'C');

            if (mask.size != a.size)
                throw new ValueError("putmask: mask and data must be the same size");

            // PyArray_FromAny(values, a.dtype, CARRAY): the 'safe' rule for an array of values (a dtype check, so empty
            // values and an empty target raise too) and every value's conversion, before nv is read.
            NDArray converted = PlaceValuesLikeNumPy(values, a, sequence);

            // Empty values ⇒ no-op. NumPy: nv <= 0 returns None without writing (probed 2.4.2).
            // Checked after writeable + mask-size + the conversion (NumPy's order), before the contig copy so an
            // empty-values call never allocates a scratch.
            if (values.size == 0)
                return;

            // Empty a ⇒ empty mask ⇒ no-op (the loop runs zero times in NumPy).
            if (a.size == 0)
                return;

            // The values the loop reads (a converted copy, or the caller's array of a's element type).
            var source = converted ?? values;

            // NumPy WRITEBACKIFCOPY / ENSURECOPY semantics (PyArray_PutMask): copy `a` when it is
            // non-contiguous OR may share memory with `values`/`mask`. NumPy's `arrays_overlap` is
            // `solve_may_share_memory(..., NPY_MAY_SHARE_BOUNDS)` == `SolveMayShareMemory(maxWork:0)`
            // (a cheap bounds-only check: `!= No` means "may share"). The fresh scratch cannot alias
            // the read operands, so a write into it never disturbs a later cyclic read of `values`
            // that aliases `a` — e.g. `putmask(a, mask, a[2:8])`, where the contiguous fast path
            // would write a[i] and then read it back as the wrapped `values[i % nv]` and diverge.
            // The kernel writes the scratch in C-order flat (matching NumPy's copy-and-writeback);
            // np.copyto pushes the result back through `a`'s strides to the parent storage. The
            // recursive call sees a fresh scratch that overlaps neither operand, so it takes the
            // direct path (no re-copy, no recursion).
            // NumPy's arrays_overlap runs on the CONVERTED values (a fresh conversion never overlaps `a`).
            bool overlaps =
                NDMemOverlap.SolveMayShareMemory(a, source, maxWork: 0) != MemOverlap.No ||
                NDMemOverlap.SolveMayShareMemory(a, mask, maxWork: 0) != MemOverlap.No;
            if (overlaps || !a.Shape.IsContiguous)
            {
                // C-contiguous a (the overlap case) → a.copy() is a fresh C-order buffer;
                // any non-contiguous layout → ascontiguousarray makes the C-order copy.
                // The recursive call opens its own nested scope; the scratch itself is released by this one. It re-runs
                // the checks on the already converted values (the same element type: nothing left to fail but the mask's).
                var scratch = a.Shape.IsContiguous ? a.copy() : np.ascontiguousarray(a);
                PutMaskCore(scratch, mask, source, sequence);
                np.copyto(a, scratch, casting: "unsafe");
                return;
            }

            // Cast mask to contig bool (a text mask already is: its truth array, made above). It is raveled in C-order, aligning flat position i of the mask with flat
            // position i of a (shapes need not match, only sizes — NumPy checks size, not shape). A converted copy is
            // built inside the scope and released with it; an operand used as-is is the caller's and is never disposed.
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

            // The values in a's element type, C-contiguous (the conversion above already is).
            NDArray valsCast = source.Shape.IsContiguous ? source : np.ascontiguousarray(source);
            if (a.Storage.IsVString)
            {
                // A StringDType target: the same putmask runs on marks (the values' element numbers, cycled by position)
                // and the marked elements receive the strings.
                var mk = maskCast;
                Backends.Strings.NDVStringMove.ScatterVia(a, valsCast, (marks, index) => PutMaskCore(marks, mk, index, false));
                return;
            }
            ExecutePutMask(a, maskCast, valsCast);
        }

        private static unsafe void ExecutePutMask(NDArray a, NDArray maskBool, NDArray valsCast)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(a.dtypesize);
            var kernel = DirectILKernelGenerator.GetPutMaskKernel(copyKind);
            if (kernel == null)
                throw new NotSupportedException("np.putmask: IL kernel unavailable");

            byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            byte* maskPtr = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            kernel(dstPtr, maskPtr, a.size, valsPtr, valsCast.size, a.dtypesize);
        }
    }
}
