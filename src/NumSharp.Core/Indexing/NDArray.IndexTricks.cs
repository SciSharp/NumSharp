using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    // ============================== lib/_index_tricks_impl.py ==============================
    // NumPy's index tricks as members of the array object (P3's batch 7): np.ix_ (Ix), np.fill_diagonal (FillDiagonal,
    // with the diagonal scatter np.diag shares) and the statements of np.r_ / np.c_'s indexer (AxisConcatenate, NumPy's
    // AxisConcatenator.__getitem__) with their helpers, and np.mgrid / np.ogrid's (MGrid, OGrid: NumPy's
    // nd_grid.__getitem__). The facades keep their numpy.matrix rules and null checks, and np.AxisConcatenator keeps
    // the C# spelling of a slice (SliceSpec, ParseSliceToken), which np.ogrid / np.mgrid share.
    public partial class NDArray
    {
        // ---------------------------- np.ix_ ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.ix_</c> after the facade: each sequence made an array (a non-ndarray one that comes out empty
        ///     cast to <c>intp</c>: an object[] or collection, never an <see cref="NDArray"/>, a typed C# array or a
        ///     <c>Memory&lt;T&gt;</c>, which are ndarrays in the boundary map), refused unless 1-D, a bool one replaced by its
        ///     <c>nonzero()</c>, then reshaped to its own axis of the open mesh.
        /// </summary>
        /// <param name="args">The sequences (the facade's <c>params</c> array; null or empty gives no arrays).</param>
        /// <returns>One array per sequence, each a NEW view of its sequence's array with length-1 axes inserted - also for a
        ///     single sequence, which gets no axis: the result never is the operand or the array a conversion made, so it
        ///     never owns its data.</returns>
        /// <exception cref="ValueError">A sequence that is not 1-D, null (NumPy's None) included: "Cross index must be 1
        ///     dimensional".</exception>
        /// <remarks>Scoped: the conversions of the sequences and the masks' index arrays are temporaries of this call (the
        ///     facade keeps its own scope).</remarks>
        [NDScoped]
        internal static NDArray[] Ix(object[] args)
        {
            if (args is null || args.Length == 0)
                return System.Array.Empty<NDArray>();

            int nd = args.Length;
            var @out = new NDArray[nd];

            for (int k = 0; k < nd; k++)
            {
                var item = args[k];
                NDArray New;

                if (item is null)
                {
                    // NumPy reaches the same error by a different route: asarray(None) is a 0-d
                    // object array, whose ndim is 0. NumSharp has no object dtype, so the outcome
                    // is reproduced directly rather than letting asanyarray raise a bare
                    // ArgumentNullException from three frames down.
                    throw new ValueError("Cross index must be 1 dimensional");
                }

                if (item is NDArray already)
                {
                    New = already;
                }
                else
                {
                    New = np.asanyarray(item);
                    // Explicitly type empty arrays to avoid the float default - NumPy's `if not isinstance(new, ndarray)`
                    // branch: only a sequence that is NOT an ndarray is cast. A typed C# array or Memory<T> is an ndarray in
                    // the boundary map (UnmanagedStorage.IsArrayLike) and keeps its dtype even when empty
                    // (np.ix_(np.array([], np.int32))[0] is int32, probed 2.4.2); an empty object[] or collection is a
                    // Python list, whose float64 [] becomes intp.
                    if (New.size == 0 && New.typecode != NPTypeCode.Int64 && !Backends.UnmanagedStorage.IsArrayLike(item))
                        New = New.astype(NPTypeCode.Int64);
                }

                if (New.ndim != 1)
                    throw new ValueError("Cross index must be 1 dimensional");

                if (New.typecode == NPTypeCode.Boolean)
                    New = np.nonzero(New)[0];

                // NumPy's reshape((1,)*k + (size,) + (1,)*(nd-k-1)), spelled as the equivalent
                // insertion of length-1 axes. expand_dims aliases the storage, so a stride-0
                // (broadcast) or otherwise read-only operand keeps its strides AND its
                // non-writeable flag — reshape materializes those and would hand back a
                // writeable copy, which NumPy does not do.
                // A single sequence inserts no axis, and NumPy's reshape to the same shape is still a NEW view
                // (PyArray_Newshape's same-shape case is PyArray_View): never the operand itself, never the owning
                // array a conversion made (np.ix_(a)[0] is not a, and np.ix_([1, 2])[0].flags.owndata is False, probed
                // 2.4.2). expand_dims with no axes gives that view, so the call is made for every nd.
                var axes = new int[nd - 1];
                for (int i = 0, w = 0; i < nd; i++)
                {
                    if (i != k)
                        axes[w++] = i;
                }

                New = np.expand_dims(New, axes);

                @out[k] = New;
            }

            return @out;
        }

        // ---------------------------- np.fill_diagonal ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.fill_diagonal</c> after the facade's numpy.matrix region and null check: the diagonal's
        ///     <c>step</c> and <c>end</c> (a rectangular 2-D array, or a hyper-cube), then <c>a.flat[:end:step] = val</c> -
        ///     the read-only check and the value's conversion (<see cref="FlatStoreValue"/>) before the slice is walked, even
        ///     when it selects nothing, then the values tiled to the diagonal's length and stored through the strided-store
        ///     kernel.
        /// </summary>
        /// <param name="a">The array, filled in place (not null).</param>
        /// <param name="val">The value or values (null is Python's None).</param>
        /// <param name="wrap">Whether a tall matrix's diagonal wraps after N columns.</param>
        /// <exception cref="ArgumentException">Fewer than 2 dimensions, a non-hyper-cubic N-D array, or a read-only
        ///     array (NumPy's ValueError texts).</exception>
        /// <exception cref="TypeError">None into an integer or decimal array (Python's <c>int(None)</c>), a complex into a
        ///     real array, or a list item the dtype cannot convert.</exception>
        /// <exception cref="ValueError">A <c>StringDType(coerce=False)</c> array given a value that is not text; a str
        ///     that is not a number of a numeric dtype's kind (Python's <c>int()</c> / <c>float()</c> / <c>complex()</c>
        ///     text, quoting the str); a NaN into an integer array.</exception>
        /// <exception cref="OverflowException">A Python int, alone or in a list, out of the dtype's range (the LP64
        ///     texts).</exception>
        /// <exception cref="NotSupportedException">A value of no NumPy type.</exception>
        /// <remarks>Scoped: the packed value, its broadcast and the tiled values are temporaries of this call.</remarks>
        [NDScoped]
        internal static void FillDiagonal(NDArray a, object val, bool wrap)
        {

            int ndim = a.ndim;
            if (ndim < 2)
                throw new ArgumentException("array must be at least 2-d");

            long step;
            long end;
            if (ndim == 2)
            {
                // Explicit, fast formula for the common case; rectangular arrays are accepted.
                long cols = a.Shape.dimensions[1];
                step = cols + 1;
                // Without wrapping, stop before a tall matrix's diagonal re-enters column 0.
                end = wrap ? a.size : cols * cols;
            }
            else
            {
                // For d > 2 the strided formula is only valid when all dimensions are equal.
                np.RequireEqualDimensions(a);

                step = 1;
                long cum = 1;
                for (int d = 0; d < ndim - 1; d++)
                {
                    cum *= a.Shape.dimensions[d];
                    step += cum;
                }

                end = a.size;
            }

            if (end > a.size) end = a.size;

            // NumPy's `a.flat[:end:step] = val` (iter_ass_subscript): the read-only check, then the value's conversion
            // PyArray_FromAny(val, a.dtype, 0, 0, NPY_ARRAY_FORCECAST), then the slice - so the conversion's errors are raised
            // even when the slice selects nothing: an empty array still refuses a read-only target, None into an integer
            // dtype (np.fill_diagonal(np.zeros((0, 0), int), None)), an out-of-range Python int and a str that int() /
            // float() refuses (np.fill_diagonal(np.zeros((0, 3), np.int8), 'x')), probed 2.4.2.
            if (!a.Shape.IsWriteable)
                throw new ArgumentException("underlying array is read-only");

            // A StringDType target converts with the array's descriptor — Python objects (None and a list's elements
            // included) by setitem, an ndarray by the cast (NDVStringMove.StoreValue), so a value coerce=False refuses
            // raises even for an empty diagonal; every other target through FlatStoreValue (a scalar packed into a 0-d
            // array, anything else converted and raveled). The descriptor answers which arrays take the first route: the
            // ones whose elements hold references (NumPy's PyDataType_REFCHK), as the item-selection family's one
            // HasReferences decision does.
            bool references = a.dtype.Meta.HasReferences;
            NDArray text = references ? Backends.Strings.NDVStringMove.StoreValue(val, a.dtype) : null;
            NDArray converted = references ? null : FlatStoreValue(val, a);

            if (end <= 0 || step <= 0)
                return;

            long count = (end + step - 1) / step;
            if (count <= 0)
                return;

            if (text is not null)
            {
                // A StringDType target: the value in its descriptor (a number becomes its str()), then the same fill on
                // marks (the value's element numbers, tiled cyclically by the numeric fill) and the marked diagonal elements
                // receive the strings — never the raw strided-store kernel.
                if (text.size == 0)
                    return; // NumPy: assigning an empty sequence leaves the diagonal untouched.
                var flatText = text.ravel();
                if (!flatText.HasSameElementType(a))
                    flatText = flatText.astype(a.dtype);
                Backends.Strings.NDVStringMove.ScatterVia(a, flatText, (marks, index) => np.fill_diagonal(marks, index, wrap));
                return;
            }

            // NumPy: assigning an empty sequence leaves the diagonal untouched (iter_ass_subscript returns when the
            // converted value has no element).
            if (converted.size == 0)
                return;

            // One element (a packed scalar, or a one-element sequence): a stride-0 broadcast view supplies the `count`
            // copies for free. More: the raveled values tiled cyclically / truncated to exactly `count` (np.resize's rule,
            // the flat iterator's repetition of a short right-hand side).
            NDArray values = converted.size == 1
                ? np.broadcast_to(converted, new Shape(count))
                : np.resize(converted, new[] {(int)count});

            var shape = a.Shape;
            long bufferSize = shape.BufferSize;

            if (ndim > 2)
            {
                // Hyper-cubic: targets are (i, i, …, i) — one constant stride.
                long diagStride = 0;
                for (int d = 0; d < ndim; d++)
                    diagStride += shape.strides[d];

                WriteDiagonalBlock(a, values, 0, count, shape.Offset, diagStride, bufferSize);
                return;
            }

            long s0 = shape.strides[0];
            long s1 = shape.strides[1];
            long colCount = a.Shape.dimensions[1];
            long blockStride = s0 + s1;

            // Each block of `cols` targets marches down the diagonal; the next block restarts
            // `cols + 1` rows lower (that is the wrap). Without wrapping count <= cols, so
            // this degenerates to a single block.
            for (long written = 0; written < count; written += colCount)
            {
                long len = System.Math.Min(colCount, count - written);
                long q = written / colCount;
                long offset = shape.Offset + q * (colCount + 1) * s0;

                WriteDiagonalBlock(a, values, written, len, offset, blockStride, bufferSize);
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FromAny(val, a.dtype, 0, 0, NPY_ARRAY_FORCECAST)</c>, the conversion np.fill_diagonal's
        ///     <c>a.flat[:end:step] = val</c> makes before it walks the slice, for an array whose elements hold no references
        ///     (a <c>StringDType</c> array converts with its descriptor's own route): ONE Python or NumPy scalar is packed
        ///     into a 0-d array of the dtype by its setitem, anything else is converted to the dtype and raveled.
        /// </summary>
        /// <param name="val">The value; null is Python's None.</param>
        /// <param name="a">The destination array, whose dtype is the request.</param>
        /// <returns>A 0-d array of <paramref name="a"/>'s dtype (a scalar), or a 1-D one (the raveled values, possibly
        ///     empty, possibly a view of <paramref name="val"/> when it already has the dtype): an intermediate of the
        ///     caller's scope.</returns>
        /// <exception cref="TypeError">None into an integer or <c>decimal</c> array (Python's <c>int(None)</c>), a complex
        ///     into a real one (<c>float()</c> / <c>int()</c> of a complex), or a list item the dtype cannot
        ///     convert.</exception>
        /// <exception cref="ValueError">A str that is not a number of the dtype's kind, quoted as the str (<c>invalid literal
        ///     for int() with base 10: 'x'</c>, <c>could not convert string to float: 'x'</c>), a NaN into an integer array,
        ///     a ragged list, or an array whose cast refuses its elements.</exception>
        /// <exception cref="OverflowException">A Python int, alone or in a list, outside an integer dtype ("Python integer
        ///     1000 out of bounds for int8"; the LP64 "Python int too large to convert to C long" past the C long), or past
        ///     the double range into a float dtype.</exception>
        /// <exception cref="NotSupportedException">A value of no NumPy type.</exception>
        /// <remarks>
        ///     <para>
        ///     A scalar (a C# number, <c>bool</c>, <see cref="BigInteger"/> or <see cref="string"/>; a NumPy scalar
        ///     <see cref="Half"/>, <c>char</c> or <c>decimal</c>) is the setitem's business: a numeric dtype packs it with
        ///     <c>PyArray_Pack</c>'s rules (<see cref="AsStoreValue"/>: a Python int range-checked, a float truncated into an
        ///     integer, a str parsed by Python's <c>int()</c> / <c>float()</c> / <c>complex()</c> - so the errors quote the
        ///     str, where converting it to a <c>U</c> array first would quote <c>np.str_('x')</c> - and a NumPy scalar cast),
        ///     an <c>S</c> / <c>U</c> dtype stores its <c>str()</c> (<see cref="Backends.UnmanagedStorage.FromAnyScalar"/>:
        ///     <c>2**70</c> into <c>S5</c> is <c>b'11805'</c>). A C# literal's 0-d array (an <see cref="NDArray"/> passed as
        ///     <c>object</c>) is the Python scalar it stands for.
        ///     </para>
        ///     <para>
        ///     Anything else: an <see cref="NDArray"/> is raveled and CAST (FORCECAST: unsafe); a C# array (an ndarray in the
        ///     boundary map), a list, a tuple or a collection goes through <see cref="np.asanyarray(in object, DType, string)"/>
        ///     with the dtype requested, whose coercion of a Python list stores every item by the setitem (<c>[1000]</c> into
        ///     int8 raises, as NumPy's <c>PyArray_AssignFromCache</c> does) and whose typed C# arrays are cast.
        ///     </para>
        /// </remarks>
        private static NDArray FlatStoreValue(object val, NDArray a)
        {
            DType descr = a.dtype;
            bool text = a.typecode == NPTypeCode.String;

            // Python's None: one element of None's setitem value (NaN, nan+nanj, False, 'None' truncated to an S / U width),
            // or int(None)'s TypeError.
            if (val is null)
                return NoneForStore(descr);

            // One scalar: PyArray_Pack into the dtype, a numeric dtype by its setitem rules, a text dtype by str().
            if (IsStoreScalar(val) || val is string)
                return text ? Backends.UnmanagedStorage.FromAnyScalar(val, descr) : AsStoreValue(val, a.typecode);

            // A C# literal's 0-d array (a number's or a str's implicit conversion) packs as the Python scalar it stands for;
            // into a text dtype its cast below already gives the str() NumPy stores.
            if (!text && val is NDArray { WasPythonLiteral: true } or NDArray { WasPythonStr: true })
                return AsStoreValue(val, a.typecode);

            // An array is raveled and cast; anything else converts through np.asanyarray with the dtype requested.
            var flat = (val as NDArray ?? np.asanyarray(val, descr)).ravel();
            // Convert unless the element type already matches (for strings: class AND width — a <U1 'Z' into a <U2 array is
            // padded first, never copied as two-unit elements).
            return flat.HasSameElementType(a) ? flat : flat.astype(descr);
        }

        /// <summary>
        ///     Write <paramref name="len"/> values into an equally-strided run of
        ///     <paramref name="a"/>'s storage.
        /// </summary>
        /// <remarks>
        ///     The fast path is the IL strided-store kernel (<see cref="TryDiagonalScatter"/>),
        ///     which writes straight through <paramref name="a"/>'s real strides — no aliased
        ///     view, no NDIter setup. The <paramref name="values"/> block is addressed by its own
        ///     stride/offset, so a stride-0 broadcast (the scalar fill) and a contiguous tiled
        ///     buffer (the sequence fill) both feed it without materialising a slice. When the
        ///     kernel is unavailable the code falls back to the original alias + <c>SetData</c>.
        /// </remarks>
        [NDScopedCovered] // only reached from [NDScoped] FillDiagonal (the AOT-fallback alias + value slice are scope-reclaimed)
        private static void WriteDiagonalBlock(
            NDArray a, NDArray values, long valueStart, long len, long offset, long stride, long bufferSize)
        {
            if (len <= 0) return;

            // The i-th target reads values[valueStart + i] = base + valuesOffset + (valueStart+i)*valuesStride.
            long valuesStride = values.ndim >= 1 ? values.Shape.strides[0] : 0;
            long valuesOffset = values.Shape.offset + valueStart * valuesStride;

            if (TryDiagonalScatter(a, offset, stride, values, valuesOffset, valuesStride, len))
                return;

            // Fallback: alias a strided 1-D view and route through the copy machinery.
            var targetShape = new Shape(new[] {len}, new[] {stride}, offset, bufferSize);
            var target = new NDArray(a.Storage.Alias(targetShape)) {TensorEngine = a.TensorEngine};

            var slice = valueStart == 0 && len == values.size
                ? values
                : values[$"{valueStart}:{valueStart + len}"];

            target.SetData(slice);
        }

        /// <summary>
        ///     Write <paramref name="count"/> elements from <paramref name="src"/> into an
        ///     equally-strided run of <paramref name="dest"/>'s buffer through the IL
        ///     strided-store kernel (<see cref="Backends.Kernels.DirectILKernelGenerator.GetDiagWriteKernel"/>).
        ///     Shared by <see cref="np.fill_diagonal"/>, <see cref="np.diag"/>'s 1-D branch and
        ///     <see cref="np.diagflat"/> — all three reduce to the same primitive: scatter a 1-D
        ///     source along a constant-stride diagonal.
        /// </summary>
        /// <remarks>
        ///     Both endpoints are addressed by REAL element strides/offsets (never memory order),
        ///     so any destination layout — contiguous, transposed, sliced, F-order, negative-stride
        ///     — writes to the correct positions, exactly as the aliased-view + <c>SetData</c> path
        ///     did but without the ~5 μs NDIter setup. A <paramref name="srcStrideElems"/> of 0
        ///     broadcasts a single source element (the scalar fill). <paramref name="dest"/> and
        ///     <paramref name="src"/> MUST share the dtype (the callers guarantee it: diag/diagflat
        ///     preserve the source dtype; fill_diagonal casts its values first), so the store is a
        ///     same-width byte copy with no cast. Returns <c>false</c> — leaving the destination
        ///     untouched — when the kernel is unavailable, so callers fall back to <c>SetData</c>.
        /// </remarks>
        internal static unsafe bool TryDiagonalScatter(
            NDArray dest, long destOffsetElems, long destStrideElems,
            NDArray src, long srcOffsetElems, long srcStrideElems, long count)
        {
            if (count <= 0) return true;

            long eb = dest.dtypesize;
            int copyKind = Backends.Kernels.DirectILKernelGenerator.CopyKindFor(eb);
            if (copyKind == 0) return false; // every NumSharp dtype is 1/2/4/8/16 B; defensive only.

            var kernel = Backends.Kernels.DirectILKernelGenerator.GetDiagWriteKernel(copyKind);
            if (kernel == null) return false;

            byte* dstPtr = (byte*)dest.Storage.Address + destOffsetElems * eb;
            byte* srcPtr = (byte*)src.Storage.Address + srcOffsetElems * eb;
            kernel(dstPtr, destStrideElems * eb, srcPtr, srcStrideElems * eb, count);
            return true;
        }

        // ---------------------------- np.r_ / np.c_ ----------------------------

        /// <summary>
        ///     <c>array(a, ndmin=ndmin)</c>: pad with leading length-1 axes, as a view.
        /// </summary>
        /// <remarks>
        ///     Deliberately UNCAPPED. NumPy validates <c>ndmin</c> against <c>NPY_MAXDIMS</c> (64) and
        ///     raises <c>ndmin must be &lt;= ndmax (64)</c>; NumSharp has no 64-dimension ceiling
        ///     anywhere, so importing one here would make the DSL refuse ranks the rest of the
        ///     library accepts. <c>ndmin</c> does arrive from a user-typed directive string, so the
        ///     padding must be cheap rather than merely bounded: <see cref="np.AtLeastNdView"/> prepends
        ///     all the axes in one alias (O(ndim)), where the per-axis loop it replaced was quadratic
        ///     — 27.6 s at ndmin=100 000, unbounded at 2**31-1. An ndmin genuinely too large to
        ///     allocate a shape for surfaces as <see cref="OutOfMemoryException"/>, the same answer
        ///     any other oversized NumSharp allocation gives. A ndmin of 0 or less stays a no-op,
        ///     as it is upstream.
        /// </remarks>
        private static NDArray AtLeastNdmin(NDArray a, int ndmin) => np.AtLeastNdView(a, ndmin);

        /// <summary>
        ///     NumPy's <c>AxisConcatenator.__getitem__</c> for <c>np.r_</c> / <c>np.c_</c>, after the indexer's numpy.matrix
        ///     rule: a leading directive string may override the object's attributes, every other entry becomes an array (a
        ///     slice string its <c>arange</c> / <c>linspace</c>, a C# scalar a weak or strong 0-d value, anything else
        ///     <c>array(item, ndmin=ndmin)</c> with the <c>trans1d</c> permutation), the result dtype is
        ///     <c>result_type</c> of the entries, and the converted entries are concatenated along the axis.
        /// </summary>
        /// <param name="key">The index expression's entries (not null).</param>
        /// <param name="axis">The object's axis (<c>np.r_</c>: 0, <c>np.c_</c>: -1).</param>
        /// <param name="matrix">The object's matrix flag (false for both; a "r" / "c" directive sets it).</param>
        /// <param name="ndmin">The object's minimum rank of every entry (<c>np.r_</c>: 1, <c>np.c_</c>: 2).</param>
        /// <param name="trans1d">Where an upgraded entry's original axes start (<c>np.r_</c>: -1, <c>np.c_</c>: 0).</param>
        /// <returns>The concatenation; an <see cref="NDMatrix"/> under a "r" / "c" directive, C-contiguous and owning its data
        ///     as NumPy's <c>matrix(res)</c> copy is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or one of its entries is null.</exception>
        /// <exception cref="ValueError">A directive after the first entry ("special directives must be the first
        ///     entry."), or an unknown one ("unknown special directive").</exception>
        /// <exception cref="OverflowException">A C# integer literal outside the adopted integer dtype, in NumPy's setitem
        ///     texts with the LP64 C <c>long</c> ("Python integer 1000 out of bounds for int8"; "Python int too large to
        ///     convert to C long" past the long range, and past 2**64 - 1 for uint32 / uint64).</exception>
        /// <exception cref="DTypePromotionError">A C# number beside text (no common DType).</exception>
        /// <exception cref="ArgumentException">A malformed slice string, or np.concatenate's refusals (no entry, entries
        ///     of different ranks or shapes).</exception>
        /// <remarks>Scoped: the converted entries and the padded views are temporaries of this call. The parameters are the
        ///     object's fields, passed as copies, which a leading directive overrides for this call only.</remarks>
        [NDScoped]
        internal static NDArray AxisConcatenate(object[] key, int axis, bool matrix, int ndmin, int trans1d)
        {
            if (key is null)
                throw new ArgumentNullException(nameof(key));

            // The attributes arrive as copies (the object's fields): the first entry may override them.
            bool col = false;

            var objs = new List<ConcatOperand>(key.Length);

            for (int k = 0; k < key.Length; k++)
            {
                object item = key[k];
                switch (item)
                {
                    case null:
                        throw new ArgumentNullException($"key[{k}]",
                            "index expression entries must not be null.");

                    case string s:
                        if (s.IndexOf(':') < 0)
                        {
                            // No colon ⇒ a NumPy special directive, which must lead.
                            if (k != 0)
                                throw new ValueError("special directives must be the first entry.");
                            ApplyDirective(s, ref axis, ref ndmin, ref trans1d, ref matrix, ref col);
                            break;
                        }

                        foreach (var token in s.Split(','))
                        {
                            if (string.IsNullOrWhiteSpace(token))
                                continue;
                            objs.Add(FromSlice(np.AxisConcatenator.ParseSliceToken(token), ndmin, trans1d));
                        }

                        break;

                    case Slice slice:
                        objs.Add(FromSlice(np.AxisConcatenator.SliceSpec.FromSlice(slice), ndmin, trans1d));
                        break;

                    case Slice[] slices:
                        foreach (var one in slices)
                            objs.Add(FromSlice(np.AxisConcatenator.SliceSpec.FromSlice(one), ndmin, trans1d));
                        break;

                    // Weak (NEP50 "python literal") scalars.
                    case bool:
                        objs.Add(ConcatOperand.Scalar(item, ConcatWeakKind.Bool));
                        break;
                    case sbyte or byte or short or ushort or int or uint or long or ulong:
                        objs.Add(ConcatOperand.Scalar(item, ConcatWeakKind.Int));
                        break;
                    case float or double:
                        objs.Add(ConcatOperand.Scalar(item, ConcatWeakKind.Float));
                        break;
                    case Complex:
                        objs.Add(ConcatOperand.Scalar(item, ConcatWeakKind.Complex));
                        break;

                    // Strong scalars — NumSharp dtypes with no Python literal form.
                    case char or Half or decimal:
                        objs.Add(ConcatOperand.StrongScalar(item));
                        break;

                    // A C# number literal that reached the key as its implicit 0-d array (an NDArray-typed entry):
                    // NumPy's Python scalar (`type(item) in ScalarType` → result_type takes the scalar itself, WEAK),
                    // exactly as the boxed literal above — np.r_[int8Array, (NDArray)1] stays int8.
                    case NDArray lit when lit.WasPythonLiteral:
                        objs.Add(ConcatOperand.Scalar(lit.Storage.GetAtIndex(0),
                            lit.typecode == NPTypeCode.Complex ? ConcatWeakKind.Complex
                            : lit.typecode is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half ? ConcatWeakKind.Float
                            : ConcatWeakKind.Int));
                        break;

                    default:
                        objs.Add(FromArrayLike(item, ndmin, trans1d));
                        break;
                }
            }

            NDArray[] arrays;
            if (objs.Count != 0 && AnyText(objs))
            {
                // A string entry: NumPy's result_type over DESCRIPTORS (the widest text, S + U giving U, a number widening
                // it to its text width, a Python number refused with DTypePromotionError), every entry converted to it.
                DType descr = ResolveTextDescr(objs);
                arrays = new NDArray[objs.Count];
                for (int i = 0; i < objs.Count; i++)
                    arrays[i] = objs[i].MaterializeText(descr, ndmin);
            }
            else if (objs.Count != 0)
            {
                NPTypeCode final = ResolveDtype(objs);
                arrays = new NDArray[objs.Count];
                for (int i = 0; i < objs.Count; i++)
                    arrays[i] = objs[i].Materialize(final, ndmin);
            }
            else
            {
                // Directives only — drops through to concatenate's own error, as in NumPy.
                arrays = System.Array.Empty<NDArray>();
            }

            var res = np.concatenate(arrays, axis);

            if (matrix)
            {
                // NumPy: res = self.makemat(res), where makemat is the matrix CLASS — the result is an
                // NDMatrix, and for the 'c' directive the transpose is the matrix's own (.T of a row is a
                // column matrix). matrix() copies; the concatenation is fresh, so the matrix takes its
                // storage over instead of copying again (the scope reclaims the plain wrapper).
                // The copy is matrix.__new__'s `new.copy()`, ndarray.copy's C ORDER: a concatenation np.concatenate laid
                // out in F order (np.r_['r', f2d], every input F-contiguous) becomes a C-contiguous matrix in NumPy, so only
                // a result that is not C-contiguous already is copied; any other one is taken over as it is.
                int oldndim = res.ndim;
                if (!res.Shape.IsContiguous)
                    res = res.copy('C');
                NDMatrix mat = NDMatrix.From(res);
                res = oldndim == 1 && col ? mat.T : mat;
            }

            return res;
        }

        #region directives

        private static void ApplyDirective(string item, ref int axis, ref int ndmin,
            ref int trans1d, ref bool matrix, ref bool col)
        {
            if (item == "r" || item == "c")
            {
                matrix = true;
                col = item == "c";
                return;
            }

            if (item.IndexOf(',') >= 0)
            {
                // "axis,ndmin" or "axis,ndmin,trans1d". NumPy reads only vec[:2] and vec[2],
                // so a fourth field is silently ignored — including a non-numeric one.
                var vec = item.Split(',');
                if (!TryParseDirectiveInt(vec[0], out int a) || !TryParseDirectiveInt(vec[1], out int n))
                    throw new ValueError($"unknown special directive '{item}'");

                int t = trans1d;
                if (vec.Length == 3 && !TryParseDirectiveInt(vec[2], out t))
                    throw new ValueError($"unknown special directive '{item}'");

                axis = a;
                ndmin = n;
                trans1d = t;
                return;
            }

            if (TryParseDirectiveInt(item, out int only))
            {
                axis = only;
                return;
            }

            throw new ValueError("unknown special directive");
        }

        private static bool TryParseDirectiveInt(string s, out int value)
            => int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);

        #endregion

        #region entry conversion

        /// <summary>
        ///     NumPy's slice branch: <c>arange</c>/<c>linspace</c>, then <c>ndmin</c> padding and the
        ///     <c>swapaxes(-1, trans1d)</c> that differs from the array branch's permutation.
        /// </summary>
        private static ConcatOperand FromSlice(in np.AxisConcatenator.SliceSpec spec, int ndmin, int trans1d)
        {
            NDArray newobj = spec.Materialize();
            if (ndmin > 1)
            {
                newobj = AtLeastNdmin(newobj, ndmin);
                if (trans1d != -1)
                    newobj = np.swapaxes(newobj, -1, trans1d);
            }

            return ConcatOperand.Array(newobj);
        }

        /// <summary>
        ///     NumPy's array branch: <c>array(item, ndmin=ndmin)</c> followed, when the entry was
        ///     upgraded, by the <c>defaxes</c> permutation that places the original axes at
        ///     <c>trans1d</c>.
        /// </summary>
        private static ConcatOperand FromArrayLike(object item, int ndmin, int trans1d)
        {
            NDArray arr = item as NDArray ?? np.asanyarray(item);
            int itemNdim = arr.ndim;

            // NumPy: array(item, subok=True, ndmin=ndmin) cannot give a numpy.matrix another axis (probed 2.4.2:
            // np.r_['0,3', m, m] joins the two matrices as they are, a (4, 3) matrix).
            NDArray newobj = NDMatrixBoundary.IsMatrix(arr) ? arr : AtLeastNdmin(arr, ndmin);
            if (trans1d != -1 && itemNdim < ndmin)
            {
                int k2 = ndmin - itemNdim;
                int k1 = trans1d;
                if (k1 < 0)
                    k1 += k2 + 1;
                newobj = np.transpose(newobj, BuildTransposeAxes(ndmin, k1, k2));
            }

            return ConcatOperand.Array(newobj);
        }

        /// <summary>
        ///     <c>defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]</c> over <c>defaxes = range(ndmin)</c>,
        ///     with Python's forgiving list-slice clamping (an out-of-range <c>trans1d</c> yields a
        ///     malformed permutation and lands on transpose's own "axes don't match array").
        /// </summary>
        private static int[] BuildTransposeAxes(int ndmin, int k1, int k2)
        {
            var axes = new List<int>(ndmin);
            AppendRangeSlice(axes, ndmin, 0, k1);
            AppendRangeSlice(axes, ndmin, k2, ndmin);
            AppendRangeSlice(axes, ndmin, k1, k2);
            return axes.ToArray();
        }

        /// <summary>Appends <c>list(range(n))[start:stop]</c> with Python's index clamping.</summary>
        private static void AppendRangeSlice(List<int> into, int n, int start, int stop)
        {
            if (start < 0)
                start = Math.Max(0, start + n);
            else if (start > n)
                start = n;

            if (stop < 0)
                stop = Math.Max(0, stop + n);
            else if (stop > n)
                stop = n;

            for (int i = start; i < stop; i++)
                into.Add(i);
        }

        #endregion

        #region dtype resolution

        /// <summary>
        ///     NEP50 kinds an entry can contribute. Ordered by dominance so the winner is the maximum.
        /// </summary>
        private enum ConcatWeakKind : byte
        {
            Strong = 0,
            Bool = 1,
            Int = 2,
            Float = 3,
            Complex = 4
        }

        /// <summary>One entry of the index expression, kept unmaterialized until the dtype is known.</summary>
        [NDBorrowed] // an entry references the caller's array until the concatenation reads it
        private readonly struct ConcatOperand
        {
            private readonly NDArray _array;
            private readonly object _scalar;
            private readonly NPTypeCode _code;
            private readonly ConcatWeakKind _weak;

            private ConcatOperand(NDArray array, object scalar, NPTypeCode code, ConcatWeakKind weak)
            {
                _array = array;
                _scalar = scalar;
                _code = code;
                _weak = weak;
            }

            public static ConcatOperand Array(NDArray a) => new ConcatOperand(a, null, a.typecode, ConcatWeakKind.Strong);

            public static ConcatOperand Scalar(object value, ConcatWeakKind weak)
                => new ConcatOperand(null, value, NPTypeCode.Empty, weak);

            public static ConcatOperand StrongScalar(object value)
                => new ConcatOperand(null, value, value.GetType().GetTypeCode(), ConcatWeakKind.Strong);

            public bool IsStrong => _weak == ConcatWeakKind.Strong;
            public NPTypeCode Code => _code;
            public ConcatWeakKind Weak => _weak;

            /// <summary>True for a string array entry (fixed-width <c>S</c> / <c>U</c> or <c>StringDType</c>).</summary>
            public bool IsText => _array is not null && _array.typecode == NPTypeCode.String;

            /// <summary>
            ///     The entry as <see cref="DTypePromotion.ResultType"/> reads it — NumPy's <c>result_type_objs</c> entry: an
            ///     array (strong), or the scalar itself (a C# integer / float / complex literal is WEAK; a bool, char,
            ///     Half or decimal is a strong dtype).
            /// </summary>
            public object PromotionOperand => _array is not null ? _array : _scalar;

            /// <summary>
            ///     The entry converted to the text descriptor <paramref name="descr"/> (an array cast, a scalar formatted as
            ///     its <c>str()</c>), then given <paramref name="ndmin"/> dimensions.
            /// </summary>
            /// <param name="descr">The result's string descriptor.</param>
            /// <param name="ndmin">The minimum dimensionality of the directive.</param>
            /// <returns>The converted entry (a new array, or the entry's own array when it already is
            ///     <paramref name="descr"/>).</returns>
            /// <exception cref="UnicodeDecodeError">An <c>S</c> entry converted to a <c>U</c> result holds a byte at or
            ///     above 0x80 (NumPy's S → U decode).</exception>
            public NDArray MaterializeText(DType descr, int ndmin)
            {
                NDArray arr = _array is not null
                    ? (_array.dtype.Equals(descr) ? _array : _array.astype(descr))
                    : np.asanyarray(_scalar).astype(descr);
                return AtLeastNdmin(arr, ndmin);
            }

            /// <summary>
            ///     A weak integer literal too large for int64 — which lifts an all-literal key's
            ///     default from int64 to uint64, as NumPy's <c>result_type(2**64-1)</c> does.
            /// </summary>
            public bool ExceedsInt64 => _scalar is ulong u && u > long.MaxValue;

            public NDArray Materialize(NPTypeCode final, int ndmin)
            {
                NDArray arr;
                if (_array is not null)
                {
                    // A numpy.matrix entry keeps its two axes here too (FromArrayLike); astype gives a plain
                    // array, so the class is read before the cast.
                    bool keepsRank = NDMatrixBoundary.IsMatrix(_array);
                    arr = _array.typecode == final ? _array : _array.astype(final);
                    if (keepsRank)
                        return arr;
                }
                else if (_weak != ConcatWeakKind.Strong)
                {
                    // NumPy: array(item, dtype=final_dtype, ndmin=ndmin) of a Python scalar is PyArray_FromAny with the
                    // dtype requested, i.e. the dtype's setitem (PyArray_Pack): a Python int is range-checked, never
                    // wrapped, with the LP64 texts NumSharp pins (np.r_[int8_a, 1000] raises "Python integer 1000 out of
                    // bounds for int8", np.r_[int64_a, 2**64 - 1] "Python int too large to convert to C long", which Linux
                    // NumPy says for every integer dtype but uint32 / uint64), a float or a complex rounds through a double,
                    // a bool is an int. A NEW 0-d array of the result dtype.
                    arr = NDArray.AsStoreValue(_scalar, final);
                }
                else
                {
                    // A NumPy scalar (char, Half, decimal) is CAST like a 0-d array (PyArray_FromAny casts a NumPy
                    // scalar of another dtype).
                    arr = np.asanyarray(_scalar);
                    if (arr.typecode != final)
                        arr = arr.astype(final);
                }

                return AtLeastNdmin(arr, ndmin);
            }
        }

        /// <summary>
        ///     NumPy's <c>result_type(*result_type_objs)</c>: arrays and NumSharp-only scalars contribute
        ///     their dtype, C# literals contribute a weak kind that adopts it.
        /// </summary>
        private static NPTypeCode ResolveDtype(List<ConcatOperand> objs)
        {
            bool anyStrong = false;
            bool anyBeyondInt64 = false;
            NPTypeCode strong = NPTypeCode.Empty;
            ConcatWeakKind weak = ConcatWeakKind.Strong;

            foreach (var op in objs)
            {
                if (op.IsStrong)
                {
                    strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, op.Code) : op.Code;
                    anyStrong = true;
                }
                else
                {
                    if (op.Weak > weak)
                        weak = op.Weak;
                    anyBeyondInt64 |= op.ExceedsInt64;
                }
            }

            if (!anyStrong)
            {
                // All literals — NEP50 defaults. An integer literal past int64 lifts the default
                // to uint64 (NumPy: result_type(2**63) and result_type(2**64-1) are both uint64).
                return weak switch
                {
                    ConcatWeakKind.Bool => NPTypeCode.Boolean,
                    ConcatWeakKind.Int => anyBeyondInt64 ? NPTypeCode.UInt64 : NPTypeCode.Int64,
                    ConcatWeakKind.Float => NPTypeCode.Double,
                    _ => NPTypeCode.Complex
                };
            }

            return weak switch
            {
                // No literal, or a bool literal (which adopts anything).
                ConcatWeakKind.Strong or ConcatWeakKind.Bool => strong,
                // An int literal adopts anything but bool, which it lifts to the default int.
                ConcatWeakKind.Int => strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong,
                // A float literal keeps a float-kind dtype's width, else forces the default float.
                ConcatWeakKind.Float => NDExprTypeRules.IsFloatKind(strong)
                                  || strong == NPTypeCode.Decimal
                                  || strong == NPTypeCode.Complex
                    ? strong
                    : NPTypeCode.Double,
                // NumSharp has a single complex width, so complex64 collapses onto Complex.
                _ => NPTypeCode.Complex
            };
        }

        /// <summary>True when any entry is a string array (the result dtype is then a text descriptor).</summary>
        /// <param name="objs">The entries.</param>
        /// <returns>Whether an entry is text.</returns>
        private static bool AnyText(List<ConcatOperand> objs)
        {
            foreach (var op in objs)
                if (op.IsText)
                    return true;
            return false;
        }

        /// <summary>
        ///     NumPy's <c>final_dtype = result_type(*result_type_objs)</c> when an entry is text: arrays and NumSharp-only
        ///     scalars contribute their descriptor, a C# number literal is a weak Python scalar — which has no common DType
        ///     with text, so <c>np.r_[u, 1]</c> raises NumPy's <c>DTypePromotionError</c>; a strong number (a 0-d array, a
        ///     bool) widens the text to its <c>str()</c> width (<c>np.r_[u, np.int64(1)]</c> is <c>&lt;U21</c>,
        ///     <c>np.r_[u, True]</c> <c>&lt;U5</c>), and <c>StringDType</c> promotes only with <c>U</c>.
        /// </summary>
        /// <param name="objs">The entries.</param>
        /// <returns>The result descriptor.</returns>
        /// <exception cref="DTypePromotionError">No common DType (a Python number beside text; StringDType beside bytes or
        ///     a number), NumPy's verbatim text with the entries' DTypes in order.</exception>
        private static DType ResolveTextDescr(List<ConcatOperand> objs)
        {
            var operands = new object[objs.Count];
            for (int i = 0; i < objs.Count; i++)
                operands[i] = objs[i].PromotionOperand;
            return DTypePromotion.ResultType(operands);
        }

        #endregion

        // ---------------------------- np.mgrid / np.ogrid (nd_grid) ----------------------------

        /// <summary>
        ///     NumPy's <c>nd_grid.__getitem__</c> with <c>sparse=False</c>: a key of one slice is the bare 1-D array of
        ///     the except branch; a key of several computes every size, the shared dtype, the dense
        ///     <c>indices(size, typ)</c> grid, then <c>nn[k] = nn[k] * step + start</c> layer by layer.
        /// </summary>
        /// <param name="key">The indexer's key: slice strings (one may hold several, comma-separated) and
        ///     <see cref="Slice"/> objects.</param>
        /// <returns>The dense grid, of shape <c>(N, size_0, ..., size_{N-1})</c>; the 1-D array of one slice; an empty
        ///     int64 array for the empty key (NumPy's <c>indices([])</c>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or an entry is null.</exception>
        /// <exception cref="ArgumentException">An entry that is neither a slice string nor a <see cref="Slice"/>.</exception>
        /// <exception cref="TypeError">See <see cref="GridSingleSlice"/>.</exception>
        /// <exception cref="ValueError">A missing stop (<see cref="GridAxisSize"/>, <see cref="GridFillLine"/>); a NaN
        ///     size; a size past int64 (<c>Maximum allowed dimension exceeded</c>); a negative one (np.indices'
        ///     <c>negative dimensions are not allowed</c>); the allocation guard's.</exception>
        /// <exception cref="DivideByZeroException">A zero real step.</exception>
        /// <exception cref="OverflowException">An infinite size; a Python int NumPy cannot convert (see
        ///     <see cref="GridFillLine"/>).</exception>
        [NDScoped] // reclaims each rescale pass's line/reshape/broadcast temps and the grid[k] layer views
        internal static NDArray MGrid(object[] key)
        {
            // Shared nd_grid parsing (same grammar/errors as ogrid).
            var specs = ParseGridSpecs(key, "mgrid");
            int n = specs.Count;

            // A key of one slice takes NumPy's except branch: a bare 1-D array, NOT a stacked grid.
            if (n == 1)
                return GridSingleSlice(specs[0]);

            // Every size first, in key order: a zero step, a missing (real-step) stop or a NaN size is raised where
            // NumPy's size loop meets it. The sizes are ceil((stop-start)/step) (imaginary: the point count) and MAY be
            // negative - indices then raises "negative dimensions are not allowed", exactly as NumPy's mgrid does (its
            // ogrid twin instead gets an empty axis from arange).
            var sizes = new BigInteger[n];
            for (int k = 0; k < n; k++)
                sizes[k] = GridAxisSize(specs[k], k, "mgrid");
            NPTypeCode typ = GridMeshDtype(specs);

            // nn = indices(size, typ): NumPy's empty() converts every dimension before it checks any, so a size past
            // int64 is "Maximum allowed dimension exceeded" even beside a negative one. The empty key gives indices([]),
            // an empty int64 array.
            long[] dims = new long[n];
            for (int k = 0; k < n; k++)
            {
                if (!np.AxisConcatenator.SliceSpec.FitsInt64(sizes[k]))
                    throw new ValueError("Maximum allowed dimension exceeded");
                dims[k] = (long)sizes[k];
            }

            NDArray grid = np.indices(dims, typ);   // (n, size_0, ..., size_{n-1})

            // nn[k] = nn[k] * step + start, layer by layer. A layer holds its coordinate at every position, so the
            // statement is its axis's line broadcast across the grid: the line is computed once and copied in. A "pure"
            // axis (real step 1, start 0) is skipped: nn[k] * 1 + 0 leaves int64 coordinates as they are, and float64
            // ones too (a coordinate is >= 0, so even a -0.0 start adds to +0.0) - the common np.mgrid[0:a, 0:b] case is
            // the single indices fill.
            var fullShape = Shape.Adopt(dims);

            for (int k = 0; k < n; k++)
            {
                var spec = specs[k];
                if (spec.Start == 0d && spec.Step == 1d && !spec.ImaginaryStep)
                    continue;

                NDArray line = new NDArray(typ, dims[k], false);
                GridFillLine(line, spec, k, typ, "mgrid");

                long[] lineShape = new long[n];
                for (int d = 0; d < n; d++)
                    lineShape[d] = 1L;
                lineShape[k] = dims[k];

                np.copyto(grid[k], np.broadcast_to(line.reshape(Shape.Adopt(lineShape)), fullShape));
            }

            return grid;
        }

        /// <summary>
        ///     Parses the index expression shared by <see cref="np.ogrid"/> and <see cref="np.mgrid"/> into one
        ///     <see cref="np.AxisConcatenator.SliceSpec"/> per slice — NumPy's
        ///     <c>numpy.lib.index_tricks.nd_grid.__getitem__</c> argument handling. Slices are spelled as
        ///     colon-bearing strings (one may hold several, comma-separated) or <see cref="Slice"/> objects;
        ///     there are no directives.
        /// </summary>
        /// <param name="key">The indexer's key.</param>
        /// <param name="caller">The public function name, for error wording (<c>"ogrid"</c>/<c>"mgrid"</c>).</param>
        /// <returns>One slice per slice of the key, in order (none for the empty key).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or an entry is null.</exception>
        /// <exception cref="ArgumentException">An entry that is neither a slice string (a string with a colon) nor a
        ///     <see cref="Slice"/> / <see cref="Slice"/>[]; a malformed slice string.</exception>
        private static List<np.AxisConcatenator.SliceSpec> ParseGridSpecs(object[] key, string caller)
        {
            if (key is null)
                throw new ArgumentNullException(nameof(key));

            var specs = new List<np.AxisConcatenator.SliceSpec>(key.Length);

            for (int k = 0; k < key.Length; k++)
            {
                object item = key[k];
                switch (item)
                {
                    case null:
                        throw new ArgumentNullException($"key[{k}]",
                            $"{caller} index entries must not be null.");

                    case string s:
                        if (s.IndexOf(':') < 0)
                            throw new ArgumentException(
                                $"{caller} indices must be slice expressions such as \"0:5\" or \"0:1:5j\"; " +
                                $"got \"{s}\".", nameof(key));

                        foreach (var token in s.Split(','))
                        {
                            if (string.IsNullOrWhiteSpace(token))
                                continue;
                            specs.Add(np.AxisConcatenator.ParseSliceToken(token));
                        }

                        break;

                    case Slice slice:
                        specs.Add(np.AxisConcatenator.SliceSpec.FromSlice(slice));
                        break;

                    case Slice[] slices:
                        foreach (var one in slices)
                            specs.Add(np.AxisConcatenator.SliceSpec.FromSlice(one));
                        break;

                    default:
                        throw new ArgumentException(
                            $"{caller} indices must be slice expressions (colon-bearing strings like \"0:5\", " +
                            $"or Slice objects); got {item.GetType().Name}.", nameof(key));
                }
            }

            return specs;
        }

        /// <summary>
        ///     NumPy's <c>nd_grid.__getitem__</c> except branch, which a key of ONE slice takes (<c>len(key)</c> of a bare
        ///     slice is the TypeError that leads there): <c>arange(start, stop, step)</c> for a real step, and for an
        ///     imaginary step <c>arange(0, length, 1, dtype=typ) * step + start</c> with <c>length = int(abs(step))</c> and
        ///     <c>step = (stop - start) / float(length - 1)</c> unless <c>length == 1</c>.
        /// </summary>
        /// <param name="spec">The slice.</param>
        /// <returns>A fresh 1-D array: arange's (int64 over Python ints, else float64), or float64 points for an
        ///     imaginary step — <c>i * step + start</c>, NOT linspace (the last point is not forced onto the stop:
        ///     <c>np.mgrid[0.3:0.9:7j]</c> ends at 0.9000000000000001).</returns>
        /// <exception cref="TypeError">An imaginary step of other than one point without a stop (Python's
        ///     <c>unsupported operand type(s) for -: 'NoneType' and 'int'</c>, or <c>'float'</c> for a float start).</exception>
        /// <exception cref="ValueError">A NaN imaginary step (<c>cannot convert float NaN to integer</c>); a length past
        ///     npy_intp (<c>Maximum allowed size exceeded</c>); arange's own errors; the allocation guard's.</exception>
        /// <exception cref="OverflowException">An infinite imaginary step (<c>cannot convert float infinity to
        ///     integer</c>); a Python int past the float range (<c>int too large to convert to float</c>).</exception>
        /// <exception cref="DivideByZeroException">A zero real step (arange's Python division).</exception>
        private static NDArray GridSingleSlice(in np.AxisConcatenator.SliceSpec spec)
        {
            if (!spec.ImaginaryStep)
                return spec.Materialize();

            // step_float = abs(step); step = length = int(step_float): NaN and infinity are int()'s errors.
            BigInteger length = PythonInt.From(Math.Abs(spec.Step));
            // A Python int 1 when length == 1: arange(...) * 1 leaves the points as they are.
            double step = 1.0;
            if (!length.IsOne)
            {
                // (key.stop - start) / float(step - 1): Python's None - start is its TypeError.
                if (!spec.HasStop)
                    throw new TypeError(
                        $"unsupported operand type(s) for -: 'NoneType' and '{(spec.StartIsInt ? "int" : "float")}'");
                step = GridPyNumber.TrueDivide(
                    GridPyNumber.Subtract(GridPyNumber.StopOf(spec), GridPyNumber.StartOf(spec)),
                    GridPyNumber.Float(GridPyNumber.IntToFloat(length - 1)));
            }

            // typ = result_type(start, stop, step_float) is float64 (step_float is a Python float); arange(0, length, 1,
            // dtype=typ) refuses a length past npy_intp before allocating anything.
            if (!np.AxisConcatenator.SliceSpec.FitsInt64(length))
                throw new ValueError("Maximum allowed size exceeded");
            var points = new NDArray(NPTypeCode.Double, (long)length, false);
            // arange * step: the step is a float, nothing to convert; + start: a Python int start converts here, so an
            // int past the float range raises after the allocation, as NumPy's does.
            GridFillFloat(points, step, GridPyNumber.StartOf(spec).AsFloat());
            return points;
        }

        /// <summary>
        ///     The size statement of NumPy's <c>nd_grid.__getitem__</c> for one slice of a key of several:
        ///     <c>int(abs(step))</c> for an imaginary step, else <c>math.ceil((stop - start) / step)</c> in Python's
        ///     arithmetic over the slice's Python numbers.
        /// </summary>
        /// <param name="spec">The slice.</param>
        /// <param name="k">The slice's position in the key, for the missing-stop error.</param>
        /// <param name="caller">The public entry's name (<c>"mgrid"</c> / <c>"ogrid"</c>), for the missing-stop error.</param>
        /// <returns>The size, a Python int: negative for a range that runs against its step, possibly past int64.</returns>
        /// <exception cref="ValueError">A real step without a stop (NumSharp's text; NumPy's <c>None - start</c> TypeError
        ///     sends it to the except branch, whose <c>key.step</c> raises <c>AttributeError: 'tuple' object has no
        ///     attribute 'step'</c>); a NaN quotient or imaginary step (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="DivideByZeroException">A zero step: <c>division by zero</c> over Python ints, <c>float division
        ///     by zero</c> once a float is involved.</exception>
        /// <exception cref="OverflowException">An infinite quotient or imaginary step (<c>cannot convert float infinity to
        ///     integer</c>); a Python int past the float range met by float arithmetic (<c>int too large to convert to
        ///     float</c>) or an int quotient past it (<c>integer division result too large for a float</c>).</exception>
        private static BigInteger GridAxisSize(in np.AxisConcatenator.SliceSpec spec, int k, string caller)
        {
            // size.append(int(abs(step))): an imaginary step's magnitude is a Python float.
            if (spec.ImaginaryStep)
                return PythonInt.From(Math.Abs(spec.Step));
            if (!spec.HasStop)
                throw GridMissingStop(caller, k);
            // size.append(math.ceil((stop - start) / step)): math.ceil of a float, whose NaN and infinity are errors.
            double quotient = GridPyNumber.TrueDivide(
                GridPyNumber.Subtract(GridPyNumber.StopOf(spec), GridPyNumber.StartOf(spec)), GridPyNumber.StepOf(spec));
            if (double.IsNaN(quotient))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(quotient))
                throw new OverflowException("cannot convert float infinity to integer");
            return new BigInteger(Math.Ceiling(quotient));
        }

        /// <summary>
        ///     The ONE dtype a multi-slice mesh shares — NumPy's <c>result_type(*num_list)</c> over every slice's
        ///     start, stop and step (an imaginary step contributes its magnitude, a Python float; a missing stop is
        ///     None, float64): int64 when every field of every slice is a Python int, else float64.
        /// </summary>
        /// <param name="specs">The slices.</param>
        /// <returns><see cref="NPTypeCode.Int64"/> or <see cref="NPTypeCode.Double"/>.</returns>
        private static NPTypeCode GridMeshDtype(List<np.AxisConcatenator.SliceSpec> specs)
        {
            bool integral = true;
            foreach (var spec in specs)
                integral &= spec.Integral;

            return integral ? NPTypeCode.Int64 : NPTypeCode.Double;
        }

        /// <summary>
        ///     The rescale statement of NumPy's <c>nd_grid.__getitem__</c> for axis <paramref name="k"/>,
        ///     <c>nn[k] = nn[k] * step + start</c>, over the axis's line: <paramref name="line"/> holds the coordinates
        ///     <c>0 .. size-1</c> (NumPy's <c>arange(size)</c>, or a layer of <c>indices</c>, which holds its coordinate at
        ///     every position), and receives <c>i * step + start</c> in the mesh dtype.
        /// </summary>
        /// <param name="line">A fresh C-contiguous 1-D array of the mesh dtype, the axis's size long; written in place.</param>
        /// <param name="spec">The axis's slice.</param>
        /// <param name="k">The axis, for the missing-stop error.</param>
        /// <param name="typ">The mesh dtype, int64 or float64 (<see cref="GridMeshDtype"/>).</param>
        /// <param name="caller">The public entry's name, for the missing-stop error.</param>
        /// <exception cref="ValueError">An imaginary step of other than one point without a stop (NumSharp's text; NumPy's
        ///     <c>kk.stop - start</c> TypeError ends in its except branch's AttributeError).</exception>
        /// <exception cref="OverflowException">A Python int step or start past int64 in an int64 mesh (NumPy's
        ///     <c>Python int too large to convert to C long</c>, the LP64 text), or past the float range in a float64 mesh
        ///     (<c>int too large to convert to float</c>) — raised before any element, even of an empty axis.</exception>
        private static void GridFillLine(NDArray line, in np.AxisConcatenator.SliceSpec spec, int k, NPTypeCode typ,
            string caller)
        {
            if (typ == NPTypeCode.Int64)
            {
                // Every slice is ints with a real step (GridMeshDtype). nn[k] * step converts the step to int64 first, then
                // + start the start: NEP 50's Python-int conversion, whose failure does not depend on the axis's size.
                GridFillInt64(line, GridToInt64(spec.IntStep), GridToInt64(spec.IntStart));
                return;
            }

            double step;
            if (spec.ImaginaryStep)
            {
                // step = int(abs(step)); if step != 1: step = (kk.stop - start) / float(step - 1). The size statement
                // already converted abs(step), so int() cannot raise here.
                BigInteger points = PythonInt.From(Math.Abs(spec.Step));
                if (points.IsOne)
                    step = 1.0;
                else
                {
                    if (!spec.HasStop)
                        throw GridMissingStop(caller, k);
                    step = GridPyNumber.TrueDivide(
                        GridPyNumber.Subtract(GridPyNumber.StopOf(spec), GridPyNumber.StartOf(spec)),
                        GridPyNumber.Float(GridPyNumber.IntToFloat(points - 1)));
                }
            }
            else
                // nn[k] * step: a Python int step converts to float64 here (CPython's float(int), raising past the range).
                step = GridPyNumber.StepOf(spec).AsFloat();

            // + start: the start converts after the step, as NumPy's second ufunc call converts it.
            GridFillFloat(line, step, GridPyNumber.StartOf(spec).AsFloat());
        }

        /// <summary>A Python int as the int64 NumPy's NEP 50 conversion gives it in an int64 operation.</summary>
        /// <param name="value">The int.</param>
        /// <returns>The value.</returns>
        /// <exception cref="OverflowException">The int does not fit int64 (NumPy's <c>Python int too large to convert to C
        ///     long</c> on LP64 platforms; Windows NumPy says <c>int too big to convert</c> — NumSharp pins the LP64 text).</exception>
        private static long GridToInt64(BigInteger value)
        {
            if (!np.AxisConcatenator.SliceSpec.FitsInt64(value))
                throw new OverflowException("Python int too large to convert to C long");
            return (long)value;
        }

        /// <summary>
        ///     Writes <c>i * step + start</c> over an int64 line: NumPy's int64 multiply and add, which wrap.
        /// </summary>
        /// <param name="line">A fresh C-contiguous int64 line; element 0 is at its storage's base address.</param>
        /// <param name="step">The step.</param>
        /// <param name="start">The start.</param>
        [MethodImpl(Optimize)]
        private static unsafe void GridFillInt64(NDArray line, long step, long start)
        {
            long* p = (long*)line.Storage.Address;
            long n = line.size;
            for (long i = 0; i < n; i++)
                p[i] = unchecked(i * step + start);
        }

        /// <summary>
        ///     Writes <c>i * step + start</c> over a float64 line: NumPy's two float64 ufunc passes, each rounded (a
        ///     product, then a sum; never a fused multiply-add — .NET does not contract them), so a <c>-0.0</c> start
        ///     gives <c>+0.0</c> at <c>i = 0</c> as NumPy's <c>0.0 + -0.0</c> does.
        /// </summary>
        /// <param name="line">A fresh C-contiguous float64 line; element 0 is at its storage's base address.</param>
        /// <param name="step">The step.</param>
        /// <param name="start">The start.</param>
        [MethodImpl(Optimize)]
        private static unsafe void GridFillFloat(NDArray line, double step, double start)
        {
            double* p = (double*)line.Storage.Address;
            long n = line.size;
            for (long i = 0; i < n; i++)
                p[i] = (double)i * step + start;
        }

        /// <summary>
        ///     The error NumSharp raises where NumPy's <c>nd_grid.__getitem__</c> meets a missing stop in a key of several
        ///     slices: Python's <c>None - start</c> TypeError sends NumPy to its except branch, whose <c>key.step</c> on a
        ///     tuple raises <c>AttributeError: 'tuple' object has no attribute 'step'</c> (a documented divergence: the
        ///     ValueError says what is missing). Raised at the same point as NumPy's: a real step's in the size loop, an
        ///     imaginary step's (of other than one point) in the rescale loop.
        /// </summary>
        /// <param name="caller">The public entry's name.</param>
        /// <param name="k">The slice's position in the key.</param>
        /// <returns>The exception, for the caller to throw.</returns>
        private static ValueError GridMissingStop(string caller, int k) =>
            new ValueError(
                $"{caller} with more than one slice requires an explicit stop for slice {k}: 'start:stop' or " +
                "'start:stop:Nj' (NumPy raises AttributeError: 'tuple' object has no attribute 'step').");

        /// <summary>
        ///     A slice field as the Python number NumPy's <c>nd_grid</c> computes with: an int (exact, any size) or a float.
        ///     Only the arithmetic nd_grid's statements perform on fields is modelled, with CPython's rules and texts.
        /// </summary>
        private readonly struct GridPyNumber
        {
            /// <summary>Whether the number is a Python int (else a float).</summary>
            private readonly bool _isInt;

            /// <summary>The int's value.</summary>
            private readonly BigInteger _int;

            /// <summary>The float's value.</summary>
            private readonly double _float;

            /// <summary>Builds a number.</summary>
            /// <param name="isInt">Whether it is a Python int.</param>
            /// <param name="i">The int's value.</param>
            /// <param name="f">The float's value.</param>
            private GridPyNumber(bool isInt, BigInteger i, double f)
            {
                _isInt = isInt;
                _int = i;
                _float = f;
            }

            /// <summary>A Python float.</summary>
            /// <param name="f">The value.</param>
            /// <returns>The number.</returns>
            public static GridPyNumber Float(double f) => new GridPyNumber(false, BigInteger.Zero, f);

            /// <summary>The slice's start (0 when omitted).</summary>
            /// <param name="spec">The slice.</param>
            /// <returns>The start as its Python number.</returns>
            public static GridPyNumber StartOf(in np.AxisConcatenator.SliceSpec spec) =>
                new GridPyNumber(spec.StartIsInt, spec.IntStart, spec.Start);

            /// <summary>The slice's stop (callers check <see cref="np.AxisConcatenator.SliceSpec.HasStop"/> first).</summary>
            /// <param name="spec">The slice.</param>
            /// <returns>The stop as its Python number.</returns>
            public static GridPyNumber StopOf(in np.AxisConcatenator.SliceSpec spec) =>
                new GridPyNumber(spec.StopIsInt, spec.IntStop, spec.Stop);

            /// <summary>The slice's real step (1 when omitted).</summary>
            /// <param name="spec">The slice.</param>
            /// <returns>The step as its Python number.</returns>
            public static GridPyNumber StepOf(in np.AxisConcatenator.SliceSpec spec) =>
                new GridPyNumber(spec.StepIsInt, spec.IntStep, spec.Step);

            /// <summary>
            ///     CPython's <c>float(int)</c> inside float arithmetic (<c>PyLong_AsDouble</c>): the nearest double, ties to
            ///     even.
            /// </summary>
            /// <param name="value">The int.</param>
            /// <returns>The double.</returns>
            /// <exception cref="OverflowException">Past the double range (<c>int too large to convert to float</c>).</exception>
            public static double IntToFloat(BigInteger value)
            {
                double d = PythonInt.ToDouble(value);
                if (double.IsInfinity(d))
                    throw new OverflowException("int too large to convert to float");
                return d;
            }

            /// <summary>The number as a float: an int converted as CPython's float arithmetic converts it.</summary>
            /// <returns>The double.</returns>
            /// <exception cref="OverflowException">An int past the double range.</exception>
            public double AsFloat() => _isInt ? IntToFloat(_int) : _float;

            /// <summary>
            ///     Python's <c>a - b</c>: exact for two ints, else float subtraction with each int converted first (left,
            ///     then right) and the left NaN kept.
            /// </summary>
            /// <param name="a">The minuend.</param>
            /// <param name="b">The subtrahend.</param>
            /// <returns>The difference.</returns>
            /// <exception cref="OverflowException">An int operand past the double range beside a float.</exception>
            public static GridPyNumber Subtract(GridPyNumber a, GridPyNumber b)
            {
                if (a._isInt && b._isInt)
                    return new GridPyNumber(true, a._int - b._int, 0d);
                double left = a.AsFloat();
                double right = b.AsFloat();
                return Float(PyScalar.FloatSub(left, right));
            }

            /// <summary>
            ///     Python's <c>a / b</c>: the correctly rounded quotient of two ints, else float division with each int
            ///     converted first.
            /// </summary>
            /// <param name="a">The dividend.</param>
            /// <param name="b">The divisor.</param>
            /// <returns>The quotient, a Python float.</returns>
            /// <exception cref="DivideByZeroException">A zero divisor: <c>division by zero</c> for two ints, <c>float
            ///     division by zero</c> otherwise.</exception>
            /// <exception cref="OverflowException">An int quotient past the double range (<c>integer division result too
            ///     large for a float</c>), or an int operand past it beside a float.</exception>
            public static double TrueDivide(GridPyNumber a, GridPyNumber b)
            {
                if (a._isInt && b._isInt)
                    return PyScalar.IntTrueDivide(a._int, b._int);
                double left = a.AsFloat();
                double right = b.AsFloat();
                return PyScalar.FloatDiv(left, right);
            }
        }

        /// <summary>
        ///     NumPy's <c>nd_grid.__getitem__</c> with <c>sparse=True</c> (<c>np.ogrid[...]</c>): one slice is the bare 1-D
        ///     array of the except branch; a key of several computes every size, the shared dtype, one
        ///     <c>arange(size, dtype=typ)</c> per axis, then <c>nn[k] = nn[k] * step + start</c> axis by axis, and returns
        ///     each line as an open-mesh view <c>(1, …, size, …, 1)</c>.
        /// </summary>
        /// <param name="key">The indexer key: slices / slice strings.</param>
        /// <param name="caller">The public entry's name, for error texts.</param>
        /// <returns>The lines: a single bare 1-D array, or one open-mesh view per slice (none for the empty key); the caller
        ///     owns them.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or an entry is null.</exception>
        /// <exception cref="ArgumentException">An entry that is neither a slice string nor a <see cref="Slice"/>.</exception>
        /// <exception cref="TypeError">See <see cref="GridSingleSlice"/>.</exception>
        /// <exception cref="ValueError">A missing stop (<see cref="GridAxisSize"/>, <see cref="GridFillLine"/>); a NaN size;
        ///     a size past npy_intp (<c>Maximum allowed size exceeded</c>); the allocation guard's.</exception>
        /// <exception cref="DivideByZeroException">A zero real step.</exception>
        /// <exception cref="OverflowException">An infinite size; a Python int NumPy cannot convert (see
        ///     <see cref="GridFillLine"/>).</exception>
        // NDScoped: each line is a temp — only its open-mesh RESHAPE VIEW is returned, and the view holds its own
        // reference on the line's buffer. Without the scope every line's wrapper stranded one pooled buffer per slice
        // (measured: 2 for a 2-slice ogrid). The weaver yields every element of the returned array.
        [NDScoped]
        internal static NDArray[] OGrid(object[] key, string caller)
        {
            var specs = ParseGridSpecs(key, caller);

            int n = specs.Count;
            // A key of one slice takes NumPy's except branch: a bare 1-D array.
            if (n == 1)
                return new[] { GridSingleSlice(specs[0]) };

            // Every size first, in key order: a zero step, a missing (real-step) stop or a NaN size is raised where
            // NumPy's size loop meets it. The empty key has no slice: NumPy's empty tuple.
            var sizes = new BigInteger[n];
            for (int k = 0; k < n; k++)
                sizes[k] = GridAxisSize(specs[k], k, caller);
            NPTypeCode typ = GridMeshDtype(specs);

            // nn = [arange(size, dtype=typ) for ...], in key order: a size past npy_intp is arange's "Maximum allowed size
            // exceeded" (its length computation overflows), a negative size an empty axis.
            var lines = new NDArray[n];
            for (int k = 0; k < n; k++)
            {
                if (!np.AxisConcatenator.SliceSpec.FitsInt64(sizes[k]))
                    throw new ValueError("Maximum allowed size exceeded");
                lines[k] = new NDArray(typ, Math.Max((long)sizes[k], 0L), false);
            }

            // nn[k] = nn[k] * step + start, axis by axis (the conversions and the missing stops of imaginary steps raise
            // here, after every line exists, as in NumPy); then each line becomes its open-mesh view.
            var @out = new NDArray[n];
            for (int k = 0; k < n; k++)
            {
                GridFillLine(lines[k], specs[k], k, typ, caller);

                long[] shape = new long[n];
                for (int d = 0; d < n; d++)
                    shape[d] = 1L;
                shape[k] = lines[k].size;

                @out[k] = lines[k].reshape(Shape.Adopt(shape));
            }

            return @out;
        }
    }
}
