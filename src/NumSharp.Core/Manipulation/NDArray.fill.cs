using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Fill the array with a scalar value, IN PLACE (NumPy's <c>ndarray.fill</c>). Every element —
        ///     across whatever memory layout this array has (contiguous, F-order, sliced, transposed,
        ///     negative-stride) — is set to <paramref name="value"/> coerced to this array's dtype.
        ///
        ///     <para>
        ///     Coercion follows NumPy's scalar-assignment (NEP50 weak-scalar) rules, probed against
        ///     NumPy 2.4.2. A C# primitive is NumSharp's analog of a Python scalar (weak): assigned to an
        ///     INTEGER dtype it is range-checked — an out-of-bounds value RAISES
        ///     (<c>OverflowException</c> "Python integer 300 out of bounds for int8") rather than wrapping,
        ///     and a float source is TRUNCATED toward zero before the check (<c>3.9</c> stores 3, <c>300.0</c>
        ///     into int8 raises <c>OverflowException</c>; NaN raises <c>ValueError</c> "cannot convert float NaN
        ///     to integer"; ±inf raises <c>OverflowException</c> "cannot convert float infinity to integer"; a
        ///     complex source raises <c>TypeError</c>, exactly as NumPy's setitem runs <c>int()</c>/<c>float()</c>
        ///     on the value). Assigned to a float/complex dtype it casts, saturating to
        ///     ±inf on overflow (<c>float32.fill(1e300)</c> → inf). A 0-d <see cref="NDArray"/> is a STRONG
        ///     scalar and WRAPS on cast (matching NumPy's 0-d array) — except a C# literal's own 0-d array
        ///     (<see cref="NDArray.WasPythonLiteral"/>), which packs as the literal; a higher-rank array is a
        ///     sequence and raises <c>ValueError("setting an array element with a sequence.")</c>. A
        ///     <see cref="Half"/> or <c>char</c> is a NumPy scalar: range-checked into a SIGNED integer dtype,
        ///     cast into every other (<c>uint8.fill((Half)300)</c> is 44, as in NumPy). The whole conversion is
        ///     NumPy's <c>PyArray_Pack</c> (<see cref="ReductionInitial.TryPack"/>), shared with every store path.
        ///     </para>
        ///
        ///     NumPy checks writeability FIRST, then packs the scalar (which may raise) BEFORE touching any
        ///     element — so a read-only destination raises the read-only error even for a bad value, and an
        ///     out-of-range value raises even on an EMPTY array. Both orderings are reproduced.
        /// </summary>
        /// <param name="value">The scalar to fill with. A C# primitive (weak) or a 0-d <see cref="NDArray"/> (strong).</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (NumSharp house convention,
        ///     as in <see cref="np.fill_diagonal"/>; NumPy instead yields NaN for a float array / TypeError otherwise).</exception>
        /// <exception cref="ValueError">This array is read-only (broadcast view / read-only memmap);
        ///     raises NumPy's exact <c>ValueError: assignment destination is read-only</c>.</exception>
        /// <exception cref="OverflowException">A weak integer/float value is out of range for an integer dtype,
        ///     or a ±inf value is assigned to an integer dtype (NumPy's <c>OverflowError</c>).</exception>
        /// <exception cref="ValueError"><paramref name="value"/> is a multi-element array (a sequence), or a NaN
        ///     value is assigned to an integer dtype (NumPy's <c>ValueError</c>).</exception>
        /// <exception cref="TypeError">A complex <paramref name="value"/> is assigned to a real (non-bool integer
        ///     or float) dtype — NumPy funnels it through <c>int()</c>/<c>float()</c>, which reject a complex.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into a bytes (<c>S</c>) array.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into a str (<c>U</c>) array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html</remarks>
        public void fill(object value)
        {
            // NumPy (convert.c :: PyArray_FillWithScalar): writeability is checked FIRST, through the same
            // guard every other write path uses.
            NumSharpException.ThrowIfNotWriteable(Shape);
            // A StringDType array stores the value through its setitem once, then into every element (NDVStringMove.Fill).
            if (Storage.IsVString)
            {
                Backends.Strings.NDVStringMove.Fill(this, value);
                return;
            }
            // A fixed-width string array packs the value into ITS OWN descriptor (encoded, truncated to the element width;
            // the ASCII checks raise here, before any element is written and even when the array is empty) and copies that
            // one element everywhere.
            if (typecode == NPTypeCode.String)
            {
                FillText(value);
                return;
            }

            // Coerce the scalar to this array's dtype with NumPy scalar-coercion. This may raise
            // (OverflowException / ValueError) and — matching NumPy's PyArray_Pack — runs regardless of
            // size, so a bad value is rejected even on an empty array, BEFORE any element is written.
            object scalar = CoerceFillValue(value);

            if (size == 0)
                return;

            // A C- or F-contiguous (non-broadcast) array occupies exactly [offset, offset+size) contiguous
            // slots, so — order-independent for a constant — the whole window can be splatted with the
            // vectorized fill (reaches NumPy's fill speed at cache-resident sizes). A view whose INNERMOST
            // axis is still a contiguous run (a row slice m[::2], a row-strided/offset row view) is a stack
            // of such runs and takes the per-run splat below (NumPy's per-row memset speed). Every remaining
            // layout (transposed / non-unit or negative inner stride / broadcast) is written through its own
            // strides by broadcasting the 0-d scalar with the same NDIter engine np.copyto uses.
            if (Shape.IsContiguous || Shape.IsFContiguous)
                FillContiguousWindow(scalar);
            else if (!TryFillInnerContiguousRuns(scalar))
                NDIter.Copy(this, NDArray.Scalar(scalar));
        }

        /// <summary>
        ///     <see cref="fill"/> for a fixed-width string array: NumPy's <c>PyArray_Pack</c> of the value into this array's
        ///     descriptor, then the broadcast copy of that one element into every position (any layout).
        /// </summary>
        /// <param name="value">A <see cref="string"/>, a <c>byte[]</c>, a <see cref="char"/>, a number (stored as its
        ///     <c>str()</c>, truncated to the width), or a 0-d array.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (the house convention of
        ///     <see cref="fill"/>).</exception>
        /// <exception cref="ValueError">An array of one or more dimensions, of any size and dtype (<c>setting an array
        ///     element with a sequence</c> — WITHOUT the period NumPy's numeric targets put there: its string pack path
        ///     words it differently, probed 2.4.2).</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into a bytes (<c>S</c>) array.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into a str (<c>U</c>) array.</exception>
        private void FillText(object value)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            if (value is NDArray nd)
            {
                // A one-element 1-D array is a sequence too: PyArray_Pack takes only a 0-d value.
                if (nd.ndim != 0)
                    throw new ValueError("setting an array element with a sequence");
                // The string copy route converts the 0-d value (S <-> U with the ASCII checks, truncation) or refuses a
                // numeric one; it runs only when there is an element to write.
                if (size > 0)
                    FillTextFrom(nd.Storage);
                return;
            }
            using var scalar = Backends.Strings.NDStringArrays.FromScalar(value, dtype);
            if (size > 0)
                FillTextFrom(scalar.Storage);
        }

        /// <summary>
        ///     Writes one 0-d string value into every element of this string array. A C- or F-contiguous array (a constant
        ///     fill does not care about order) converts the value into its FIRST element only — the string copy route's
        ///     conversion and checks — and then replicates that element's bytes across the window by doubling block moves,
        ///     each source a cache-resident prefix of the window; any other layout broadcasts the value through the copy
        ///     route.
        /// </summary>
        /// <param name="value">A 0-d string storage (converted to this array's descriptor here).</param>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into a bytes (<c>S</c>) array.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into a str (<c>U</c>) array.</exception>
        private unsafe void FillTextFrom(UnmanagedStorage value)
        {
            var shape = Shape;
            if (!(shape.IsContiguous || shape.IsFContiguous) || shape.IsBroadcasted)
            {
                NDIter.Copy(Storage, value);
                return;
            }

            long itemsize = Storage.StringSlice.ItemSize;
            byte* window = Storage.Address + shape.offset * itemsize;
            // The first element, as a 0-d view: the copy route converts the value into it (or raises before any write).
            var first = Storage.Alias(new Shape(System.Array.Empty<long>(), System.Array.Empty<long>(), shape.offset, shape.bufferSize));
            NDIter.Copy(first, value);

            // Doubling block moves from the start of the window; past ~64 KB the source stays a fixed, cache-resident
            // prefix (a whole number of elements, so the pattern stays aligned) and only the writes stream.
            long total = shape.size * itemsize;
            long cap = Math.Max(itemsize, (65536 / itemsize) * itemsize);
            long filled = itemsize;
            while (filled < total)
            {
                long chunk = Math.Min(Math.Min(filled, cap), total - filled);
                Buffer.MemoryCopy(window, window + filled, chunk, chunk);
                filled += chunk;
            }
        }

        /// <summary>
        ///     Splat <paramref name="scalar"/> (already this array's exact dtype) across the contiguous
        ///     logical window [offset, offset+size) via the vectorized <see cref="UnmanagedSpan{T}"/> fill.
        /// </summary>
        private void FillContiguousWindow(object scalar)
        {
            IArraySlice window = Storage.InternalArray;
            long n = size;
            // A contiguous VIEW may sit at [offset, offset+size) inside a larger buffer (np.split children,
            // a row of a matrix); narrow to exactly the logical window before splatting (zero-copy, writes
            // through) — mirrors UnmanagedStorage.GetData<T>()'s own narrowing.
            if (Shape.offset != 0 || window.Count != n)
                window = window.Slice(Shape.offset, n);
            GetWindowFiller(dtype.type)(window, scalar);
        }

        // Dtype-agnostic dispatch to the generic SIMD fill: one cached delegate per dtype (the reflection-
        // cache pattern), each closing over FillWindowGeneric<T> so the loop lives in the shared vectorized
        // UnmanagedSpan.Fill — not a hand-written per-dtype switch.
        private static readonly ConcurrentDictionary<Type, Action<IArraySlice, object>> _windowFillers
            = new ConcurrentDictionary<Type, Action<IArraySlice, object>>();

        private static Action<IArraySlice, object> GetWindowFiller(Type dtype)
            => _windowFillers.GetOrAdd(dtype, static t => (Action<IArraySlice, object>)typeof(NDArray)
                .GetMethod(nameof(FillWindowGeneric), BindingFlags.NonPublic | BindingFlags.Static)
                .MakeGenericMethod(t)
                .CreateDelegate(typeof(Action<IArraySlice, object>)));

        private static void FillWindowGeneric<T>(IArraySlice window, object value) where T : unmanaged
            => window.AsSpan<T>().Fill((T)value);

        /// <summary>
        ///     Fast path for a NON-contiguous view whose INNERMOST axis is still a unit-stride contiguous run
        ///     (length &gt; 1) — a row slice <c>m[::2]</c>, a row-strided or offset row view — which is a stack
        ///     of contiguous runs. Each run is splatted with the same vectorized <see cref="UnmanagedSpan{T}"/>
        ///     fill the contiguous window uses (reaching NumPy's per-row memset speed), walking the outer axes
        ///     as an odometer so the inner run never touches the element-scatter iterator. Works at ANY rank
        ///     (NumSharp has no ndim cap; the odometer heap-allocates past the stack budget). Returns
        ///     <see langword="false"/> — so the caller falls back to <see cref="NDIter.Copy(NDArray, NDArray)"/>
        ///     — for any layout without a real contiguous inner run (broadcast, non-unit or negative inner
        ///     stride, or a length-≤1 inner axis).
        /// </summary>
        private bool TryFillInnerContiguousRuns(object scalar)
        {
            Shape shape = Shape;
            int ndim = shape.NDim;
            if (ndim < 1 || shape.IsBroadcasted)
                return false;                                  // 0-d handled above; stride-0 dims -> NDIter
            long[] strides = shape.strides;
            long[] dims = shape.dimensions;
            if (strides[ndim - 1] != 1 || dims[ndim - 1] <= 1)
                return false;                                  // innermost must be a unit-stride run of len > 1
            GetInnerRunFiller(dtype.type)(Storage.InternalArray, shape, scalar);
            return true;
        }

        // Cached per-dtype delegate for the inner-run fill (reflection-cache pattern, like GetWindowFiller):
        // each closes over FillInnerRunsGeneric<T> so the run splat lives in the shared vectorized span fill.
        private static readonly ConcurrentDictionary<Type, Action<IArraySlice, Shape, object>> _innerRunFillers
            = new ConcurrentDictionary<Type, Action<IArraySlice, Shape, object>>();

        private static Action<IArraySlice, Shape, object> GetInnerRunFiller(Type dtype)
            => _innerRunFillers.GetOrAdd(dtype, static t => (Action<IArraySlice, Shape, object>)typeof(NDArray)
                .GetMethod(nameof(FillInnerRunsGeneric), BindingFlags.NonPublic | BindingFlags.Static)
                .MakeGenericMethod(t)
                .CreateDelegate(typeof(Action<IArraySlice, Shape, object>)));

        // Splat `value` across each contiguous inner run of `shape` (innermost axis unit-stride, length > 1),
        // advancing over the outer axes as an odometer. Runs directly on the whole-buffer span via a struct
        // Slice (no per-run allocation); the run splat is the SIMD UnmanagedSpan<T>.Fill.
        private static void FillInnerRunsGeneric<T>(IArraySlice buffer, Shape shape, object value) where T : unmanaged
        {
            var span = buffer.AsSpan<T>();                     // whole backing buffer; indexed from shape.offset
            long[] dims = shape.dimensions;
            long[] strides = shape.strides;
            int outerNd = shape.NDim - 1;                      // axes above the contiguous inner run
            long innerLen = dims[outerNd];
            T v = (T)value;

            // Odometer over the outer axes — supports ANY rank (NumSharp has no ndim cap): stackalloc for the
            // common shallow ranks, heap only for a pathologically deep one (the FiniteScan.cs pattern).
            Span<long> coord = outerNd <= 64 ? stackalloc long[outerNd] : new long[outerNd];
            coord.Clear();
            long outerCount = 1;
            for (int i = 0; i < outerNd; i++) outerCount *= dims[i];

            for (long o = 0; o < outerCount; o++)
            {
                long off = shape.offset;
                for (int i = 0; i < outerNd; i++) off += coord[i] * strides[i];
                span.Slice(off, innerLen).Fill(v);
                for (int i = outerNd - 1; i >= 0; i--)         // advance odometer, innermost outer axis fastest
                {
                    if (++coord[i] < dims[i]) break;
                    coord[i] = 0;
                }
            }
        }

        /// <summary>
        ///     Coerces <paramref name="value"/> to this array's dtype the way NumPy's <c>PyArray_FillWithScalar</c> does — one
        ///     <c>PyArray_Pack</c> of the value (<see cref="ReductionInitial.TryPack"/>) — and returns it boxed as the
        ///     dtype's EXACT C# type, the form the typed fill routes consume.
        /// </summary>
        /// <param name="value">The fill value.</param>
        /// <returns>The element every slot receives.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (house convention, see <see cref="fill"/>).</exception>
        /// <exception cref="OverflowException">A Python int (any C# integer, or a C# literal's 0-d array) out of an integer
        ///     dtype's range — "Python integer 300 out of bounds for int8", or past the C <c>long</c> the LP64 text "Python int
        ///     too large to convert to C long" —; a float infinity into an integer ("cannot convert float infinity to
        ///     integer"); a NumPy scalar (<see cref="Half"/>, <c>char</c>) whose int() does not fit a SIGNED integer dtype.</exception>
        /// <exception cref="ValueError">A float NaN into an integer dtype ("cannot convert float NaN to integer"), or an array
        ///     with dimensions ("setting an array element with a sequence.").</exception>
        /// <exception cref="TypeError">A complex into a real dtype (Python's int() / float() texts), or a sequence into an
        ///     integer or complex dtype.</exception>
        /// <remarks>
        ///     <para>
        ///     The C# boundary map decides the kind of each value, so every rule is NumPy's for that kind: an integer
        ///     primitive / <see cref="BigInteger"/> is a Python int (range-checked, never wrapped), <c>float</c> /
        ///     <c>double</c> a Python float (truncated toward zero into an integer, then range-checked; rounded ONCE from the
        ///     double into float16 / float32, so a <c>long</c> past 2^53 rounds twice into float32, as in NumPy), a
        ///     <see cref="Complex"/> a Python complex, <see cref="Half"/> / <c>char</c> NumPy scalars (int()-converted with
        ///     the range check into a SIGNED integer dtype, cast into every other: <c>uint8.fill((Half)300)</c> is 44), a
        ///     bool dtype takes the value's truth, a 0-d <see cref="NDArray"/> is CAST (wrapping, like NumPy's 0-d array)
        ///     unless it is a C# literal's (<see cref="NDArray.WasPythonLiteral"/>), which packs as the literal.
        ///     </para>
        ///     <para>
        ///     A value of the dtype's own C# type is already the stored element (the hot <c>int32.fill(42)</c> /
        ///     <c>float64.fill(3.5)</c>): it returns as is, with no packing and no allocation.
        ///     </para>
        /// </remarks>
        private unsafe object CoerceFillValue(object value)
        {
            if (value is null)
                // NumPy reaches its dtype converter (float → NaN, else TypeError); NumSharp treats a null
                // scalar as a caller bug, matching np.fill_diagonal's deliberate divergence.
                throw new ArgumentNullException(nameof(value), "fill() value must not be null.");

            // Fast path: a number already of the exact dtype IS the stored representation — it always fits
            // and needs no coercion (the hot `int32.fill(42)` / `float64.fill(3.5)` case, no allocation). Only a
            // number qualifies: an array or a list is not an element, whatever its element type.
            if (IsStoreScalar(value) && value.GetType().GetTypeCode() == typecode)
                return value;

            NPTypeCode tc = typecode;
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.TryPack(value, tc, new Span<byte>(slot, ReductionInitial.SlotBytes));
            return ReductionInitial.Box(new ReadOnlySpan<byte>(slot, ReductionInitial.SlotBytes), tc);
        }
    }
}
