using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// NDPolyVander.cs — numpy.polynomial's pseudo-Vandermonde matrices ({p}vander, {p}vander2d, {p}vander3d; plan U5)
// =============================================================================
//
// NumPy 2.4.2 (numpy/polynomial/*.py, polyutils._vander_nd / _vander_nd_flat), statement by statement — each statement
// is also where its error comes from, so the order below IS NumPy's error order:
//
// {p}vander(x, deg)
//   1. ideg = pu._as_int(deg, "deg")               TypeError "deg must be an integer, received {deg}" (PolyIndexArgument)
//   2. if ideg < 0: raise                          ValueError "deg must be non-negative"
//   3. x = np.array(x, copy=None, ndmin=1)         a ragged list's ValueError; a str / None / object array is refused
//        + 0.0                                     (NumSharp has no str / object dtype — NumPy fails at this `+ 0.0`)
//   4. v = np.empty((ideg + 1,) + x.shape)         "Maximum allowed dimension exceeded" (ideg + 1 past npy_intp), "array
//                                                  is too big; ..." (AllocationGuard), MemoryError (the allocator)
//   5. the recurrence (the IL kernel, DirectILKernelGenerator.PolyVander.cs); 6. np.moveaxis(v, 0, -1) — a VIEW
//      (OWNDATA false): the points' C-order strides, then the degree axis with stride npts.
//
// {p}vander2d / {p}vander3d (x, y[, z], deg) = _vander_nd_flat:
//   1. len(deg)                                    TypeError "object of type 'int' has no len()" / "len() of unsized object"
//      != n_dims                                   ValueError "Expected {n} dimensions of degrees, got {len}"
//   2. np.asarray((x, y[, z])) + 0.0               ONE array of the points' promoted dtype — ragged shapes raise np.array's
//                                                  inhomogeneous ValueError (no broadcasting); a str / object stack is refused
//   3. per dimension k, in order (functools.reduce over a generator): {p}vander(points[k], deg[k]) — its steps 1, 2 and 4
//      — then, from k = 1 on, the outer product of the dimensions so far (its own np.empty-like allocation)
//   4. reshape(points.shape + (-1,))               for NO points: IncorrectShapeException/ValueError "cannot reshape array
//                                                  of size 0 into shape (0,newaxis)"; otherwise a VIEW of the last product
//                                                  (rows r = (a*(dy+1) + b)[*(dz+1) + c] over the points in C order).
// The allocations of step 3 are reproduced as NumPy makes them — AllocationGuard's check, and for an array past the pool's
// size a real map/unmap probe (nothing written) — so a MemoryError surfaces BEFORE a later degree's TypeError exactly as
// in NumPy, without the kernel ever materializing the per-dimension matrices (they live in its per-block scratch).
//
// DTYPES. T = (x + 0.0).dtype: float16/float32/float64/complex128 kept, bool / every integer / char -> float64, decimal kept.
// For 2-D / 3-D the stacking dtype S is np.promote_types over the points (array coercion's strong promotion: (float16,
// int8) is float16, (float32, int32) float64) and T = S + 0.0. Each point is loaded straight from its own dtype into T:
// the two-step NumPy conversion (P_k -> S exactly for integer promotion, S -> T rounding once; or P_k -> S = T) rounds
// at most once, the same value, so no stacked copy is made when every point is an NDArray of the same shape.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> Vandermonde family (see the file header): one shared implementation behind the six basis
    ///     facades; the basis is data (<see cref="PolyBasis"/>).
    /// </summary>
    internal static unsafe class NDPolyVander
    {
        /// <summary>NumPy's text for a dimension past <c>npy_intp</c> (np.empty's shape conversion of <c>ideg + 1</c>).</summary>
        private const string MaxDimension = "Maximum allowed dimension exceeded";

        /// <summary>Scratch a 1-D block aims for: its slots plus the three rows a step touches stay in L1 (48 KB).</summary>
        private const long Block1DBytes = 48 * 1024;

        /// <summary>Scratch a 2-D / 3-D block aims for: every dimension's matrix rows for the block, L2-resident.</summary>
        private const long BlockNdBytes = 192 * 1024;

        /// <summary>The widest block: past it the per-step overhead is already negligible.</summary>
        private const long MaxBlock = 4096;

        /// <summary>
        ///     TEST HOOK: when positive, every kernel call on this thread uses blocks of at most this many points instead of
        ///     the cache-sized default — so the unit tests can drive the block loop, its partial last block and the
        ///     per-block scratch reuse with small inputs (the 2-D / 3-D budget otherwise needs megabytes of output to span
        ///     two blocks). Thread-static: a test sets and resets it around its own calls; production code never touches it.
        /// </summary>
        [ThreadStatic] internal static long BlockOverride;

        /// <summary>
        ///     The 2-D / 3-D result size from which the product stage streams its output rows with non-temporal stores
        ///     (<see cref="PolyVanderKey.NonTemporal"/>). Past a last-level cache the result is evicted to memory before any
        ///     consumer reads it, so keeping it cached buys nothing, while a normal store reads every line before writing it:
        ///     a 96.8 MB <c>(100000, 121)</c> float64 result took 17.9 ms with normal stores and 13.4 ms streamed (the fresh
        ///     pages' demand-zero faults are ~10.4 ms of both). Below it the result may still be cached for the caller's next
        ///     step (a least-squares fit reads it at once), so normal stores stay.
        /// </summary>
        internal const long NonTemporalMinBytes = 32L * 1024 * 1024;

        /// <summary>
        ///     TEST HOOK: 1 forces the non-temporal product stores on, -1 forces them off, 0 (the default) decides by
        ///     <see cref="NonTemporalMinBytes"/> — so the unit tests can prove the streamed kernel writes the same bytes as the
        ///     normal one (every row's scalar head, vector body and tail) without allocating 32 MB per case. Thread-static.
        /// </summary>
        [ThreadStatic] internal static int NonTemporalOverride;

        // ---------------------------------------------------------------------------------------------
        //  Entry points
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>{p}vander(x, deg)</c>: the pseudo-Vandermonde matrix of the basis — <c>V[..., i]</c> is the i-th basis polynomial
        ///     at the points, by NumPy's forward recurrence.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">The points: an <see cref="NDArray"/> (any layout; a 0-d one is one point), a typed C# array, a
        ///     Python list / tuple (np.array's coercion) or a scalar.</param>
        /// <param name="deg">The degree: anything <c>operator.index</c> accepts (<see cref="PolyIndexArgument.AsInt"/>).</param>
        /// <returns>A view of shape <c>x.shape + (deg + 1,)</c> (<c>(1, deg + 1)</c> for a scalar), of <c>(x + 0.0).dtype</c>,
        ///     over a new array — NumPy's <c>np.moveaxis</c> result, OWNDATA false.</returns>
        /// <exception cref="TypeError">A non-integer degree (NumPy's <c>deg must be an integer, received …</c>).</exception>
        /// <exception cref="ValueError"><c>deg must be non-negative</c>; a ragged list x; <c>Maximum allowed dimension
        ///     exceeded</c> (deg ≥ 2^63 - 1) or <c>array is too big; …</c> (the matrix's byte count).</exception>
        /// <exception cref="OutOfMemoryException">The matrix cannot be allocated (NumPy's MemoryError).</exception>
        /// <exception cref="NotSupportedException">A str / null x, or a list holding a str / None / a Python int past uint64 —
        ///     NumPy builds a str or object array (and its <c>+ 0.0</c> raises or computes with Python objects).</exception>
        internal static NDArray Vander(PolyBasis basis, object x, object deg)
        {
            // 1-2. ideg = pu._as_int(deg, "deg"); if ideg < 0: raise ValueError(...)
            BigInteger ideg = PolyIndexArgument.AsInt(deg, "deg");
            if (ideg.Sign < 0)
                throw new ValueError("deg must be non-negative");

            NDArray converted = null, temp = null;
            try
            {
                // 3. x = np.array(x, copy=None, ndmin=1) + 0.0 — the + 0.0 happens in the kernel's load stage.
                NDArray xa = Points(x, ref converted);
                NPTypeCode t = LoopType(xa.typecode);
                long[] pdims = PointDims(xa.Shape);
                long npts = Count(pdims);

                // 4. v = np.empty((ideg + 1,) + x.shape, dtype=x.dtype): NumPy's dimension conversion, then the allocation
                //    (AllocationGuard's "array is too big", the allocator's MemoryError).
                long rows = Rows(ideg);
                var buf = new NDArray(t, new Shape(Prepend(rows, pdims)), false);

                // 5. The recurrence, straight into the matrix's rows (none for no points: NumPy's loop over empty rows).
                if (npts > 0)
                {
                    byte** srcs = stackalloc byte*[1];
                    long* strides = stackalloc long[1];
                    long* nrows = stackalloc long[1];
                    SourceOf(xa, pdims, out srcs[0], out strides[0], ref temp);
                    nrows[0] = rows;
                    Run(basis, t, 1, srcs, strides, stackalloc[] { xa.typecode }, nrows, buf, npts);
                }

                // 6. np.moveaxis(v, 0, -1): the degree axis last, a view.
                return DegreeAxisLast(buf, pdims, rows);
            }
            finally
            {
                temp?.Dispose();
                converted?.Dispose();
            }
        }

        /// <summary>
        ///     <c>{p}vander2d(x, y, deg)</c> / <c>{p}vander3d(x, y, z, deg)</c> — <c>polyutils._vander_nd_flat</c>: the outer product
        ///     of the per-axis matrices, its degree axes flattened, <c>(a*(dy+1) + b)[*(dz+1) + c]</c> being the column of
        ///     <c>x^a y^b [z^c]</c> (in the basis).
        /// </summary>
        /// <param name="basis">The basis (every axis).</param>
        /// <param name="points">The 2 or 3 point arrays (anything <c>np.asarray</c> stacks; all of one shape).</param>
        /// <param name="deg">The degrees: a sequence of 2 / 3 integers (a list / tuple, a typed C# array, an NDArray, a str of
        ///     that length); each item as <see cref="Vander"/>'s <c>deg</c>.</param>
        /// <returns>A view of shape <c>points.shape + (prod(deg + 1),)</c> over a new array (OWNDATA false), of the stacked
        ///     points' <c>(S + 0.0).dtype</c>.</returns>
        /// <exception cref="TypeError"><c>object of type '…' has no len()</c> / <c>len() of unsized object</c> (a scalar or 0-d
        ///     deg), or a non-integer degree (<c>deg must be an integer, received …</c>).</exception>
        /// <exception cref="ValueError"><c>Expected {n} dimensions of degrees, got {len}</c>; points of different shapes
        ///     (np.array's inhomogeneous text); <c>deg must be non-negative</c>; <c>Maximum allowed dimension exceeded</c>;
        ///     <c>array is too big; …</c>.</exception>
        /// <exception cref="IncorrectShapeException">No points: NumPy's reshape ValueError text (<c>cannot reshape array of size
        ///     0 into shape (0,newaxis)</c>).</exception>
        /// <exception cref="OutOfMemoryException">A matrix NumPy allocates on the way cannot be allocated (MemoryError).</exception>
        /// <exception cref="NotSupportedException">Points NumPy stacks into a str or object array.</exception>
        internal static NDArray VanderNd(PolyBasis basis, object[] points, object deg)
        {
            int n = points.Length;
            NDArray degArray = null, stacked = null;
            var views = new List<NDArray>(3);
            var temps = new NDArray[n];
            try
            {
                // 1. len(degrees) and its items (Python's len(), then degrees[k] when dimension k runs).
                object[] degItems = DegreeItems(deg, n, ref degArray, views);

                // 2. points = tuple(np.asarray(tuple(points)) + 0.0): one stacked array of the promoted dtype S. Points that are
                //    all NDArrays of one shape are read in place (the stack is only a copy); anything else is stacked by
                //    np.array's coercion, which also raises the ragged ValueError / refuses a str or object stack.
                NPTypeCode s;
                long[] pdims;
                byte** srcs = stackalloc byte*[3];
                long* strides = stackalloc long[3];
                Span<NPTypeCode> srcTypes = stackalloc NPTypeCode[3];
                if (SameShapeArrays(points, out s))
                {
                    var first = (NDArray)points[0];
                    pdims = PointDims(first.Shape);
                    if (Count(pdims) > 0)
                    {
                        for (int k = 0; k < n; k++)
                        {
                            var p = (NDArray)points[k];
                            SourceOf(p, pdims, out srcs[k], out strides[k], ref temps[k]);
                            srcTypes[k] = p.typecode;
                        }
                    }
                }
                else
                {
                    stacked = PolySequence.ToArrayOrNonNumeric(PolySequence.MakeTuple(points, n), out var nonNumeric);
                    if (stacked is null)
                        throw nonNumeric.Refusal ?? new NotSupportedException(
                            "the points stack into a str array (NumPy's `+ 0.0` then raises), a dtype NumSharp does not have");
                    s = stacked.typecode;
                    var sd = stacked.Shape.dimensions;
                    pdims = sd.Length == 1 ? new long[] { 1 } : sd[1..];
                    long each = Count(pdims);
                    int ss = stacked.dtypesize;
                    for (int k = 0; k < n; k++)
                    {
                        // Row k of the C-contiguous stack: dimension k's points, contiguous.
                        srcs[k] = Ptr(stacked) + k * each * ss;
                        strides[k] = ss;
                        srcTypes[k] = s;
                    }
                }
                NPTypeCode t = LoopType(s);
                long npts = Count(pdims);

                // 3. Dimension by dimension, NumPy's reduce: the degree checks and the dimension's own matrix allocation, then
                //    (from the second dimension on) the outer product's allocation.
                long* nrows = stackalloc long[3];
                for (int k = 0; k < n; k++)
                {
                    BigInteger ideg = PolyIndexArgument.AsInt(degItems[k], "deg");
                    if (ideg.Sign < 0)
                        throw new ValueError("deg must be non-negative");
                    nrows[k] = Rows(ideg);
                    Reserve(Prepend(nrows[k], pdims), t);
                    if (k >= 1)
                    {
                        var productDims = new long[pdims.Length + k + 1];
                        pdims.CopyTo(productDims, 0);
                        for (int q = 0; q <= k; q++) productDims[pdims.Length + q] = nrows[q];
                        // The last product is the result's own array, allocated for real below; an earlier one (3-D's first)
                        // is a NumPy temporary whose allocation is reproduced here.
                        if (k < n - 1) Reserve(productDims, t);
                        else AllocationGuard.CheckDimensions(productDims, DirectILKernelGenerator.GetTypeSize(t));
                    }
                }

                // 4. reshape(points.shape + (-1,)): an empty stack has nothing to infer the flattened length from.
                if (npts == 0)
                {
                    var requested = new long[pdims.Length + 1];
                    pdims.CopyTo(requested, 0);
                    requested[^1] = -1;
                    throw new IncorrectShapeException($"cannot reshape array of size 0 into shape {Shape.ConvertShapeToString(requested)}");
                }
                long r = 1;
                for (int k = 0; k < n; k++) r *= nrows[k];   // fits: the product's allocation check passed
                var buf = new NDArray(t, new Shape(r, npts), false);
                Run(basis, t, n, srcs, strides, srcTypes, nrows, buf, npts);
                return DegreeAxisLast(buf, pdims, r);
            }
            finally
            {
                foreach (var tmp in temps) tmp?.Dispose();
                foreach (var v in views) v.Dispose();
                stacked?.Dispose();
                degArray?.Dispose();
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  The kernel call
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Runs the Vandermonde kernel of <paramref name="basis"/> into <paramref name="buf"/>: sizes the point block, takes the
        ///     per-block scratch from the house pool (slots, and for 2-D / 3-D every dimension's matrix rows plus the product row)
        ///     and hands every dimension's source to the kernel.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="t">The compute dtype.</param>
        /// <param name="n">Dimensions (1, 2 or 3).</param>
        /// <param name="srcs">Per dimension, the first point (C order).</param>
        /// <param name="strides">Per dimension, the byte stride between points.</param>
        /// <param name="srcTypes">Per dimension, the points' dtype.</param>
        /// <param name="nrows">Per dimension, <c>deg + 1</c>.</param>
        /// <param name="buf">The C-contiguous result buffer, <c>(rows, npts)</c>.</param>
        /// <param name="npts">Points (≥ 1).</param>
        [SkipLocalsInit]
        private static void Run(PolyBasis basis, NPTypeCode t, int n, byte** srcs, long* strides, ReadOnlySpan<NPTypeCode> srcTypes,
            long* nrows, NDArray buf, long npts)
        {
            int size = DirectILKernelGenerator.GetTypeSize(t);
            int slots = PolyVanderRoutines.Slots(basis);
            // A 2-D / 3-D result too large to stay cached streams its rows past the cache (the 1-D recurrence reads its
            // output rows back, so it never does). The override is the unit tests' switch; production leaves it 0.
            bool nonTemporal = n > 1 && (NonTemporalOverride != 0
                ? NonTemporalOverride > 0
                : buf.size * size >= NonTemporalMinBytes);
            var kernel = DirectILKernelGenerator.GetPolyVanderKernel(new PolyVanderKey(basis, t, n, srcTypes[0],
                n > 1 ? srcTypes[1] : NPTypeCode.Empty, n > 2 ? srcTypes[2] : NPTypeCode.Empty, nonTemporal));

            // Scratch rows per block: the slots (1-D — its matrix rows are the result's own), or every dimension's slots and
            // matrix rows plus 3-D's product row.
            long scratchRows = n == 1 ? slots : (n == 3 ? 1 : 0);
            if (n > 1)
                for (int k = 0; k < n; k++) scratchRows += slots + nrows[k];
            long block = n == 1 ? Block(npts, size, slots + 3, Block1DBytes) : Block(npts, size, scratchRows, BlockNdBytes);
            long rowBytes = block * size;
            long scratchBytes = scratchRows * rowBytes;

            IntPtr scratch = SizeBucketedBufferPool.Take(scratchBytes);
            try
            {
                byte** areas = stackalloc byte*[4];
                byte* cur = (byte*)scratch;
                for (int k = 0; k < n; k++)
                {
                    areas[k] = cur;
                    cur += (n == 1 ? slots : slots + nrows[k]) * rowBytes;
                }
                areas[3] = n == 3 ? cur : null;
                kernel(srcs, strides, Ptr(buf), npts * size, npts, nrows, block, areas);
            }
            finally
            {
                SizeBucketedBufferPool.Return(scratch, scratchBytes);
            }
        }

        /// <summary>
        ///     Points per kernel block: enough that <paramref name="rowsPerBlock"/> block-wide rows of the dtype fill
        ///     <paramref name="budget"/> bytes, at least one, at most <see cref="MaxBlock"/>, a multiple of 16 when it can be
        ///     (whole vectors for every lane width), and no more than there are points — or, when the test hook
        ///     <see cref="BlockOverride"/> is set on this thread, that width (capped by the points).
        /// </summary>
        /// <param name="npts">Points (≥ 1).</param><param name="size">Element size.</param>
        /// <param name="rowsPerBlock">Block-wide rows the budget must hold (≥ 1).</param><param name="budget">Bytes.</param>
        /// <returns>The block (≥ 1).</returns>
        private static long Block(long npts, int size, long rowsPerBlock, long budget)
        {
            if (BlockOverride > 0)
                return Math.Min(BlockOverride, npts);
            // A per-row byte count past the budget (a degree in the tens of thousands) still gets one point per block.
            long perPoint = rowsPerBlock > budget / size ? budget + 1 : rowsPerBlock * size;
            long b = Math.Clamp(budget / perPoint, 1, MaxBlock);
            if (b >= 16) b &= ~15L;
            return Math.Min(b, npts);
        }

        // ---------------------------------------------------------------------------------------------
        //  Arguments
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.array(x, copy=None, ndmin=1)</c> without the copy (the kernel only reads it): an NDArray is used as is (a 0-d
        ///     one reads as one point), anything else converted by the house mapping (<see cref="NDPolySeries.AsCoefficientArray"/>:
        ///     a typed C# array is an ndarray, a list / tuple goes through np.array's coercion, a scalar is a one-element array
        ///     of its discovered dtype).
        /// </summary>
        /// <param name="x">The points.</param>
        /// <param name="converted">Receives the array built here (the caller disposes it), untouched for an NDArray.</param>
        /// <returns>The points array.</returns>
        /// <exception cref="ValueError">A ragged list / tuple.</exception>
        /// <exception cref="NotSupportedException">A str / null x, or a list NumPy makes a str / object array of.</exception>
        private static NDArray Points(object x, ref NDArray converted)
        {
            if (x is NDArray nd)
                return nd;
            var v = NDPolySeries.AsCoefficientArray(x);
            // NumPy fails at `x + 0.0` on a str array (UFuncTypeError) and computes with Python objects on an object array:
            // NumSharp has neither dtype, so both are refused here — where NumPy would compute with it.
            if (v.NonNumeric)
                throw v.Refusal ?? new NotSupportedException(
                    "a str x makes NumPy build a str array (its `+ 0.0` then raises), a dtype NumSharp does not have");
            converted = v.Source;
            return v.Source;
        }

        /// <summary>
        ///     Python's <c>len(degrees)</c>, the count check, and the items <c>degrees[k]</c> (k &lt; n) NumPy reads: a str yields
        ///     one-character strs, an ndarray (an NDArray, a typed C# array) its NumPy-scalar elements / sub-arrays as views, a
        ///     list / tuple its items.
        /// </summary>
        /// <param name="deg">The degrees argument.</param>
        /// <param name="n">The number of dimensions.</param>
        /// <param name="converted">Receives an ndarray built from a typed C# array (the caller disposes it).</param>
        /// <param name="views">Receives the element views taken of an ndarray (the caller disposes them).</param>
        /// <returns>The n items.</returns>
        /// <exception cref="TypeError"><c>object of type '…' has no len()</c> (null, a scalar), <c>len() of unsized object</c>
        ///     (a 0-d array).</exception>
        /// <exception cref="ValueError"><c>Expected {n} dimensions of degrees, got {len}</c>.</exception>
        private static object[] DegreeItems(object deg, int n, ref NDArray converted, List<NDArray> views)
        {
            NDArray arr = null;
            object[] seq = null;
            string str = null;
            long len;
            switch (deg)
            {
                case null:
                    throw new TypeError("object of type 'NoneType' has no len()");
                case string s:
                    str = s;
                    len = s.Length;
                    break;
                case NDArray nd:
                    arr = nd;
                    len = LenOf(nd);
                    break;
                default:
                    if (PolySequence.IsArrayLike(deg))
                    {
                        arr = converted = np.asanyarray(deg);
                        len = LenOf(arr);
                    }
                    else if (PolySequence.IsSequence(deg))
                    {
                        seq = PolySequence.Items(deg);
                        len = seq.Length;
                    }
                    else
                        throw new TypeError($"object of type '{NDPolySeries.PythonTypeName(deg)}' has no len()");
                    break;
            }
            if (len != n)
                throw new ValueError($"Expected {n} dimensions of degrees, got {len}");

            var items = new object[n];
            for (int k = 0; k < n; k++)
            {
                if (str is not null)
                    items[k] = str[k].ToString();   // a str's items are one-character strs
                else if (seq is not null)
                    items[k] = seq[k];
                else
                {
                    // An ndarray's item: a NumPy scalar (1-D — a 0-d view keeps its dtype, so np.True_ / np.float64 are refused
                    // by _as_int as in NumPy) or a sub-array (N-D).
                    var item = arr[k];
                    views.Add(item);
                    items[k] = item;
                }
            }
            return items;

            static long LenOf(NDArray a)
            {
                if (a.ndim == 0)
                    throw new TypeError("len() of unsized object");
                return a.Shape.dimensions[0];
            }
        }

        /// <summary>
        ///     Whether every point is an <see cref="NDArray"/> of one shape — then np.asarray's stack is a pure copy, skipped — and
        ///     the dtype that stack would have (np.promote_types over the points, array coercion's strong promotion).
        /// </summary>
        /// <param name="points">The points.</param>
        /// <param name="s">The stacking dtype when the result is true.</param>
        /// <returns>True for same-shape NDArrays.</returns>
        private static bool SameShapeArrays(object[] points, out NPTypeCode s)
        {
            s = NPTypeCode.Empty;
            if (points[0] is not NDArray first)
                return false;
            var dims = first.Shape.dimensions;
            s = first.typecode;
            for (int k = 1; k < points.Length; k++)
            {
                if (points[k] is not NDArray p || p.ndim != first.ndim)
                    return false;
                var pd = p.Shape.dimensions;
                for (int d = 0; d < dims.Length; d++)
                    if (pd[d] != dims[d])
                        return false;
                s = PolyTyping.Promote(s, p.typecode);
            }
            return true;
        }

        /// <summary>
        ///     Where the kernel reads a points array: the first point and ONE byte stride between consecutive points in C order —
        ///     every 0-d / 1-D array (any stride, reversed, broadcast) and every N-D array whose axes merge in C order; anything
        ///     else is first copied C-contiguous by the house iterator (its own dtype, so the kernel still converts once).
        /// </summary>
        /// <param name="a">The points (at least one).</param>
        /// <param name="pdims">The points' dims (<c>ndmin=1</c> applied).</param>
        /// <param name="p">The first point.</param>
        /// <param name="strideBytes">The byte stride between points.</param>
        /// <param name="temp">Receives the contiguous copy when one is made (the caller disposes it).</param>
        private static void SourceOf(NDArray a, long[] pdims, out byte* p, out long strideBytes, ref NDArray temp)
        {
            int size = a.dtypesize;
            p = Ptr(a);
            strideBytes = size;
            if (a.ndim == 0)
                return;
            var dims = a.Shape.dimensions;
            var st = a.Shape.strides;
            long col = 1, expect = 0;
            bool have = false, flat = true;
            for (int d = dims.Length - 1; d >= 0 && flat; d--)
            {
                if (dims[d] == 1) continue;
                if (!have)
                {
                    col = st[d];
                    have = true;
                }
                else if (st[d] != expect)
                    flat = false;
                expect = st[d] * dims[d];
            }
            if (flat)
            {
                strideBytes = col * size;
                return;
            }
            temp = new NDArray(a.typecode, new Shape(pdims), false);
            NDIter.Copy(temp, a);
            p = Ptr(temp);
            strideBytes = size;
        }

        // ---------------------------------------------------------------------------------------------
        //  Shapes and allocations
        // ---------------------------------------------------------------------------------------------

        /// <summary><c>(x + 0.0).dtype</c> of points of dtype <paramref name="s"/>: bool / integers / char become float64, every
        ///     inexact dtype (and NumSharp's decimal) is kept.</summary>
        /// <param name="s">The points' dtype.</param>
        /// <returns>The compute dtype.</returns>
        private static NPTypeCode LoopType(NPTypeCode s) => PolyTyping.IsIntLike(s) ? NPTypeCode.Double : s;

        /// <summary>The points' dims after <c>ndmin=1</c>: a 0-d array is one point.</summary>
        /// <param name="s">The points' shape.</param><returns>The dims (a copy).</returns>
        private static long[] PointDims(in Shape s) => s.NDim == 0 ? new long[] { 1 } : (long[])s.dimensions.Clone();

        /// <summary>The product of <paramref name="dims"/> (0 when any is 0).</summary>
        /// <param name="dims">The dims.</param><returns>The count.</returns>
        private static long Count(long[] dims)
        {
            long c = 1;
            foreach (long d in dims) c *= d;
            return c;
        }

        /// <summary><c>(first,) + rest</c>.</summary>
        /// <param name="first">The leading dim.</param><param name="rest">The other dims.</param><returns>The dims.</returns>
        private static long[] Prepend(long first, long[] rest)
        {
            var d = new long[rest.Length + 1];
            d[0] = first;
            rest.CopyTo(d, 1);
            return d;
        }

        /// <summary>
        ///     <c>ideg + 1</c> as NumPy's shape conversion reads it: a Python int past <c>npy_intp</c> is <c>Maximum allowed
        ///     dimension exceeded</c>.
        /// </summary>
        /// <param name="ideg">The degree (≥ 0).</param>
        /// <returns>The row count.</returns>
        /// <exception cref="ValueError">The degree is 2^63 - 1 or more.</exception>
        private static long Rows(BigInteger ideg)
        {
            if (ideg >= long.MaxValue)
                throw new ValueError(MaxDimension);
            return (long)ideg + 1;
        }

        /// <summary>
        ///     An array NumPy allocates on the way (a dimension's own matrix, 3-D's first outer product) that NumSharp's kernel
        ///     never materializes: AllocationGuard's check (<c>array is too big; …</c>), and — for an array past the pool's size,
        ///     where the OS can refuse it — a real map/unmap probe of the same size (nothing is written), so NumPy's MemoryError
        ///     surfaces at the same statement.
        /// </summary>
        /// <param name="dims">The array's dims.</param>
        /// <param name="t">Its dtype.</param>
        /// <exception cref="ValueError"><c>array is too big; …</c>.</exception>
        /// <exception cref="OutOfMemoryException">The allocation fails (NumPy's MemoryError).</exception>
        private static void Reserve(long[] dims, NPTypeCode t)
        {
            int size = DirectILKernelGenerator.GetTypeSize(t);
            AllocationGuard.CheckDimensions(dims, size);
            if (Count(dims) * size > SizeBucketedBufferPool.MaxPoolableBytes)
            {
                var probe = new NDArray(t, new Shape(dims), false);
                probe.Dispose();
            }
        }

        /// <summary>
        ///     The result view over a C-contiguous <c>(degLen, npts)</c> buffer: the points' axes first (C-order strides), the
        ///     degree axis last with stride <c>npts</c> — NumPy's <c>np.moveaxis(v, 0, -1)</c> (1-D) and its reshape of the
        ///     outer product (2-D / 3-D). OWNDATA is false.
        /// </summary>
        /// <param name="buf">The buffer.</param><param name="pdims">The points' dims.</param>
        /// <param name="degLen">The degree axis' length.</param>
        /// <returns>The view.</returns>
        private static NDArray DegreeAxisLast(NDArray buf, long[] pdims, long degLen)
        {
            int nd = pdims.Length;
            var dims = new long[nd + 1];
            var strides = new long[nd + 1];
            long acc = 1;
            for (int d = nd - 1; d >= 0; d--)
            {
                dims[d] = pdims[d];
                strides[d] = acc;
                acc *= Math.Max(pdims[d], 1);   // a zero dim counts as 1, as the buffer's C strides do (no element is read)
            }
            dims[nd] = degLen;
            strides[nd] = acc;
            return NDPolyCalc.View(buf, dims, strides, buf.Shape.offset);
        }

        /// <summary>The address of an array's element 0.</summary>
        /// <param name="a">The array.</param><returns>The pointer.</returns>
        private static byte* Ptr(NDArray a) => (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
    }
}
