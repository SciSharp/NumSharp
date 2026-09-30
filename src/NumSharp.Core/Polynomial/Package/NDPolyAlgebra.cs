using System;
using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// NDPolyAlgebra.cs — numpy.polynomial's series algebra (plan U2): the engine
// =============================================================================
//
// WHAT NUMPY DOES
// ---------------
// {p}mulx / {p}mul / {p}div / {p}pow / {p}fromroots and the basis conversions X2poly / poly2X are Python loops over
// SHORT 1-D series (NumPy 2.4.2 numpy/polynomial/*.py, polyutils._div / _pow / _fromroots). Every statement of those
// loops is one of: polyutils.as_series (trim + np.common_type + copy), _add / _sub, a {p}mulx, an ARRAY op with a
// Python int or NumPy scalar (`c[-i] * xs`, `(c1 * (nd - 1)) / nd`, `c1[i:j] -= c2 * c1[j]`), np.convolve, trimseq,
// or scalarmath on one element (`q = rem[-1] / p[-1]`). NumPy pays ~0.5-1 µs of interpreter and allocation cost per
// statement; legmul of two 50-term series runs ~2 ms, legdiv 50/16 ~9 ms.
//
// HOW IT RUNS HERE
// ----------------
// The Python is replayed statement for statement, in NumPy's order and with NumPy's dtypes, over an ARENA of raw
// series (PolySer: pointer + length + dtype), and each statement runs through the kernel NumPy's own statement runs
// through in NumSharp:
//   * array ops                 the house ufunc kernels (GetPolyHouseBinaryKernel: simd_cmul / Smith / HALF loop);
//   * as_series / _add / _sub   the U1 substrate (TrimLength, CopyInto, CombineInto — the same code {p}add runs);
//   * {p}mulx                   an integral-layout calculus kernel (PolyCalcKey.Mulx); chebmulx of two or more terms —
//                               NumPy's three ARRAY statements, not a loop — one fused pass (GetPolyChebMulxKernel);
//   * np.convolve               the sliding-dot engine np.convolve runs (NDArray.SlidingCorrelateInto), the byte-parity
//                               OpenBLAS route included;
//   * scalarmath                the U1 one-element kernels (naive complex product, NumPy's scalar division).
// So the only C# here is NumPy's own Python — its loops are the ones over SERIES, never over elements — and the
// element work is the house's, which is why every result is NumPy's to the bit (the convolution bases' long real and
// all complex products excepted without the backend: np.convolve itself is BLAS-bound there, see NDArray.SlidingDot).
//
// DTYPES CARRIED PER SERIES
// -------------------------
// A series carries its dtype because NumPy's changes mid-algorithm: legmul's `len(c) == 1` branch binds the Python
// int 0, which as_series makes a float64 [0.] — so a float16/float32 product with a one-term factor is float64; _div's
// divisor products are mul_f([0]*i + [1], c2), an int list promoted with c2 (float64 again), so the remainder of a
// float32 legdiv is float64 while its quotient is float32; poly2X starts from res = 0 (float64).
//
// THE ARENA
// ---------
// Intermediates live in a per-thread bump arena (PolyArena): a series costs a pointer bump, not an NDArray (~200 ns
// each, the §9 port's whole cost). Its blocks come from — and growth blocks return to — the house buffer pool, so a long
// product's megabytes of temporaries are warm memory on the next call instead of fresh page faults. Every internal function allocates freely and returns its result through Keep, which
// moves the result down to the function's entry mark and frees everything else it allocated — so a call's memory is
// bounded by its live series, not by how many statements ran. Only the final result becomes an NDArray, with NumPy's
// ownership: a trimseq slice is returned as a VIEW of the full result array (flags.owndata false), exactly as NumPy
// returns ret[:k].
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     One series of the series-algebra engine: contiguous raw memory (the arena, or an as_series copy there), its
    ///     length and dtype, and what NumPy would RETURN for it — the array itself, or a slice of a longer array.
    /// </summary>
    internal readonly unsafe struct PolySer
    {
        /// <summary>Element 0 (also element 0 of the underlying NumPy array when <see cref="Slice"/>).</summary>
        public readonly byte* P;
        /// <summary>Length.</summary>
        public readonly long N;
        /// <summary>Dtype.</summary>
        public readonly NPTypeCode T;
        /// <summary>Bytes per element of <see cref="T"/>.</summary>
        public readonly int Size;
        /// <summary>Length of the NumPy array this series is (<see cref="N"/>) or is a leading slice of (a trimseq
        ///     result <c>seq[:k]</c>, or polydiv's <c>c1[:j+1]</c>).</summary>
        public readonly long Full;
        /// <summary>NumPy returns this series as a VIEW (<c>arr[:N]</c> of the <see cref="Full"/>-element array) — true even
        ///     when N == Full (trimseq's all-zero <c>seq[:1]</c>).</summary>
        public readonly bool Slice;

        /// <summary>A whole array: <paramref name="n"/> elements of <paramref name="t"/> at <paramref name="p"/>.</summary>
        /// <param name="p">Element 0.</param><param name="n">Length.</param><param name="t">Dtype.</param>
        public PolySer(byte* p, long n, NPTypeCode t) : this(p, n, t, n, false) { }

        /// <summary>A series with explicit return semantics.</summary>
        /// <param name="p">Element 0.</param><param name="n">Length.</param><param name="t">Dtype.</param>
        /// <param name="full">Length of the underlying array (≥ n).</param><param name="slice">NumPy returns a view of it.</param>
        public PolySer(byte* p, long n, NPTypeCode t, long full, bool slice)
        {
            P = p; N = n; T = t; Size = DirectILKernelGenerator.GetTypeSize(t); Full = full; Slice = slice;
        }

        /// <summary>Address of element <paramref name="i"/> (a NumPy scalar <c>s[i]</c>, read in place).</summary>
        /// <param name="i">The index (0 ≤ i &lt; N).</param>
        /// <returns>The address.</returns>
        public byte* At(long i) => P + i * Size;

        /// <summary>NumPy's <c>s[:k]</c>: a view of the same underlying array.</summary>
        /// <param name="k">Elements kept (≤ N).</param>
        /// <returns>The slice.</returns>
        public PolySer Prefix(long k) => new PolySer(P, k, T, Full, true);

        /// <summary>NumPy's <c>s[start:start+n]</c> as an OPERAND (never returned): the elements, as a whole array.</summary>
        /// <param name="start">First element.</param><param name="n">Length.</param>
        /// <returns>The sub-series.</returns>
        public PolySer Sub(long start, long n) => new PolySer(P + start * Size, n, T);

        /// <summary>The series as a substrate view (unit stride), for the U1 kernels.</summary>
        public PolySeriesView View => PolySeriesView.Raw(P, N, T);

        /// <summary>Bytes the underlying array occupies (<see cref="Full"/> elements).</summary>
        public long FullBytes => Full * Size;
    }

    /// <summary>
    ///     The per-thread bump allocator of the series-algebra engine (see the file header): a stack of native blocks,
    ///     positions compared as (block, offset). Memory is never freed piecemeal — <see cref="Release"/> moves the top
    ///     back — and the growth blocks go back to the house buffer pool (<see cref="SizeBucketedBufferPool"/>) when the
    ///     outermost <see cref="Enter"/> exits; the base block (64 KB) stays with the thread until the arena is finalized.
    /// </summary>
    /// <remarks>
    ///     Why the pool and not the OS: a product of long series needs megabytes of temporaries (chebmul of a 100000-term
    ///     series: a 200000-term z-series and its convolution), and memory fresh from the OS is committed page by page on
    ///     first touch — measured ~150 µs of page faults per 800 KB, per call, more than the convolution itself. Growth
    ///     blocks are sized in powers of two so repeated calls of similar sizes land on the same pool bucket and get warm,
    ///     already-committed memory back; the pool's own per-bucket caps bound what stays resident.
    /// </remarks>
    internal sealed unsafe class PolyArena
    {
        [ThreadStatic] private static PolyArena t_arena;

        /// <summary>Bytes of the base block every thread keeps.</summary>
        private const long BaseBytes = 64 * 1024;

        /// <summary>Allocation granularity: a cache line, so no two series share one and every vector load is aligned.</summary>
        private const long Align = 64;

        private IntPtr[] _block = new IntPtr[16];   // block bases (IntPtr: an array of pointers cannot be resized generically)
        private long[] _cap = new long[16];
        private int _count;   // blocks allocated
        private int _cur;     // block the top is in
        private long _top;    // bytes used in the current block
        private int _depth;   // Enter/Exit nesting

        /// <summary>A position of the arena's top: everything allocated after it is freed by <see cref="Release"/>.</summary>
        internal readonly struct Mark
        {
            /// <summary>The block index.</summary>
            public readonly int Block;
            /// <summary>Bytes used in that block.</summary>
            public readonly long Top;

            /// <summary>A position.</summary>
            /// <param name="block">Block index.</param><param name="top">Offset in the block.</param>
            public Mark(int block, long top) { Block = block; Top = top; }

            /// <summary>Whether this position is at or after <paramref name="other"/> in allocation order.</summary>
            /// <param name="other">The other position.</param>
            /// <returns>True when not before it.</returns>
            public bool AtOrAfter(Mark other) => Block > other.Block || (Block == other.Block && Top >= other.Top);
        }

        private PolyArena()
        {
            NewBlock(BaseBytes);
            _cur = 0;
            _top = 0;
        }

        /// <summary>
        ///     The calling thread's arena, entered: pair with <see cref="Exit"/> in a <c>finally</c>. Entries nest (an
        ///     algebra call made while another is running on the same thread keeps the outer call's memory).
        /// </summary>
        /// <returns>The arena.</returns>
        public static PolyArena Enter()
        {
            var a = t_arena ??= new PolyArena();
            a._depth++;
            return a;
        }

        /// <summary>
        ///     Leaves the arena; the outermost exit frees every allocation and returns the growth blocks to the house buffer
        ///     pool (a series beyond the 64 KB base block does not pin its memory to the thread — the pool decides what
        ///     stays warm for the next call, of this thread or any other).
        /// </summary>
        public void Exit()
        {
            if (--_depth > 0)
                return;
            _cur = 0;
            _top = 0;
            for (int i = 1; i < _count; i++)
            {
                SizeBucketedBufferPool.Return(_block[i], _cap[i]);
                _block[i] = IntPtr.Zero;
                _cap[i] = 0;
            }
            _count = 1;
        }

        /// <summary>
        ///     Returns the base block (and any growth block a call left behind — none, when every Enter was paired with its
        ///     Exit) to the house buffer pool once the arena is unreachable: its thread has ended, since only the thread-static
        ///     field references it.
        /// </summary>
        ~PolyArena()
        {
            for (int i = 0; i < _count; i++)
                SizeBucketedBufferPool.Return(_block[i], _cap[i]);
        }

        /// <summary>The current top.</summary>
        public Mark Position => new Mark(_cur, _top);

        /// <summary>
        ///     Blocks the arena holds right now — 1 (the base block) between calls, whatever the last call needed. A
        ///     test hook for the release-on-exit contract (a thread that once multiplied long series must not keep that
        ///     memory).
        /// </summary>
        internal int BlockCount => _count;

        /// <summary>
        ///     The calling thread's arena WITHOUT entering it: null before the thread's first series-algebra call. A test
        ///     hook (the entry points always go through <see cref="Enter"/>).
        /// </summary>
        internal static PolyArena PeekCurrent => t_arena;

        /// <summary>Frees everything allocated after <paramref name="m"/> (a position taken earlier in this call).</summary>
        /// <param name="m">The position.</param>
        public void Release(Mark m)
        {
            _cur = m.Block;
            _top = m.Top;
        }

        /// <summary>
        ///     Bump-allocates <paramref name="bytes"/> (uninitialized, 64-byte aligned): in the current block, else at the
        ///     start of the first later block with room (blocks past the top hold only freed memory), else in a new block
        ///     of at least twice the last one.
        /// </summary>
        /// <param name="bytes">Byte count (0 allocates one line).</param>
        /// <returns>The memory.</returns>
        /// <exception cref="OutOfMemoryException">The OS refuses a new block.</exception>
        public byte* Alloc(long bytes)
        {
            long need = bytes <= 0 ? Align : (bytes + (Align - 1)) & ~(Align - 1);
            if (_top + need <= _cap[_cur])
            {
                byte* p = (byte*)_block[_cur] + _top;
                _top += need;
                return p;
            }
            for (int b = _cur + 1; b < _count; b++)
            {
                if (need <= _cap[b])
                {
                    _cur = b;
                    _top = need;
                    return (byte*)_block[b];
                }
            }
            // A power of two at least twice the last block: repeated calls of similar sizes then take the same pool
            // bucket (warm memory, no first-touch page faults), and the doubling bounds the block count by log2.
            NewBlock((long)BitOperations.RoundUpToPowerOf2((ulong)Math.Max(need, _cap[_count - 1] * 2)));
            _cur = _count - 1;
            _top = need;
            return (byte*)_block[_cur];
        }

        /// <summary><see cref="Alloc"/>, zero-filled (NumPy's <c>np.zeros</c>: all-zero bits are +0 in every float
        ///     dtype, (0+0j) in complex, 0 in decimal).</summary>
        /// <param name="bytes">Byte count.</param>
        /// <returns>The memory.</returns>
        public byte* AllocZeroed(long bytes)
        {
            byte* p = Alloc(bytes);
            NativeMemory.Clear(p, (nuint)bytes);
            return p;
        }

        /// <summary>
        ///     The allocation-order position of <paramref name="p"/>, or null when it is not arena memory (an as_series
        ///     copy never is — but a caller's view might be).
        /// </summary>
        /// <param name="p">An address.</param>
        /// <returns>Its (block, offset), or null.</returns>
        public Mark? PositionOf(byte* p)
        {
            for (int b = 0; b < _count; b++)
                if (p >= (byte*)_block[b] && p < (byte*)_block[b] + _cap[b])
                    return new Mark(b, p - (byte*)_block[b]);
            return null;
        }

        /// <summary>Takes one more block from the house buffer pool (uninitialized memory — every series is written before
        ///     it is read, and <see cref="AllocZeroed"/> clears its own).</summary>
        /// <param name="cap">Its byte capacity.</param>
        /// <exception cref="OutOfMemoryException">The pool cannot supply the block.</exception>
        private void NewBlock(long cap)
        {
            if (_count == _block.Length)
            {
                Array.Resize(ref _block, _count * 2);
                Array.Resize(ref _cap, _count * 2);
            }
            _block[_count] = SizeBucketedBufferPool.Take(cap);
            _cap[_count] = cap;
            _count++;
        }
    }

    /// <summary>
    ///     numpy.polynomial's series algebra (plan U2) — see the file header. The entry points (<see cref="Mulx"/>,
    ///     <see cref="Mul"/>, <see cref="Div"/>, <see cref="Pow(PolyBasis, object, long, long?)"/>, <see cref="FromRoots"/>,
    ///     <see cref="ToPower"/>, <see cref="FromPower"/>) back the six basis facades; the rest replays NumPy's Python.
    /// </summary>
    internal static unsafe partial class NDPolyAlgebra
    {
        // =============================================================================================
        //  Entry points
        // =============================================================================================

        /// <summary>
        ///     <c>{p}mulx(c)</c>: <c>[c] = as_series([c])</c>; the zero series <c>[0]</c> is returned as is; otherwise
        ///     <c>prd = np.empty(len(c) + 1)</c> filled by the basis's recurrence.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The series (anything <c>np.array</c> accepts, 1-D).</param>
        /// <returns>A new array of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (a bool series).</exception>
        /// <exception cref="NotSupportedException">A null or str series (NumPy's object / str arrays).</exception>
        public static NDArray Mulx(PolyBasis basis, object c)
        {
            var v = NDPolySeries.AsCoefficientArray(c);
            NDPolySeries.Validate(v);
            long n = NDPolySeries.TrimLength(v);
            NPTypeCode t = NDPolySeries.CommonType(new[] { v });
            // `if len(c) == 1 and c[0] == 0: return c` — the as_series copy, -0.0 kept.
            if (n == 1 && IsZero(v.Ptr, v.Dtype))
                return NDPolySeries.CopyAs(v, 1, t);
            // prd written straight into the result: prd[1:] = c (as_series' converted copy lands there — one pass), then
            // the kernel computes NumPy's prd in place.
            var r = new NDArray(t, new Shape(n + 1), false);
            byte* rp = (byte*)r.Storage.Address;
            int size = DirectILKernelGenerator.GetTypeSize(t);
            if (basis == PolyBasis.Chebyshev && n >= ChebMulxFusedMin)
            {
                // The fused kernel reads c wherever it lies: a contiguous series already of the result dtype in place
                // (as_series' copy of it would be a plain byte copy), anything else after its conversion into prd[1:].
                byte* src = v.Dtype == t && v.Stride == size ? v.Ptr : rp + size;
                if (src != v.Ptr)
                    NDPolySeries.CopyInto(v, n, t, rp + size);
                DirectILKernelGenerator.GetPolyChebMulxKernel(t)(src, n, rp);
                return r;
            }
            NDPolySeries.CopyInto(v, n, t, rp + size);
            DirectILKernelGenerator.GetPolyMulxKernel(basis, t)(null, 0, 0, rp, size, 1, n, 1, 1, null);
            return r;
        }

        /// <summary><c>{p}mul(c1, c2)</c> (see the basis's facade).</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c1">First series.</param>
        /// <param name="c2">Second series.</param>
        /// <returns>The product: a new array, or a view of one when trailing zeros were trimmed (as in NumPy).</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <exception cref="NotSupportedException">A null or str series.</exception>
        public static NDArray Mul(PolyBasis basis, object c1, object c2)
        {
            var (v1, v2, t) = AsSeriesViews(c1, c2);
            var a = PolyArena.Enter();
            try
            {
                var s1 = Load(a, v1, NDPolySeries.TrimLength(v1), t);
                var s2 = Load(a, v2, NDPolySeries.TrimLength(v2), t);
                return ToNDArray(MulCore(a, basis, s1, s2));
            }
            finally
            {
                a.Exit();
            }
        }

        /// <summary><c>{p}div(c1, c2)</c> (see the basis's facade).</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c1">Dividend.</param>
        /// <param name="c2">Divisor.</param>
        /// <returns>(quotient, remainder), each a new array or a view of one.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <exception cref="DivideByZeroException">The divisor is zero (NumPy's bare ZeroDivisionError).</exception>
        /// <exception cref="NotSupportedException">A null or str series.</exception>
        public static (NDArray quo, NDArray rem) Div(PolyBasis basis, object c1, object c2)
        {
            var (v1, v2, t) = AsSeriesViews(c1, c2);
            var a = PolyArena.Enter();
            try
            {
                var s1 = Load(a, v1, NDPolySeries.TrimLength(v1), t);
                var s2 = Load(a, v2, NDPolySeries.TrimLength(v2), t);
                var (q, r) = DivCore(a, basis, s1, s2);
                return (ToNDArray(q), ToNDArray(r));
            }
            finally
            {
                a.Exit();
            }
        }

        /// <summary>
        ///     <c>{p}pow(c, pow, maxpower)</c> for an integer power: <c>[c] = as_series([c])</c>, then NumPy's checks —
        ///     <c>Power must be a non-negative integer.</c> before <c>Power is too large</c> — then 0 → <c>[1]</c>, 1 → c,
        ///     otherwise the repeated product (np.convolve for power series, the z-series convolution for Chebyshev,
        ///     <c>{p}mul</c> for the others).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The series.</param>
        /// <param name="power">The power, already <c>int(pow)</c> and checked to equal it.</param>
        /// <param name="maxpower">The limit, or null for none.</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">An invalid series, a negative power, or a power above <paramref name="maxpower"/>.</exception>
        public static NDArray Pow(PolyBasis basis, object c, long power, long? maxpower)
        {
            var v = NDPolySeries.AsCoefficientArray(c);
            NDPolySeries.Validate(v);
            NPTypeCode t = NDPolySeries.CommonType(new[] { v });
            return PowChecked(basis, v, t, power, maxpower);
        }

        /// <summary>
        ///     <see cref="Pow(PolyBasis, object, long, long?)"/> past <c>as_series</c> and <c>int(pow)</c>: NumPy's power
        ///     checks in order — <c>power != pow or power &lt; 0</c> (only the sign is left to test), then maxpower — and
        ///     the product.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="v">The series' view (validated).</param>
        /// <param name="t">Its common type.</param>
        /// <param name="power">The integer power.</param>
        /// <param name="maxpower">The limit, or null for none.</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">A negative power, or one above <paramref name="maxpower"/>.</exception>
        private static NDArray PowChecked(PolyBasis basis, in PolySeriesView v, NPTypeCode t, long power, long? maxpower)
        {
            if (power < 0)
                throw new ValueError("Power must be a non-negative integer.");
            if (maxpower is long mp && power > mp)
                throw new ValueError("Power is too large");
            long n = NDPolySeries.TrimLength(v);
            var a = PolyArena.Enter();
            try
            {
                return ToNDArray(PowCore(a, basis, Load(a, v, n, t), power));
            }
            finally
            {
                a.Exit();
            }
        }

        /// <summary>
        ///     <c>{p}pow(c, pow, maxpower)</c> for a Python-float power: <c>power = int(pow)</c> (truncation; NaN and inf
        ///     raise CPython's errors), then <see cref="Pow(PolyBasis, object, long, long?)"/> — a non-integral value fails
        ///     NumPy's <c>power != pow</c> test.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The series.</param>
        /// <param name="pow">The power as a Python float.</param>
        /// <param name="maxpower">The limit, or null for none.</param>
        /// <returns>The power series.</returns>
        /// <exception cref="ValueError">An invalid series; <c>cannot convert float NaN to integer</c>; a non-integral or
        ///     negative power; a power above <paramref name="maxpower"/>.</exception>
        /// <exception cref="OverflowException"><c>cannot convert float infinity to integer</c>, or a power beyond int64
        ///     (NumPy would loop until memory runs out).</exception>
        public static NDArray Pow(PolyBasis basis, object c, double pow, long? maxpower)
        {
            // as_series runs BEFORE int(pow): an invalid series is reported first.
            var v = NDPolySeries.AsCoefficientArray(c);
            NDPolySeries.Validate(v);
            NPTypeCode t = NDPolySeries.CommonType(new[] { v });
            if (double.IsNaN(pow))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(pow))
                throw new OverflowException("cannot convert float infinity to integer");
            double truncated = Math.Truncate(pow);
            if (truncated != pow || pow < 0)
                throw new ValueError("Power must be a non-negative integer.");
            if (truncated >= 9.2233720368547758E18)
            {
                // int(pow) is a Python int past int64: `power > maxpower` still decides for a finite limit; without one NumPy
                // would multiply until memory runs out.
                if (maxpower is not null)
                    throw new ValueError("Power is too large");
                throw new OverflowException($"power {pow} is beyond int64: NumPy would run out of memory computing it");
            }
            return PowChecked(basis, v, t, (long)truncated, maxpower);
        }

        /// <summary><c>{p}fromroots(roots)</c> (see the basis's facade).</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="roots">The roots.</param>
        /// <returns>The series whose roots they are, float64 (complex128 for complex roots).</returns>
        /// <exception cref="TypeError"><c>len()</c> fails: a scalar or a 0-d array (CPython's texts).</exception>
        /// <exception cref="ValueError">A non-1-d root array, an empty one of nonzero length, or no common type.</exception>
        /// <exception cref="NotSupportedException">A str / null root (NumPy's str / object arrays).</exception>
        public static NDArray FromRoots(PolyBasis basis, object roots)
        {
            // `if len(roots) == 0: return np.ones(1)` — Python's len() of the argument itself, before any conversion.
            if (PythonLen(roots) == 0)
                return np.ones(new Shape(1), NPTypeCode.Double);
            // `[roots] = as_series([roots], trim=False)`, then `roots.sort()` — on the copy.
            var v = NDPolySeries.AsCoefficientArray(roots);
            NDPolySeries.Validate(v);
            NPTypeCode tr = NDPolySeries.CommonType(new[] { v });
            using var sorted = NDPolySeries.CopyAs(v, v.Len, tr);
            sorted.sort();
            var a = PolyArena.Enter();
            try
            {
                var rs = new PolySer((byte*)sorted.Storage.Address + sorted.Shape.offset * sorted.dtypesize, sorted.size, tr);
                return ToNDArray(FromRootsCore(a, basis, rs));
            }
            finally
            {
                a.Exit();
            }
        }

        /// <summary><c>X2poly(c)</c> — a basis series converted to a power series (see the basis's facade).</summary>
        /// <param name="basis">The source basis (not <see cref="PolyBasis.Power"/>).</param>
        /// <param name="c">The series.</param>
        /// <returns>The power-series coefficients, of <c>np.common_type(c)</c>.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <exception cref="NotSupportedException">A null or str series.</exception>
        public static NDArray ToPower(PolyBasis basis, object c)
        {
            var v = NDPolySeries.AsCoefficientArray(c);
            NDPolySeries.Validate(v);
            long n = NDPolySeries.TrimLength(v);
            NPTypeCode t = NDPolySeries.CommonType(new[] { v });
            var a = PolyArena.Enter();
            try
            {
                return ToNDArray(ToPowerCore(a, basis, Load(a, v, n, t)));
            }
            finally
            {
                a.Exit();
            }
        }

        /// <summary><c>poly2X(pol)</c> — a power series converted to basis <paramref name="basis"/> (see its facade).</summary>
        /// <param name="basis">The target basis (not <see cref="PolyBasis.Power"/>).</param>
        /// <param name="pol">The power series.</param>
        /// <returns>The basis coefficients: float64 for a real series (NumPy starts from the Python int 0), complex128 for a
        ///     complex one.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        /// <exception cref="NotSupportedException">A null or str series.</exception>
        public static NDArray FromPower(PolyBasis basis, object pol)
        {
            var v = NDPolySeries.AsCoefficientArray(pol);
            NDPolySeries.Validate(v);
            long n = NDPolySeries.TrimLength(v);
            NPTypeCode t = NDPolySeries.CommonType(new[] { v });
            var a = PolyArena.Enter();
            try
            {
                return ToNDArray(FromPowerCore(a, basis, Load(a, v, n, t)));
            }
            finally
            {
                a.Exit();
            }
        }

        // =============================================================================================
        //  Arguments and results
        // =============================================================================================

        /// <summary>
        ///     <c>[c1, c2] = as_series([c1, c2])</c> up to the copies: both converted (np.array, ndmin=1) FIRST, then each
        ///     checked (empty, then 1-d) in order, then the common type.
        /// </summary>
        /// <param name="c1">First argument.</param>
        /// <param name="c2">Second argument.</param>
        /// <returns>Both views and their common type.</returns>
        /// <exception cref="ValueError">NumPy's as_series texts.</exception>
        /// <exception cref="NotSupportedException">A null or str argument.</exception>
        private static (PolySeriesView v1, PolySeriesView v2, NPTypeCode t) AsSeriesViews(object c1, object c2)
        {
            var v1 = NDPolySeries.AsCoefficientArray(c1);
            var v2 = NDPolySeries.AsCoefficientArray(c2);
            NDPolySeries.Validate(v1);
            NDPolySeries.Validate(v2);
            return (v1, v2, NDPolySeries.CommonType(new[] { v1, v2 }));
        }

        /// <summary>as_series' <c>np.array(a, copy=True, dtype=t)</c> of a trimmed user argument, into the arena.</summary>
        /// <param name="a">The arena.</param>
        /// <param name="v">The argument's view.</param>
        /// <param name="n">Its trimmed length.</param>
        /// <param name="t">The common type.</param>
        /// <returns>The copy.</returns>
        private static PolySer Load(PolyArena a, in PolySeriesView v, long n, NPTypeCode t)
        {
            int size = DirectILKernelGenerator.GetTypeSize(t);
            byte* p = a.Alloc(n * size);
            NDPolySeries.CopyInto(v, n, t, p);
            return new PolySer(p, n, t);
        }

        /// <summary>
        ///     The NDArray NumPy returns for <paramref name="s"/>: a copy of its whole underlying array, and — when NumPy
        ///     returns a slice of that array — the leading view of it (flags.owndata false), exactly <c>ret[:k]</c>.
        /// </summary>
        /// <param name="s">The result series.</param>
        /// <returns>The array.</returns>
        private static NDArray ToNDArray(in PolySer s)
        {
            var r = new NDArray(s.T, new Shape(s.Full), false);
            long bytes = s.FullBytes;
            Buffer.MemoryCopy(s.P, (byte*)r.Storage.Address, bytes, bytes);
            return s.Slice ? NDPolySeries.Prefix(r, s.N) : r;
        }

        /// <summary>
        ///     Python's <c>len(o)</c> under the house mapping: an ndarray's first dimension (a 0-d one has none), a sequence's
        ///     item count, a string's character count; a scalar has no len.
        /// </summary>
        /// <param name="o">The value.</param>
        /// <returns>The length.</returns>
        /// <exception cref="TypeError">CPython's <c>object of type 'X' has no len()</c>, or NumPy's <c>len() of unsized
        ///     object</c> for a 0-d array.</exception>
        private static long PythonLen(object o)
        {
            switch (o)
            {
                case null:
                    throw new TypeError("object of type 'NoneType' has no len()");
                case NDArray nd:
                    if (nd.ndim == 0)
                        throw new TypeError("len() of unsized object");
                    return nd.Shape.dimensions[0];
                case string s:
                    return s.Length;
                case Array arr:
                    return arr.Rank == 1 ? arr.LongLength : arr.GetLongLength(0);
                case ITuple tuple:
                    return tuple.Length;
                case ICollection col:
                    return col.Count;
            }
            if (PolySequence.IsArrayLike(o) || PolySequence.IsSequence(o))
                return PolySequence.IsArrayLike(o) ? np.asanyarray(o).Shape.dimensions[0] : PolySequence.Items(o).Length;
            throw new TypeError($"object of type '{NDPolySeries.PythonTypeName(o)}' has no len()");
        }

        // =============================================================================================
        //  Element-level helpers (one kernel call each)
        // =============================================================================================

        /// <summary>NumPy's <c>x == 0</c> on one element (NaN is nonzero, -0.0 is zero).</summary>
        /// <param name="p">The element.</param>
        /// <param name="t">Its dtype.</param>
        /// <returns>True when it equals zero.</returns>
        private static bool IsZero(byte* p, NPTypeCode t) => !DirectILKernelGenerator.GetPolyScalarPredicateKernel(t, PolyZeroTest.NotEqual)(p);

        /// <summary>
        ///     NumPy's NEP 50 conversion of the Python int <paramref name="k"/> into dtype <paramref name="t"/> — how every
        ///     weak int of these algorithms (<c>nd - 1</c>, <c>2</c>, <c>0</c>) enters an array op: exact into decimal,
        ///     otherwise through a double (exact for every int an index or length reaches) and the house cast, so float16
        ///     rounds to nearest-even (npy_double_to_half) and complex is <c>k + 0j</c>.
        /// </summary>
        /// <param name="k">The int.</param>
        /// <param name="t">The target dtype.</param>
        /// <param name="dst">One element of <paramref name="t"/>.</param>
        private static void WeakInt(long k, NPTypeCode t, byte* dst)
        {
            if (t == NPTypeCode.Decimal)
            {
                DirectILKernelGenerator.GetPolyCastKernel(NPTypeCode.Int64, t)((byte*)&k, 1, sizeof(long), dst);
                return;
            }
            double d = k;
            DirectILKernelGenerator.GetPolyCastKernel(NPTypeCode.Double, t)((byte*)&d, 1, sizeof(double), dst);
        }

        /// <summary>One element converted: <c>*dst = (tr)*src</c> (NumPy's setitem / scalar promotion cast).</summary>
        /// <param name="src">The element.</param><param name="ts">Its dtype.</param>
        /// <param name="tr">Target dtype.</param><param name="dst">One element of <paramref name="tr"/>.</param>
        private static void CastElement(byte* src, NPTypeCode ts, NPTypeCode tr, byte* dst)
        {
            if (ts == tr)
            {
                int size = DirectILKernelGenerator.GetTypeSize(ts);
                Buffer.MemoryCopy(src, dst, size, size);
                return;
            }
            DirectILKernelGenerator.GetPolyCastKernel(ts, tr)(src, 1, DirectILKernelGenerator.GetTypeSize(ts), dst);
        }

        /// <summary>
        ///     One house ufunc loop over contiguous memory (see <see cref="DirectILKernelGenerator.GetPolyHouseBinaryKernel"/>):
        ///     <c>r = lhs OP rhs</c> elementwise, with the scalar side of <see cref="ExecutionPath.SimdScalarRight"/> /
        ///     <see cref="ExecutionPath.SimdScalarLeft"/> read once. <paramref name="r"/> may be an operand (in place).
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <param name="path">Which operand (if any) is a scalar.</param>
        /// <param name="t">The dtype of both operands and the result.</param>
        /// <param name="lhs">Left operand.</param><param name="rhs">Right operand.</param>
        /// <param name="r">Result.</param><param name="n">Element count (0 does nothing).</param>
        private static void House(BinaryOp op, ExecutionPath path, NPTypeCode t, byte* lhs, byte* rhs, byte* r, long n)
        {
            if (n <= 0)
                return;
            var k = DirectILKernelGenerator.GetPolyHouseBinaryKernel(op, path, t);
            // These three paths read only the pointers and the count; the stride/shape arguments are well-formed anyway.
            long one = 1, zero = 0, extent = n;
            k(lhs, rhs, r, path == ExecutionPath.SimdScalarLeft ? &zero : &one, path == ExecutionPath.SimdScalarRight ? &zero : &one,
                &extent, 1, n);
        }

        // =============================================================================================
        //  Arena bookkeeping
        // =============================================================================================

        /// <summary>A fresh (uninitialized) series of <paramref name="n"/> elements — NumPy's <c>np.empty</c>.</summary>
        /// <param name="a">The arena.</param><param name="n">Length.</param><param name="t">Dtype.</param>
        /// <returns>The series.</returns>
        private static PolySer New(PolyArena a, long n, NPTypeCode t)
            => new PolySer(a.Alloc(n * DirectILKernelGenerator.GetTypeSize(t)), n, t);

        /// <summary>A fresh copy of <paramref name="s"/>'s elements (NumPy's <c>.copy()</c>).</summary>
        /// <param name="a">The arena.</param><param name="s">The series.</param>
        /// <returns>The copy.</returns>
        private static PolySer Copy(PolyArena a, in PolySer s)
        {
            var r = New(a, s.N, s.T);
            long bytes = s.N * s.Size;
            Buffer.MemoryCopy(s.P, r.P, bytes, bytes);
            return r;
        }

        /// <summary>
        ///     Moves <paramref name="s"/> down to <paramref name="mark"/> and frees everything else allocated after it — how
        ///     a function hands back its result and drops its intermediates. Its whole underlying array
        ///     (<see cref="PolySer.Full"/>) moves, so a slice result keeps its base. A series that is not above the mark
        ///     (a caller's) stays where it is.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="mark">The function's entry position.</param>
        /// <param name="s">The live series (updated).</param>
        private static void Keep(PolyArena a, PolyArena.Mark mark, ref PolySer s)
        {
            var pos = a.PositionOf(s.P);
            if (pos is not { } at || !at.AtOrAfter(mark))
            {
                a.Release(mark);
                return;
            }
            a.Release(mark);
            s = Move(a, s);
        }

        /// <summary>
        ///     <see cref="Keep(PolyArena, PolyArena.Mark, ref PolySer)"/> for two live series: they are moved in allocation
        ///     order, each destination at or before its source, so an earlier move never overwrites a later source (the
        ///     stack argument: destinations are allocated in the same order from a lower start). Two views of one array
        ///     move once.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="mark">The entry position.</param>
        /// <param name="s0">First live series (updated).</param>
        /// <param name="s1">Second live series (updated).</param>
        private static void Keep(PolyArena a, PolyArena.Mark mark, ref PolySer s0, ref PolySer s1)
        {
            var p0 = a.PositionOf(s0.P);
            var p1 = a.PositionOf(s1.P);
            bool m0 = p0 is { } q0 && q0.AtOrAfter(mark);
            bool m1 = p1 is { } q1 && q1.AtOrAfter(mark);
            a.Release(mark);
            if (s0.P == s1.P)
            {
                // One underlying array, two views of it (a series and its own slice): move it once, re-point both.
                if (m0)
                {
                    var moved = Move(a, s0.Full >= s1.Full ? s0 : s1);
                    s0 = new PolySer(moved.P, s0.N, s0.T, s0.Full, s0.Slice);
                    s1 = new PolySer(moved.P, s1.N, s1.T, s1.Full, s1.Slice);
                }
                return;
            }
            bool firstIs0 = !m1 || (m0 && !p1.Value.AtOrAfter(p0.Value));
            if (firstIs0)
            {
                if (m0) s0 = Move(a, s0);
                if (m1) s1 = Move(a, s1);
            }
            else
            {
                if (m1) s1 = Move(a, s1);
                if (m0) s0 = Move(a, s0);
            }
        }

        /// <summary>Re-allocates <paramref name="s"/> at the top and copies its underlying array there (memmove: the
        ///     destination may overlap the source, which lies at or after it).</summary>
        /// <param name="a">The arena (its top already released below the source).</param>
        /// <param name="s">The series.</param>
        /// <returns>The moved series (same length, dtype and return semantics).</returns>
        private static PolySer Move(PolyArena a, in PolySer s)
        {
            long bytes = s.FullBytes;
            byte* dst = a.Alloc(bytes);
            if (dst != s.P)
                Buffer.MemoryCopy(s.P, dst, bytes, bytes);
            return new PolySer(dst, s.N, s.T, s.Full, s.Slice);
        }

        // =============================================================================================
        //  polyutils: trimseq / as_series / _add / _sub
        // =============================================================================================

        /// <summary><c>polyutils.trimseq(s)</c>: s itself, or the slice <c>s[:k]</c> ending at the last nonzero element.</summary>
        /// <param name="s">The series.</param>
        /// <returns>The trimmed series (a slice of the same array).</returns>
        private static PolySer TrimSeq(in PolySer s)
        {
            long k = DirectILKernelGenerator.GetPolyTrimLenKernel(s.T)(s.P, s.N, s.Size);
            return k >= 0 ? s : s.Prefix(-k);
        }

        /// <summary>The length <c>trimseq(s)</c> keeps.</summary>
        /// <param name="s">The series.</param>
        /// <returns>The trimmed length (≥ 1 for a non-empty series).</returns>
        private static long TrimLen(in PolySer s)
        {
            long k = DirectILKernelGenerator.GetPolyTrimLenKernel(s.T)(s.P, s.N, s.Size);
            return k < 0 ? -k : k;
        }

        /// <summary>
        ///     <c>np.common_type</c> of two series dtypes (after as_series: float16/float32/float64/complex128/decimal — and
        ///     int64 for a Python int list, which common_type counts as float64).
        /// </summary>
        /// <param name="x">First dtype.</param><param name="y">Second dtype.</param>
        /// <returns>The common type.</returns>
        private static NPTypeCode Common(NPTypeCode x, NPTypeCode y) => x == y && !PolyTyping.IsIntLike(x) ? x : np.common_type_code(x, y);

        /// <summary>
        ///     One series through <c>as_series</c>: trimmed, converted to <paramref name="t"/>. NumPy copies; the engine
        ///     never writes into a series it did not just create, so an already-<paramref name="t"/> series is kept in
        ///     place (its trimmed prefix) and only a conversion copies.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="s">The series.</param>
        /// <param name="t">The common type.</param>
        /// <returns>The as_series series.</returns>
        private static PolySer AsSeries(PolyArena a, in PolySer s, NPTypeCode t)
        {
            long n = TrimLen(s);
            if (s.T == t)
                return new PolySer(s.P, n, t);
            var r = New(a, n, t);
            NDPolySeries.CopyInto(s.View, n, t, r.P);
            return r;
        }

        /// <summary>
        ///     <c>polyutils._add(x, y)</c> / <c>_sub(x, y)</c>: as_series both (trim, common type), the longer updated in
        ///     place (y on a tie; for <c>_sub</c> with a subtrahend at least as long, y negated first), trimseq'd — the U1
        ///     combine NumPy's <c>{p}add</c> runs, into a fresh arena series.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="x">First series (NumPy's c1).</param>
        /// <param name="y">Second series (c2).</param>
        /// <param name="subtract">Run <c>_sub</c>.</param>
        /// <returns>The sum / difference (a trimseq slice when trailing zeros were trimmed).</returns>
        private static PolySer AddSub(PolyArena a, in PolySer x, in PolySer y, bool subtract)
        {
            long n1 = TrimLen(x), n2 = TrimLen(y);
            NPTypeCode t = Common(x.T, y.T);
            var r = New(a, n1 > n2 ? n1 : n2, t);
            long k = NDPolySeries.CombineInto(PolySeriesView.Raw(x.P, n1, x.T), n1, PolySeriesView.Raw(y.P, n2, y.T), n2, t, subtract, r.P, out long na);
            return k >= 0 ? new PolySer(r.P, na, t) : new PolySer(r.P, -k, t, na, true);
        }

        /// <summary>The float64 series <c>[0.]</c>: what as_series makes of the Python int 0 (<c>c1 = 0</c> in the
        ///     one-term product branch, <c>res = 0</c> in poly2X).</summary>
        /// <param name="a">The arena.</param>
        /// <returns>A fresh one-element float64 series holding +0.0.</returns>
        private static PolySer PythonZero(PolyArena a) => new PolySer(a.AllocZeroed(sizeof(double)), 1, NPTypeCode.Double);

        // =============================================================================================
        //  Array ops (one ufunc call each)
        // =============================================================================================

        /// <summary><c>s OP k</c> — an array op with a Python int on the right (<c>c1 * (nd - 1)</c>, <c>... / nd</c>): k
        ///     converted into s's dtype (NEP 50), one house loop into <paramref name="dst"/>.</summary>
        /// <param name="op">Multiply or Divide (true division).</param>
        /// <param name="s">The series.</param>
        /// <param name="k">The Python int.</param>
        /// <param name="dst">The result's memory (s.N elements; may be s.P for an in-place temporary).</param>
        /// <returns>The result.</returns>
        private static PolySer OpInt(BinaryOp op, in PolySer s, long k, byte* dst)
        {
            ulong* kv = stackalloc ulong[2];
            WeakInt(k, s.T, (byte*)kv);
            House(op, ExecutionPath.SimdScalarRight, s.T, s.P, (byte*)kv, dst, s.N);
            return new PolySer(dst, s.N, s.T);
        }

        /// <summary><c>k OP s</c> — an array op with the Python int on the LEFT (lagmul's <c>(2 * nd - 1) * c1</c>).</summary>
        /// <param name="op">The op.</param>
        /// <param name="k">The Python int.</param>
        /// <param name="s">The series.</param>
        /// <param name="dst">The result's memory (s.N elements; may be s.P).</param>
        /// <returns>The result.</returns>
        private static PolySer IntOp(BinaryOp op, long k, in PolySer s, byte* dst)
        {
            ulong* kv = stackalloc ulong[2];
            WeakInt(k, s.T, (byte*)kv);
            House(op, ExecutionPath.SimdScalarLeft, s.T, (byte*)kv, s.P, dst, s.N);
            return new PolySer(dst, s.N, s.T);
        }

        /// <summary><c>x OP s</c> — an array op with a NumPy scalar of s's dtype on the left (<c>c[-i] * xs</c>, <c>q * p[:-1]</c>,
        ///     <c>r * z2</c>).</summary>
        /// <param name="op">The op.</param>
        /// <param name="x">The scalar (one element of s.T).</param>
        /// <param name="s">The series.</param>
        /// <param name="dst">The result's memory (s.N elements).</param>
        /// <returns>The result.</returns>
        private static PolySer ScalarOp(BinaryOp op, byte* x, in PolySer s, byte* dst)
        {
            House(op, ExecutionPath.SimdScalarLeft, s.T, x, s.P, dst, s.N);
            return new PolySer(dst, s.N, s.T);
        }

        /// <summary><c>s OP x</c> — an array op with a NumPy scalar of s's dtype on the right (<c>c2 * c1[j]</c>,
        ///     <c>c1 / c2[-1]</c>, <c>quo /= scl</c>).</summary>
        /// <param name="op">The op.</param>
        /// <param name="s">The series.</param>
        /// <param name="x">The scalar (one element of s.T).</param>
        /// <param name="dst">The result's memory (s.N elements; may be s.P).</param>
        /// <returns>The result.</returns>
        private static PolySer OpScalar(BinaryOp op, in PolySer s, byte* x, byte* dst)
        {
            House(op, ExecutionPath.SimdScalarRight, s.T, s.P, x, dst, s.N);
            return new PolySer(dst, s.N, s.T);
        }

        /// <summary>
        ///     <c>{p}mulx(c)</c> of an intermediate: as_series (trim), the zero series returned as is (a copy of
        ///     <c>[c[0]]</c>), otherwise NumPy's <c>prd</c> — c copied to rows 1..n, then the mulx kernel in place.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The series (any as_series dtype).</param>
        /// <returns>A fresh series.</returns>
        private static PolySer Mulx(PolyArena a, PolyBasis basis, in PolySer c)
        {
            long n = TrimLen(c);
            if (n == 1 && IsZero(c.P, c.T))
                return Copy(a, c.Sub(0, 1));
            var prd = New(a, n + 1, c.T);
            if (basis == PolyBasis.Chebyshev && n >= ChebMulxFusedMin)
            {
                // The fused kernel reads the intermediate where it lies (an arena series: contiguous, not prd's memory).
                DirectILKernelGenerator.GetPolyChebMulxKernel(c.T)(c.P, n, prd.P);
                return prd;
            }
            long bytes = n * c.Size;
            Buffer.MemoryCopy(c.P, prd.P + c.Size, bytes, bytes);
            DirectILKernelGenerator.GetPolyMulxKernel(basis, c.T)(null, 0, 0, prd.P, c.Size, 1, n, 1, 1, null);
            return prd;
        }

        /// <summary>
        ///     The series length from which chebmulx runs the fused kernel (<see cref="DirectILKernelGenerator.GetPolyChebMulxKernel"/>)
        ///     instead of the integral-layout calculus kernel: every series with a term past c[0]. NumPy's chebmulx is the one
        ///     mulx that is NOT a Python loop — three array statements — so it streams, where the calculus kernel (built to
        ///     vectorize over the COLUMNS of an N-D series) walks a 1-D series one row at a time with a true division per
        ///     element: measured 0.43x NumPy on a 10000-term float16 series, 0.74x float32. The fused pass is never slower,
        ///     even at two terms; the calculus kernel keeps the one-term series (<c>prd = [c[0]*0, c[0]]</c>).
        /// </summary>
        private const long ChebMulxFusedMin = 2;

        /// <summary>
        ///     <c>np.convolve(x, y)</c> of two series of one dtype: the longer is the data, the other reversed into the
        ///     kernel (<c>correlate(a, v[::-1])</c>), NumPy's full-mode <c>_pyarray_correlate</c> — through the installed
        ///     byte-parity backend when there is one, exactly as <c>np.convolve</c>.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="x">First series.</param>
        /// <param name="y">Second series (same dtype).</param>
        /// <returns>A fresh series of <c>len(x) + len(y) - 1</c> elements.</returns>
        private static PolySer Convolve(PolyArena a, in PolySer x, in PolySer y)
        {
            // `if len(v) > len(a): a, v = v, a` — the swap is on length only (convolve is commutative).
            var (d, v) = y.N > x.N ? (y, x) : (x, y);
            var r = New(a, d.N + v.N - 1, d.T);
            var mark = a.Position;
            byte* k = a.Alloc(v.N * v.Size);
            NDPolySeries.CopyInto(v.View.Reversed(), v.N, v.T, k);
            NDArray.SlidingCorrelateInto(d.P, d.N, k, v.N, r.P, d.T, NDArray.SlidingMode.Full,
                BackendFactory.GetEngine().Blas as ISlidingDotBackend);
            a.Release(mark);
            return r;
        }

        // =============================================================================================
        //  Chebyshev z-series (chebyshev._cseries_to_zseries / _zseries_to_cseries)
        // =============================================================================================

        /// <summary>
        ///     <c>_cseries_to_zseries(c)</c>: <c>zs = np.zeros(2n - 1); zs[n-1:] = c / 2; return zs + zs[::-1]</c> — element
        ///     k is <c>zs[k] + zs[2n-2-k]</c>: <c>0 + h[n-1-k]</c> below the middle, <c>h[0] + h[0]</c> at it, <c>h[k-n+1] + 0</c>
        ///     above (h = c / 2), each an array op with NumPy's operand order (the +0.0 turns -0.0 into +0.0).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="c">The c-series.</param>
        /// <returns>A fresh z-series of <c>2n - 1</c> elements.</returns>
        private static PolySer CToZ(PolyArena a, in PolySer c)
        {
            long n = c.N;
            var z = New(a, 2 * n - 1, c.T);
            var mark = a.Position;
            ulong* two = stackalloc ulong[2];
            ulong* zero = stackalloc ulong[2];
            zero[0] = zero[1] = 0;   // np.zeros' +0 in every dtype
            WeakInt(2, c.T, (byte*)two);
            var h = New(a, n, c.T);
            House(BinaryOp.Divide, ExecutionPath.SimdScalarRight, c.T, c.P, (byte*)two, h.P, n);            // c / 2
            House(BinaryOp.Add, ExecutionPath.SimdFull, c.T, h.P, h.P, z.At(n - 1), 1);                        // zs[n-1] + zs[n-1]
            if (n > 1)
            {
                House(BinaryOp.Add, ExecutionPath.SimdScalarRight, c.T, h.At(1), (byte*)zero, z.At(n), n - 1);   // h[k] + 0
                // Below the middle: z[j] = 0 + h[n-1-j] — h[1:] reversed into place, then the zero added on the left.
                NDPolySeries.CopyInto(h.Sub(1, n - 1).View.Reversed(), n - 1, c.T, z.P);
                House(BinaryOp.Add, ExecutionPath.SimdScalarLeft, c.T, (byte*)zero, z.P, z.P, n - 1);
            }
            a.Release(mark);
            return z;
        }

        /// <summary><c>_zseries_to_cseries(zs)</c>: <c>n = (len + 1) // 2; c = zs[n-1:].copy(); c[1:n] *= 2</c>.</summary>
        /// <param name="a">The arena.</param>
        /// <param name="zs">The z-series.</param>
        /// <returns>A fresh c-series of n elements.</returns>
        private static PolySer ZToC(PolyArena a, in PolySer zs)
        {
            long n = (zs.N + 1) / 2;
            var c = Copy(a, zs.Sub(n - 1, n));
            if (n > 1)
                OpInt(BinaryOp.Multiply, c.Sub(1, n - 1), 2, c.At(1));
            return c;
        }

        // =============================================================================================
        //  The shared drivers: {p}mul / {p}div / _pow / _fromroots
        // =============================================================================================

        /// <summary>
        ///     <c>{p}mul(x, y)</c> of two intermediates, as_series included (trim; np.common_type — a float64 factor makes a
        ///     float32 product float64).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis.</param>
        /// <param name="x">First series.</param>
        /// <param name="y">Second series.</param>
        /// <returns>The product (fresh, possibly a trimseq slice).</returns>
        private static PolySer MulCore(PolyArena a, PolyBasis basis, in PolySer x, in PolySer y)
        {
            var mark = a.Position;
            NPTypeCode t = Common(x.T, y.T);
            var c1 = AsSeries(a, x, t);
            var c2 = AsSeries(a, y, t);
            PolySer r = basis switch
            {
                // polymul: ret = np.convolve(c1, c2); return trimseq(ret)
                PolyBasis.Power => TrimSeq(Convolve(a, c1, c2)),
                // chebmul: z1 = c2z(c1); z2 = c2z(c2); prd = np.convolve(z1, z2); ret = z2c(prd); return trimseq(ret)
                PolyBasis.Chebyshev => TrimSeq(ZToC(a, Convolve(a, CToZ(a, c1), CToZ(a, c2)))),
                _ => RecurrenceMul(a, basis, c1, c2),
            };
            Keep(a, mark, ref r);
            return r;
        }

        /// <summary>
        ///     <c>{p}pow</c> after <c>as_series([c])</c> and NumPy's checks: <c>np.array([1], dtype=c.dtype)</c> for power 0,
        ///     c for 1, otherwise the repeated product — <c>_pow(np.convolve, …)</c> for power series (no trimming between
        ///     steps), chebpow's z-series convolutions (the c-series is not trimmed at the end), <c>_pow({p}mul, …)</c>
        ///     for the rest (each step a full <c>{p}mul</c>).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis.</param>
        /// <param name="c">The as_series copy.</param>
        /// <param name="power">The power (≥ 0, within maxpower).</param>
        /// <returns>The power series (fresh).</returns>
        private static PolySer PowCore(PolyArena a, PolyBasis basis, in PolySer c, long power)
        {
            if (power == 0)
            {
                var one = New(a, 1, c.T);
                WeakInt(1, c.T, one.P);
                return one;
            }
            if (power == 1)
                return c;
            var mark = a.Position;
            PolySer prd;
            if (basis == PolyBasis.Chebyshev)
            {
                // zs = c2z(c); prd = zs; for i in range(2, power + 1): prd = np.convolve(prd, zs); return z2c(prd)
                var zs = CToZ(a, c);
                var loop = a.Position;
                prd = zs;
                for (long i = 2; i <= power; i++)
                {
                    prd = Convolve(a, prd, zs);
                    Keep(a, loop, ref prd);
                }
                prd = ZToC(a, prd);
            }
            else
            {
                // prd = c; for i in range(2, power + 1): prd = mul_f(prd, c)
                prd = c;
                for (long i = 2; i <= power; i++)
                {
                    prd = basis == PolyBasis.Power ? Convolve(a, prd, c) : MulCore(a, basis, prd, c);
                    Keep(a, mark, ref prd);
                }
            }
            Keep(a, mark, ref prd);
            return prd;
        }

        /// <summary>
        ///     <c>{p}div</c> after <c>as_series([c1, c2])</c>: the zero-divisor ZeroDivisionError, NumPy's two short cuts —
        ///     <c>(c1[:1]*0, c1)</c> for a shorter dividend, <c>(c1/c2[-1], c1[:1]*0)</c> for a constant divisor — then the
        ///     basis's division.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis.</param>
        /// <param name="c1">Dividend (as_series copy).</param>
        /// <param name="c2">Divisor (as_series copy, same dtype).</param>
        /// <returns>(quotient, remainder).</returns>
        /// <exception cref="DivideByZeroException">c2's last coefficient is zero (only an all-zero divisor, after trimming).</exception>
        private static (PolySer quo, PolySer rem) DivCore(PolyArena a, PolyBasis basis, in PolySer c1, in PolySer c2)
        {
            if (IsZero(c2.At(c2.N - 1), c2.T))
                throw new DivideByZeroException("");
            long lc1 = c1.N, lc2 = c2.N;
            if (lc1 < lc2)
                return (ZeroLike(a, c1), c1);
            if (lc2 == 1)
            {
                var q = OpScalar(BinaryOp.Divide, c1, c2.At(0), New(a, lc1, c1.T).P);
                return (q, ZeroLike(a, c1));
            }
            return basis switch
            {
                PolyBasis.Power => PolyDiv(a, c1, c2),
                PolyBasis.Chebyshev => ChebDiv(a, c1, c2),
                _ => RecurrenceDiv(a, basis, c1, c2),
            };
        }

        /// <summary><c>c[:1] * 0</c>: a fresh one-element array op (±0, or NaN for an infinite / NaN c[0]).</summary>
        /// <param name="a">The arena.</param>
        /// <param name="c">The series.</param>
        /// <returns>The one-element series.</returns>
        private static PolySer ZeroLike(PolyArena a, in PolySer c) => OpInt(BinaryOp.Multiply, c.Sub(0, 1), 0, New(a, 1, c.T).P);

        /// <summary>
        ///     <c>polyutils._div(mul_f, c1, c2)</c> past its short cuts: for i from lc1 - lc2 down to 0,
        ///     <c>p = mul_f([0]*i + [1], c2); q = rem[-1] / p[-1]; rem = rem[:-1] - q * p[:-1]; quo[i] = q</c>.
        /// </summary>
        /// <remarks>
        ///     The int list promotes with c2 in as_series (<c>np.common_type</c> counts int64 as float64), so p — and from
        ///     the first step on, rem — are float64 for a float16/float32 divisor (complex128 / decimal otherwise), while
        ///     quo keeps c1's dtype (<c>quo[i] = q</c> casts). The dividend is converted to that dtype up front: NumPy's first
        ///     step reads c1 through <c>rem[-1] / p[-1]</c> and <c>rem[:-1] - …</c>, which both widen it exactly first. c2 is
        ///     converted once for all the products (NumPy's as_series converts it inside every one — the same values).
        /// </remarks>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis (Legendre, Laguerre, Hermite or HermiteE).</param>
        /// <param name="c1">Dividend (lc1 ≥ lc2 ≥ 2).</param>
        /// <param name="c2">Divisor.</param>
        /// <returns>(quotient, remainder).</returns>
        /// <exception cref="IndexError">A remainder emptied by NumPy's broadcasting (an underflowed product; NumPy's
        ///     <c>rem[-1]</c> text).</exception>
        /// <exception cref="IncorrectShapeException">The subtraction's operands do not broadcast (NumPy's ValueError text).</exception>
        private static (PolySer quo, PolySer rem) RecurrenceDiv(PolyArena a, PolyBasis basis, in PolySer c1, in PolySer c2)
        {
            long lc1 = c1.N, lc2 = c2.N;
            NPTypeCode tp = Common(NPTypeCode.Int64, c2.T);
            var quo = New(a, lc1 - lc2 + 1, c1.T);
            // The running remainder, shrinking in place from lc1 elements (the subtraction reads rem[k] before writing it).
            var rem = New(a, lc1, tp);
            NDPolySeries.CopyInto(c1.View, lc1, tp, rem.P);
            var c2p = AsSeries(a, c2, tp);
            if (c2p.P == c2.P)
                c2p = Copy(a, c2p);   // the products below read it while rem shrinks — keep it in this call's own memory
            var loop = a.Position;
            ulong* q = stackalloc ulong[2];
            long lr = lc1;
            for (long i = lc1 - lc2; i >= 0; i--)
            {
                // p = mul_f([0] * i + [1], c2)
                var e = new PolySer(a.AllocZeroed((i + 1) * c2p.Size), i + 1, tp);
                WeakInt(1, tp, e.At(i));
                var p = MulCore(a, basis, e, c2p);
                if (lr == 0)
                    throw new IndexError("index -1 is out of bounds for axis 0 with size 0");
                // q = rem[-1] / p[-1]   (scalarmath: both NumPy scalars of tp)
                DirectILKernelGenerator.GetPolyScalarBinaryKernel(BinaryOp.Divide, tp, PolyComplexProduct.Naive)(rem.At(lr - 1), p.At(p.N - 1), (byte*)q);
                // rem = rem[:-1] - q * p[:-1]
                var qp = ScalarOp(BinaryOp.Multiply, (byte*)q, p.Sub(0, p.N - 1), New(a, Math.Max(p.N - 1, 1), tp).P);
                lr = BroadcastSubtract(new PolySer(rem.P, lr - 1, tp), qp, rem.P);
                // quo[i] = q
                CastElement((byte*)q, tp, c1.T, quo.At(i));
                a.Release(loop);
            }
            // return quo, trimseq(rem) — rem is the last difference, a fresh array of lr elements.
            return (quo, TrimSeq(new PolySer(rem.P, lr, tp)));
        }

        /// <summary>
        ///     NumPy's <c>x - y</c> for two 1-D arrays of one dtype, broadcasting a one-element operand, into
        ///     <paramref name="dst"/> (which may be x's memory: each element is read before it is written, and a broadcast x
        ///     is read once, up front).
        /// </summary>
        /// <param name="x">Minuend.</param>
        /// <param name="y">Subtrahend.</param>
        /// <param name="dst">Result memory (room for the broadcast length).</param>
        /// <returns>The result length.</returns>
        /// <exception cref="IncorrectShapeException">Lengths that do not broadcast (NumPy's ValueError text).</exception>
        private static long BroadcastSubtract(in PolySer x, in PolySer y, byte* dst)
        {
            if (x.N == y.N)
            {
                House(BinaryOp.Subtract, ExecutionPath.SimdFull, x.T, x.P, y.P, dst, x.N);
                return x.N;
            }
            if (y.N == 1)
            {
                ulong* s = stackalloc ulong[2];
                CastElement(y.P, y.T, y.T, (byte*)s);
                House(BinaryOp.Subtract, ExecutionPath.SimdScalarRight, x.T, x.P, (byte*)s, dst, x.N);
                return x.N;
            }
            if (x.N == 1)
            {
                ulong* s = stackalloc ulong[2];
                CastElement(x.P, x.T, x.T, (byte*)s);
                House(BinaryOp.Subtract, ExecutionPath.SimdScalarLeft, x.T, (byte*)s, y.P, dst, y.N);
                return y.N;
            }
            throw new IncorrectShapeException($"operands could not be broadcast together with shapes ({x.N},) ({y.N},) ");
        }

        /// <summary>
        ///     <c>polyutils._fromroots(line_f, mul_f, roots)</c> after the empty check, as_series and the sort:
        ///     <c>p = [line_f(-r, 1) for r in roots]</c>, then the pairwise product tree
        ///     (<c>m, r = divmod(n, 2); tmp = [mul_f(p[i], p[i+m]) for i in range(m)]; if r: tmp[0] = mul_f(tmp[0], p[-1])</c>).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis.</param>
        /// <param name="roots">The sorted roots (as_series copy).</param>
        /// <returns><c>p[0]</c>: the lone line for one root (unmultiplied), else the last product.</returns>
        private static PolySer FromRootsCore(PolyArena a, PolyBasis basis, in PolySer roots)
        {
            long n = roots.N;
            var p = new PolySer[n];
            for (long i = 0; i < n; i++)
                p[i] = Line(a, basis, roots.At(i), roots.T);
            while (n > 1)
            {
                long m = n / 2, rem = n % 2;
                var tmp = new PolySer[m];
                for (long i = 0; i < m; i++)
                    tmp[i] = MulCore(a, basis, p[i], p[i + m]);
                if (rem != 0)
                    tmp[0] = MulCore(a, basis, tmp[0], p[n - 1]);
                p = tmp;
                n = m;
            }
            return p[0];
        }

        /// <summary>
        ///     <c>line_f(-r, 1)</c> for one root: <c>{p}line</c>'s <c>np.array([off, scl])</c> (lagline
        ///     <c>[off + scl, -scl]</c>, hermline <c>[off, scl / 2]</c>) with <c>off = -r</c> a NumPy scalar of the roots'
        ///     dtype and <c>scl</c> the Python int 1 — array coercion promotes the pair STRONGLY, so a float16/float32 root
        ///     gives a float64 line (the root negated, and lagline's <c>-r + 1</c> added, in the root's own dtype first).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The basis.</param>
        /// <param name="r">The root.</param>
        /// <param name="t">Its dtype.</param>
        /// <returns>A fresh two-element series.</returns>
        private static PolySer Line(PolyArena a, PolyBasis basis, byte* r, NPTypeCode t)
        {
            ulong* neg = stackalloc ulong[2];
            DirectILKernelGenerator.GetPolyScalarNegateKernel(t)(r, (byte*)neg);
            var off = PolyNumber.FromScalar(t, neg);
            var scl = PolyNumber.FromPython(PyScalar.Int(1));
            PolyNumber first = off, second = scl;
            switch (basis)
            {
                case PolyBasis.Laguerre:
                    first = PolyNumber.Binary(BinaryOp.Add, off, scl);   // Python evaluates the list left to right
                    second = PolyNumber.Negate(scl);
                    break;
                case PolyBasis.Hermite:
                    second = PolyNumber.Binary(BinaryOp.Divide, scl, PolyNumber.FromPython(PyScalar.Int(2)));
                    break;
            }
            NPTypeCode tl = PolyTyping.Promote(first.DiscoveredDtype(), second.DiscoveredDtype());
            var line = New(a, 2, tl);
            first.WriteElement(tl, line.At(0));
            second.WriteElement(tl, line.At(1));
            return line;
        }
    }
}
