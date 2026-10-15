using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Polynomial.ConstPool.cs — the weak (Python) values a kernel reads
// =============================================================================
//
// A step tree's Python values (2, (nd-1)/nd, 2*(nd-1), a folded 2*x, ...) have no dtype until they meet
// an array operand; the typing pass decides WHICH dtype for each (constant, use) pair. The emitter then
// registers one REGION per (constant, dtype) and reads it through a table of region base pointers the
// kernel receives in auxdata. A region is an array of `rows` elements of its dtype, filled through the
// house cast path astype uses — NumPy's own conversion — so a float16 kernel reads (nd-1)/nd rounded to float16 exactly
// as NumPy rounds the Python float, and a Python int that does not fit an integer dtype raises
// NumPy's OverflowError before the kernel runs.
//
// CACHING
// -------
// x-independent regions are cached per kernel and grown geometrically (a region's row r never depends
// on how many rows exist, so a larger table extends a smaller one). Old tables are kept alive, never
// freed, because a concurrent call may still hold their pointer; geometric growth bounds that to 2x.
// Calls beyond CacheRowLimit rows, and regions that depend on a Python-scalar x, are materialized per
// call into one scratch block the caller frees.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The weak-value regions of one polynomial kernel and the per-call tables built from them (see the
    ///     file header). Registration happens only while the kernel is emitted (single-threaded, inside the
    ///     kernel cache's factory); <see cref="Prepare"/> is thread-safe.
    /// </summary>
    internal sealed unsafe class PolyConstPool
    {
        /// <summary>Largest row count served from the cached tables; bigger calls materialize per call so a
        ///     one-off million-coefficient series never pins its constants for the process lifetime.</summary>
        private const long CacheRowLimit = 1 << 16;

        /// <summary>Registered regions, in emission order (the index is baked into the IL).</summary>
        private readonly List<(PolyWeakConst W, NPTypeCode T)> _regions = new();

        private readonly object _growLock = new();
        private readonly List<IntPtr> _retired = new();   // superseded cached blocks, kept alive (see header)
        private Snapshot _snap;
        private bool _sealed, _hasXDependent;

        /// <summary>One published cached table: pointers into a single native block, plus, per region, the
        ///     first row whose value does not fit the region's integer dtype (long.MaxValue when none).</summary>
        private sealed class Snapshot
        {
            /// <summary>The region base pointers (region r at index r), inside the snapshot's native block.</summary>
            public long* Table;
            /// <summary>Rows every region of this snapshot holds (per-row constants valid for any call up to it).</summary>
            public long Capacity;
            /// <summary>Per region, the first row whose value does not fit its integer dtype (long.MaxValue when none).</summary>
            public long[] FirstBad;
        }

        /// <summary>Number of regions.</summary>
        public int Count => _regions.Count;

        /// <summary>Returns the region index for (<paramref name="w"/>, <paramref name="t"/>), adding it on first use.</summary>
        /// <param name="w">The constant.</param><param name="t">The dtype it is materialized in.</param>
        /// <returns>The index (stable for the kernel's lifetime).</returns>
        /// <exception cref="InvalidOperationException">Called after <see cref="Seal"/> (the IL is final).</exception>
        public int Region(PolyWeakConst w, NPTypeCode t)
        {
            if (_sealed) throw new InvalidOperationException("the constant pool is sealed");
            for (int r = 0; r < _regions.Count; r++)
                if (ReferenceEquals(_regions[r].W, w) && _regions[r].T == t) return r;
            _regions.Add((w, t));
            return _regions.Count - 1;
        }

        /// <summary>Freezes the region list once the kernel is emitted.</summary>
        public void Seal()
        {
            _sealed = true;
            foreach (var (w, _) in _regions)
                _hasXDependent |= w.XDependent;
        }

        /// <summary>
        ///     The region table for a call reading rows <c>[0, rows)</c> (per-row constants are read at rows
        ///     <c>2..rows-1</c>, fixed ones at row 0). Raises NumPy's OverflowError, in NumPy's evaluation
        ///     order, when a Python int does not fit the integer dtype it meets.
        /// </summary>
        /// <param name="rows">The coefficient count of the call.</param>
        /// <param name="x">The call's Python-scalar x (read only by x-dependent regions).</param>
        /// <param name="scratch">A native block the caller must free with <see cref="NativeMemory.Free"/> after
        ///     the kernel returns (<see cref="IntPtr.Zero"/> when the cached table serves the call).</param>
        /// <returns>The address of the table of region base pointers.</returns>
        /// <exception cref="OverflowException">A Python int out of bounds for its integer dtype — NumPy's
        ///     <c>OverflowError: Python integer N out of bounds for int8</c> (e.g. lagval on int8 x with 70
        ///     coefficients reads <c>2*69 - 1 = 137</c>).</exception>
        public long Prepare(long rows, in PyScalar x, out IntPtr scratch)
        {
            scratch = IntPtr.Zero;
            if (_regions.Count == 0) return 0;
            rows = Math.Max(rows, 1);

            // x-independent regions come from the cached snapshot whenever the row count allows it.
            Snapshot snap = rows <= CacheRowLimit ? EnsureCapacity(rows) : null;
            if (snap is not null && !_hasXDependent)
            {
                for (int r = 0; r < _regions.Count; r++)
                    ThrowIfBad(r, snap.FirstBad[r], rows, x);
                return (long)snap.Table;
            }

            // Per call: one block holding the table plus the data of every region the snapshot cannot serve
            // (the x-dependent ones, or all of them past CacheRowLimit). Regions are visited in registration
            // order so an overflow is raised where NumPy's evaluation would raise it.
            long dataBytes = 0;
            for (int r = 0; r < _regions.Count; r++)
                if (snap is null || _regions[r].W.XDependent)
                    dataBytes += Align16(rows * DirectILKernelGenerator.GetTypeSize(_regions[r].T));
            long tableBytes = Align16(_regions.Count * 8L);
            var block = (byte*)NativeMemory.AllocZeroed((nuint)(tableBytes + dataBytes));
            try
            {
                var table = (long*)block;
                byte* data = block + tableBytes;
                for (int r = 0; r < _regions.Count; r++)
                {
                    var (w, t) = _regions[r];
                    if (snap is not null && !w.XDependent)
                    {
                        ThrowIfBad(r, snap.FirstBad[r], rows, x);
                        table[r] = snap.Table[r];
                        continue;
                    }
                    long firstBad = Fill(w, t, rows, x, data);
                    ThrowIfBad(r, firstBad, rows, x);
                    table[r] = (long)data;
                    data += Align16(rows * DirectILKernelGenerator.GetTypeSize(t));
                }
                scratch = (IntPtr)block;
                return (long)table;
            }
            catch
            {
                NativeMemory.Free(block);
                throw;
            }
        }

        /// <summary>Raises NumPy's OverflowError when region <paramref name="r"/> is read at a row whose value
        ///     does not fit its integer dtype.</summary>
        /// <param name="r">Region index.</param><param name="firstBad">First non-fitting row.</param>
        /// <param name="rows">The call's row count.</param><param name="x">The call's scalar x.</param>
        /// <exception cref="OverflowException">The value read does not fit.</exception>
        private void ThrowIfBad(int r, long firstBad, long rows, in PyScalar x)
        {
            if (firstBad == long.MaxValue) return;
            var (w, t) = _regions[r];
            // The first step NumPy runs reads the LARGEST row (nd = len(c) - 1) and every per-row integer
            // constant grows with the row, so that step is where the first failure happens.
            long row = w.PerRow ? rows - 1 : 0;
            if (w.PerRow && row < 2) return;          // no step reads this region at this length
            if (row < firstBad) return;
            var v = w.Value(row, x);
            throw new OverflowException($"Python integer {v} out of bounds for {t.AsNumpyDtypeName()}");
        }

        /// <summary>Returns a cached snapshot with at least <paramref name="rows"/> rows, growing it geometrically.</summary>
        /// <param name="rows">Rows needed.</param><returns>The snapshot.</returns>
        private Snapshot EnsureCapacity(long rows)
        {
            var snap = Volatile.Read(ref _snap);
            if (snap is not null && snap.Capacity >= rows) return snap;
            lock (_growLock)
            {
                snap = _snap;
                if (snap is not null && snap.Capacity >= rows) return snap;
                long cap = Math.Min(CacheRowLimit, Math.Max(rows, Math.Max(16, (snap?.Capacity ?? 0) * 2)));
                long tableBytes = Align16(_regions.Count * 8L), dataBytes = 0;
                foreach (var (_, t) in _regions)
                    dataBytes += Align16(cap * DirectILKernelGenerator.GetTypeSize(t));
                var block = (byte*)NativeMemory.AllocZeroed((nuint)(tableBytes + dataBytes));
                var next = new Snapshot { Table = (long*)block, Capacity = cap, FirstBad = new long[_regions.Count] };
                byte* data = block + tableBytes;
                for (int r = 0; r < _regions.Count; r++)
                {
                    var (w, t) = _regions[r];
                    // An x-dependent region's values change per call: Prepare fills it into the call's own block.
                    next.FirstBad[r] = w.XDependent ? long.MaxValue : Fill(w, t, cap, default, data);
                    next.Table[r] = (long)data;
                    data += Align16(cap * DirectILKernelGenerator.GetTypeSize(t));
                }
                if (snap is not null) _retired.Add((IntPtr)snap.Table);
                Volatile.Write(ref _snap, next);
                return next;
            }
        }

        /// <summary>
        ///     Materializes rows <c>[0, rows)</c> of one region into <paramref name="dst"/> with NumPy's conversion
        ///     of a Python value to <paramref name="t"/> — the house contiguous cast path (<see cref="ConvertBuffer"/>),
        ///     so the bits are exactly <c>np.array(values).astype(t)</c>'s. Rows 0 and 1 of a per-row constant are
        ///     never read (the recurrences start at <c>nd = 2</c>), so they stay 0 instead of holding a value that
        ///     could spuriously fail the integer range check.
        /// </summary>
        /// <param name="w">The constant.</param><param name="t">Target dtype.</param><param name="rows">Row count.</param>
        /// <param name="x">The call's scalar x (x-dependent constants only).</param>
        /// <param name="dst">Destination (rows elements of t, zeroed).</param>
        /// <returns>The first row whose value does not fit an integer <paramref name="t"/>, or long.MaxValue.</returns>
        /// <exception cref="OverflowException">A Python int too large for a double (CPython's own error), or
        ///     beyond a decimal's range.</exception>
        private static long Fill(PolyWeakConst w, NPTypeCode t, long rows, in PyScalar x, byte* dst)
        {
            long first = w.PerRow ? Math.Min(2, rows) : 0;
            long last = w.PerRow ? rows : 1;            // a fixed constant fills row 0 only
            long n = last - first;
            if (n <= 0) return long.MaxValue;
            int size = DirectILKernelGenerator.GetTypeSize(t);
            byte* target = dst + first * size;

            if (w.Kind == PyKind.Complex)
            {
                var v = new Complex[n];
                for (long i = 0; i < n; i++) v[i] = w.Value(first + i, x).C;
                fixed (Complex* pv = v) ConvertBuffer(pv, NPTypeCode.Complex, target, t, n);
                return long.MaxValue;
            }

            if (w.Kind == PyKind.Int && (PolyTyping.IsIntLike(t) || t == NPTypeCode.Decimal))
            {
                // Python int -> integer (or decimal) dtype: exact, range-checked. A value beyond a long fits no
                // NumSharp integer dtype either, so it is recorded as bad directly (its slot stays 0, unread).
                var ints = new long[n];
                long bad = long.MaxValue;
                for (long i = 0; i < n; i++)
                {
                    var bi = w.Value(first + i, x).I;
                    if (bi < long.MinValue || bi > long.MaxValue) { if (bad == long.MaxValue) bad = first + i; continue; }
                    ints[i] = (long)bi;
                }
                if (bad != long.MaxValue && t == NPTypeCode.Decimal)
                    throw new OverflowException("Python int too large to convert to a decimal value");
                fixed (long* pi = ints) ConvertBuffer(pi, NPTypeCode.Int64, target, t, n);
                if (t == NPTypeCode.Decimal) return long.MaxValue;
                // Dtype-agnostic range check: a value survives the int64 round trip iff it fits t — except a
                // negative value into an unsigned dtype, whose wrap can round-trip back to itself.
                bool unsigned = t is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64
                    or NPTypeCode.Char or NPTypeCode.Boolean;
                var back = new long[n];
                fixed (long* pb = back) ConvertBuffer(target, t, pb, NPTypeCode.Int64, n);
                for (long i = 0; i < n && first + i < bad; i++)
                    if (back[i] != ints[i] || (unsigned && ints[i] < 0))
                        return first + i;
                return bad;
            }

            // Python int/float -> inexact dtype: through a double, as NumPy's scalar setitem does
            // (PyFloat_AsDouble / PyLong_AsDouble), then the house cast to t.
            var d = new double[n];
            for (long i = 0; i < n; i++) d[i] = w.Value(first + i, x).AsDouble();
            fixed (double* pd = d) ConvertBuffer(pd, NPTypeCode.Double, target, t, n);
            return long.MaxValue;
        }

        /// <summary>
        ///     Converts <paramref name="n"/> contiguous elements exactly as <c>astype</c> converts a contiguous
        ///     array: the IL cast kernel when the pair has one (<c>NDIter.Copy</c>'s fast path 1), otherwise the
        ///     per-element <see cref="NDIterCasting.ConvertValue"/> (its scalar dispatch). No NDArray is built —
        ///     this runs per call for a Python-scalar x, where two allocations would cost more than the kernel.
        ///     Also the one-element conversion of <c>numpy.polynomial</c>'s scalar engine (<c>PolyNumber</c>), where
        ///     it carries every NumPy scalar and weak Python value into its NEP 50 loop dtype.
        /// </summary>
        /// <param name="src">Source elements.</param><param name="from">Source dtype.</param>
        /// <param name="dst">Destination elements.</param><param name="to">Target dtype.</param><param name="n">Count.</param>
        internal static void ConvertBuffer(void* src, NPTypeCode from, void* dst, NPTypeCode to, long n)
        {
            if (from == to)
            {
                long bytes = n * DirectILKernelGenerator.GetTypeSize(to);
                Buffer.MemoryCopy(src, dst, bytes, bytes);
                return;
            }
            var kernel = DirectILKernelGenerator.TryGetCastKernel(from, to);
            if (kernel != null)
            {
                kernel(src, dst, n);
                return;
            }
            int ss = DirectILKernelGenerator.GetTypeSize(from), ds = DirectILKernelGenerator.GetTypeSize(to);
            for (long i = 0; i < n; i++)
                NDIterCasting.ConvertValue((byte*)src + i * ss, (byte*)dst + i * ds, from, to);
        }

        /// <summary>Rounds up to a 16-byte multiple (every region starts aligned for vector-free scalar loads).</summary>
        /// <param name="bytes">Byte count.</param><returns>The aligned count.</returns>
        private static long Align16(long bytes) => (bytes + 15) & ~15L;
    }
}
