using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    /// Along-axis sort / argsort, structured exactly like NumPy's <c>_new_sortlike</c> /
    /// <c>_new_argsortlike</c> (item_selection.c): an NpyIter <b>IterAllButAxis</b> drive (the
    /// sort axis is dropped from the iterator via <c>op_axes</c> so it can't coalesce) hands one
    /// 1-D line per call to a high-performance inner sort kernel.
    ///
    /// Inner kernel = LSD <see cref="RadixSort"/> for every fixed-width numeric dtype (POC: int
    /// sort 0.7-0.9x NumPy, argsort 2-4x FASTER than NumPy); scalar BCL introsort with the exact
    /// NumPy comparators for Half/Complex/Decimal. Floats partition NaN to the end (NumPy sorts
    /// NaN last). Radix is stable, so argsort ties resolve in ascending index order (NumPy 'stable').
    /// </summary>
    internal static unsafe class AxisSort
    {
        // ----- key adapters: monotonic unsigned transform so ascending key order == NumPy value order -----
        private interface IKey32<T> where T : unmanaged { uint To(T v); T From(uint k); int Bytes { get; } }
        private interface IKey64<T> where T : unmanaged { ulong To(T v); T From(ulong k); }

        private readonly struct KBool : IKey32<byte> { public uint To(byte v) => v; public byte From(uint k) => (byte)k; public int Bytes => 1; }
        private readonly struct KU8 : IKey32<byte> { public uint To(byte v) => v; public byte From(uint k) => (byte)k; public int Bytes => 1; }
        private readonly struct KI8 : IKey32<sbyte> { public uint To(sbyte v) => (byte)(v ^ unchecked((sbyte)0x80)); public sbyte From(uint k) => (sbyte)((byte)k ^ 0x80); public int Bytes => 1; }
        private readonly struct KI16 : IKey32<short> { public uint To(short v) => (ushort)(v ^ unchecked((short)0x8000)); public short From(uint k) => (short)((ushort)k ^ 0x8000); public int Bytes => 2; }
        private readonly struct KU16 : IKey32<ushort> { public uint To(ushort v) => v; public ushort From(uint k) => (ushort)k; public int Bytes => 2; }
        private readonly struct KChar : IKey32<char> { public uint To(char v) => v; public char From(uint k) => (char)k; public int Bytes => 2; }
        private readonly struct KI32 : IKey32<int> { public uint To(int v) => (uint)v ^ 0x80000000u; public int From(uint k) => (int)(k ^ 0x80000000u); public int Bytes => 4; }
        private readonly struct KU32 : IKey32<uint> { public uint To(uint v) => v; public uint From(uint k) => k; public int Bytes => 4; }
        private readonly struct KI64 : IKey64<long> { public ulong To(long v) => (ulong)v ^ 0x8000000000000000UL; public long From(ulong k) => (long)(k ^ 0x8000000000000000UL); }
        private readonly struct KU64 : IKey64<ulong> { public ulong To(ulong v) => v; public ulong From(ulong k) => k; }

        // ============================ public entry points ============================

        /// <summary>np.sort: returns a new C-contiguous sorted array (axis=null flattens).</summary>
        public static NDArray Sort(NDArray a, int? axis)
        {
            if (axis == null)
            {
                var flat = a.ravel().copy('C');
                SortInPlace(flat, 0);
                return flat;
            }
            var res = a.copy('C');
            SortInPlace(res, NormalizeAxis(axis.Value, a.ndim));
            return res;
        }

        /// <summary>ndarray.sort: sorts <paramref name="a"/> in place along the axis.</summary>
        public static void SortInPlace(NDArray a, int? axis)
        {
            if (!a.Shape.IsWriteable)
                throw new InvalidOperationException("sort: cannot sort a read-only (broadcast) array in place.");
            if (axis == null)
            {
                // In-place flatten-sort only well-defined for contiguous; NumPy raises otherwise.
                SortInPlace(a.reshape(a.size), 0);
                return;
            }
            SortInPlace(a, NormalizeAxis(axis.Value, a.ndim));
        }

        /// <summary>np.argsort: returns int64 indices (same shape; axis=null flattens).</summary>
        public static NDArray ArgSort(NDArray a, int? axis)
        {
            if (axis == null)
            {
                var flat = a.Shape.IsContiguous ? a.reshape(a.size) : a.ravel().copy('C');
                var outFlat = new NDArray(NPTypeCode.Int64, new Shape((int)a.size), false);
                ArgSortInto(flat, outFlat, 0);
                return outFlat;
            }
            int ax = NormalizeAxis(axis.Value, a.ndim);
            var src = a.Shape.IsContiguous ? a : a.copy('C');
            var ret = new NDArray(NPTypeCode.Int64, new Shape((long[])a.Shape.dimensions.Clone()), false);
            ArgSortInto(src, ret, ax);
            return ret;
        }

        private static int NormalizeAxis(int axis, int ndim)
        {
            int ax = axis < 0 ? axis + ndim : axis;
            if (ax < 0 || ax >= ndim)
                throw new ArgumentException($"axis {axis} is out of bounds for array of dimension {ndim}");
            return ax;
        }

        // ============================ in-place line sort ============================

        /// <summary>
        /// Radix key width (bytes) the line kernel needs for <paramref name="tc"/>, or 0 for the
        /// scalar BCL path (Half/Complex/Decimal sort their own buffer). 8 for the 8-byte numeric
        /// dtypes (Int64/UInt64/Double), else 4 (1/2/4-byte ints, Single via its float32 key).
        /// Drives which scratch buffers <see cref="SortInPlace"/>/<see cref="ArgSortInto"/> allocate:
        /// the old code allocated BOTH the u32 and u64 double-buffers (plus the histogram) for every
        /// dtype, so a 10M-element int32 sort allocated 160 MB of u64 buffers it never touched — and
        /// the CLR zero-fills new arrays, ≈16 ms of pure memset wasted per large sort.
        /// </summary>
        private static int KeyWidth(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Double => 8,
            NPTypeCode.Half or NPTypeCode.Complex or NPTypeCode.Decimal => 0,
            _ => 4,
        };

        private static void SortInPlace(NDArray target, int axis)
        {
            int N = (int)target.shape[axis];
            if (N <= 1 || target.size == 0) return;

            var tc = target.GetTypeCode;
            int elsize = tc.SizeOf();
            long axisStride = (long)target.Shape.strides[axis] * elsize; // byte stride along the sort axis

            // scratch (sized to the line length, reused across all lines) — only the width the
            // chosen kernel touches; unused buffers stay Array.Empty (fix to a null pointer).
            int w = KeyWidth(tc);
            var ctx = new LineCtx { n = N, inStride = axisStride, outStride = axisStride };
            var k32 = w == 4 ? new uint[N] : Array.Empty<uint>();
            var t32 = w == 4 ? new uint[N] : Array.Empty<uint>();
            var k64 = w == 8 ? new ulong[N] : Array.Empty<ulong>();
            var t64 = w == 8 ? new ulong[N] : Array.Empty<ulong>();
            var cnt = w == 0 ? Array.Empty<int>() : new int[256];

            fixed (uint* pk = k32, pt = t32)
            fixed (ulong* pk6 = k64, pt6 = t64)
            fixed (int* pc = cnt)
            {
                ctx.k32 = pk; ctx.t32 = pt; ctx.k64 = pk6; ctx.t64 = pt6; ctx.count = pc;
                NpyInnerLoopFunc kern = GetSortKernel(tc);
                DriveAllButAxis(new[] { target }, new[] { NpyIterPerOpFlags.READWRITE }, axis, kern, &ctx);
            }
        }

        private static void ArgSortInto(NDArray src, NDArray dst, int axis)
        {
            int N = (int)src.shape[axis];
            var tc = src.GetTypeCode;
            int elsize = tc.SizeOf();
            var ctx = new LineCtx
            {
                n = N,
                inStride = (long)src.Shape.strides[axis] * elsize,
                outStride = (long)dst.Shape.strides[axis] * sizeof(long),
            };
            if (N == 0) return;

            // Only the key width the kernel touches; the index column (idx/it) is radix-only, so the
            // scalar BCL argsort path (Half/Complex/Decimal, w==0) allocates none of it.
            int w = KeyWidth(tc);
            var k32 = w == 4 ? new uint[N] : Array.Empty<uint>();
            var t32 = w == 4 ? new uint[N] : Array.Empty<uint>();
            var k64 = w == 8 ? new ulong[N] : Array.Empty<ulong>();
            var t64 = w == 8 ? new ulong[N] : Array.Empty<ulong>();
            var idx = w == 0 ? Array.Empty<long>() : new long[N];
            var it = w == 0 ? Array.Empty<long>() : new long[N];
            var cnt = w == 0 ? Array.Empty<int>() : new int[256];

            fixed (uint* pk = k32, pt = t32)
            fixed (ulong* pk6 = k64, pt6 = t64)
            fixed (long* pi = idx, pit = it)
            fixed (int* pc = cnt)
            {
                ctx.k32 = pk; ctx.t32 = pt; ctx.k64 = pk6; ctx.t64 = pt6; ctx.idx = pi; ctx.it = pit; ctx.count = pc;
                NpyInnerLoopFunc kern = GetArgSortKernel(tc);
                DriveAllButAxis(new[] { src, dst }, new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY }, axis, kern, &ctx);
            }
        }

        /// <summary>NumPy IterAllButAxis: iterate every axis EXCEPT <paramref name="axis"/> (dropped via
        /// op_axes so it can't coalesce); the kernel receives each operand's line start per call.</summary>
        private static void DriveAllButAxis(NDArray[] ops, NpyIterPerOpFlags[] flags, int axis, NpyInnerLoopFunc kern, void* aux)
        {
            int ndim = ops[0].ndim;

            // 1-D input drops its only axis -> a 0-dimensional all-but-axis iterator. Our NpyIter
            // mis-drives that degenerate shape as `size` single-element iterations, and because each
            // line kernel re-sorts the whole line (it walks by LineCtx.n, ignoring the per-call count)
            // the total work is N x O(N) = O(N^2). Promote every operand to a (1, N) memory-sharing
            // view (expand_dims aliases storage, so an in-place sort still mutates the original, and a
            // leading size-1 axis leaves the data-axis stride -> LineCtx in/out strides unchanged): now
            // exactly one axis (size 1) is kept, so the iterator makes exactly one line kernel call.
            if (ndim == 1)
            {
                var promoted = new NDArray[ops.Length];
                for (int i = 0; i < ops.Length; i++) promoted[i] = np.expand_dims(ops[i], 0);
                ops = promoted;
                axis = 1;
                ndim = 2;
            }

            var kept = new int[ndim - 1];
            for (int d = 0, w = 0; d < ndim; d++) if (d != axis) kept[w++] = d;
            var opAxes = new int[ops.Length][];
            for (int i = 0; i < ops.Length; i++) opAxes[i] = kept;

            var iter = NpyIterRef.AdvancedNew(ops.Length, ops, NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, flags, null, ndim - 1, opAxes);
            try { iter.ForEach(kern, aux); }
            finally { iter.Dispose(); }
        }

        // line-sort context carried via NpyIter auxdata (no per-call captures/allocations)
        private struct LineCtx
        {
            public uint* k32; public uint* t32; public ulong* k64; public ulong* t64;
            public long* idx; public long* it; public int* count;
            public long inStride; public long outStride; public int n;
        }

        // ============================ generic line kernels ============================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SortLine32<T, K>(byte* line, LineCtx* c) where T : unmanaged where K : struct, IKey32<T>
        {
            K k = default; int n = c->n; long s = c->inStride;
            for (int i = 0; i < n; i++) c->k32[i] = k.To(*(T*)(line + i * s));
            uint* r = RadixSort.SortU32(c->k32, c->t32, n, k.Bytes, c->count);
            for (int i = 0; i < n; i++) *(T*)(line + i * s) = k.From(r[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SortLine64<T, K>(byte* line, LineCtx* c) where T : unmanaged where K : struct, IKey64<T>
        {
            K k = default; int n = c->n; long s = c->inStride;
            for (int i = 0; i < n; i++) c->k64[i] = k.To(*(T*)(line + i * s));
            ulong* r = RadixSort.SortU64(c->k64, c->t64, n, c->count);
            for (int i = 0; i < n; i++) *(T*)(line + i * s) = k.From(r[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ArgLine32<T, K>(byte* inLine, byte* outLine, LineCtx* c) where T : unmanaged where K : struct, IKey32<T>
        {
            K k = default; int n = c->n; long si = c->inStride, so = c->outStride;
            for (int i = 0; i < n; i++) { c->k32[i] = k.To(*(T*)(inLine + i * si)); c->idx[i] = i; }
            long* r = RadixSort.ArgSortU32(c->k32, c->t32, c->idx, c->it, n, k.Bytes, c->count);
            for (int i = 0; i < n; i++) *(long*)(outLine + i * so) = r[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ArgLine64<T, K>(byte* inLine, byte* outLine, LineCtx* c) where T : unmanaged where K : struct, IKey64<T>
        {
            K k = default; int n = c->n; long si = c->inStride, so = c->outStride;
            for (int i = 0; i < n; i++) { c->k64[i] = k.To(*(T*)(inLine + i * si)); c->idx[i] = i; }
            long* r = RadixSort.ArgSortU64(c->k64, c->t64, c->idx, c->it, n, c->count);
            for (int i = 0; i < n; i++) *(long*)(outLine + i * so) = r[i];
        }

        // ----- float radix with NaN-last partition -----
        private static void SortLineF32(byte* line, LineCtx* c)
        {
            int n = c->n; long s = c->inStride; int m = 0;
            for (int i = 0; i < n; i++) { float v = *(float*)(line + i * s); if (!float.IsNaN(v)) c->k32[m++] = FKey32(v); }
            uint* r = RadixSort.SortU32(c->k32, c->t32, m, 4, c->count);
            for (int i = 0; i < m; i++) *(float*)(line + i * s) = FVal32(r[i]);
            for (int i = m; i < n; i++) *(float*)(line + i * s) = float.NaN;
        }
        private static void SortLineF64(byte* line, LineCtx* c)
        {
            int n = c->n; long s = c->inStride; int m = 0;
            for (int i = 0; i < n; i++) { double v = *(double*)(line + i * s); if (!double.IsNaN(v)) c->k64[m++] = FKey64(v); }
            ulong* r = RadixSort.SortU64(c->k64, c->t64, m, c->count);
            for (int i = 0; i < m; i++) *(double*)(line + i * s) = FVal64(r[i]);
            for (int i = m; i < n; i++) *(double*)(line + i * s) = double.NaN;
        }
        private static void ArgLineF32(byte* inLine, byte* outLine, LineCtx* c)
        {
            int n = c->n; long si = c->inStride, so = c->outStride; int m = 0;
            for (int i = 0; i < n; i++) { float v = *(float*)(inLine + i * si); if (!float.IsNaN(v)) { c->k32[m] = FKey32(v); c->idx[m] = i; m++; } }
            long* r = RadixSort.ArgSortU32(c->k32, c->t32, c->idx, c->it, m, 4, c->count);
            for (int i = 0; i < m; i++) *(long*)(outLine + i * so) = r[i];
            int q = m; for (int i = 0; i < n; i++) if (float.IsNaN(*(float*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }
        private static void ArgLineF64(byte* inLine, byte* outLine, LineCtx* c)
        {
            int n = c->n; long si = c->inStride, so = c->outStride; int m = 0;
            for (int i = 0; i < n; i++) { double v = *(double*)(inLine + i * si); if (!double.IsNaN(v)) { c->k64[m] = FKey64(v); c->idx[m] = i; m++; } }
            long* r = RadixSort.ArgSortU64(c->k64, c->t64, c->idx, c->it, m, c->count);
            for (int i = 0; i < m; i++) *(long*)(outLine + i * so) = r[i];
            int q = m; for (int i = 0; i < n; i++) if (double.IsNaN(*(double*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] private static uint FKey32(float v) { uint b = BitConverter.SingleToUInt32Bits(v); return b ^ ((uint)((int)b >> 31) | 0x80000000u); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] private static float FVal32(uint k) { uint b = k ^ (((k >> 31) - 1) | 0x80000000u); return BitConverter.UInt32BitsToSingle(b); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] private static ulong FKey64(double v) { ulong b = BitConverter.DoubleToUInt64Bits(v); return b ^ ((ulong)((long)b >> 63) | 0x8000000000000000UL); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] private static double FVal64(ulong k) { ulong b = k ^ (((k >> 63) - 1) | 0x8000000000000000UL); return BitConverter.UInt64BitsToDouble(b); }

        // ----- scalar BCL introsort + exact NumPy comparators (Half / Complex / Decimal) -----
        private static void SortLineScalar<T, TCmp>(byte* line, LineCtx* c) where T : unmanaged where TCmp : struct, IComparer<T>
        {
            int n = c->n; long s = c->inStride;
            Span<T> buf = n <= 1024 ? stackalloc T[n] : new T[n];
            for (int i = 0; i < n; i++) buf[i] = *(T*)(line + i * s);
            buf.Sort(default(TCmp));
            for (int i = 0; i < n; i++) *(T*)(line + i * s) = buf[i];
        }
        private static void ArgLineScalar<T, TCmp>(byte* inLine, byte* outLine, LineCtx* c) where T : unmanaged where TCmp : struct, IComparer<T>
        {
            int n = c->n; long si = c->inStride, so = c->outStride;
            var buf = new T[n]; var ix = new long[n];
            for (int i = 0; i < n; i++) { buf[i] = *(T*)(inLine + i * si); ix[i] = i; }
            Array.Sort(ix, new IndexedCmp<T, TCmp>(buf));
            for (int i = 0; i < n; i++) *(long*)(outLine + i * so) = ix[i];
        }

        // stable indirect comparer for scalar argsort (ties -> ascending index)
        private sealed class IndexedCmp<T, TCmp> : IComparer<long> where TCmp : struct, IComparer<T>
        {
            private readonly T[] _v; public IndexedCmp(T[] v) { _v = v; }
            public int Compare(long i, long j) { int r = default(TCmp).Compare(_v[i], _v[j]); return r != 0 ? r : i.CompareTo(j); }
        }

        private readonly struct HalfCmp : IComparer<Half>
        {
            public int Compare(Half a, Half b)
            {
                float x = (float)a, y = (float)b;
                if (x < y) return -1; if (x > y) return 1;
                bool xn = float.IsNaN(x), yn = float.IsNaN(y);
                if (xn && yn) return 0; if (xn) return 1; if (yn) return -1; return 0;
            }
        }
        private readonly struct DecimalCmp : IComparer<decimal> { public int Compare(decimal a, decimal b) => a.CompareTo(b); }
        private readonly struct ComplexCmp : IComparer<Complex>
        {
            // NumPy CDOUBLE_LT (npysort_common.h): lexicographic real-then-imag, any-NaN-part sorts last.
            public int Compare(Complex a, Complex b) { if (Lt(a, b)) return -1; if (Lt(b, a)) return 1; return 0; }
            private static bool Lt(Complex a, Complex b)
            {
                double ar = a.Real, ai = a.Imaginary, br = b.Real, bi = b.Imaginary;
                if (ar < br) return ai == ai || bi != bi;
                if (ar > br) return bi != bi && ai == ai;
                if (ar == br || (ar != ar && br != br)) return ai < bi || (bi != bi && ai == ai);
                return br != br;
            }
        }

        // ============================ dtype dispatch (one type-switch each) ============================

        private static NpyInnerLoopFunc GetSortKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => SortLine32<byte, KBool>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => SortLine32<byte, KU8>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => SortLine32<sbyte, KI8>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => SortLine32<short, KI16>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => SortLine32<ushort, KU16>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => SortLine32<char, KChar>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => SortLine32<int, KI32>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => SortLine32<uint, KU32>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => SortLine64<long, KI64>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => SortLine64<ulong, KU64>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => SortLineF32((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => SortLineF64((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => SortLineScalar<Half, HalfCmp>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => SortLineScalar<Complex, ComplexCmp>((byte*)p[0], (LineCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => SortLineScalar<decimal, DecimalCmp>((byte*)p[0], (LineCtx*)a),
            _ => throw new NotSupportedException($"sort not supported for dtype {tc}"),
        };

        private static NpyInnerLoopFunc GetArgSortKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => ArgLine32<byte, KBool>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => ArgLine32<byte, KU8>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => ArgLine32<sbyte, KI8>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => ArgLine32<short, KI16>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => ArgLine32<ushort, KU16>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => ArgLine32<char, KChar>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => ArgLine32<int, KI32>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => ArgLine32<uint, KU32>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => ArgLine64<long, KI64>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => ArgLine64<ulong, KU64>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => ArgLineF32((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => ArgLineF64((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => ArgLineScalar<Half, HalfCmp>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => ArgLineScalar<Complex, ComplexCmp>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => ArgLineScalar<decimal, DecimalCmp>((byte*)p[0], (byte*)p[1], (LineCtx*)a),
            _ => throw new NotSupportedException($"argsort not supported for dtype {tc}"),
        };
    }
}
