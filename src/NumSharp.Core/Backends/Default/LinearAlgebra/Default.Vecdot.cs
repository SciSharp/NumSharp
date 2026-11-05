using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    ///     np.vecdot's loops: NumPy's <c>@TYPE@_vecdot</c> (matmul.c.src) calls the loop dtype's dot function once per
    ///     output, over the two core runs at the strides its gufunc iterator hands it. For float32, float64 and complex128
    ///     those dot functions sum in BLAS order (sdot / ddot / zdotc), which an elementwise product and its reduction do
    ///     not reproduce, so they compute here with NumPy's models (<see cref="NDDot"/>).
    /// </summary>
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     <para>float32, float64 and complex128 loops run NumPy's <c>FLOAT_vecdot</c> / <c>DOUBLE_vecdot</c> /
        ///     <c>CDOUBLE_vecdot</c>: each output is <c>FLOAT_dot</c> / <c>DOUBLE_dot</c> (<see cref="NDDot.FloatDotFor"/>,
        ///     <see cref="NDDot.DoubleDotFor"/>) or the conjugating <c>CDOUBLE_dotc</c> (<see cref="NDDot.ComplexDotcFor"/>)
        ///     over the core runs at the strides NumPy's gufunc gives them: an operand of the loop dtype at its own core
        ///     stride (a reversed or broadcast core takes the dot function's own loop, as in NumPy), an operand cast to the
        ///     loop dtype at unit stride (the iterator's copy lays an input's core axes out innermost,
        ///     <c>npyiter_new_temp_array</c>), and stride 0 for a core of one element (NumPy forces it, so the own loop runs:
        ///     a complex infinite imaginary part leaves the real part alone there). The outputs are written in C order over
        ///     the broadcast loop axes, which the gufunc member relabels into NumPy's layout.</para>
        ///     <para>Every other loop takes the base composition, whose values are NumPy's: the integers' dot functions
        ///     wrap as the in-dtype product and sum do, bool's ORs the ANDs, float16 runs HALF_dot through the matmul
        ///     kernel. So does a direct engine call the gufunc checks would refuse (0-d operands, unequal core lengths,
        ///     loop axes that do not broadcast), which keeps the composition's answer or error.</para>
        /// </remarks>
        /// <exception cref="IncorrectShapeException">Through the base composition only: loop axes that do not broadcast
        ///     (a direct call np.vecdot's checks did not validate).</exception>
        [NDScoped] // engine boundary: an operand's cast to the loop dtype is reclaimed, and both operands stay rooted
        public override NDArray Vecdot(NDArray x1, NDArray x2)
        {
            // The loop dtype NumPy's type resolution picks, as np.vecdot's gufunc member resolves it; only the NumPy
            // numeric dtypes (no decimal, char or text, whose answers the composition owns) take part.
            if (!IsVecdotOperand(x1.typecode) || !IsVecdotOperand(x2.typecode) || x1.ndim == 0 || x2.ndim == 0)
                return base.Vecdot(x1, x2);
            var loop = x1.typecode == x2.typecode
                ? x1.typecode
                : DTypePromotion.PromoteTypes(x1.dtype, x2.dtype).GetTypeCode();
            if (loop != NPTypeCode.Single && loop != NPTypeCode.Double && loop != NPTypeCode.Complex)
                return base.Vecdot(x1, x2);

            long n = x1.Shape.dimensions[x1.ndim - 1];
            if (x2.Shape.dimensions[x2.ndim - 1] != n)
                return base.Vecdot(x1, x2);

            // The broadcast loop axes, right-aligned as the gufunc iterator aligns them (an axis an operand lacks, or
            // has with length 1, broadcasts). Validated before the Blas hook so a refused shape keeps the base's error.
            int l1 = x1.ndim - 1, l2 = x2.ndim - 1, nd = Math.Max(l1, l2);
            var dims = new long[nd];
            for (int i = 0; i < nd; i++)
            {
                long d1 = i >= nd - l1 ? x1.Shape.dimensions[i - (nd - l1)] : 1;
                long d2 = i >= nd - l2 ? x2.Shape.dimensions[i - (nd - l2)] : 1;
                if (d1 != d2 && d1 != 1 && d2 != 1)
                    return base.Vecdot(x1, x2);
                dims[i] = d1 == 1 ? d2 : d1;
            }

            var blas = Blas;
            if (blas != null && blas.TryVecdot(x1, x2, out var accelerated))
                return accelerated;

            // An empty core sums nothing: every dot function writes +0 (complex (+0, +0)), so a zeroed result is the
            // answer and no operand is read or cast.
            var result = NewArray(loop, Shape.Adopt(dims), n == 0);
            if (result.size == 0 || n == 0)
                return result;

            // A core of one element: NumPy forces both core strides to 0 ("so that broadcasting works right"), so every
            // dot function takes its own loop, whose one term is an elementwise operation (VecdotOneWalk).
            if (n == 1)
            {
                VecdotOne(x1, x2, loop, result, dims);
                return result;
            }

            // NumPy's iterator copies an operand that needs a cast into the loop dtype (NPY_ITER_COPY) with its core axis
            // innermost, so the dot function reads that copy at unit stride whatever the operand's own layout (probed
            // 2.4.2: a float32 A.T against float64 rows gives the C copy's bits, not the F copy's). The cast runs here a
            // block of consecutive runs at a time into cache-sized scratch, read there at unit stride: the same values
            // at the same stride, without writing and re-reading a whole converted array.
            if (x1.typecode != loop || x2.typecode != loop)
            {
                VecdotBlockCast(x1, x2, loop, result, n, dims);
                return result;
            }

            // The per-axis element strides of the loop axes (0 where an operand broadcasts), and the core strides.
            var s1 = LoopStrides(x1, nd);
            var s2 = LoopStrides(x2, nd);
            long c1 = x1.Shape.strides[x1.ndim - 1];
            long c2 = x2.Shape.strides[x2.ndim - 1];
            unsafe
            {
                // A stride cblas is not handed (zero or negative) runs the dot function's own loop: four outputs' chains
                // side by side along the innermost loop axis (the pair rows); every other class one dot per output.
                bool own = !(c1 > 0 && c2 > 0);
                switch (loop)
                {
                    case NPTypeCode.Single:
                        VecdotWalk((float*)x1.Address + x1.Shape.offset, c1, (float*)x2.Address + x2.Shape.offset, c2,
                            (float*)result.Address, n, dims, s1, s2, NDDot.FloatDotFor(c1, c2),
                            own ? &NDDot.FloatDotOwnPairRow : null);
                        break;
                    case NPTypeCode.Double:
                        VecdotWalk((double*)x1.Address + x1.Shape.offset, c1, (double*)x2.Address + x2.Shape.offset, c2,
                            (double*)result.Address, n, dims, s1, s2, NDDot.DoubleDotFor(c1, c2),
                            own ? &NDDot.DoubleDotOwnPairRow : null);
                        break;
                    default:
                        VecdotWalk((Complex*)x1.Address + x1.Shape.offset, c1, (Complex*)x2.Address + x2.Shape.offset, c2,
                            (Complex*)result.Address, n, dims, s1, s2, NDDot.ComplexDotcFor(c1, c2),
                            own ? &NDDot.ComplexDotcOwnPairRow : null);
                        break;
                }
            }

            return result;
        }

        /// <summary>
        ///     Whether an operand's dtype is one of NumPy's numeric dtypes, the only ones whose np.vecdot loops this
        ///     engine computes through dot functions (after promotion); decimal, char and text keep the composition.
        /// </summary>
        /// <param name="tc">The operand's dtype.</param>
        /// <returns>True for bool, the integers, float16, float32, float64 and complex128.</returns>
        private static bool IsVecdotOperand(NPTypeCode tc)
            => tc is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Half
                or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex;

        /// <summary>
        ///     An operand's element strides over the broadcast loop axes: its own stride on an axis it has with more than
        ///     one element, 0 on an axis it lacks (right alignment) or has with one element, where it broadcasts.
        /// </summary>
        /// <param name="x">The operand, its core axis last.</param>
        /// <param name="nd">The number of broadcast loop axes.</param>
        /// <returns>One stride per loop axis, outermost first.</returns>
        private static long[] LoopStrides(NDArray x, int nd)
        {
            int lx = x.ndim - 1;
            var s = new long[nd];
            for (int i = nd - lx; i < nd; i++)
            {
                int k = i - (nd - lx);
                s[i] = x.Shape.dimensions[k] == 1 ? 0 : x.Shape.strides[k];
            }
            return s;
        }

        /// <summary>
        ///     NumPy's <c>@TYPE@_vecdot</c> outer loop: one dot per output, the outputs in C order over the broadcast loop
        ///     axes (the last one innermost, walked by pointer steps; the others by an odometer).
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <param name="p1">The first operand's first core run.</param>
        /// <param name="c1">Its core element stride (the dot function's <c>is1</c>).</param>
        /// <param name="p2">The second operand's first core run.</param>
        /// <param name="c2">Its core element stride (<c>is2</c>).</param>
        /// <param name="o">The first output of a fresh C-ordered result holding every output.</param>
        /// <param name="n">The core length.</param>
        /// <param name="dims">The broadcast loop axes (empty for one output).</param>
        /// <param name="s1">The first operand's element strides over them (0 where it broadcasts).</param>
        /// <param name="s2">The second operand's element strides over them.</param>
        /// <param name="dot">The dot function for the core strides (chosen once: every output shares them).</param>
        /// <param name="pairRow">The own loop's pair row (<see cref="NDDot.FloatDotOwnPairRow"/> and its kin), which
        ///     computes a whole innermost axis with four outputs' chains side by side; null to call
        ///     <paramref name="dot"/> per output.</param>
        /// <remarks>The caller allocated <paramref name="o"/> with every loop axis non-empty, so each output is written
        ///     once; the dot functions read only the core runs, which the operands' views bound.</remarks>
        private static unsafe void VecdotWalk<T>(T* p1, long c1, T* p2, long c2, T* o, long n, long[] dims, long[] s1,
            long[] s2, delegate*<T*, long, T*, long, long, T> dot,
            delegate*<T*, long, long, T*, long, long, T*, long, long, void> pairRow) where T : unmanaged
        {
            int nd = dims.Length;
            if (nd == 0)
            {
                *o = dot(p1, c1, p2, c2, n);
                return;
            }

            long inner = dims[nd - 1], i1 = s1[nd - 1], i2 = s2[nd - 1];
            long rows = 1;
            for (int k = 0; k < nd - 1; k++)
                rows *= dims[k];
            // The odometer over the outer loop axes: idx[k] counts axis k's position; stackalloc is zeroed here (the
            // method has no SkipLocalsInit).
            long* idx = stackalloc long[nd];
            for (long r = 0; r < rows; r++)
            {
                if (pairRow != null)
                {
                    pairRow(p1, c1, i1, p2, c2, i2, o, inner, n);
                    o += inner;
                }
                else
                {
                    T* q1 = p1, q2 = p2;
                    for (long j = 0; j < inner; j++, q1 += i1, q2 += i2)
                        *o++ = dot(q1, c1, q2, c2, n);
                }

                for (int k = nd - 2; k >= 0; k--)
                {
                    p1 += s1[k];
                    p2 += s2[k];
                    if (++idx[k] < dims[k])
                        break;
                    // Axis k wrapped: rewind it and carry into the next outer axis.
                    p1 -= s1[k] * dims[k];
                    p2 -= s2[k] * dims[k];
                    idx[k] = 0;
                }
            }
        }

        // --------------------------------------------------------------------------------------------------------------
        //  A pair that needs a cast: blocks of runs cast into scratch
        // --------------------------------------------------------------------------------------------------------------

        /// <summary>The scratch budget of one cast operand's block, in bytes: well inside L1 with the other operand's.</summary>
        private const long VecdotBlockBytes = 16 * 1024;

        /// <summary>
        ///     np.vecdot of a pair in which an operand needs a cast to the loop dtype, every output into
        ///     <paramref name="result"/>: <see cref="VecdotBlockWalk{T}"/> instantiated for the loop dtype.
        /// </summary>
        /// <param name="x1">The first operand, its core axis last.</param>
        /// <param name="x2">The second operand, its core axis last (the same core length).</param>
        /// <param name="loop">The loop dtype (float32, float64 or complex128).</param>
        /// <param name="result">The fresh C-ordered result, non-empty.</param>
        /// <param name="n">The core length (at least 2: a core of one element takes <see cref="VecdotOne"/>).</param>
        /// <param name="dims">The broadcast loop axes.</param>
        private static unsafe void VecdotBlockCast(NDArray x1, NDArray x2, NPTypeCode loop, NDArray result, long n,
            long[] dims)
        {
            // A cast run sits in scratch at unit stride and an operand of the loop dtype is read at its own core stride:
            // the dot function for those strides.
            long e1 = x1.typecode != loop ? 1 : x1.Shape.strides[x1.ndim - 1];
            long e2 = x2.typecode != loop ? 1 : x2.Shape.strides[x2.ndim - 1];
            bool own = !(e1 > 0 && e2 > 0);
            switch (loop)
            {
                case NPTypeCode.Single:
                    VecdotBlockWalk(x1, x2, loop, (float*)result.Address, n, dims, e1, e2, NDDot.FloatDotFor(e1, e2),
                        own ? &NDDot.FloatDotOwnPairRow : null);
                    break;
                case NPTypeCode.Double:
                    VecdotBlockWalk(x1, x2, loop, (double*)result.Address, n, dims, e1, e2, NDDot.DoubleDotFor(e1, e2),
                        own ? &NDDot.DoubleDotOwnPairRow : null);
                    break;
                default:
                    VecdotBlockWalk(x1, x2, loop, (Complex*)result.Address, n, dims, e1, e2, NDDot.ComplexDotcFor(e1, e2),
                        own ? &NDDot.ComplexDotcOwnPairRow : null);
                    break;
            }
        }

        /// <summary>
        ///     <see cref="VecdotWalk{T}"/> for a pair in which an operand needs a cast to the loop dtype: along the
        ///     innermost loop axis the outputs go a block at a time, each cast operand's block of runs converted by one
        ///     cast-kernel call into scratch (at most <see cref="VecdotBlockBytes"/>, or one run when a run is longer) and
        ///     read there at unit stride - the stride of NumPy's copy - while an operand of the loop dtype is read in
        ///     place at its own strides.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <param name="x1">The first operand, its core axis last.</param>
        /// <param name="x2">The second operand, its core axis last (the same core length).</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="o">The first output of a fresh C-ordered result holding every output.</param>
        /// <param name="n">The core length (at least 2).</param>
        /// <param name="dims">The broadcast loop axes, none empty (empty for one output).</param>
        /// <param name="e1">The dot function's stride for x1's runs (<see cref="VecdotBlockCast"/>).</param>
        /// <param name="e2">The dot function's stride for x2's runs.</param>
        /// <param name="dot">The dot function for those strides.</param>
        /// <param name="pairRow">The own loop's pair row for a block (the other operand's core stride is zero or
        ///     negative), or null to call <paramref name="dot"/> per output.</param>
        /// <remarks>Why: casting each whole operand first writes and re-reads a converted copy, which costs more than the
        ///     dots on arrays beyond the caches (measured 0.73x of the former composition for int32 · float32 at 1000 x
        ///     1000); casting each run alone pays a kernel call per output, too much for short cores (0.55x of the whole
        ///     cast at 8 terms). A block of L1-sized scratch keeps both costs down.</remarks>
        private static unsafe void VecdotBlockWalk<T>(NDArray x1, NDArray x2, NPTypeCode loop, T* o, long n,
            long[] dims, long e1, long e2, delegate*<T*, long, T*, long, long, T> dot,
            delegate*<T*, long, long, T*, long, long, T*, long, long, void> pairRow) where T : unmanaged
        {
            int nd = dims.Length;
            long inner = nd == 0 ? 1 : dims[nd - 1];
            // Outputs per block: as many runs as the scratch budget holds, at least one (a run longer than the budget
            // gets scratch of its own length), at most the innermost axis.
            long block = Math.Max(1, Math.Min(inner, VecdotBlockBytes / (n * sizeof(T))));
            var s1 = LoopStrides(x1, nd);
            var s2 = LoopStrides(x2, nd);
            long i1 = nd == 0 ? 0 : s1[nd - 1], i2 = nd == 0 ? 0 : s2[nd - 1];
            // Both sources start empty (nothing to return) and take their scratch inside the try, so a failing second
            // source still returns the first one's.
            BlockSource<T> r1 = default, r2 = default;
            try
            {
                r1 = new BlockSource<T>(x1, loop, n, block, i1);
                r2 = new BlockSource<T>(x2, loop, n, block, i2);
                long rows = 1;
                for (int k = 0; k < nd - 1; k++)
                    rows *= dims[k];
                long off1 = 0, off2 = 0;   // the current row's first run, in each operand's own elements
                long* idx = stackalloc long[Math.Max(nd, 1)];   // zeroed: no SkipLocalsInit here
                for (long r = 0; r < rows; r++)
                {
                    for (long j0 = 0; j0 < inner; j0 += block)
                    {
                        long count = Math.Min(block, inner - j0);
                        T* b1 = r1.Block(off1 + j0 * i1, count, out long step1);
                        T* b2 = r2.Block(off2 + j0 * i2, count, out long step2);
                        if (pairRow != null)
                        {
                            pairRow(b1, e1, step1, b2, e2, step2, o, count, n);
                            o += count;
                        }
                        else
                        {
                            for (long j = 0; j < count; j++, b1 += step1, b2 += step2)
                                *o++ = dot(b1, e1, b2, e2, n);
                        }
                    }

                    for (int k = nd - 2; k >= 0; k--)
                    {
                        off1 += s1[k];
                        off2 += s2[k];
                        if (++idx[k] < dims[k])
                            break;
                        // Axis k wrapped: rewind it and carry into the next outer axis.
                        off1 -= s1[k] * dims[k];
                        off2 -= s2[k] * dims[k];
                        idx[k] = 0;
                    }
                }
            }
            finally
            {
                // The scratch is pooled; returned even when a cast kernel throws.
                r1.Release();
                r2.Release();
            }
        }

        /// <summary>
        ///     Where <see cref="VecdotBlockWalk{T}"/> and <see cref="VecdotOneWalk{T, TOne}"/> read one operand's runs: in
        ///     place (the operand has the loop dtype), or cast a block at a time into pooled scratch - by the cast kernels
        ///     where they convert the pair (the contiguous one for a contiguous block), else by a typed managed
        ///     conversion (<see cref="ConvertBlock{TS, T}"/>: bool's strided runs, every dtype to complex128). The last
        ///     block cast is remembered, so a block a broadcast loop axis repeats (and a run the innermost axis
        ///     broadcasts) is cast once.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        private unsafe struct BlockSource<T> where T : unmanaged
        {
            /// <summary>The operand's element 0 (its own dtype), offset included.</summary>
            private readonly byte* _base;
            /// <summary>The operand's item size in bytes.</summary>
            private readonly long _itemSize;
            /// <summary>The operand's core element stride.</summary>
            private readonly long _coreStride;
            /// <summary>The operand's element stride between consecutive runs along the innermost loop axis.</summary>
            private readonly long _runStride;
            /// <summary>The core length.</summary>
            private readonly long _n;
            /// <summary>The scratch (null when the operand is read in place).</summary>
            private readonly T* _scratch;
            /// <summary>The scratch's byte size, for its return to the pool.</summary>
            private readonly long _scratchBytes;
            /// <summary>The contiguous cast kernel, for a block that is contiguous in the operand (null otherwise).</summary>
            private readonly DirectILKernelGenerator.CastKernel _contiguous;
            /// <summary>The strided cast kernel (any layout), or null.</summary>
            private readonly DirectILKernelGenerator.StridedCastKernel _strided;
            /// <summary>The managed conversion, for a block neither kernel takes (null when a kernel always does).</summary>
            private readonly delegate*<byte*, long, long, long, long, long, T*, void> _managed;
            /// <summary>The first run of the block in scratch (-1 before the first cast).</summary>
            private long _cachedOffset;
            /// <summary>The number of runs of the block in scratch.</summary>
            private long _cachedCount;

            /// <summary>Prepares an operand's runs.</summary>
            /// <param name="x">The operand, its core axis last.</param>
            /// <param name="loop">The loop dtype.</param>
            /// <param name="n">The core length.</param>
            /// <param name="block">The most runs a block holds.</param>
            /// <param name="runStride">The operand's element stride between consecutive outputs' runs along the innermost
            ///     loop axis (0 where it broadcasts).</param>
            /// <exception cref="NotSupportedException">The operand's dtype is not one of NumPy's numeric dtypes (excluded
            ///     by np.vecdot's dispatch).</exception>
            public BlockSource(NDArray x, NPTypeCode loop, long n, long block, long runStride)
            {
                _itemSize = x.dtypesize;
                _base = (byte*)x.Address + x.Shape.offset * _itemSize;
                _coreStride = x.Shape.strides[x.ndim - 1];
                _runStride = runStride;
                _n = n;
                _cachedOffset = -1;
                _cachedCount = 0;
                if (x.typecode == loop)
                {
                    _scratch = null;
                    _scratchBytes = 0;
                    _contiguous = null;
                    _strided = null;
                    _managed = null;
                    return;
                }

                // A block is contiguous in the operand when its runs are (unit core stride, or a single element) and follow
                // each other, or when it is a single run (one output per block, or a run the innermost axis broadcasts).
                bool contiguousBlocks = (_coreStride == 1 || n == 1) && (runStride == n || runStride == 0 || block == 1);
                _contiguous = contiguousBlocks ? DirectILKernelGenerator.TryGetCastKernel(x.typecode, loop) : null;
                _strided = _contiguous == null ? DirectILKernelGenerator.TryGetStridedCastKernel(x.typecode, loop) : null;
                _managed = _contiguous == null && _strided == null ? ManagedConversion<T>(x.typecode) : null;
                // A broadcast run (runStride 0) is one run however many outputs read it.
                _scratchBytes = (runStride == 0 ? 1 : block) * n * sizeof(T);
                _scratch = (T*)SizeBucketedBufferPool.Take(_scratchBytes);
            }

            /// <summary>
            ///     The block of <paramref name="count"/> runs starting at element offset <paramref name="offset"/>: in
            ///     place, or its cast in scratch.
            /// </summary>
            /// <param name="offset">The first run's first element, in the operand's elements from element 0.</param>
            /// <param name="count">The number of runs (consecutive outputs along the innermost loop axis).</param>
            /// <param name="step">Set to the element distance from one run to the next where they are read: the run
            ///     stride in place, the core length in scratch, 0 for a broadcast run.</param>
            /// <returns>The first run's first element in the loop dtype.</returns>
            public T* Block(long offset, long count, out long step)
            {
                if (_scratch == null)
                {
                    step = _runStride;
                    return (T*)_base + offset;
                }

                if (_runStride == 0)
                    count = 1;   // every output of the block reads the same run
                if (offset != _cachedOffset || count != _cachedCount)
                {
                    byte* src = _base + offset * _itemSize;
                    if (_contiguous != null)
                    {
                        _contiguous(src, _scratch, count * _n);
                    }
                    else if (_managed != null)
                    {
                        _managed(src, _itemSize, _runStride, _coreStride, count, _n, _scratch);
                    }
                    else
                    {
                        // The block as a (count, n) array: runs at the run stride, elements at the core stride, written
                        // C-ordered into scratch.
                        long* srcStrides = stackalloc long[2] { _runStride, _coreStride };
                        long* dstStrides = stackalloc long[2] { _n, 1 };
                        long* shape = stackalloc long[2] { count, _n };
                        _strided(src, _scratch, srcStrides, dstStrides, shape, 2);
                    }

                    _cachedOffset = offset;
                    _cachedCount = count;
                }

                step = _runStride == 0 ? 0 : _n;
                return _scratch;
            }

            /// <summary>Returns the scratch to the pool (nothing for an operand read in place, or a source never
            ///     constructed: the default value holds no scratch).</summary>
            public void Release()
            {
                if (_scratch != null)
                    SizeBucketedBufferPool.Return((IntPtr)_scratch, _scratchBytes);
            }
        }

        // --------------------------------------------------------------------------------------------------------------
        //  The managed conversion: the pairs the cast kernels do not take
        // --------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     One source dtype's element read as a float32 or a float64 the way NumPy's cast to that dtype converts it:
        ///     one C conversion, rounded once (int64 to float32 directly, not through float64, which could round twice).
        ///     A complex128 target takes the float64 as its real part and +0 as its imaginary part.
        /// </summary>
        private unsafe interface IVecdotSource
        {
            /// <summary>The element at <paramref name="p"/> cast to float32.</summary>
            /// <param name="p">The element's address.</param>
            /// <returns>NumPy's float32 value of it.</returns>
            static abstract float ToSingle(byte* p);

            /// <summary>The element at <paramref name="p"/> cast to float64.</summary>
            /// <param name="p">The element's address.</param>
            /// <returns>NumPy's float64 value of it.</returns>
            static abstract double ToDouble(byte* p);
        }

        /// <summary>bool: 1 for a nonzero byte, else 0.</summary>
        private unsafe struct BoolSource : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *p != 0 ? 1f : 0f;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *p != 0 ? 1.0 : 0.0;
        }

        /// <summary>uint8.</summary>
        private unsafe struct ByteSource : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *p;
        }

        /// <summary>int8.</summary>
        private unsafe struct SByteSource : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(sbyte*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(sbyte*)p;
        }

        /// <summary>int16.</summary>
        private unsafe struct Int16Source : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(short*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(short*)p;
        }

        /// <summary>uint16.</summary>
        private unsafe struct UInt16Source : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(ushort*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(ushort*)p;
        }

        /// <summary>int32 (to float32 rounded once).</summary>
        private unsafe struct Int32Source : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(int*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(int*)p;
        }

        /// <summary>uint32 (to float32 rounded once).</summary>
        private unsafe struct UInt32Source : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(uint*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(uint*)p;
        }

        /// <summary>int64 (rounded once to either width).</summary>
        private unsafe struct Int64Source : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(long*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(long*)p;
        }

        /// <summary>uint64 (rounded once to either width).</summary>
        private unsafe struct UInt64Source : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(ulong*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(ulong*)p;
        }

        /// <summary>float16 (exact in both widths; a NaN keeps its payload's top bits, quieted).</summary>
        private unsafe struct HalfSource : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => (float)*(Half*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => (double)*(Half*)p;
        }

        /// <summary>float32 (to float64 exact).</summary>
        private unsafe struct SingleSource : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => *(float*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(float*)p;
        }

        /// <summary>float64 (to float32 rounded once).</summary>
        private unsafe struct DoubleSource : IVecdotSource
        {
            /// <inheritdoc/>
            public static float ToSingle(byte* p) => (float)*(double*)p;
            /// <inheritdoc/>
            public static double ToDouble(byte* p) => *(double*)p;
        }

        /// <summary>
        ///     The managed conversion of a block for a source dtype: <see cref="ConvertBlock{TS, T}"/> instantiated for
        ///     it, chosen once per operand.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type (float, double or Complex).</typeparam>
        /// <param name="source">The operand's dtype.</param>
        /// <returns>The conversion.</returns>
        /// <exception cref="NotSupportedException">A dtype outside NumPy's real numeric dtypes (np.vecdot's dispatch admits
        ///     none here: a complex128 operand only meets a complex128 loop, which reads it in place).</exception>
        private static unsafe delegate*<byte*, long, long, long, long, long, T*, void> ManagedConversion<T>(
            NPTypeCode source) where T : unmanaged
        {
            return source switch
            {
                NPTypeCode.Boolean => &ConvertBlock<BoolSource, T>,
                NPTypeCode.Byte => &ConvertBlock<ByteSource, T>,
                NPTypeCode.SByte => &ConvertBlock<SByteSource, T>,
                NPTypeCode.Int16 => &ConvertBlock<Int16Source, T>,
                NPTypeCode.UInt16 => &ConvertBlock<UInt16Source, T>,
                NPTypeCode.Int32 => &ConvertBlock<Int32Source, T>,
                NPTypeCode.UInt32 => &ConvertBlock<UInt32Source, T>,
                NPTypeCode.Int64 => &ConvertBlock<Int64Source, T>,
                NPTypeCode.UInt64 => &ConvertBlock<UInt64Source, T>,
                NPTypeCode.Half => &ConvertBlock<HalfSource, T>,
                NPTypeCode.Single => &ConvertBlock<SingleSource, T>,
                NPTypeCode.Double => &ConvertBlock<DoubleSource, T>,
                _ => throw new NotSupportedException($"np.vecdot has no conversion from {source} to {typeof(T).Name}"),
            };
        }

        /// <summary>
        ///     Converts a block of <paramref name="count"/> runs of <paramref name="n"/> elements into C-ordered scratch:
        ///     run r's element k from <c>src + (r * runStride + k * coreStride) * itemSize</c>.
        /// </summary>
        /// <typeparam name="TS">The source dtype's reader.</typeparam>
        /// <typeparam name="T">The loop dtype's element type: float (<see cref="IVecdotSource.ToSingle"/>), double
        ///     (<see cref="IVecdotSource.ToDouble"/>) or Complex (the double as the real part, +0 imaginary).</typeparam>
        /// <param name="src">The block's first element.</param>
        /// <param name="itemSize">The source item size in bytes.</param>
        /// <param name="runStride">The element stride between runs.</param>
        /// <param name="coreStride">The element stride within a run.</param>
        /// <param name="count">The number of runs.</param>
        /// <param name="n">The run length.</param>
        /// <param name="dst">The scratch: room for <c>count * n</c> elements.</param>
        private static unsafe void ConvertBlock<TS, T>(byte* src, long itemSize, long runStride, long coreStride,
            long count, long n, T* dst) where TS : struct, IVecdotSource where T : unmanaged
        {
            long rs = runStride * itemSize, cs = coreStride * itemSize;
            for (long r = 0; r < count; r++, src += rs)
            {
                byte* q = src;
                for (long k = 0; k < n; k++, q += cs, dst++)
                {
                    // typeof(T) tests fold in each instantiation: one store per element.
                    if (typeof(T) == typeof(float))
                        *(float*)dst = TS.ToSingle(q);
                    else if (typeof(T) == typeof(double))
                        *(double*)dst = TS.ToDouble(q);
                    else
                        *(Complex*)dst = new Complex(TS.ToDouble(q), 0.0);
                }
            }
        }

        // --------------------------------------------------------------------------------------------------------------
        //  A core of one element: the own loop's single term
        // --------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The value every np.vecdot dot function writes for a core of one element: NumPy forces the core strides to
        ///     0, so the own loop runs, and its one term is added into the loop's +0 start.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        private unsafe interface IVecdotOne<T> where T : unmanaged
        {
            /// <summary>The dot of the one-element runs at <paramref name="a"/> and <paramref name="b"/>.</summary>
            /// <param name="a">The first operand's element (the conjugated one for complex128).</param>
            /// <param name="b">The second operand's element.</param>
            /// <returns>The value NumPy writes.</returns>
            static abstract T One(T* a, T* b);

            /// <summary>
            ///     <c>o[j] = One(a + j * sa, b + j * sb)</c> for a block of outputs: vector lanes where both steps are 1 (the
            ///     same IEEE operations per lane, so the same bits), <see cref="One"/> otherwise.
            /// </summary>
            /// <param name="a">The first output's first-operand element.</param>
            /// <param name="sa">The element step between outputs' first operands (0 when broadcast).</param>
            /// <param name="b">The first output's second-operand element.</param>
            /// <param name="sb">The element step between outputs' second operands.</param>
            /// <param name="o">The first output; the block's outputs are consecutive.</param>
            /// <param name="count">The number of outputs.</param>
            static abstract void Many(T* a, long sa, T* b, long sb, T* o, long count);
        }

        /// <summary>FLOAT_dot's own loop for one term: <c>0f + a*b</c> (a -0.0 product gives +0.0).</summary>
        private unsafe struct FloatOne : IVecdotOne<float>
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static float One(float* a, float* b)
            {
                float s = 0f;
                s += *a * *b;
                return s;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Many(float* a, long sa, float* b, long sb, float* o, long count)
            {
                long j = 0;
                if (sa == 1 && sb == 1 && Vector256.IsHardwareAccelerated)
                {
                    // +0 + product per lane: the add keeps a -0.0 product from surviving, as the scalar sum's start does
                    // (only `x + -0.0` folds to x; `+0.0 + x` is computed).
                    for (; j + 8 <= count; j += 8)
                        Vector256.Store(Vector256<float>.Zero + Vector256.Load(a + j) * Vector256.Load(b + j), o + j);
                }
                else if (sa == 1 && sb == 0 && Vector256.IsHardwareAccelerated)
                {
                    // A broadcast second operand: its one element in every lane, still the right factor of each product.
                    var vb = Vector256.Create(*b);
                    for (; j + 8 <= count; j += 8)
                        Vector256.Store(Vector256<float>.Zero + Vector256.Load(a + j) * vb, o + j);
                }
                else if (sa == 0 && sb == 1 && Vector256.IsHardwareAccelerated)
                {
                    var va = Vector256.Create(*a);
                    for (; j + 8 <= count; j += 8)
                        Vector256.Store(Vector256<float>.Zero + va * Vector256.Load(b + j), o + j);
                }

                for (; j < count; j++)
                    o[j] = One(a + j * sa, b + j * sb);
            }
        }

        /// <summary>DOUBLE_dot's own loop for one term: <c>0.0 + a*b</c>.</summary>
        private unsafe struct DoubleOne : IVecdotOne<double>
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static double One(double* a, double* b)
            {
                double s = 0.0;
                s += *a * *b;
                return s;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Many(double* a, long sa, double* b, long sb, double* o, long count)
            {
                long j = 0;
                if (sa == 1 && sb == 1 && Vector256.IsHardwareAccelerated)
                {
                    for (; j + 4 <= count; j += 4)
                        Vector256.Store(Vector256<double>.Zero + Vector256.Load(a + j) * Vector256.Load(b + j), o + j);
                }
                else if (sa == 1 && sb == 0 && Vector256.IsHardwareAccelerated)
                {
                    var vb = Vector256.Create(*b);
                    for (; j + 4 <= count; j += 4)
                        Vector256.Store(Vector256<double>.Zero + Vector256.Load(a + j) * vb, o + j);
                }
                else if (sa == 0 && sb == 1 && Vector256.IsHardwareAccelerated)
                {
                    var va = Vector256.Create(*a);
                    for (; j + 4 <= count; j += 4)
                        Vector256.Store(Vector256<double>.Zero + va * Vector256.Load(b + j), o + j);
                }

                for (; j < count; j++)
                    o[j] = One(a + j * sa, b + j * sb);
            }
        }

        /// <summary>CDOUBLE_dotc's own loop for one term: <c>(0 + (ar*br + ai*bi), 0 + (ar*bi - ai*br))</c>.</summary>
        private unsafe struct ComplexcOne : IVecdotOne<Complex>
        {
            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Complex One(Complex* a, Complex* b)
            {
                double* x = (double*)a, y = (double*)b;
                double ar = x[0], ai = x[1], br = y[0], bi = y[1];
                double sumr = 0.0, sumi = 0.0;
                sumr += ar * br + ai * bi;
                sumi += ar * bi - ai * br;
                return new Complex(sumr, sumi);
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Many(Complex* a, long sa, Complex* b, long sb, Complex* o, long count)
            {
                long j = 0;
                if ((sa == 1 || sa == 0) && (sb == 1 || sb == 0) && sa + sb > 0 && Avx.IsSupported)
                {
                    // A broadcast operand (step 0) repeats its one complex in both halves of the register.
                    var ra = sa == 0 ? Vector256.Create(((double*)a)[0], ((double*)a)[1], ((double*)a)[0], ((double*)a)[1]) : default;
                    var rb = sb == 0 ? Vector256.Create(((double*)b)[0], ((double*)b)[1], ((double*)b)[0], ((double*)b)[1]) : default;
                    for (; j + 2 <= count; j += 2)
                    {
                        // Two outputs: p = [ar·br, ai·bi] and q = [ar·bi, ai·br] per complex; hadd gives ar·br + ai·bi and
                        // hsub ar·bi − ai·br (each lane's left operand first, as the C sums read), the blend takes the real
                        // parts from hadd and the imaginary parts from hsub, then the +0 start is added.
                        var va = sa == 0 ? ra : Avx.LoadVector256((double*)(a + j));
                        var vb = sb == 0 ? rb : Avx.LoadVector256((double*)(b + j));
                        var pr = Avx.Multiply(va, vb);
                        var qi = Avx.Multiply(va, Avx.Permute(vb, 0b0101));
                        var sums = Avx.Blend(Avx.HorizontalAdd(pr, pr), Avx.HorizontalSubtract(qi, qi), 0b1010);
                        Avx.Store((double*)(o + j), Avx.Add(Vector256<double>.Zero, sums));
                    }
                }

                for (; j < count; j++)
                    o[j] = One(a + j * sa, b + j * sb);
            }
        }

        /// <summary>
        ///     np.vecdot with a core of one element, every output into <paramref name="result"/>:
        ///     <see cref="VecdotOneWalk{T, TOne}"/> instantiated for the loop dtype.
        /// </summary>
        /// <param name="x1">The first operand, its core axis (of one element) last.</param>
        /// <param name="x2">The second operand, likewise.</param>
        /// <param name="loop">The loop dtype (float32, float64 or complex128).</param>
        /// <param name="result">The fresh C-ordered result, non-empty.</param>
        /// <param name="dims">The broadcast loop axes.</param>
        private static unsafe void VecdotOne(NDArray x1, NDArray x2, NPTypeCode loop, NDArray result, long[] dims)
        {
            switch (loop)
            {
                case NPTypeCode.Single:
                    VecdotOneWalk<float, FloatOne>(x1, x2, loop, (float*)result.Address, dims);
                    break;
                case NPTypeCode.Double:
                    VecdotOneWalk<double, DoubleOne>(x1, x2, loop, (double*)result.Address, dims);
                    break;
                default:
                    VecdotOneWalk<Complex, ComplexcOne>(x1, x2, loop, (Complex*)result.Address, dims);
                    break;
            }
        }

        /// <summary>
        ///     <see cref="VecdotBlockWalk{T}"/> for a core of one element: the same blocks (an operand needing a cast
        ///     converted a block at a time, any other read in place), each output the own loop's single term inlined
        ///     (<typeparamref name="TOne"/>) instead of a dot function called per output.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <typeparam name="TOne">The loop dtype's one-term dot.</typeparam>
        /// <param name="x1">The first operand, its core axis (of one element) last.</param>
        /// <param name="x2">The second operand, likewise.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="o">The first output of a fresh C-ordered result holding every output.</param>
        /// <param name="dims">The broadcast loop axes, none empty (empty for one output).</param>
        /// <remarks>Why: with one term an output is an elementwise product, and a dot function called per output
        ///     cost 2.5-4x the former composition's elementwise kernels (measured at 1,000,000 x 1); the blocks run
        ///     vectorized where both operands' runs follow each other.</remarks>
        private static unsafe void VecdotOneWalk<T, TOne>(NDArray x1, NDArray x2, NPTypeCode loop, T* o, long[] dims)
            where T : unmanaged where TOne : struct, IVecdotOne<T>
        {
            int nd = dims.Length;
            long inner = nd == 0 ? 1 : dims[nd - 1];
            long block = Math.Max(1, Math.Min(inner, VecdotBlockBytes / sizeof(T)));
            var s1 = LoopStrides(x1, nd);
            var s2 = LoopStrides(x2, nd);
            long i1 = nd == 0 ? 0 : s1[nd - 1], i2 = nd == 0 ? 0 : s2[nd - 1];
            BlockSource<T> r1 = default, r2 = default;   // see VecdotBlockWalk: scratch taken inside the try
            try
            {
                r1 = new BlockSource<T>(x1, loop, 1, block, i1);
                r2 = new BlockSource<T>(x2, loop, 1, block, i2);
                long rows = 1;
                for (int k = 0; k < nd - 1; k++)
                    rows *= dims[k];
                long off1 = 0, off2 = 0;
                long* idx = stackalloc long[Math.Max(nd, 1)];   // zeroed: no SkipLocalsInit here
                for (long r = 0; r < rows; r++)
                {
                    for (long j0 = 0; j0 < inner; j0 += block)
                    {
                        long count = Math.Min(block, inner - j0);
                        T* b1 = r1.Block(off1 + j0 * i1, count, out long step1);
                        T* b2 = r2.Block(off2 + j0 * i2, count, out long step2);
                        TOne.Many(b1, step1, b2, step2, o, count);
                        o += count;
                    }

                    for (int k = nd - 2; k >= 0; k--)
                    {
                        off1 += s1[k];
                        off2 += s2[k];
                        if (++idx[k] < dims[k])
                            break;
                        // Axis k wrapped: rewind it and carry into the next outer axis.
                        off1 -= s1[k] * dims[k];
                        off2 -= s2[k] * dims[k];
                        idx[k] = 0;
                    }
                }
            }
            finally
            {
                r1.Release();
                r2.Release();
            }
        }
    }
}
