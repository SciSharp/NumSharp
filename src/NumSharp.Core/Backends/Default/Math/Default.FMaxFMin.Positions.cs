using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // np.fmax / np.fmin at a float32 / float64 loop: the answer depends on the element's position
        // =============================================================================
        //
        // NumPy 2.4.2's FLOAT_fmax / DOUBLE_fmax / FLOAT_fmin / DOUBLE_fmin (numpy/_core/src/umath/
        // loops_minmax.dispatch.c.src) run every inner-loop call in two parts:
        //   - when the call's input and output ranges do not overlap (is_mem_overlap) and its steps are contiguous
        //     (simd_binary_ccc) or loadable / storable (simd_binary), the first floor(len / W) * W elements go through
        //     whole SIMD vectors, npyv_maxp / npyv_minp: x86 MAXPS / MINPS keep the SECOND operand on a ±0 tie, and
        //     the ordered-compare blend keeps the FIRST operand when both are NaN (a lone NaN is skipped);
        //   - every other element — the tail of such a call, a whole call shorter than one vector, a call whose ranges
        //     overlap or whose steps are not loadable — calls the C library's fmax / fmin, and the win-amd64 UCRT gives
        //     +0.0 (fmax) / -0.0 (fmin) on a ±0 tie whatever the order and the SECOND operand when both are NaN (its
        //     bits verbatim: a signalling NaN stays signalling).
        // So np.fmax([0.0], [-0.0]) is [0.0] while the first eight of a 9-element float64 fmax of +0 and -0 are -0.
        //
        // The two rules differ ONLY on a pair of zeros of opposite signs and on a pair of NaNs with different bits; on
        // every other pair both return the larger (smaller) operand, or the non-NaN one. NumSharp's kernels apply the
        // vector rule at every position (the vector bodies, DirectILKernelGenerator.EmitVectorFMinOrFMax, and the scalar
        // helpers FloatMaxNum / DoubleMaxNum agree), so they are right everywhere except at NumPy's scalar positions
        // where such a pair meets. This file finds those positions from NumPy's own calls (NumPyUfuncCalls: the trivial
        // loop, or NpyIter's sorted, flipped, coalesced and buffered iteration; a where= mask splits each call into its
        // true runs, each a call of its own), computes the C library's answer there BEFORE the kernels run (an out= may
        // alias an input, which the kernels then overwrite) and writes it into the result afterwards.
        //
        // Which positions are vector positions also depends on the SIMD target and on its stride limits. The answer
        // pinned here is numpy 2.4.2's win-amd64 wheel on an AVX2 CPU, on EVERY host: that wheel dispatches
        // loops_minmax to X86_V3 (AVX2, 32-byte vectors: 8 float32 / 4 float64 lanes) and has no AVX-512 target at all
        // (MSVC builds disable X86_V4), and pinning it keeps the answer independent of the machine — a Linux AVX-512 or
        // an ARM NumPy splits at other places, and the corpus comparing ±0 signs runs on all three CI operating systems.
        // The AVX2 stride limit carries an integer quirk: npyv_loadable_stride divides the signed byte stride by an
        // unsigned sizeof, so a NEGATIVE stride becomes a huge one and fails the float32 limit, while float64 has no limit
        // on AVX2. A reversed float32 operand therefore runs every element through the C library, where a reversed float64
        // one keeps its vectors. The C library rule itself is the win-amd64 UCRT's (glibc's fmax keeps the first operand
        // on a tie): the whole answer is a platform pin, like the add / multiply NaN order.
        // =============================================================================

        /// <summary>
        /// The lanes of one NumPy fmax / fmin vector at <paramref name="itemSize"/> (W in the file header): the win-amd64
        /// AVX2 build's 32-byte vector — 8 float32 / 4 float64 lanes — on every host (see the file header).
        /// </summary>
        /// <param name="itemSize">4 (float32) or 8 (float64).</param>
        /// <returns>The vector width in elements.</returns>
        private static int NumPyMinMaxLanes(int itemSize) => 32 / itemSize;

        /// <summary>
        /// <c>npyv_loadable_stride_f32/f64</c> of the AVX2 build, with its integer quirk: the byte stride is divided by an
        /// UNSIGNED <c>sizeof</c>, so a negative float32 stride wraps to a huge one and fails
        /// <c>NPY_SIMD_MAXLOAD_STRIDE32</c> (<c>0x7fffffff / 8</c>); AVX2 defines no float64 limit, so every float64 stride
        /// loads.
        /// </summary>
        /// <param name="stride">The operand's byte step in the call.</param>
        /// <param name="itemSize">4 (float32) or 8 (float64).</param>
        /// <returns>True when the strided SIMD loop may load this operand.</returns>
        private static bool NumPyMinMaxLoadable(long stride, int itemSize)
            => itemSize == 8 || (stride >= 0 && stride / itemSize <= 0x7fffffff / 8);

        /// <summary>
        /// NumPy's <c>nomemoverlap(ip, ip_step, op, op_step, len)</c> (loops_utils.h.src): true when the two strided
        /// ranges are disjoint or exactly the same range (an in-place call), compared over inclusive byte ranges.
        /// </summary>
        /// <param name="ip">The input's first address.</param>
        /// <param name="ipStep">The input's byte step.</param>
        /// <param name="op">The output's first address.</param>
        /// <param name="opStep">The output's byte step.</param>
        /// <param name="len">The call's element count (at least 1).</param>
        /// <returns>True when the inner loop treats the pair as not overlapping.</returns>
        private static bool NumPyNoMemOverlap(long ip, long ipStep, long op, long opStep, long len)
        {
            long ipOffset = ipStep * (len - 1), opOffset = opStep * (len - 1);
            long ipStart = ipStep < 0 ? ip + ipOffset : ip, ipEnd = ipStep < 0 ? ip : ip + ipOffset;
            long opStart = opStep < 0 ? op + opOffset : op, opEnd = opStep < 0 ? op : op + opOffset;
            return (ipStart == opStart && opEnd == ipEnd) || ipStart > opEnd || opStart > ipEnd;
        }

        /// <summary>
        /// Whether the C library's answer to a float64 pair differs from NumPy's vector rule, and the C library's answer
        /// when it does. The vector rule (<c>npyv_maxp_f64</c> / <c>npyv_minp_f64</c>, <c>blendv(a, max(a, b), b == b)</c>):
        /// a NaN second operand returns the first (so of two NaNs, the first), a NaN first operand the second, otherwise
        /// MAXPD / MINPD — the second operand unless the first is strictly larger (smaller). The UCRT's <c>fmax</c> /
        /// <c>fmin</c>: a NaN first operand returns the second (so of two NaNs, the second, verbatim), a NaN second operand
        /// the first, unequal values the larger (smaller), equal values their bitwise AND (OR). The two agree on every pair
        /// but two kinds, which is all this tests: zeros of opposite signs (the vector rule keeps <paramref name="b"/>, the
        /// C library gives +0.0 for fmax and -0.0 for fmin) and NaNs with different bits (the vector rule keeps
        /// <paramref name="a"/>, the C library <paramref name="b"/>).
        /// </summary>
        /// <param name="a">The first operand's bits.</param>
        /// <param name="b">The second operand's bits.</param>
        /// <param name="isMax">True for fmax, false for fmin.</param>
        /// <param name="answer">Receives the C library's answer when this returns true; 0 otherwise.</param>
        /// <returns>True when the two rules give different bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool CrtDiffers64(ulong a, ulong b, bool isMax, out ulong answer)
        {
            answer = 0;
            // Identical bits: both rules return them. This is the exit for almost every pair.
            if (a == b)
                return false;
            double x = BitConverter.UInt64BitsToDouble(a), y = BitConverter.UInt64BitsToDouble(b);
            if (x == y)
            {
                // Equal values with different bits: +0.0 against -0.0. The vector rule's b is already the C library's
                // answer when b has fmax's + (fmin's -) sign.
                if (isMax ? (long)b >= 0 : (long)b < 0)
                    return false;
                answer = isMax ? a & b : a | b;
                return true;
            }
            if (double.IsNaN(x) && double.IsNaN(y))
            {
                answer = b;
                return true;
            }
            return false;
        }

        /// <summary>
        /// The float32 twin of <see cref="CrtDiffers64"/> (<c>npyv_maxp_f32</c> / <c>npyv_minp_f32</c> against the UCRT's
        /// <c>fmaxf</c> / <c>fminf</c>).
        /// </summary>
        /// <param name="a">The first operand's bits.</param>
        /// <param name="b">The second operand's bits.</param>
        /// <param name="isMax">True for fmax, false for fmin.</param>
        /// <param name="answer">Receives the C library's answer when this returns true; 0 otherwise.</param>
        /// <returns>True when the two rules give different bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool CrtDiffers32(uint a, uint b, bool isMax, out uint answer)
        {
            answer = 0;
            if (a == b)
                return false;
            float x = BitConverter.UInt32BitsToSingle(a), y = BitConverter.UInt32BitsToSingle(b);
            if (x == y)
            {
                if (isMax ? (int)b >= 0 : (int)b < 0)
                    return false;
                answer = isMax ? a & b : a | b;
                return true;
            }
            if (float.IsNaN(x) && float.IsNaN(y))
            {
                answer = b;
                return true;
            }
            return false;
        }

        /// <summary>
        /// The answers np.fmax / np.fmin give at NumPy's scalar positions where they differ from the vector rule
        /// NumSharp's kernels apply (see the file header), or null when there is none — or when the shapes do not agree,
        /// which the kernels' run then reports with NumPy's error.
        /// </summary>
        /// <param name="lhs">The first input.</param>
        /// <param name="rhs">The second input.</param>
        /// <param name="isMax">True for fmax, false for fmin.</param>
        /// <param name="loop">float32 or float64.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>The patches to write after the kernels ran, or null.</returns>
        private static unsafe FMaxFMinPatches NumPyFMaxFMinPatches(NDArray lhs, NDArray rhs, bool isMax, NPTypeCode loop,
                                                                   NDArray @out, NDArray where)
        {
            // The common call — no out=, no mask, inputs NumPy's trivial loop takes as they are — needs no plan.
            if (@out is null && where is null && TryTrivialFMaxFMinPatches(lhs, rhs, isMax, loop, out var trivialPatches))
                return trivialPatches;

            if (!TryUfuncIterationDims(lhs, rhs, @out, where, out long[] dims))
                return null;
            int nd = dims.Length;
            long size = 1;
            for (int k = 0; k < nd; k++)
                size *= dims[k];
            if (size == 0)
                return null;

            int itemSize = loop.SizeOf();
            int lanes = NumPyMinMaxLanes(itemSize);
            var inputs = new[] { UfuncLoopOperand.OfInput(lhs, loop), UfuncLoopOperand.OfInput(rhs, loop) };
            var plan = NumPyUfuncCalls(inputs, @out, where, loop, dims);

            // ccc / simd_binary need loadable inputs (AVX2 defines no store limit, so the output's step never refuses
            // them) and no overlap between an input's and the output's range. NpyIter's COPY_IF_OVERLAP leaves only
            // disjoint operands or exact aliases (which nomemoverlap accepts), so only the trivial loop's call can overlap.
            long s0 = plan.Steps[0], s1 = plan.Steps[1], so = plan.Steps[2];
            bool simd = NumPyMinMaxLoadable(s0, itemSize) && NumPyMinMaxLoadable(s1, itemSize);
            if (simd && plan.Trivial && plan.Starts[2] != 0)
                for (int i = 0; i < 2; i++)
                    if (plan.Starts[i] != 0 && !NumPyNoMemOverlap(plan.Starts[i], plan.Steps[i], plan.Starts[2], so, size))
                        simd = false;

            // Without SIMD a call is all scalar; with it only the elements past its last whole vector are. A full-width
            // vector call with no tail (every call a multiple of the lanes, no mask) has no scalar position at all.
            if (simd && where is null && plan.CallSize % lanes == 0 && plan.SlabSize % plan.CallSize == 0)
                return null;

            var walker = new FMaxFMinWalker(plan, dims, lhs, rhs, where, loop, isMax);
            for (long slab = 0; slab < size; slab += plan.SlabSize)
            {
                for (long q = 0; q < plan.SlabSize; q += plan.CallSize)
                {
                    long start = slab + q, length = Math.Min(plan.CallSize, plan.SlabSize - q);
                    if (where is null)
                        walker.ScanScalarPart(start, length, simd ? lanes : 0);
                    else
                        walker.ScanMaskedCall(start, length, simd ? lanes : 0);
                }
            }

            return walker.Patches;
        }

        /// <summary>
        /// The allocation-free form of <see cref="NumPyFMaxFMinPatches"/> for the call NumPy runs most: no out=, no mask,
        /// and inputs its trivial loop takes without a copy that changes their layout — each a 0-d array (step 0; a cast
        /// one is copied, still step 0), or an aligned array of the loop dtype that is 1-D (any stride) or C-contiguous
        /// with the other input's dims. NumPy then makes ONE call over every element in C order, so the scalar positions
        /// are the tail past the last whole vector, or every element when a step is not loadable.
        /// </summary>
        /// <param name="lhs">The first input.</param>
        /// <param name="rhs">The second input.</param>
        /// <param name="isMax">True for fmax, false for fmin.</param>
        /// <param name="loop">float32 or float64.</param>
        /// <param name="patches">When this returns true: the answers to write, or null when no scalar position differs.</param>
        /// <returns>True when the call is this shape (and <paramref name="patches"/> is final); false to take the plan.</returns>
        private static unsafe bool TryTrivialFMaxFMinPatches(NDArray lhs, NDArray rhs, bool isMax, NPTypeCode loop,
                                                             out FMaxFMinPatches patches)
        {
            patches = null;
            int itemSize = loop.SizeOf();
            if (!TrivialFMaxFMinStep(lhs, loop, itemSize, out long sa) || !TrivialFMaxFMinStep(rhs, loop, itemSize, out long sb))
                return false;

            // Two non-0-d inputs must agree exactly (a broadcast leaves the trivial loop): the same rank and dims.
            NDArray shaped = lhs.ndim != 0 ? lhs : rhs;
            if (lhs.ndim != 0 && rhs.ndim != 0)
            {
                var ld = lhs.Shape.dimensions;
                var rd = rhs.Shape.dimensions;
                if (ld.Length != rd.Length)
                    return false;
                for (int k = 0; k < ld.Length; k++)
                    if (ld[k] != rd[k])
                        return false;
            }

            long n = shaped.size;
            int lanes = NumPyMinMaxLanes(itemSize);
            // The allocated output's step is the item size (AVX2 has no store limit anyway).
            bool simd = NumPyMinMaxLoadable(sa, itemSize) && NumPyMinMaxLoadable(sb, itemSize);
            long first = simd ? n - n % lanes : 0;
            if (first >= n)
                return true;

            bool asDouble = loop == NPTypeCode.Double;
            byte* pa = (byte*)lhs.Address + lhs.Shape.offset * lhs.dtypesize;
            byte* pb = (byte*)rhs.Address + rhs.Shape.offset * rhs.dtypesize;
            // A 0-d input (the only kind here that may need a cast) is converted once and read from a local with step 0,
            // so the loops below read every input at the loop dtype. float32 bits sit in the low half of the ulong, which
            // a little-endian uint read at its address returns.
            ulong ca = 0, cb = 0;
            if (lhs.ndim == 0)
            {
                ca = ReadAsLoopBits(pa, lhs.typecode, asDouble);
                pa = (byte*)&ca;
            }
            if (rhs.ndim == 0)
            {
                cb = ReadAsLoopBits(pb, rhs.typecode, asDouble);
                pb = (byte*)&cb;
            }

            // One call in C order: the call's i-th element IS the result's C-order element i.
            if (asDouble)
            {
                for (long i = first; i < n; i++)
                    if (CrtDiffers64(*(ulong*)(pa + i * sa), *(ulong*)(pb + i * sb), isMax, out ulong answer))
                        (patches ??= new FMaxFMinPatches((long[])shaped.Shape.dimensions.Clone(), loop)).Add(i, answer);
            }
            else
            {
                for (long i = first; i < n; i++)
                    if (CrtDiffers32(*(uint*)(pa + i * sa), *(uint*)(pb + i * sb), isMax, out uint answer))
                        (patches ??= new FMaxFMinPatches((long[])shaped.Shape.dimensions.Clone(), loop)).Add(i, answer);
            }

            // The arrays' memory was read through raw pointers: keep both alive past the last read.
            GC.KeepAlive(lhs);
            GC.KeepAlive(rhs);
            return true;
        }

        /// <summary>
        /// The step of an input in NumPy's trivial loop when that loop takes it without changing its layout: 0 for a 0-d
        /// input (a cast or unaligned one is copied, which keeps step 0), the byte stride of an aligned 1-D input of the
        /// loop dtype, the item size of an aligned C-contiguous one; false for anything else (a cast or unaligned 1-D or
        /// N-D input, a non-C-contiguous N-D one), which <see cref="NumPyUfuncCalls"/> decides instead.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="itemSize">The loop dtype's item size.</param>
        /// <param name="step">Receives the step when this returns true.</param>
        /// <returns>True when the fast trivial route may take this input.</returns>
        private static bool TrivialFMaxFMinStep(NDArray x, NPTypeCode loop, int itemSize, out long step)
        {
            step = 0;
            if (x.ndim == 0)
                return true;
            if (x.typecode != loop || !UfuncLoopOperand.IsAligned(x))
                return false;
            if (x.ndim == 1)
            {
                step = x.Shape.strides[0] * (long)itemSize;
                return true;
            }
            if (!x.Shape.IsContiguous)
                return false;
            step = itemSize;
            return true;
        }

        /// <summary>
        /// The broadcast iteration shape of a binary ufunc call — the inputs, the out= array and the mask together, NumPy's
        /// NpyIter broadcast, where an out is never stretched — or false when the shapes do not agree.
        /// </summary>
        /// <param name="lhs">The first input.</param>
        /// <param name="rhs">The second input.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <param name="dims">Receives the shape when this returns true.</param>
        /// <returns>True when every operand broadcasts and out= has exactly the broadcast shape.</returns>
        private static bool TryUfuncIterationDims(NDArray lhs, NDArray rhs, NDArray @out, NDArray where, out long[] dims)
        {
            int nd = Math.Max(lhs.ndim, rhs.ndim);
            if (@out is not null)
                nd = Math.Max(nd, @out.ndim);
            if (where is not null)
                nd = Math.Max(nd, where.ndim);
            dims = new long[nd];
            for (int k = 0; k < nd; k++)
                dims[k] = 1;

            // Right-aligned, a 1 stretches, any other extent must agree (NumPy's broadcast rule).
            foreach (var x in new[] { lhs, rhs, @out, where })
            {
                if (x is null)
                    continue;
                var xd = x.Shape.dimensions;
                int off = nd - xd.Length;
                for (int k = 0; k < xd.Length; k++)
                {
                    long v = xd[k];
                    if (v == 1)
                        continue;
                    if (dims[off + k] == 1)
                        dims[off + k] = v;
                    else if (dims[off + k] != v)
                        return false;
                }
            }

            // NO_BROADCAST on the output: it must already have every dimension of the broadcast.
            if (@out is not null)
            {
                if (@out.ndim != nd)
                    return false;
                var od = @out.Shape.dimensions;
                for (int k = 0; k < nd; k++)
                    if (od[k] != dims[k])
                        return false;
            }

            return true;
        }

        /// <summary>
        /// The C library's answers at the scalar positions where they differ from the vector rule: each a position in the
        /// broadcast result (its C-order index) and the answer's bits at the loop dtype. Applied to the result after the
        /// kernels ran.
        /// </summary>
        private sealed class FMaxFMinPatches
        {
            /// <summary>The broadcast result's shape, which the C-order indices refer to.</summary>
            private readonly long[] _dims;

            /// <summary>The loop dtype the bits are encoded in (float32 bits in the low half for <see cref="NPTypeCode.Single"/>).</summary>
            private readonly NPTypeCode _loop;

            /// <summary>The C-order index of each patched element.</summary>
            private readonly List<long> _positions = new List<long>();

            /// <summary>The answer's bits for each patched element, in <see cref="_positions"/>' order.</summary>
            private readonly List<ulong> _bits = new List<ulong>();

            /// <summary>Creates an empty patch list for a result of shape <paramref name="dims"/>.</summary>
            /// <param name="dims">The broadcast result's shape (kept by reference; not modified).</param>
            /// <param name="loop">The loop dtype the bits are encoded in.</param>
            internal FMaxFMinPatches(long[] dims, NPTypeCode loop)
            {
                _dims = dims;
                _loop = loop;
            }

            /// <summary>Records the answer for one element.</summary>
            /// <param name="position">The element's C-order index in the broadcast result.</param>
            /// <param name="bits">The answer's bits at the loop dtype.</param>
            internal void Add(long position, ulong bits)
            {
                _positions.Add(position);
                _bits.Add(bits);
            }

            /// <summary>
            /// Writes every recorded answer into <paramref name="result"/> (the call's out= or its fresh result, of the
            /// broadcast shape). A result of another dtype than the loop's (an out= reached by a same_kind cast) receives
            /// each answer through the engine's own cast, as NumPy's buffered output cast delivers it.
            /// </summary>
            /// <param name="result">The array the kernels wrote.</param>
            /// <param name="engine">The engine whose cast converts the answers for a result of another dtype.</param>
            internal unsafe void ApplyTo(NDArray result, DefaultEngine engine)
            {
                int n = _positions.Count;
                if (n == 0)
                    return;

                int nd = _dims.Length;
                int itemSize = result.dtypesize;
                var strides = result.Shape.strides;
                byte* basePtr = (byte*)result.Address + result.Shape.offset * itemSize;

                // The answers at the result's dtype: the loop's bits as they are, or one engine cast of all of them.
                NDArray converted = null;
                byte* values = null;
                try
                {
                    if (result.typecode != _loop)
                    {
                        using var raw = engine.NewArray(_loop, Shape.Vector(n), false);
                        for (int i = 0; i < n; i++)
                        {
                            if (_loop == NPTypeCode.Double)
                                ((ulong*)raw.Address)[i] = _bits[i];
                            else
                                ((uint*)raw.Address)[i] = (uint)_bits[i];
                        }
                        converted = engine.Cast(raw, result.typecode, copy: true);
                        values = (byte*)converted.Address + converted.Shape.offset * itemSize;
                    }

                    for (int i = 0; i < n; i++)
                    {
                        // The C-order index unravelled over the shape, then through the result's own strides.
                        long rest = _positions[i], offset = 0;
                        for (int k = nd - 1; k >= 0; k--)
                        {
                            long extent = _dims[k];
                            long digit = rest % extent;
                            rest /= extent;
                            offset += digit * strides[k];
                        }
                        byte* dst = basePtr + offset * itemSize;
                        if (values is not null)
                            Buffer.MemoryCopy(values + (long)i * itemSize, dst, itemSize, itemSize);
                        else if (_loop == NPTypeCode.Double)
                            *(ulong*)dst = _bits[i];
                        else
                            *(uint*)dst = (uint)_bits[i];
                    }
                }
                finally
                {
                    converted?.Dispose();
                }
                GC.KeepAlive(result);
            }
        }

        /// <summary>
        /// Walks NumPy's scalar positions of one fmax / fmin call (see the file header) and records the C library's answer
        /// wherever it differs from the vector rule. It reads each input at a position through that input's broadcast
        /// strides, converted to the loop dtype as NumPy's cast would deliver it, and walks a run of consecutive positions
        /// with an odometer over the plan's iterator axes (one division per axis per run, not per element).
        /// </summary>
        private sealed unsafe class FMaxFMinWalker
        {
            /// <summary>The call plan whose positions are walked.</summary>
            private readonly NumPyUfuncCallPlan _plan;

            /// <summary>The broadcast result's shape.</summary>
            private readonly long[] _dims;

            /// <summary>The first input's logical element 0, its dtype and its byte step per iterator axis (0 where broadcast).</summary>
            private readonly byte* _a;
            private readonly NPTypeCode _aType;
            private readonly long[] _aStep;

            /// <summary>The second input's logical element 0, dtype and byte step per iterator axis.</summary>
            private readonly byte* _b;
            private readonly NPTypeCode _bType;
            private readonly long[] _bStep;

            /// <summary>The mask's logical element 0 and byte step per iterator axis (null pointer without a mask).</summary>
            private readonly byte* _m;
            private readonly long[] _mStep;

            /// <summary>The C-order index step per iterator axis (negative along a reversed axis).</summary>
            private readonly long[] _cStep;

            /// <summary>The loop dtype and the op.</summary>
            private readonly bool _double, _isMax;

            /// <summary>The odometer: the digit of each iterator axis and the running offsets of the current position.</summary>
            private readonly long[] _digit;
            private long _aOff, _bOff, _mOff, _cIndex;

            /// <summary>The arrays read through raw pointers, kept reachable until the walk ends.</summary>
            private readonly NDArray _lhs, _rhs, _where;

            /// <summary>The answers found so far (allocated on the first difference).</summary>
            internal FMaxFMinPatches Patches { get; private set; }

            /// <summary>
            /// Prepares the walk: each operand's start address and its byte step along every iterator axis (the result axis
            /// that iterator axis walks, right-aligned to the operand's dims, 0 where the operand broadcasts), negated
            /// along a reversed axis so that one step always moves to the next position.
            /// </summary>
            /// <param name="plan">NumPy's calls.</param>
            /// <param name="dims">The broadcast result's shape.</param>
            /// <param name="lhs">The first input.</param>
            /// <param name="rhs">The second input.</param>
            /// <param name="where">The mask, or null.</param>
            /// <param name="loop">float32 or float64.</param>
            /// <param name="isMax">True for fmax, false for fmin.</param>
            internal FMaxFMinWalker(NumPyUfuncCallPlan plan, long[] dims, NDArray lhs, NDArray rhs, NDArray where,
                                    NPTypeCode loop, bool isMax)
            {
                _plan = plan;
                _dims = dims;
                _lhs = lhs;
                _rhs = rhs;
                _where = where;
                _double = loop == NPTypeCode.Double;
                _isMax = isMax;
                int naxes = plan.AxisShape.Length;
                _digit = new long[naxes];

                _a = Start(lhs);
                _aType = lhs.typecode;
                _aStep = AxisSteps(lhs);
                _b = Start(rhs);
                _bType = rhs.typecode;
                _bStep = AxisSteps(rhs);
                if (where is not null)
                {
                    _m = Start(where);
                    _mStep = AxisSteps(where);
                }

                // C-order index step of each result axis, then per iterator axis (signed by the walking direction).
                int nd = dims.Length;
                var cStride = new long[nd];
                long s = 1;
                for (int k = nd - 1; k >= 0; k--)
                {
                    cStride[k] = s;
                    s *= dims[k];
                }
                _cStep = new long[naxes];
                for (int k = 0; k < naxes; k++)
                    _cStep[k] = plan.AxisReversed[k] ? -cStride[plan.AxisArray[k]] : cStride[plan.AxisArray[k]];
            }

            /// <summary>The address of an array's logical element 0.</summary>
            /// <param name="x">The array.</param>
            /// <returns>Its first element's address (the base address plus the view's offset).</returns>
            private static byte* Start(NDArray x) => (byte*)x.Address + x.Shape.offset * x.dtypesize;

            /// <summary>
            /// An operand's byte step along each iterator axis: the stride of the result axis the iterator axis walks,
            /// right-aligned to the operand (0 where it has no such axis or extent 1), negated along a reversed axis.
            /// </summary>
            /// <param name="x">The operand (broadcastable to the result).</param>
            /// <returns>One signed byte step per iterator axis.</returns>
            private long[] AxisSteps(NDArray x)
            {
                int naxes = _plan.AxisShape.Length;
                var steps = new long[naxes];
                var xd = x.Shape.dimensions;
                var xs = x.Shape.strides;
                int off = _dims.Length - xd.Length;
                int item = x.dtypesize;
                for (int k = 0; k < naxes; k++)
                {
                    int axis = _plan.AxisArray[k] - off;
                    long step = axis < 0 || xd[axis] == 1 ? 0 : xs[axis] * (long)item;
                    steps[k] = _plan.AxisReversed[k] ? -step : step;
                }
                return steps;
            }

            /// <summary>
            /// Moves the odometer to position <paramref name="p"/>: its digits and every running offset.
            /// </summary>
            /// <param name="p">The position.</param>
            private void Seek(long p)
            {
                _aOff = _bOff = _mOff = 0;
                // Position 0 is the result's first index along a forward axis and its last along a reversed one.
                _cIndex = 0;
                long cBase = 0, aBase = 0, bBase = 0, mBase = 0;
                for (int k = 0; k < _digit.Length; k++)
                {
                    long extent = _plan.AxisShape[k];
                    long d = p % extent;
                    p /= extent;
                    _digit[k] = d;
                    if (_plan.AxisReversed[k])
                    {
                        // The reversed axis starts at the logical index extent-1; its steps are already negated.
                        cBase -= _cStep[k] * (extent - 1);
                        aBase -= _aStep[k] * (extent - 1);
                        bBase -= _bStep[k] * (extent - 1);
                        if (_mStep is not null)
                            mBase -= _mStep[k] * (extent - 1);
                    }
                    _cIndex += _cStep[k] * d;
                    _aOff += _aStep[k] * d;
                    _bOff += _bStep[k] * d;
                    if (_mStep is not null)
                        _mOff += _mStep[k] * d;
                }
                _cIndex += cBase;
                _aOff += aBase;
                _bOff += bBase;
                _mOff += mBase;
            }

            /// <summary>
            /// Advances the odometer one position: the fastest axis steps, and a full axis wraps to its first digit and
            /// carries into the next.
            /// </summary>
            private void Next()
            {
                for (int k = 0; k < _digit.Length; k++)
                {
                    _cIndex += _cStep[k];
                    _aOff += _aStep[k];
                    _bOff += _bStep[k];
                    if (_mStep is not null)
                        _mOff += _mStep[k];
                    if (++_digit[k] < _plan.AxisShape[k])
                        return;
                    // Wrap: undo the axis's full sweep and carry.
                    long sweep = _plan.AxisShape[k];
                    _cIndex -= _cStep[k] * sweep;
                    _aOff -= _aStep[k] * sweep;
                    _bOff -= _bStep[k] * sweep;
                    if (_mStep is not null)
                        _mOff -= _mStep[k] * sweep;
                    _digit[k] = 0;
                }
            }

            /// <summary>
            /// Checks the scalar part of one inner-loop call of <paramref name="length"/> positions from
            /// <paramref name="start"/>: everything past its last whole vector (<paramref name="lanes"/> elements each), or
            /// the whole call when it takes no SIMD (<paramref name="lanes"/> = 0).
            /// </summary>
            /// <param name="start">The call's first position.</param>
            /// <param name="length">The call's number of positions.</param>
            /// <param name="lanes">The vector width when the call runs SIMD, else 0.</param>
            internal void ScanScalarPart(long start, long length, int lanes)
            {
                long first = lanes == 0 ? start : start + length / lanes * lanes;
                long end = start + length;
                if (first >= end)
                    return;
                Seek(first);
                for (long p = first; p < end; p++)
                {
                    Check();
                    if (p + 1 < end)
                        Next();
                }
            }

            /// <summary>
            /// Checks one masked call: NumPy's masked strided loop calls the inner loop once per run of true mask
            /// elements, with the call's steps, so each run gets its own vector part and scalar tail.
            /// </summary>
            /// <param name="start">The call's first position.</param>
            /// <param name="length">The call's number of positions.</param>
            /// <param name="lanes">The vector width when the call runs SIMD, else 0.</param>
            internal void ScanMaskedCall(long start, long length, int lanes)
            {
                // Collect the runs first (one pass over the mask), then check each run's scalar part.
                var runs = new List<(long Start, long Length)>();
                Seek(start);
                long runStart = -1;
                long end = start + length;
                for (long p = start; p < end; p++)
                {
                    bool on = _m[_mOff] != 0;
                    if (on && runStart < 0)
                        runStart = p;
                    else if (!on && runStart >= 0)
                    {
                        runs.Add((runStart, p - runStart));
                        runStart = -1;
                    }
                    if (p + 1 < end)
                        Next();
                }
                if (runStart >= 0)
                    runs.Add((runStart, end - runStart));

                foreach (var run in runs)
                    ScanScalarPart(run.Start, run.Length, lanes);
            }

            /// <summary>
            /// Compares the two rules at the current position and records the C library's answer when it differs.
            /// </summary>
            private void Check()
            {
                ulong a = ReadInput(_a + _aOff, _aType), b = ReadInput(_b + _bOff, _bType);
                ulong answer;
                bool differs;
                if (_double)
                    differs = CrtDiffers64(a, b, _isMax, out answer);
                else
                {
                    differs = CrtDiffers32((uint)a, (uint)b, _isMax, out uint answer32);
                    answer = answer32;
                }
                if (!differs)
                    return;
                Patches ??= new FMaxFMinPatches(_dims, _double ? NPTypeCode.Double : NPTypeCode.Single);
                Patches.Add(_cIndex, answer);
            }

            /// <summary>
            /// An input element at the loop dtype: read directly when the input already has it (the common case),
            /// otherwise through <see cref="ReadAsLoopBits"/>'s NumPy cast.
            /// </summary>
            /// <param name="p">The element's address.</param>
            /// <param name="type">The input's dtype.</param>
            /// <returns>The element's bits at the loop dtype (float32 bits in the low half).</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private ulong ReadInput(byte* p, NPTypeCode type)
            {
                if (_double)
                    return type == NPTypeCode.Double ? *(ulong*)p : ReadAsLoopBits(p, type, true);
                return type == NPTypeCode.Single ? *(uint*)p : ReadAsLoopBits(p, type, false);
            }
        }

        /// <summary>
        /// One input element converted to the loop dtype as NumPy's cast delivers it: float16 widened bit for bit (a
        /// signalling NaN stays signalling), float32 to float64 and float64 to float32 by the hardware conversion (which
        /// quiets a NaN, as the C cast does), integers, bool and char by value.
        /// </summary>
        /// <param name="p">The element's address.</param>
        /// <param name="type">The element's dtype.</param>
        /// <param name="asDouble">True for a float64 loop, false for float32.</param>
        /// <returns>The converted value's bits (float32 bits in the low half for a float32 loop).</returns>
        /// <exception cref="NotSupportedException">The dtype has no float cast (a dtype the float loops never receive).</exception>
        private static unsafe ulong ReadAsLoopBits(byte* p, NPTypeCode type, bool asDouble)
        {
            if (asDouble)
            {
                switch (type)
                {
                    case NPTypeCode.Double: return *(ulong*)p;
                    case NPTypeCode.Single: return BitConverter.DoubleToUInt64Bits(*(float*)p);
                    case NPTypeCode.Half: return DirectILKernelGenerator.HalfToDoubleBitsExact(*(ushort*)p);
                    default: return BitConverter.DoubleToUInt64Bits(ReadIntegralAsDouble(p, type));
                }
            }

            switch (type)
            {
                case NPTypeCode.Single: return *(uint*)p;
                case NPTypeCode.Double: return BitConverter.SingleToUInt32Bits((float)*(double*)p);
                case NPTypeCode.Half: return BitConverter.SingleToUInt32Bits(DirectILKernelGenerator.HalfToFloatScalarExact(*(ushort*)p));
                default: return BitConverter.SingleToUInt32Bits((float)ReadIntegralAsDouble(p, type));
            }
        }

        /// <summary>
        /// An integer, bool, char or decimal element as a double (exact for every type but the 64-bit integers and decimal,
        /// which round to nearest like NumPy's cast; only a zero can meet the two rules' difference, and a zero is exact).
        /// </summary>
        /// <param name="p">The element's address.</param>
        /// <param name="type">The element's dtype.</param>
        /// <returns>The value.</returns>
        /// <exception cref="NotSupportedException">A dtype the float loops never receive (complex, text).</exception>
        private static unsafe double ReadIntegralAsDouble(byte* p, NPTypeCode type) => type switch
        {
            NPTypeCode.Boolean => *p != 0 ? 1.0 : 0.0,
            NPTypeCode.Byte => *p,
            NPTypeCode.SByte => *(sbyte*)p,
            NPTypeCode.Int16 => *(short*)p,
            NPTypeCode.UInt16 => *(ushort*)p,
            NPTypeCode.Char => *(char*)p,
            NPTypeCode.Int32 => *(int*)p,
            NPTypeCode.UInt32 => *(uint*)p,
            NPTypeCode.Int64 => *(long*)p,
            NPTypeCode.UInt64 => *(ulong*)p,
            NPTypeCode.Decimal => (double)*(decimal*)p,
            _ => throw new NotSupportedException($"np.fmax / np.fmin: no float cast from {type}."),
        };
    }
}
