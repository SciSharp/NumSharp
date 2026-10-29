using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// The fast axis argmax / argmin: <c>np.argmax(a, axis)</c> / <c>np.argmin(a, axis)</c> folded straight from the
    /// operand's strides, with the loop order chosen by the MEMORY layout rather than by the output order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The per-output IL kernel this replaces (<c>AxisArgReductionHelper</c>) recomputed every output's base offset with
    /// a division/modulo chain and then walked its axis with a scalar strided loop — so reducing the axis that is NOT
    /// contiguous (axis 0 of a C array) touched one element per cache line per step, and reducing a contiguous axis
    /// never used the SIMD row kernels the flat reduction already has. NumPy itself answers an axis argmax by
    /// transposing the axis last and copying the operand C-contiguous before one SIMD argmax per row, so its axis-0
    /// cells pay a full copy.
    /// </para>
    /// <para>
    /// Here the axis is folded in place, one of four ways picked from the strides (element units):
    /// <list type="bullet">
    ///   <item><b>Zero</b> — the axis has stride 0 (a broadcast): every element along it is the same element, so every
    ///   answer is 0 and the result is just allocated zeroed.</item>
    ///   <item><b>Rows</b> — the axis is contiguous: each output folds one contiguous row with the flat SIMD argmax
    ///   kernels (<see cref="DirectILKernelGenerator.ArgMaxDoubleNaNHelper"/> and friends).</item>
    ///   <item><b>Slab</b> — some non-axis dimension (the "lane" run) has a smaller stride than the axis: a chunk of up to
    ///   <see cref="AxisArgMaxChunkLanes"/> outputs is folded together, row after row, keeping each lane's running best
    ///   in L1 scratch and writing the winning row index into the result only when a lane improves. With a ±1 lane
    ///   stride the fold is a <see cref="Vector256{T}"/> compare per row (a −1 run is flipped first); otherwise it is a
    ///   scalar row-major walk that still reads memory in stride order.</item>
    ///   <item><b>AxisWalk</b> — the axis has the smallest stride but is not contiguous: each output folds its axis with
    ///   a scalar strided loop.</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Exact by construction.</b> An argmax is an index, not arithmetic, so the only contract is NumPy's selection
    /// rule — the FIRST occurrence of the extreme value under IEEE comparison (<c>-0.0 == +0.0</c>, so the earlier of the
    /// two wins), with the first NaN winning for float argmax AND argmin (<c>!(v &lt;= best)</c> then stop, NumPy's
    /// <c>@TYPE@_argmax</c>), integers by strict <c>&gt;</c>/<c>&lt;</c>, bool argmax = first nonzero byte (0 when
    /// none), bool argmin = first zero byte (0 when none). Every mode applies that rule in LOGICAL axis order, whatever
    /// the memory order, so the result is the same index sequence NumPy computes.
    /// </para>
    /// <para>
    /// Serves Boolean, the eight integer widths, Char (as its UInt16 bits — identical ordering) and Single / Double.
    /// Half, Decimal and Complex decline to their existing kernels.
    /// </para>
    /// </remarks>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test and benchmark switch for the fast axis argmax / argmin. While <see langword="true"/> on the calling
        /// thread, <see cref="TryExecuteAxisArgFast"/> declines every call and the per-output IL kernel it replaced runs
        /// instead — which is what lets a test compare the two routes on one input, byte for byte.
        /// </summary>
        /// <remarks>
        /// Thread-static on purpose: MSTest runs test classes in parallel, and a process-wide switch flipped by one test
        /// would silently re-route another test's reductions. Production code never sets it.
        /// </remarks>
        [ThreadStatic] internal static bool DisableFastAxisArg;

        /// <summary>
        /// Counts the calls <see cref="TryExecuteAxisArgFast"/> served on the calling thread. A test reads it before and
        /// after a reduction to prove the fast route actually ran — a decline would otherwise pass every value check,
        /// because the fallback computes the same answer.
        /// </summary>
        [ThreadStatic] internal static long FastAxisArgRuns;

        /// <summary>
        /// Bytes of stack scratch one slab chunk keeps its running best values in. Sized to stay L1-resident next to the
        /// chunk's slice of the result, which the fold rewrites whenever a lane improves.
        /// </summary>
        private const int AxisArgScratchBytes = 8192;

        /// <summary>
        /// Most lanes one slab chunk folds at a time. Every improving lane also writes its row index into the int64
        /// result, so a chunk's working set is its best values PLUS 8 result bytes per lane: 1024 lanes keep both inside
        /// L1 even for 8-byte elements (8 KB + 8 KB), where a byte chunk sized by the scratch alone (8192 lanes) would
        /// drag 64 KB of result through the cache on every row of a monotone input.
        /// </summary>
        private const int AxisArgMaxChunkLanes = 1024;

        /// <summary>
        /// Rows shorter than this are folded by the inline scalar loop; longer rows go to the SIMD row kernels. Those
        /// kernels already fall back to scalar below their own vector block (16–64 elements), so this only saves the
        /// call — which matters for many short rows, e.g. <c>(1M, 4)</c> reduced along axis 1.
        /// </summary>
        private const int AxisArgRowSimdMin = 32;

        /// <summary>
        /// Ranks above this decline to the legacy kernel: the plan lives in <c>stackalloc</c> buffers sized by the rank,
        /// and NumSharp has no dimension cap of its own (<c>np.r_</c> can build a 100 000-dim array), so an unbounded
        /// rank would turn the plan into a stack overflow.
        /// </summary>
        private const int AxisArgMaxDims = 64;

        /// <summary>
        /// How the fast axis argmax / argmin walks memory for one call — picked from the strides, never from the dtype.
        /// </summary>
        private enum AxisArgMode
        {
            /// <summary>The reduced axis has stride 0 (broadcast): every element along it is the same one, so every answer is 0.</summary>
            Zero,

            /// <summary>The reduced axis is contiguous (stride 1): each output folds one contiguous row with the SIMD row kernels.</summary>
            Rows,

            /// <summary>The reduced axis has the smallest non-zero stride but is not contiguous: each output folds its axis with a scalar strided loop.</summary>
            AxisWalk,

            /// <summary>A non-axis run of lanes has a smaller stride than the reduced axis: lanes are folded together, row by row.</summary>
            Slab,
        }

        /// <summary>
        /// Everything the fast axis argmax / argmin runner needs, resolved once per call from the operand's dimensions and
        /// strides (element units, not bytes).
        /// </summary>
        /// <remarks>
        /// The four pointer fields point at <c>stackalloc</c> buffers in <see cref="TryExecuteAxisArgFast"/>'s frame, so a
        /// plan must never outlive that call — which is why it is only ever passed down by <see langword="in"/>.
        /// </remarks>
        private unsafe struct AxisArgPlan
        {
            /// <summary>The traversal picked for this call.</summary>
            public AxisArgMode Mode;

            /// <summary>Elements along the reduced axis (always at least 2 here — shorter axes are answered by the caller).</summary>
            public long AxisLength;

            /// <summary>Element stride along the reduced axis; negative for a reversed view.</summary>
            public long AxisStride;

            /// <summary>Slab only: outputs in the lane run (the coalesced non-axis run with the smallest stride).</summary>
            public long LaneCount;

            /// <summary>Slab only: input elements between adjacent lanes, made positive by the flip.</summary>
            public long LaneStride;

            /// <summary>Slab only: result elements between adjacent lanes; negative when the flip reversed the run.</summary>
            public long LaneOutStride;

            /// <summary>Number of outer (odometer) dimensions: every coalesced non-axis run, minus the lane run for Slab.</summary>
            public int OuterCount;

            /// <summary>Extent of each outer dimension, outermost first.</summary>
            public long* OuterDims;

            /// <summary>Input element stride of each outer dimension.</summary>
            public long* OuterIn;

            /// <summary>Result element stride of each outer dimension.</summary>
            public long* OuterOut;

            /// <summary>The odometer's coordinate buffer (reset by <see cref="AxisArgCursor"/>'s constructor).</summary>
            public long* OuterCoord;

            /// <summary>Outer iterations: the product of <see cref="OuterDims"/> (1 when there are none).</summary>
            public long OuterTotal;

            /// <summary>Element offset, from the storage's data start, of the first element the runner reads.</summary>
            public long SrcBase;

            /// <summary>Result element offset of the first lane (non-zero only when the flip reversed the lane run).</summary>
            public long DstBase;
        }

        /// <summary>
        /// C-order odometer over a plan's outer dimensions, tracking the input and result element offsets of the current
        /// outer index incrementally — one add per step, a carry only when a dimension wraps.
        /// </summary>
        private unsafe ref struct AxisArgCursor
        {
            /// <summary>Number of outer dimensions.</summary>
            private readonly int _count;

            /// <summary>Outer extents.</summary>
            private readonly long* _dims;

            /// <summary>Outer input strides.</summary>
            private readonly long* _in;

            /// <summary>Outer result strides.</summary>
            private readonly long* _out;

            /// <summary>Current coordinate per outer dimension.</summary>
            private readonly long* _coord;

            /// <summary>Input element offset of the current outer index, relative to the plan's source base.</summary>
            public long In;

            /// <summary>Result element offset of the current outer index, relative to the plan's result base.</summary>
            public long Out;

            /// <summary>Starts the odometer at the all-zero outer index.</summary>
            /// <param name="p">The plan whose outer dimensions are walked; its coordinate buffer is cleared here.</param>
            public AxisArgCursor(in AxisArgPlan p)
            {
                _count = p.OuterCount;
                _dims = p.OuterDims;
                _in = p.OuterIn;
                _out = p.OuterOut;
                _coord = p.OuterCoord;
                In = 0;
                Out = 0;
                // The coordinate buffer comes from a [SkipLocalsInit] stackalloc: clear it before the first carry reads it.
                for (int d = 0; d < _count; d++)
                    _coord[d] = 0;
            }

            /// <summary>Steps to the next outer index in C order (innermost dimension fastest).</summary>
            /// <remarks>After the last index it wraps to the first one; callers bound the walk by the plan's total.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Advance()
            {
                for (int d = _count - 1; d >= 0; d--)
                {
                    if (++_coord[d] < _dims[d])
                    {
                        In += _in[d];
                        Out += _out[d];
                        return;
                    }

                    // This dimension wrapped: rewind its whole extent and carry into the next outer one.
                    _coord[d] = 0;
                    In -= (_dims[d] - 1) * _in[d];
                    Out -= (_dims[d] - 1) * _out[d];
                }
            }
        }

        /// <summary>
        /// One argmax / argmin selection rule — NumPy's per-dtype <c>@TYPE@_argmax</c> / <c>_argmin</c> predicate — in its
        /// scalar, vector and whole-row forms, as static members so each <c>(T, rule)</c> pair specializes the fold loops
        /// with no call or branch left in the hot path.
        /// </summary>
        /// <typeparam name="T">The lane type as stored (Boolean is read as its bytes, Char as its UInt16 bits).</typeparam>
        internal unsafe interface IArgRule<T> where T : unmanaged
        {
            /// <summary>
            /// Whether <paramref name="v"/>, met LATER along the axis, replaces the running <paramref name="best"/>. Strict,
            /// so an equal value never does — which is what makes the first occurrence win a tie.
            /// </summary>
            /// <param name="v">The candidate element.</param>
            /// <param name="best">The running best (an element met earlier along the axis).</param>
            /// <returns><see langword="true"/> when <paramref name="v"/> becomes the new best.</returns>
            static abstract bool Better(T v, T best);

            /// <summary>
            /// Whether nothing can ever replace <paramref name="best"/> — a NaN for float rules, the dtype's extreme for
            /// integers, a nonzero (argmax) / zero (argmin) byte for Boolean. Lets a fold stop early.
            /// </summary>
            /// <param name="best">The running best.</param>
            /// <returns><see langword="true"/> when the answer can no longer change.</returns>
            static abstract bool Decided(T best);

            /// <summary>The per-lane <see cref="Better"/>: all-ones lanes where <paramref name="v"/> replaces <paramref name="best"/>.</summary>
            /// <param name="v">Candidates from one later row.</param>
            /// <param name="best">The running best of the same lanes.</param>
            /// <returns>The replace mask.</returns>
            static abstract Vector256<T> BetterMask(Vector256<T> v, Vector256<T> best);

            /// <summary>
            /// Whether the slab fold should track undecided lanes to stop early. True only for Boolean, where a lane is
            /// decided by its FIRST hit; for floats (NaN) and integers (the extreme value) the check would cost more than
            /// the rare early exit saves.
            /// </summary>
            static abstract bool TracksUndecided { get; }

            /// <summary>All-ones lanes whose running best can still change (only called when <see cref="TracksUndecided"/>).</summary>
            /// <param name="best">The running best after this row.</param>
            /// <returns>The undecided mask.</returns>
            static abstract Vector256<T> UndecidedMask(Vector256<T> best);

            /// <summary>
            /// Folds one CONTIGUOUS row with the flat SIMD argmax / argmin kernel of the dtype — the same kernel
            /// <c>np.argmax(a)</c> runs, so a row answer is exactly what a flat call on that row would return.
            /// </summary>
            /// <param name="row">First element of the row.</param>
            /// <param name="n">Row length (at least 2).</param>
            /// <returns>The row's argmax / argmin index.</returns>
            static abstract long RowSimd(T* row, long n);
        }

        /// <summary>
        /// Float argmax (Single / Double): NumPy's <c>!(v &lt;= best)</c> rule — the first NaN wins and is final, ties keep
        /// the earlier index (<c>-0.0 == +0.0</c>).
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        internal readonly unsafe struct ArgMaxFloatRule<T> : IArgRule<T> where T : unmanaged, IFloatingPointIeee754<T>
        {
            /// <inheritdoc />
            /// <remarks>
            /// <c>!(v &lt;= best)</c> is true for a larger <paramref name="v"/> AND for a NaN <paramref name="v"/>; the
            /// <c>!IsNaN(best)</c> guard freezes a lane once its best is NaN — the row fold stops there anyway, but the
            /// slab fold keeps visiting such lanes and must not let a later value replace the NaN.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Better(T v, T best) => !T.IsNaN(best) && !(v <= best);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Decided(T best) => T.IsNaN(best);

            /// <inheritdoc />
            /// <remarks><c>ordered(best) AND NOT (v &lt;= best)</c> — the vector spelling of <see cref="Better"/>.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> BetterMask(Vector256<T> v, Vector256<T> best)
                => Vector256.AndNot(Vector256.Equals(best, best), Vector256.LessThanOrEqual(v, best));

            /// <inheritdoc />
            public static bool TracksUndecided
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => false;
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> UndecidedMask(Vector256<T> best) => Vector256<T>.Zero;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static long RowSimd(T* row, long n) => typeof(T) == typeof(double)
                ? DirectILKernelGenerator.ArgMaxDoubleNaNHelper(row, n)
                : DirectILKernelGenerator.ArgMaxFloatNaNHelper(row, n);
        }

        /// <summary>
        /// Float argmin (Single / Double): NumPy's <c>!(v &gt;= best)</c> rule — the first NaN wins and is final, ties keep
        /// the earlier index (<c>-0.0 == +0.0</c>).
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        internal readonly unsafe struct ArgMinFloatRule<T> : IArgRule<T> where T : unmanaged, IFloatingPointIeee754<T>
        {
            /// <inheritdoc />
            /// <remarks>Mirror of <see cref="ArgMaxFloatRule{T}.Better"/>, including the NaN-freezing guard.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Better(T v, T best) => !T.IsNaN(best) && !(v >= best);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Decided(T best) => T.IsNaN(best);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> BetterMask(Vector256<T> v, Vector256<T> best)
                => Vector256.AndNot(Vector256.Equals(best, best), Vector256.GreaterThanOrEqual(v, best));

            /// <inheritdoc />
            public static bool TracksUndecided
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => false;
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> UndecidedMask(Vector256<T> best) => Vector256<T>.Zero;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static long RowSimd(T* row, long n) => typeof(T) == typeof(double)
                ? DirectILKernelGenerator.ArgMinDoubleNaNHelper(row, n)
                : DirectILKernelGenerator.ArgMinFloatNaNHelper(row, n);
        }

        /// <summary>Integer argmax (every signed/unsigned width, Char as UInt16): strict <c>&gt;</c>, so the first maximum wins.</summary>
        /// <typeparam name="T">The integer lane type.</typeparam>
        internal readonly unsafe struct ArgMaxIntRule<T> : IArgRule<T> where T : unmanaged, IBinaryInteger<T>, IMinMaxValue<T>
        {
            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Better(T v, T best) => v > best;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Decided(T best) => best == T.MaxValue;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> BetterMask(Vector256<T> v, Vector256<T> best) => Vector256.GreaterThan(v, best);

            /// <inheritdoc />
            public static bool TracksUndecided
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => false;
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> UndecidedMask(Vector256<T> best) => Vector256<T>.Zero;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static long RowSimd(T* row, long n) => DirectILKernelGenerator.ArgMaxSimdHelper<T>(row, n);
        }

        /// <summary>Integer argmin (every signed/unsigned width, Char as UInt16): strict <c>&lt;</c>, so the first minimum wins.</summary>
        /// <typeparam name="T">The integer lane type.</typeparam>
        internal readonly unsafe struct ArgMinIntRule<T> : IArgRule<T> where T : unmanaged, IBinaryInteger<T>, IMinMaxValue<T>
        {
            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Better(T v, T best) => v < best;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Decided(T best) => best == T.MinValue;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> BetterMask(Vector256<T> v, Vector256<T> best) => Vector256.LessThan(v, best);

            /// <inheritdoc />
            public static bool TracksUndecided
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => false;
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> UndecidedMask(Vector256<T> best) => Vector256<T>.Zero;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static long RowSimd(T* row, long n) => DirectILKernelGenerator.ArgMinSimdHelper<T>(row, n);
        }

        /// <summary>
        /// Boolean argmax over the raw bytes: the FIRST nonzero byte wins (any nonzero byte is True, as in NumPy — a
        /// <c>frombuffer</c> 0x80 counts), and an all-False lane answers 0.
        /// </summary>
        internal readonly unsafe struct ArgMaxBoolRule : IArgRule<byte>
        {
            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Better(byte v, byte best) => v != 0 && best == 0;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Decided(byte best) => best != 0;

            /// <inheritdoc />
            /// <remarks><c>(best == 0) AND NOT (v == 0)</c>.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<byte> BetterMask(Vector256<byte> v, Vector256<byte> best)
                => Vector256.AndNot(Vector256.Equals(best, Vector256<byte>.Zero), Vector256.Equals(v, Vector256<byte>.Zero));

            /// <inheritdoc />
            public static bool TracksUndecided
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => true;
            }

            /// <inheritdoc />
            /// <remarks>A lane is still undecided while it has seen no True.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<byte> UndecidedMask(Vector256<byte> best) => Vector256.Equals(best, Vector256<byte>.Zero);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static long RowSimd(byte* row, long n) => DirectILKernelGenerator.ArgMaxBoolHelper(row, n);
        }

        /// <summary>
        /// Boolean argmin over the raw bytes: the FIRST zero byte wins (NumPy's <c>BOOL_argmin</c> is a <c>memchr</c> for
        /// 0), and an all-True lane answers 0.
        /// </summary>
        internal readonly unsafe struct ArgMinBoolRule : IArgRule<byte>
        {
            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Better(byte v, byte best) => v == 0 && best != 0;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Decided(byte best) => best == 0;

            /// <inheritdoc />
            /// <remarks><c>(v == 0) AND NOT (best == 0)</c>.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<byte> BetterMask(Vector256<byte> v, Vector256<byte> best)
                => Vector256.AndNot(Vector256.Equals(v, Vector256<byte>.Zero), Vector256.Equals(best, Vector256<byte>.Zero));

            /// <inheritdoc />
            public static bool TracksUndecided
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => true;
            }

            /// <inheritdoc />
            /// <remarks>A lane is still undecided while it has seen no False.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<byte> UndecidedMask(Vector256<byte> best) => ~Vector256.Equals(best, Vector256<byte>.Zero);

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static long RowSimd(byte* row, long n) => DirectILKernelGenerator.ArgMinBoolHelper(row, n);
        }

        /// <summary>
        /// Answers <c>np.argmax(arr, axis)</c> / <c>np.argmin(arr, axis)</c> for the dtypes and ranks the fast fold serves,
        /// or returns <see langword="null"/> — having allocated nothing — so the caller runs its legacy kernel.
        /// </summary>
        /// <param name="arr">The operand. Any layout: C, F, strided, reversed, sliced (non-zero offset) or broadcast.</param>
        /// <param name="axis">The reduced axis, already normalized to <c>[0, ndim)</c>.</param>
        /// <param name="axisedShape">
        /// The result shape without keepdims — <paramref name="arr"/>'s dimensions with <paramref name="axis"/> removed. The
        /// result is allocated C-contiguous in this shape, as NumPy's argmax result always is.
        /// </param>
        /// <param name="op"><see cref="ReductionOp.ArgMax"/> or <see cref="ReductionOp.ArgMin"/>.</param>
        /// <returns>
        /// A fresh int64 array of <paramref name="axisedShape"/> holding the first-occurrence index of each output's
        /// extreme along the axis, or <see langword="null"/> when declined: <see cref="DisableFastAxisArg"/> is set, the
        /// rank exceeds <see cref="AxisArgMaxDims"/>, or the dtype is Half / Decimal / Complex.
        /// </returns>
        /// <remarks>
        /// The caller has already answered every degenerate input — empty operands, a 0-d or single-element operand, and an
        /// axis of length 1 — so here the axis length is at least 2 and the operand has at least one element.
        /// </remarks>
        [SkipLocalsInit]
        internal unsafe NDArray TryExecuteAxisArgFast(NDArray arr, int axis, Shape axisedShape, ReductionOp op)
        {
            var shape = arr.Shape;
            int nd = shape.NDim;
            if (DisableFastAxisArg || nd > AxisArgMaxDims)
                return null;

            NPTypeCode tc = arr.GetTypeCode;
            switch (tc)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                    break;
                default:
                    // Half / Decimal / Complex keep their dedicated kernels (no vector compare, 16-byte structs).
                    return null;
            }

            long[] dims = shape.dimensions;
            long[] strides = shape.strides;
            long axisLength = dims[axis];
            long axisStride = strides[axis];

            // The result is C-contiguous over the non-axis dimensions (NumPy's argmax result always is), so each
            // non-axis dimension's result stride is the product of the non-axis extents to its right.
            long* outStride = stackalloc long[nd];
            long outCount = 1;
            for (int d = nd - 1; d >= 0; d--)
            {
                if (d == axis)
                    continue;
                outStride[d] = outCount;
                outCount *= dims[d];
            }

            // Coalesce the non-axis dimensions, outermost first: an outer run merges into the next inner one when it
            // continues it in memory (its stride equals the inner extent × the inner stride). The result strides of two
            // adjacent non-axis dimensions always chain (the result is C-contiguous over exactly these dimensions), so
            // only the input condition decides — and a merged run walks input AND result linearly in one index.
            // Extent-1 dimensions contribute no index and are dropped.
            long* runDim = stackalloc long[nd];
            long* runIn = stackalloc long[nd];
            long* runOut = stackalloc long[nd];
            int runs = 0;
            for (int d = 0; d < nd; d++)
            {
                if (d == axis || dims[d] == 1)
                    continue;
                if (runs > 0 && runIn[runs - 1] == dims[d] * strides[d])
                {
                    runDim[runs - 1] *= dims[d];
                    runIn[runs - 1] = strides[d];
                    runOut[runs - 1] = outStride[d];
                }
                else
                {
                    runDim[runs] = dims[d];
                    runIn[runs] = strides[d];
                    runOut[runs] = outStride[d];
                    runs++;
                }
            }

            // Mode: iterate the SMALLER stride innermost. A stride-0 axis is a broadcast (every answer 0); a contiguous
            // axis is one SIMD row per output. Otherwise compare the axis against the smallest non-zero run stride: when a
            // run is at least as tight (a ±1 run always is, and vectorizes), fold its lanes together row by row; when the
            // axis is tighter, walk each output's axis. Stride-0 runs are never lanes — every lane would re-read one
            // element — so an all-broadcast set of runs walks the axis per output.
            var plan = new AxisArgPlan
            {
                AxisLength = axisLength,
                AxisStride = axisStride,
                SrcBase = shape.offset,
            };
            int lane = -1;
            if (axisStride == 0)
                plan.Mode = AxisArgMode.Zero;
            else if (axisStride == 1)
                plan.Mode = AxisArgMode.Rows;
            else
            {
                long laneAbs = long.MaxValue;
                for (int e = 0; e < runs; e++)
                {
                    long s = Math.Abs(runIn[e]);
                    // <= prefers the innermost of equally tight runs: its result stride is the smallest.
                    if (s != 0 && s <= laneAbs)
                    {
                        laneAbs = s;
                        lane = e;
                    }
                }

                if (lane >= 0 && (laneAbs == 1 || Math.Abs(axisStride) > laneAbs))
                    plan.Mode = AxisArgMode.Slab;
                else
                {
                    plan.Mode = AxisArgMode.AxisWalk;
                    lane = -1;
                }
            }

            // Zero mode is complete the moment the result exists: allocate it zeroed and skip the runner entirely.
            // Every other mode writes every result element (Slab clears its lanes per chunk), so an uninitialized
            // allocation is safe and saves the memset.
            var ret = new NDArray(NPTypeCode.Int64, axisedShape, plan.Mode == AxisArgMode.Zero);
            if (plan.Mode != AxisArgMode.Zero)
            {
                long* outerDims = stackalloc long[nd];
                long* outerIn = stackalloc long[nd];
                long* outerOut = stackalloc long[nd];
                long* outerCoord = stackalloc long[nd];
                int outer = 0;
                long outerTotal = outCount;
                if (plan.Mode == AxisArgMode.Slab)
                {
                    long laneCount = runDim[lane];
                    long laneStride = runIn[lane];
                    long laneOut = runOut[lane];
                    if (laneStride < 0)
                    {
                        // Flip a reversed lane run so the fold reads it ascending: memory lane m is logical lane
                        // L-1-m, so the source starts L-1 elements back and the result is written from its far end
                        // with a negated stride. Only the lanes' storage order changes; each lane still folds its own
                        // axis in logical order, so the answer is unchanged.
                        plan.SrcBase += (laneCount - 1) * laneStride;
                        plan.DstBase += (laneCount - 1) * laneOut;
                        laneStride = -laneStride;
                        laneOut = -laneOut;
                    }

                    plan.LaneCount = laneCount;
                    plan.LaneStride = laneStride;
                    plan.LaneOutStride = laneOut;
                    outerTotal = outCount / laneCount;
                    for (int e = 0; e < runs; e++)
                    {
                        if (e == lane)
                            continue;
                        outerDims[outer] = runDim[e];
                        outerIn[outer] = runIn[e];
                        outerOut[outer] = runOut[e];
                        outer++;
                    }
                }
                else
                {
                    // Rows / AxisWalk: every run is an outer dimension, walked in C order — so the t-th output visited
                    // is result element t and the runner can write the result sequentially.
                    for (int e = 0; e < runs; e++)
                    {
                        outerDims[outer] = runDim[e];
                        outerIn[outer] = runIn[e];
                        outerOut[outer] = runOut[e];
                        outer++;
                    }
                }

                plan.OuterCount = outer;
                plan.OuterDims = outerDims;
                plan.OuterIn = outerIn;
                plan.OuterOut = outerOut;
                plan.OuterCoord = outerCoord;
                plan.OuterTotal = outerTotal;

                void* src = arr.Address;
                long* dst = (long*)ret.Address;
                bool isMax = op == ReductionOp.ArgMax;
                switch (tc)
                {
                    case NPTypeCode.Boolean:
                        if (isMax)
                            RunAxisArg<byte, ArgMaxBoolRule>(plan, src, dst);
                        else
                            RunAxisArg<byte, ArgMinBoolRule>(plan, src, dst);
                        break;
                    case NPTypeCode.Byte: RunAxisArgInt<byte>(isMax, plan, src, dst); break;
                    case NPTypeCode.SByte: RunAxisArgInt<sbyte>(isMax, plan, src, dst); break;
                    case NPTypeCode.Int16: RunAxisArgInt<short>(isMax, plan, src, dst); break;
                    // Char orders exactly like its UInt16 code unit, so it rides the UInt16 instantiation.
                    case NPTypeCode.UInt16:
                    case NPTypeCode.Char: RunAxisArgInt<ushort>(isMax, plan, src, dst); break;
                    case NPTypeCode.Int32: RunAxisArgInt<int>(isMax, plan, src, dst); break;
                    case NPTypeCode.UInt32: RunAxisArgInt<uint>(isMax, plan, src, dst); break;
                    case NPTypeCode.Int64: RunAxisArgInt<long>(isMax, plan, src, dst); break;
                    case NPTypeCode.UInt64: RunAxisArgInt<ulong>(isMax, plan, src, dst); break;
                    case NPTypeCode.Single: RunAxisArgFloat<float>(isMax, plan, src, dst); break;
                    case NPTypeCode.Double: RunAxisArgFloat<double>(isMax, plan, src, dst); break;
                }
            }

            // The kernel read arr's unmanaged buffer through a raw pointer; keep arr reachable until here so a temporary
            // operand (np.argmax(a * b, 1)) cannot be finalized — and its buffer released — mid-fold.
            GC.KeepAlive(arr);
            FastAxisArgRuns++;
            return ret;
        }

        /// <summary>Runs the fast fold for an integer lane type with the argmax or argmin rule.</summary>
        /// <typeparam name="T">The integer lane type (Char arrives as <see cref="ushort"/>).</typeparam>
        /// <param name="isMax"><see langword="true"/> for argmax, <see langword="false"/> for argmin.</param>
        /// <param name="p">The call's plan.</param>
        /// <param name="src">The operand's data start (the plan's offsets are relative to it).</param>
        /// <param name="dst">The result's data start.</param>
        private static unsafe void RunAxisArgInt<T>(bool isMax, in AxisArgPlan p, void* src, long* dst)
            where T : unmanaged, IBinaryInteger<T>, IMinMaxValue<T>
        {
            if (isMax)
                RunAxisArg<T, ArgMaxIntRule<T>>(p, src, dst);
            else
                RunAxisArg<T, ArgMinIntRule<T>>(p, src, dst);
        }

        /// <summary>Runs the fast fold for a float lane type with the argmax or argmin rule.</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="isMax"><see langword="true"/> for argmax, <see langword="false"/> for argmin.</param>
        /// <param name="p">The call's plan.</param>
        /// <param name="src">The operand's data start (the plan's offsets are relative to it).</param>
        /// <param name="dst">The result's data start.</param>
        private static unsafe void RunAxisArgFloat<T>(bool isMax, in AxisArgPlan p, void* src, long* dst)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (isMax)
                RunAxisArg<T, ArgMaxFloatRule<T>>(p, src, dst);
            else
                RunAxisArg<T, ArgMinFloatRule<T>>(p, src, dst);
        }

        /// <summary>
        /// The fast fold's driver: walks the plan's outer dimensions and folds each output (Rows / AxisWalk) or each
        /// chunk of lanes (Slab), writing every result element exactly once per its mode's contract.
        /// </summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule; a struct so the fold specializes per rule.</typeparam>
        /// <param name="p">The call's plan (never Zero mode — that one needs no fold).</param>
        /// <param name="src">The operand's data start; the plan's source offsets are element offsets from it.</param>
        /// <param name="dst">The result's data start (C-contiguous int64).</param>
        [SkipLocalsInit]
        private static unsafe void RunAxisArg<T, TRule>(in AxisArgPlan p, void* src, long* dst)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            T* baseT = (T*)src + p.SrcBase;
            var cur = new AxisArgCursor(p);
            long total = p.OuterTotal;
            long axisLength = p.AxisLength;
            long axisStride = p.AxisStride;

            switch (p.Mode)
            {
                case AxisArgMode.Rows:
                    // Outputs are visited in result order (see the plan), so the result is written sequentially.
                    if (axisLength < AxisArgRowSimdMin)
                    {
                        for (long t = 0; t < total; t++)
                        {
                            dst[t] = ArgFoldStrided<T, TRule>(baseT + cur.In, axisLength, 1);
                            cur.Advance();
                        }
                    }
                    else
                    {
                        for (long t = 0; t < total; t++)
                        {
                            dst[t] = TRule.RowSimd(baseT + cur.In, axisLength);
                            cur.Advance();
                        }
                    }

                    break;

                case AxisArgMode.AxisWalk:
                    for (long t = 0; t < total; t++)
                    {
                        dst[t] = ArgFoldStrided<T, TRule>(baseT + cur.In, axisLength, axisStride);
                        cur.Advance();
                    }

                    break;

                case AxisArgMode.Slab:
                {
                    // One chunk's running best values; fully seeded by FoldSlabChunk before any read, hence
                    // [SkipLocalsInit] above (zeroing 8 KB per call would cost more than a small reduction itself).
                    byte* scratch = stackalloc byte[AxisArgScratchBytes];
                    T* best = (T*)scratch;
                    long laneCount = p.LaneCount;
                    long laneStride = p.LaneStride;
                    long laneOut = p.LaneOutStride;
                    long chunk = Math.Min(AxisArgScratchBytes / sizeof(T), AxisArgMaxChunkLanes);
                    long* dstBase = dst + p.DstBase;
                    for (long t = 0; t < total; t++)
                    {
                        T* row0 = baseT + cur.In;
                        long* lanes = dstBase + cur.Out;
                        for (long l0 = 0; l0 < laneCount; l0 += chunk)
                        {
                            FoldSlabChunk<T, TRule>(row0 + l0 * laneStride, axisStride, axisLength, laneStride,
                                (int)Math.Min(chunk, laneCount - l0), lanes + l0 * laneOut, laneOut, best);
                        }

                        cur.Advance();
                    }

                    break;
                }
            }
        }

        /// <summary>
        /// Scalar fold of one output's axis: NumPy's <c>@TYPE@_argmax</c> loop — seed with element 0, replace on a strict
        /// improvement, stop the moment the best can no longer change.
        /// </summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="p">The output's first element along the axis.</param>
        /// <param name="n">Elements along the axis (at least 1).</param>
        /// <param name="stride">Element stride along the axis (any sign; 1 for a contiguous row).</param>
        /// <returns>The index, in logical axis order, of the first occurrence of the extreme.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long ArgFoldStrided<T, TRule>(T* p, long n, long stride)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            T best = *p;
            if (TRule.Decided(best))
                return 0;
            long bestIndex = 0;
            for (long k = 1; k < n; k++)
            {
                p += stride;
                T v = *p;
                if (TRule.Better(v, best))
                {
                    best = v;
                    bestIndex = k;
                    // The Decided test sits inside the rare improvement branch, so it costs nothing on the common path.
                    if (TRule.Decided(best))
                        break;
                }
            }

            return bestIndex;
        }

        /// <summary>
        /// Folds one chunk of lanes (outputs whose elements sit side by side in memory) along the axis, row by row: row 0
        /// seeds every lane's best with index 0, and each later row replaces the lanes it strictly improves, writing that
        /// row's index into those lanes' result slots.
        /// </summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="row0">Lane 0's element on axis row 0.</param>
        /// <param name="axisStride">Element stride between consecutive axis rows (any sign).</param>
        /// <param name="axisLength">Axis rows (at least 2).</param>
        /// <param name="laneStride">Element stride between adjacent lanes (positive; 1 enables the vector fold).</param>
        /// <param name="lanes">Lanes in this chunk (1 to <see cref="AxisArgMaxChunkLanes"/>, and fitting the scratch).</param>
        /// <param name="dst">Lane 0's result slot.</param>
        /// <param name="dstStride">Result element stride between adjacent lanes (any sign; 1 for a C result).</param>
        /// <param name="best">Scratch for the lanes' running best values (at least <paramref name="lanes"/> elements).</param>
        /// <remarks>
        /// The engine's whole-axis form: <see cref="SeedSlabChunk"/> with row 0, then <see cref="FoldSlabRows"/> over rows
        /// 1..A-1 read in place. The evaluate stream (<c>StreamArgSlabs</c>) calls the same two halves on rows it produces
        /// block by block, which is why they are separate: one fold, two row sources.
        /// </remarks>
        private static unsafe void FoldSlabChunk<T, TRule>(T* row0, long axisStride, long axisLength, long laneStride,
            int lanes, long* dst, long dstStride, T* best)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            if (!SeedSlabChunk<T, TRule>(row0, laneStride, lanes, dst, dstStride, best))
                return;
            FoldSlabRows<T, TRule>(row0 + axisStride, axisStride, 1, axisLength - 1, laneStride, lanes, dst, dstStride, best);
        }

        /// <summary>
        /// Seeds one chunk of lanes with axis row 0: every lane's running best becomes its row-0 element and every
        /// lane's result slot becomes index 0 (the answer for any lane no later row improves).
        /// </summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="row0">Lane 0's element on axis row 0.</param>
        /// <param name="laneStride">Element stride between adjacent lanes (positive).</param>
        /// <param name="lanes">Lanes in this chunk (fitting the scratch).</param>
        /// <param name="dst">Lane 0's result slot.</param>
        /// <param name="dstStride">Result element stride between adjacent lanes (any sign).</param>
        /// <param name="best">Scratch receiving the lanes' running best values.</param>
        /// <returns>
        /// <see langword="false"/> when every lane is already decided (Boolean only — e.g. an argmax chunk whose row 0 is
        /// all True), so no later row can change an answer and the caller may skip the rest of the axis; otherwise
        /// <see langword="true"/>.
        /// </returns>
        /// <remarks>
        /// The result is allocated uninitialized by both callers (the engine and the evaluate stream), so clearing the
        /// slots here is what makes "index 0 unless improved" true.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool SeedSlabChunk<T, TRule>(T* row0, long laneStride, int lanes, long* dst, long dstStride, T* best)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            if (laneStride == 1)
                Unsafe.CopyBlockUnaligned(best, row0, (uint)(lanes * sizeof(T)));
            else
                for (int j = 0; j < lanes; j++)
                    best[j] = row0[j * laneStride];
            if (dstStride == 1)
                new Span<long>(dst, lanes).Clear();
            else
                for (int j = 0; j < lanes; j++)
                    dst[j * dstStride] = 0;

            return !TRule.TracksUndecided || AnyUndecided<T, TRule>(best, lanes);
        }

        /// <summary>
        /// Folds <paramref name="rows"/> consecutive axis rows into a seeded chunk of lanes: row <c>r</c> (at
        /// <c>rowFirst + r·rowStride</c>) carries axis index <c>kFirst + r</c>, and replaces the lanes it strictly improves,
        /// writing that index into their result slots.
        /// </summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="rowFirst">Lane 0's element on the first row folded here.</param>
        /// <param name="rowStride">Element stride between consecutive rows (any sign; unused when <paramref name="rows"/> is 1).</param>
        /// <param name="kFirst">The axis index of the first row folded here (≥ 1: row 0 is the seed).</param>
        /// <param name="rows">Rows to fold (≥ 0).</param>
        /// <param name="laneStride">Element stride between adjacent lanes (positive; 1 enables the vector fold).</param>
        /// <param name="lanes">Lanes in this chunk.</param>
        /// <param name="dst">Lane 0's result slot.</param>
        /// <param name="dstStride">Result element stride between adjacent lanes (any sign; 1 for a C result).</param>
        /// <param name="best">The chunk's running best values, seeded by <see cref="SeedSlabChunk"/>.</param>
        /// <returns>
        /// <see langword="false"/> once every lane is decided (Boolean only), telling a caller that streams the axis in
        /// pieces to stop producing rows; otherwise <see langword="true"/>.
        /// </returns>
        /// <remarks>
        /// Rows must arrive in increasing axis order across calls — the strict "replace only when better" rule is what
        /// keeps the FIRST occurrence, and it only holds when no later row is folded before an earlier one. The vector
        /// fold covers the chunk with full vectors and ends on one OVERLAPPING vector at <c>lanes - C</c> rather than a
        /// scalar tail: re-folding a lane with the same row is idempotent (its best already holds the better of the two
        /// values and <c>Better(x, x)</c> is false for every rule), so the overlap neither changes a best nor writes a
        /// different index.
        /// </remarks>
        private static unsafe bool FoldSlabRows<T, TRule>(T* rowFirst, long rowStride, long kFirst, long rows,
            long laneStride, int lanes, long* dst, long dstStride, T* best)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            int width = Vector256<T>.Count;
            if (laneStride == 1 && Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && lanes >= width)
            {
                int last = lanes - width;
                // ExtractMostSignificantBits sets one bit per lane; all lanes improving is the full mask.
                uint full = width == 32 ? uint.MaxValue : (1u << width) - 1;
                T* rk = rowFirst;
                for (long r = 0; r < rows; r++, rk += rowStride)
                {
                    long k = kFirst + r;
                    Vector256<T> undecided = Vector256<T>.Zero;
                    for (int i = 0; i < last; i += width)
                        FoldStep<T, TRule>(rk, best, dst, dstStride, i, k, full, ref undecided);
                    // The final (possibly overlapping) vector — see remarks.
                    FoldStep<T, TRule>(rk, best, dst, dstStride, last, k, full, ref undecided);
                    // Boolean only: once every lane has met its deciding byte, no later row can change an answer.
                    if (TRule.TracksUndecided && undecided == Vector256<T>.Zero)
                        return false;
                }

                return true;
            }

            // Scalar lanes: a strided lane run, fewer lanes than one vector, or no 256-bit SIMD on this host. Rows are
            // still visited in order and each row's lanes in memory order, so the reads stay sequential.
            T* rp = rowFirst;
            for (long r = 0; r < rows; r++, rp += rowStride)
            {
                long k = kFirst + r;
                bool anyUndecided = false;
                T* q = rp;
                for (int j = 0; j < lanes; j++, q += laneStride)
                {
                    T v = *q;
                    if (TRule.Better(v, best[j]))
                    {
                        best[j] = v;
                        dst[j * dstStride] = k;
                    }

                    if (TRule.TracksUndecided && !anyUndecided && !TRule.Decided(best[j]))
                        anyUndecided = true;
                }

                if (TRule.TracksUndecided && !anyUndecided)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// One vector of one axis row against the running best: replaces the improved lanes, records the row index for
        /// them, and (Boolean only) accumulates the lanes still undecided afterwards.
        /// </summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="rk">Lane 0's element on axis row <paramref name="k"/>.</param>
        /// <param name="best">The chunk's running best values.</param>
        /// <param name="dst">Lane 0's result slot.</param>
        /// <param name="dstStride">Result element stride between adjacent lanes.</param>
        /// <param name="i">First lane of this vector.</param>
        /// <param name="k">The axis row index written into improved lanes.</param>
        /// <param name="full">The extracted-mask value meaning "every lane improved".</param>
        /// <param name="undecided">Accumulates the undecided-lane mask across the row's vectors (Boolean only).</param>
        /// <remarks>
        /// Improvements are rare after the first few rows of a random operand, so the common path is load, compare,
        /// extract, branch-not-taken. A full-vector improvement with a contiguous result (every lane of a monotone
        /// operand, every row) stores the index as whole vectors instead of walking the mask bit by bit.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void FoldStep<T, TRule>(T* rk, T* best, long* dst, long dstStride, int i, long k, uint full,
            ref Vector256<T> undecided)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            var v = Vector256.Load(rk + i);
            var b = Vector256.Load(best + i);
            var improve = TRule.BetterMask(v, b);
            uint bits = Vector256.ExtractMostSignificantBits(improve);
            if (bits != 0)
            {
                b = Vector256.ConditionalSelect(improve, v, b);
                b.Store(best + i);
                long* dp = dst + i * dstStride;
                if (bits == full && dstStride == 1)
                {
                    var kv = Vector256.Create(k);
                    for (int g = 0; g < Vector256<T>.Count; g += Vector256<long>.Count)
                        kv.Store(dp + g);
                }
                else
                {
                    do
                    {
                        dp[BitOperations.TrailingZeroCount(bits) * dstStride] = k;
                        bits &= bits - 1;
                    } while (bits != 0);
                }
            }

            if (TRule.TracksUndecided)
                undecided |= TRule.UndecidedMask(b);
        }

        /// <summary>Whether any lane of a seeded chunk can still change its answer (Boolean early exit).</summary>
        /// <typeparam name="T">The lane type as stored.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="best">The chunk's running best values.</param>
        /// <param name="lanes">Lanes in the chunk.</param>
        /// <returns><see langword="true"/> when at least one lane is not yet decided.</returns>
        private static unsafe bool AnyUndecided<T, TRule>(T* best, int lanes)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            for (int j = 0; j < lanes; j++)
                if (!TRule.Decided(best[j]))
                    return true;
            return false;
        }
    }
}
