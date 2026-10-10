using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using NumSharp.Backends.Iteration;

// =============================================================================
// ComparisonRun.cs — one comparison (==, !=, <, <=, >, >=) of one dtype pair over
// one inner-loop chunk of ANY layout, through the vector comparison kernels
// =============================================================================
//
// WHAT NUMPY DOES (numpy/_core/src/umath/loops_comparison.dispatch.c.src)
//   Its comparison loops are vectorized for exactly three operand forms: both
//   inputs contiguous, or one input a single value (stride 0), with a contiguous
//   bool output. A stepped or reversed input, an input of another dtype (NumPy
//   casts it through a buffer first) and a stepped output all run its scalar
//   BINARY_LOOP, 0.6 - 1 ns an element.
//
// WHAT THIS CLASS DOES
//   It turns EVERY chunk an iterator hands out into those three vectorized forms:
//
//     direct     an input already is a contiguous run of the dtype the kernel
//                compares in, or one value: the kernel reads it in place;
//     staged     any other input (a step, a reversed run, another dtype) is copied
//                and converted 8 KB at a time into a stack block, and the kernel
//                reads the block while it is in the L1 cache. The stride is
//                removed by the house SIMD copy (a step of two, a reversed run:
//                NDPolySeries.CopyInto) or by the unrolled strided move
//                (DirectILKernelGenerator.StridedMove.cs: every other stride),
//                the dtype is converted by the house SIMD cast;
//     flipped    inputs that run BACKWARDS in the kernel's dtype are read forwards
//                from their last element and the bool results are stored reversed,
//                which moves one byte an element where staging moves 8 or 16;
//     scattered  a bool output that is not contiguous receives each block's results
//                through a strided byte store;
//     adopted    an input of another dtype against ONE value: the value is restated
//                in the array's own dtype (or the comparison is decided for the
//                whole chunk), so the array is compared as it is stored. See
//                ScalarAdoption below: the restatement is exact, never a cast.
//
//   A chunk too short to pay for a staging pass (under 64 elements) runs the
//   per-element kernel the iterator routes always used. A long contiguous run that
//   starts off a vector boundary gets a short first kernel call, so that the rest
//   starts on one (AlignmentPeel).
//
//   Two forms belong to a whole iteration of rows (RunRows), not to one chunk:
//
//     tiled      an operand whose successive ROWS are adjacent elements (an F-ordered
//                matrix beside a C-ordered one). One of its rows is a run with a
//                stride of one column, which a row-by-row staging reads one element
//                per cache line. RunTiles copies a tile of several rows at a time
//                (DirectILKernelGenerator.TransposeTile reads each column's rows as
//                one contiguous piece) and compares the tile;
//     repeated   a matrix against ONE short row: the row is repeated in a stack
//                block and several matrix rows are compared per kernel call
//                (RunRepeatedRow), instead of one call a row.
//
// WHO CALLS IT
//   DefaultEngine.TryTrivialContiguousComparisonOp   Run: two contiguous arrays of
//                                                    one shape, or an array and one
//                                                    value, of any two dtypes
//   DefaultEngine.TryRowsComparisonOp                RunRows: an iteration of R runs
//     (DefaultEngine.CompareOp.Rows.cs)              of N elements (a matrix against
//                                                    a row or column, an outer
//                                                    product, a stepped or reversed
//                                                    view), without an iterator
//   NDIterRef.TryExecuteComparisonRun                Chunk: one call per inner-loop
//     (DefaultEngine.TryExecuteComparisonOpViaNDIter, chunk of every other iteration,
//      DefaultEngine.ExecuteComparisonUfuncInto)     and of every out= call
//
// RESULTS
//   Every route gives the bytes of the per-element comparison at NumPy's comparison
//   dtype (result_type of the operands; int64 against uint64 exactly, NumPy's qQ
//   loops). test/NumSharp.Tests/Backends/Kernels/ComparisonRunTests.cs holds each
//   route to an element reference written from NumPy's rules, the raw entry points
//   to "no byte outside the output slots changes", and the run to the routes it
//   replaced (the Disabled switch).
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     One comparison operator over one (left dtype, right dtype) pair, runnable on an inner-loop chunk of any
    ///     stride through the contiguous vector comparison kernels. Instances are cached for the life of the process
    ///     (<see cref="Get"/>); they are immutable after construction and safe to call from any thread.
    /// </summary>
    /// <remarks>
    ///     The chunk contract is NumPy's inner-loop contract (<see cref="NDInnerLoopFunc"/>): three data pointers
    ///     (left, right, bool output), their byte strides, and an element count. A stride of 0 means one value for
    ///     the whole chunk. The output is written one byte per element, 0 or 1; nothing outside the
    ///     <c>count</c> output slots is written.
    /// </remarks>
    internal sealed unsafe class ComparisonRun
    {
        // ---------------------------------------------------------------------------------------------------------
        //  Thresholds
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Elements from which a chunk that needs a staging pass (a stepped or reversed run, a conversion, a
        ///     strided output) takes it. Below it the per-element kernel is faster: the house copy and cast kernels
        ///     are scalar under <see cref="DirectILKernelGenerator.PolyHouseKernelThreshold"/> elements, so a staged
        ///     short chunk would pay two passes of scalar work for one.
        /// </summary>
        internal const long StagedMinElements = DirectILKernelGenerator.PolyHouseKernelThreshold;

        /// <summary>
        ///     Elements per ROW from which an iteration of many direct rows (a broadcast row or column, an outer
        ///     product) is driven one vector-kernel call per row. Narrower rows keep the iterator's 2-D block kernel,
        ///     which loops the rows inside one call: a kernel call costs about 12 ns, a row of 16 scalar compares
        ///     about the same.
        /// </summary>
        internal const long DirectRowMinElements = 16;

        /// <summary>
        ///     Bytes of one staging block when ONE input is staged. The block, the source lines it is filled from
        ///     (twice the block for a step of two), the other input's part and the results stay inside a 48 KB L1
        ///     data cache. With BOTH inputs staged each takes half of it, for the same reason.
        /// </summary>
        private const int BlockBytes = 8192;

        /// <summary>
        ///     Bytes from which a contiguous input run is worth aligning to the vector width before the kernel reads
        ///     it (see <see cref="AlignmentPeel"/>). Under it the run is L1-resident, where a load that crosses a
        ///     cache line costs about a nanosecond for the whole run, less than the extra kernel call.
        /// </summary>
        private const long AlignMinBytes = 4096;

        /// <summary>Bytes of one vector load of the comparison kernels on this host (0: no vector kernels).</summary>
        private static readonly int s_vectorBytes = DirectILKernelGenerator.VectorBits / 8;

        /// <summary>
        ///     Trailing one-byte elements of a row from which <see cref="RunRows"/> compares the row's last full vector
        ///     a second time instead of leaving them to the kernel's scalar tail: the second kernel call costs about
        ///     as much as five scalar compares.
        /// </summary>
        private const long OverlapTailMinElements = 8;

        /// <summary>
        ///     Bytes of one row up to which <see cref="RunRows"/> compares a matrix against ONE row by repeating the
        ///     row in a stack block and comparing several matrix rows per kernel call. A kernel call costs about 8 ns
        ///     whatever it compares, which is most of a short row (400 one-byte elements take 10 ns). Measured, kernel
        ///     time of the repeated form against a call per row: 0.31x for (25, 40) of one byte, 0.52x for (250, 400)
        ///     of one byte, 0.86x for (1000, 100) of eight bytes (800 bytes a row), 0.95x for (250, 400) of four
        ///     bytes (1,600), and 1.05x for (250, 400) of eight bytes (3,200), where the block's copy costs more than
        ///     the calls it saves.
        /// </summary>
        private const long RepeatedRowMaxBytes = 2048;

        /// <summary>
        ///     Fewest rows from which <see cref="RunRows"/> repeats a row: with fewer rows the block's copy saves at
        ///     most two kernel calls.
        /// </summary>
        private const long RepeatedRowMinRows = 16;

        /// <summary>
        ///     Fewest rows of one tile of <see cref="RunTiles"/> (when the operand has that many, and never less than
        ///     one vector group of the transposition). More rows per tile give more contiguous bytes for every cache
        ///     line and page the tile touches, but fewer columns per tile, and so more kernel calls a row once a row
        ///     no longer fits a tile. Measured (a C-ordered (250, 400) matrix against an F-ordered one, us a call,
        ///     for 4 / 8 / 16 rows): float64 40.5 / 35.3 / 43.2, float32 19.2 / 21.2 / 27.0; and for (1000, 1000):
        ///     float64 791 / 641 / 753, float32 414 / 340 / 334. Eight rows make the row pieces of a tile 128 to 512
        ///     elements long, about twice the cost of the kernel call that compares them.
        /// </summary>
        private const long TileMinRows = 8;

        /// <summary>
        ///     Bytes one row of a tiled operand spans in memory (elements a row times the column stride) from which
        ///     <see cref="RunTiles"/> makes its tiles taller (<see cref="TallTileRows"/>). From about this span each
        ///     column piece is a cache miss and, further up, a TLB miss, which more rows a piece pay for better than
        ///     longer row pieces do. Measured (us a call for 8 / 16 / 32 rows): (1448, 1448) float32 1550 / 1355 /
        ///     1456, int16 625 / 449 / 514; (3162, 3162) float64 17330 / 15261 / 17221, float32 11993 / 8726 / 8855,
        ///     int16 9350 / 5545 / 4686, uint8 (16 / 32 / 64 rows) 3557 / 2926 / 2643. At (1000, 1000) float64, a span
        ///     of 8 MB, 16 rows cost 6 % (796 against 751).
        /// </summary>
        private const long TileTallSpanBytes = 4 << 20;

        // ---------------------------------------------------------------------------------------------------------
        //  Cache
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>Slots per dtype in <see cref="s_runs"/>: every dtype code is under 32 except
        ///     <see cref="NPTypeCode.Complex"/> (128), which masks to the unused slot 0.</summary>
        private const int DtypeSlots = 32;

        /// <summary>Number of <see cref="ComparisonOp"/> values.</summary>
        private const int OpCount = 6;

        /// <summary>
        ///     One entry per (operator, left dtype, right dtype). An entry is replaced only by an equivalent one (two
        ///     threads racing on a first call both build it), so a reader needs no lock.
        /// </summary>
        private static readonly ComparisonRun[] s_runs = new ComparisonRun[OpCount * DtypeSlots * DtypeSlots];

        /// <summary>
        ///     Test and measurement hook: when true, <see cref="Get"/> answers null for every pair, so every comparison
        ///     takes the routes it took before this class existed (the per-element kernels and the iterator). It is a
        ///     process-wide switch read on every comparison call; set it only from single-threaded test code and
        ///     restore it in a <c>finally</c>.
        /// </summary>
        internal static bool Disabled;

        /// <summary>
        ///     Test and measurement hook: when true, <see cref="RunRows"/> stages a tiled operand one row at a time,
        ///     as it did before <see cref="RunTiles"/> existed. The results are the same bytes. Process-wide; set it
        ///     only from single-threaded test code and restore it in a <c>finally</c>.
        /// </summary>
        internal static bool TilesDisabled;

        /// <summary>
        ///     Test and measurement hook: when true, <see cref="RunRows"/> compares a matrix against one row with one
        ///     kernel call a row (and <see cref="RepeatsRow"/> answers false). The results are the same bytes.
        ///     Process-wide; set it only from single-threaded test code and restore it in a <c>finally</c>.
        /// </summary>
        internal static bool RepeatedRowDisabled;

        /// <summary>
        ///     Test hook: calls of <see cref="RunRows"/> that ran tile by tile (<see cref="RunTiles"/>) since the
        ///     process started. A plain counter, not synchronized: tests read the difference around calls they make
        ///     on one thread, and a lost increment under concurrent comparisons changes nothing else.
        /// </summary>
        internal static long TileRuns;

        /// <summary>
        ///     Test hook: calls of <see cref="RunRows"/> that compared a matrix against one repeated row
        ///     (<see cref="RunRepeatedRow"/>). See <see cref="TileRuns"/> for its limits.
        /// </summary>
        internal static long RepeatedRowRuns;

        /// <summary>
        ///     The run for <paramref name="op"/> over the dtype pair, or null when this host cannot serve it (runtime
        ///     IL generation is off, or a kernel the run needs does not exist for the pair) or the run is switched off
        ///     (<see cref="Disabled"/>). Apart from that switch a null answer is final for the process; the caller
        ///     then keeps its per-element route.
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">Stored dtype of the left operand.</param>
        /// <param name="rhsType">Stored dtype of the right operand.</param>
        /// <returns>The cached run, or null.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ComparisonRun Get(ComparisonOp op, NPTypeCode lhsType, NPTypeCode rhsType)
        {
            if ((uint)op >= OpCount || Disabled)
                return null;
            int slot = ((int)op * DtypeSlots + ((int)lhsType & (DtypeSlots - 1))) * DtypeSlots + ((int)rhsType & (DtypeSlots - 1));
            var run = Volatile.Read(ref s_runs[slot]);
            // The slot index folds dtype codes modulo 32, so an entry carries its exact pair and a colliding pair
            // rebuilds (no storage dtype collides today; the check keeps that true for a future one).
            if (run is null || run.LhsType != lhsType || run.RhsType != rhsType)
            {
                run = new ComparisonRun(op, lhsType, rhsType);
                Volatile.Write(ref s_runs[slot], run);
            }
            return run._available ? run : null;
        }

        // ---------------------------------------------------------------------------------------------------------
        //  State
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>The comparison, as <c>left op right</c>.</summary>
        public readonly ComparisonOp Op;

        /// <summary>Stored dtype of the left operand.</summary>
        public readonly NPTypeCode LhsType;

        /// <summary>Stored dtype of the right operand.</summary>
        public readonly NPTypeCode RhsType;

        /// <summary>
        ///     The inner-loop kernel: <see cref="Chunk"/> as a delegate, for <c>NDIterRef.ForEach</c>. Null when the
        ///     run is not available (such a run is never returned by <see cref="Get"/>).
        /// </summary>
        public readonly NDInnerLoopFunc InnerLoop;

        /// <summary>False when construction could not resolve every kernel; <see cref="Get"/> then returns null.</summary>
        private readonly bool _available;

        /// <summary>Dtype the kernels read on the left: the comparison dtype, or the stored dtype itself for the
        ///     int64 / uint64 pair (compared exactly without a conversion).</summary>
        private readonly NPTypeCode _lhsKernelType;

        /// <summary>Dtype the kernels read on the right (see <see cref="_lhsKernelType"/>).</summary>
        private readonly NPTypeCode _rhsKernelType;

        /// <summary>Item size of <see cref="LhsType"/>.</summary>
        private readonly int _lhsSize;

        /// <summary>Item size of <see cref="RhsType"/>.</summary>
        private readonly int _rhsSize;

        /// <summary>Item size of the larger kernel dtype: sizes one staging block.</summary>
        private readonly int _kernelUnit;

        /// <summary>The left operand is stored in another dtype than the kernels read.</summary>
        private readonly bool _convertLhs;

        /// <summary>The right operand is stored in another dtype than the kernels read.</summary>
        private readonly bool _convertRhs;

        /// <summary>Both inputs contiguous runs of the kernel dtypes.</summary>
        private readonly ComparisonKernel _full;

        /// <summary>Left a contiguous run, right one value (both of the kernel dtypes).</summary>
        private readonly ComparisonKernel _scalarRight;

        /// <summary>Left one value, right a contiguous run (both of the kernel dtypes).</summary>
        private readonly ComparisonKernel _scalarLeft;

        /// <summary>Converts ONE left value into the kernel dtype (null when no conversion is needed).</summary>
        private readonly PolyCastKernel _lhsValueCast;

        /// <summary>Converts ONE right value into the kernel dtype (null when no conversion is needed).</summary>
        private readonly PolyCastKernel _rhsValueCast;

        /// <summary>The per-element kernel (any strides, converts each element): short chunks.</summary>
        private readonly NDInnerLoopFunc _elementwise;

        /// <summary>
        ///     Strided one-byte move: a block of results into an output that is not contiguous, and one result byte
        ///     into every slot of a decided chunk (source stride 0).
        /// </summary>
        private readonly StridedMoveKernel _byteMove;

        /// <summary>Strided move of left-dtype elements: stages a left run of a stride no vector copy reads.</summary>
        private readonly StridedMoveKernel _lhsMove;

        /// <summary>Strided move of right-dtype elements (see <see cref="_lhsMove"/>).</summary>
        private readonly StridedMoveKernel _rhsMove;

        /// <summary>
        ///     SIMD strided one-byte copy, used to store a block of results reversed (null when the house has none;
        ///     <see cref="_byteMove"/> then serves). It is the uint8 copy: a bool result is one byte, and the house
        ///     has the shuffle-reverse kernel for the integer byte dtypes only (33 ns for 1,024 bytes, against 418 ns
        ///     one byte at a time).
        /// </summary>
        private readonly DirectILKernelGenerator.StridedCastKernel _byteCopy;

        /// <summary>
        ///     The dtype a LEFT run is copied as when it is brought to unit stride without a conversion: the stored
        ///     dtype, except bool, which is copied as uint8 (the same bytes; the house vector copies for a step of two
        ///     and for a reversed run exist for the integer byte dtypes, not for bool: 0.04 ns an element against 0.38).
        /// </summary>
        private readonly NPTypeCode _lhsCopyType;

        /// <summary>The dtype a RIGHT run is copied as (see <see cref="_lhsCopyType"/>).</summary>
        private readonly NPTypeCode _rhsCopyType;

        /// <summary>Restates a right VALUE in the left array's dtype (null when the left array needs no conversion,
        ///     or the pair cannot be restated exactly).</summary>
        private readonly ScalarAdoption _adoptRight;

        /// <summary>Restates a left VALUE in the right array's dtype (see <see cref="_adoptRight"/>).</summary>
        private readonly ScalarAdoption _adoptLeft;

        /// <summary>Sixteen bytes of stack for one converted value of any dtype.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct ValueSlot
        {
            /// <summary>Low eight bytes.</summary>
            public ulong Lo;
            /// <summary>High eight bytes (complex128 and decimal).</summary>
            public ulong Hi;
        }

        // ---------------------------------------------------------------------------------------------------------
        //  Construction
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Resolves every kernel the run can need. On a host that cannot provide one, the run is left unavailable
        ///     (no exception leaves the constructor: a missing kernel only means the caller keeps its other route).
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">Stored dtype of the left operand.</param>
        /// <param name="rhsType">Stored dtype of the right operand.</param>
        private ComparisonRun(ComparisonOp op, NPTypeCode lhsType, NPTypeCode rhsType)
        {
            Op = op;
            LhsType = lhsType;
            RhsType = rhsType;
            try
            {
                if (!DirectILKernelGenerator.Enabled)
                    return;

                // The dtype each side is compared in. NumPy's rule: the operands' result_type — except int64 with
                // uint64, which NumPy compares exactly in its qQ / Qq loops instead of in float64. The mixed-sign
                // kernels read both stored dtypes directly, so that pair needs no conversion at all.
                bool mixedSign = (lhsType == NPTypeCode.Int64 && rhsType == NPTypeCode.UInt64)
                                 || (lhsType == NPTypeCode.UInt64 && rhsType == NPTypeCode.Int64);
                if (lhsType == rhsType || mixedSign)
                {
                    _lhsKernelType = lhsType;
                    _rhsKernelType = rhsType;
                }
                else
                {
                    _lhsKernelType = _rhsKernelType = np._FindCommonScalarType(lhsType, rhsType);
                }

                _lhsSize = DirectILKernelGenerator.GetTypeSize(lhsType);
                _rhsSize = DirectILKernelGenerator.GetTypeSize(rhsType);
                _kernelUnit = Math.Max(DirectILKernelGenerator.GetTypeSize(_lhsKernelType), DirectILKernelGenerator.GetTypeSize(_rhsKernelType));
                _convertLhs = lhsType != _lhsKernelType;
                _convertRhs = rhsType != _rhsKernelType;

                _full = DirectILKernelGenerator.GetComparisonKernel(new ComparisonKernelKey(_lhsKernelType, _rhsKernelType, op, ExecutionPath.SimdFull));
                _scalarRight = DirectILKernelGenerator.GetComparisonKernel(new ComparisonKernelKey(_lhsKernelType, _rhsKernelType, op, ExecutionPath.SimdScalarRight));
                _scalarLeft = DirectILKernelGenerator.GetComparisonKernel(new ComparisonKernelKey(_lhsKernelType, _rhsKernelType, op, ExecutionPath.SimdScalarLeft));

                // The staging copy's last resort for a pair is the scalar convert-copy; resolving it here proves that
                // CopyInto can serve the pair (it throws NotSupportedException for one the house cannot convert).
                var lhsCast = DirectILKernelGenerator.GetPolyCastKernel(lhsType, _lhsKernelType);
                var rhsCast = DirectILKernelGenerator.GetPolyCastKernel(rhsType, _rhsKernelType);
                _lhsValueCast = _convertLhs ? lhsCast : null;
                _rhsValueCast = _convertRhs ? rhsCast : null;

                _byteMove = DirectILKernelGenerator.GetStridedMoveKernel(1);
                _lhsMove = DirectILKernelGenerator.GetStridedMoveKernel(_lhsSize);
                _rhsMove = DirectILKernelGenerator.GetStridedMoveKernel(_rhsSize);
                if (_byteMove is null || _lhsMove is null || _rhsMove is null)
                    return;
                _byteCopy = DirectILKernelGenerator.GetPolyStridedCast(NPTypeCode.Byte, NPTypeCode.Byte);
                _lhsCopyType = lhsType == NPTypeCode.Boolean ? NPTypeCode.Byte : lhsType;
                _rhsCopyType = rhsType == NPTypeCode.Boolean ? NPTypeCode.Byte : rhsType;

                _elementwise = CompileElementwise(op, lhsType, rhsType);

                _adoptRight = _convertLhs ? ScalarAdoption.TryCreate(this, lhsType, rhsType, _lhsKernelType, scalarOnLeft: false) : null;
                _adoptLeft = _convertRhs ? ScalarAdoption.TryCreate(this, rhsType, lhsType, _rhsKernelType, scalarOnLeft: true) : null;

                InnerLoop = Chunk;
                _available = true;
            }
            catch (NotSupportedException)
            {
                // A dtype pair the house conversion or comparison emitter does not support.
            }
            catch (InvalidOperationException)
            {
                // IL generation switched off between the Enabled test and a kernel request.
            }
        }

        /// <summary>
        ///     The per-element kernel of the pair: loads one element of each stored dtype, converts both to NumPy's
        ///     comparison dtype and compares. It serves any strides, so it is the route of chunks too short to stage.
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">Stored dtype of the left operand.</param>
        /// <param name="rhsType">Stored dtype of the right operand.</param>
        /// <returns>The compiled (cached) kernel.</returns>
        /// <exception cref="NotSupportedException">The emitter has no conversion or comparison for the pair.</exception>
        /// <remarks>
        ///     The body is the one the iterator routes of <c>DefaultEngine</c> emit, with the comparison dtype of
        ///     <see cref="ComparisonKernelKey.GetComparisonType"/> (int64 against uint64 in <see cref="decimal"/>:
        ///     exact), and it is registered under the same key, so the run and those routes share one compiled kernel.
        /// </remarks>
        private static NDInnerLoopFunc CompileElementwise(ComparisonOp op, NPTypeCode lhsType, NPTypeCode rhsType)
        {
            NPTypeCode comparisonType = ComparisonKernelKey.GetComparisonType(lhsType, rhsType);
            var key = InnerLoopKernelKey.Comparison(op, lhsType, rhsType);
            return DirectILKernelGenerator.CompileInnerLoop(
                new[] { lhsType, rhsType, NPTypeCode.Boolean },
                il =>
                {
                    // Stack on entry: [lhs value, rhs value].
                    if (lhsType == rhsType && lhsType == comparisonType)
                    {
                        DirectILKernelGenerator.EmitComparisonOperation(il, op, comparisonType);
                        return;
                    }
                    // The left value is under the right one: park the right, convert the left, bring the right back.
                    var right = il.DeclareLocal(DirectILKernelGenerator.GetClrType(rhsType));
                    il.Emit(OpCodes.Stloc, right);
                    if (lhsType != comparisonType)
                        DirectILKernelGenerator.EmitConvertTo(il, lhsType, comparisonType);
                    il.Emit(OpCodes.Ldloc, right);
                    if (rhsType != comparisonType)
                        DirectILKernelGenerator.EmitConvertTo(il, rhsType, comparisonType);
                    DirectILKernelGenerator.EmitComparisonOperation(il, op, comparisonType);
                },
                vectorBody: null,
                in key);
        }

        // ---------------------------------------------------------------------------------------------------------
        //  Entry points
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Compares one run: <c>out[i] = left[i] op right[i]</c> for <paramref name="n"/> elements.
        /// </summary>
        /// <param name="l">First left element.</param>
        /// <param name="ls">Byte stride of the left operand (0: one value).</param>
        /// <param name="r">First right element.</param>
        /// <param name="rs">Byte stride of the right operand (0: one value).</param>
        /// <param name="o">First output byte.</param>
        /// <param name="os">Byte stride of the output.</param>
        /// <param name="n">Element count.</param>
        /// <remarks>The output may be the same memory as an input of the same item size walked with the same stride
        ///     (an in-place <c>out=</c>): every route reads a block or a vector completely before it stores it.</remarks>
        public void Run(byte* l, long ls, byte* r, long rs, byte* o, long os, long n)
        {
            void** ptrs = stackalloc void*[3];
            long* strides = stackalloc long[3];
            ptrs[0] = l;
            ptrs[1] = r;
            ptrs[2] = o;
            strides[0] = ls;
            strides[1] = rs;
            strides[2] = os;
            Chunk(ptrs, strides, n, null);
        }

        /// <summary>
        ///     Whether an iteration whose every chunk has these inner strides and this length is faster through this
        ///     run than through the iterator's per-element kernels. The strides of an unbuffered iteration do not
        ///     change between chunks, so the caller asks once.
        /// </summary>
        /// <param name="ls">Inner byte stride of the left operand.</param>
        /// <param name="rs">Inner byte stride of the right operand.</param>
        /// <param name="os">Inner byte stride of the output.</param>
        /// <param name="count">Elements per chunk.</param>
        /// <param name="singleChunk">The whole iteration is one chunk.</param>
        /// <returns>True to drive <see cref="InnerLoop"/>; false to keep the per-element route.</returns>
        /// <remarks>
        ///     A chunk the kernels read in place costs one kernel call, so one chunk is always worth it and many
        ///     chunks are from <see cref="DirectRowMinElements"/> elements each; so are chunks of runs that go
        ///     backwards in the kernel dtype (one kernel call and one reversed byte store). A chunk that needs a
        ///     staging pass is worth it from <see cref="StagedMinElements"/> elements (see that constant).
        /// </remarks>
        public bool Prefers(long ls, long rs, long os, long count, bool singleChunk)
        {
            bool lRun = ls == _lhsSize && !_convertLhs;
            bool rRun = rs == _rhsSize && !_convertRhs;
            bool direct = os == 1 && ((lRun && (rRun || rs == 0)) || (ls == 0 && rRun));
            if (direct)
                return singleChunk || count >= DirectRowMinElements;
            if (IsBackwardForm(ls, rs, os))
                return count >= DirectRowMinElements;
            return count >= StagedMinElements;
        }

        /// <summary>
        ///     Whether a chunk with a contiguous output is one the kernels read in place once it is turned around:
        ///     every input is a run that goes BACKWARDS in the kernel's own dtype or is one value, and at least one is
        ///     such a run.
        /// </summary>
        /// <param name="ls">Left byte stride.</param>
        /// <param name="rs">Right byte stride.</param>
        /// <param name="os">Output byte stride.</param>
        /// <returns>True for the backward form.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool IsBackwardForm(long ls, long rs, long os)
        {
            if (os != 1)
                return false;
            bool lBackward = ls == -_lhsSize && !_convertLhs;
            bool rBackward = rs == -_rhsSize && !_convertRhs;
            return (lBackward || rBackward) && (lBackward || ls == 0) && (rBackward || rs == 0);
        }

        // ---------------------------------------------------------------------------------------------------------
        //  The chunk
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The inner-loop kernel (<see cref="NDInnerLoopFunc"/>): operands 0 and 1 are the inputs, operand 2 the
        ///     bool output.
        /// </summary>
        /// <param name="dataptrs">The three data pointers.</param>
        /// <param name="strides">The three byte strides.</param>
        /// <param name="count">Element count of the chunk.</param>
        /// <param name="auxdata">Unused (passed on to the per-element kernel).</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Chunk(void** dataptrs, long* strides, long count, void* auxdata)
        {
            if (count <= 0)
                return;

            byte* l = (byte*)dataptrs[0], r = (byte*)dataptrs[1], o = (byte*)dataptrs[2];
            long ls = strides[0], rs = strides[1], os = strides[2];

            // The three forms the vector kernels read in place. A single value of another dtype is converted once
            // (it costs one scalar conversion; the array beside it is already what the kernel compares).
            if (os == 1)
            {
                if (ls == _lhsSize && !_convertLhs)
                {
                    if (rs == _rhsSize && !_convertRhs)
                    {
                        InPlace(_full, l, ls, r, rs, o, count, AlignmentPeel(l, _lhsSize, o, count));
                        return;
                    }
                    if (rs == 0)
                    {
                        ValueSlot value;
                        if (_convertRhs)
                        {
                            _rhsValueCast(r, 1, 0, (byte*)&value);
                            r = (byte*)&value;
                        }
                        InPlace(_scalarRight, l, ls, r, 0, o, count, AlignmentPeel(l, _lhsSize, o, count));
                        return;
                    }
                }
                else if (ls == 0 && rs == _rhsSize && !_convertRhs)
                {
                    ValueSlot value;
                    if (_convertLhs)
                    {
                        _lhsValueCast(l, 1, 0, (byte*)&value);
                        l = (byte*)&value;
                    }
                    InPlace(_scalarLeft, l, 0, r, rs, o, count, AlignmentPeel(r, _rhsSize, o, count));
                    return;
                }
            }

            // Too short to pay for a staging pass: one element at a time, any strides. A chunk of backward runs needs
            // no staging (one kernel call and one reversed store), so it is taken from fewer elements.
            if (count < StagedMinElements && !(count >= DirectRowMinElements && IsBackwardForm(ls, rs, os)))
            {
                _elementwise(dataptrs, strides, count, auxdata);
                return;
            }

            Staged(l, ls, r, rs, o, os, count);
        }

        /// <summary>
        ///     Elements to compare BEFORE the kernel's main call so that the rest of the chunk starts on a vector-width
        ///     boundary of the stream whose misalignment costs most; 0 when that stream is aligned already, the chunk
        ///     is too short to matter, or the stream cannot be aligned.
        /// </summary>
        /// <param name="run">First element of the contiguous input run (the array operand of the form).</param>
        /// <param name="itemSize">Its item size (a power of two).</param>
        /// <param name="o">First output byte (contiguous).</param>
        /// <param name="n">Element count.</param>
        /// <returns>The element count of the prefix.</returns>
        /// <remarks>
        ///     <para>A run that starts in the middle of a 32-byte vector makes every second vector load or store cross
        ///     a cache line. Measured on the kernels, in ns an element with the operands in the L2 cache (100,000
        ///     elements; input offset / output offset in bytes past a 64-byte boundary):</para>
        ///     <code>
        ///                          0/0     16/0    0/16    16/16
        ///     float64  two arrays  0.141   0.250   0.141   0.222
        ///     float64  a value     0.093   0.129   0.093   0.126
        ///     float32  two arrays  0.074   0.126   0.076   0.114
        ///     int16    a value     0.035   0.043   0.044   0.048
        ///     uint8    two arrays  0.033   0.040   0.042   0.045
        ///     uint8    a value     0.026   0.029   0.031   0.036
        ///     </code>
        ///     <para>So an input of two or more bytes an element is the stream to align (a split load every second
        ///     vector, against one split store in 2 to 8 input vectors), and for one-byte elements it is the OUTPUT
        ///     (as many stores as loads, and a split store costs more than a split load). There is no difference
        ///     once the operands are in DRAM, and under <see cref="AlignMinBytes"/> the gain is less than the extra
        ///     kernel call.</para>
        ///     <para>A prefix aligns ONE stream. Aligning every stream is the allocator's job, and the buffer pool
        ///     does it: a buffer of one page or more starts on a 64-byte boundary
        ///     (<c>SizeBucketedBufferPool.AlignedMinBytes</c>), so for whole arrays the prefix is zero. It still
        ///     serves a view that starts inside its buffer (<c>a[1:]</c>, a row of a matrix whose width is not a
        ///     multiple of the vector) and memory the pool did not allocate.</para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long AlignmentPeel(byte* run, int itemSize, byte* o, long n)
        {
            int vector = s_vectorBytes;
            if (vector <= 16 || n * itemSize < AlignMinBytes)
                return 0;                                   // 16 bytes is the allocator's own alignment
            if (itemSize == 1)
                return (vector - ((long)o & (vector - 1))) & (vector - 1);
            long misaligned = (long)run & (vector - 1);
            // A run that is not even item-aligned (a byte offset into a buffer) never reaches a vector boundary.
            if (misaligned == 0 || (misaligned & (itemSize - 1)) != 0)
                return 0;
            return (vector - misaligned) / itemSize;
        }

        /// <summary>
        ///     One of the three in-place forms: the kernel over <paramref name="n"/> elements, with the first
        ///     <paramref name="peel"/> of them in a call of their own (see <see cref="AlignmentPeel"/>).
        /// </summary>
        /// <param name="kernel">The kernel of the form.</param>
        /// <param name="l">Left operand (a run of the kernel dtype, or one value).</param>
        /// <param name="ls">Left byte stride: the item size, or 0 for one value.</param>
        /// <param name="r">Right operand.</param>
        /// <param name="rs">Right byte stride: the item size, or 0 for one value.</param>
        /// <param name="o">Output (contiguous).</param>
        /// <param name="n">Element count.</param>
        /// <param name="peel">Leading elements to compare separately (0: none).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void InPlace(ComparisonKernel kernel, byte* l, long ls, byte* r, long rs, byte* o, long n, long peel)
        {
            if (peel != 0)
            {
                // Fewer elements than one vector: the kernel's own scalar tail compares them.
                kernel(l, r, (bool*)o, null, null, null, 1, peel);
                l += peel * ls;
                r += peel * rs;
                o += peel;
                n -= peel;
            }
            kernel(l, r, (bool*)o, null, null, null, 1, n);
        }

        /// <summary>
        ///     Compares <paramref name="rows"/> runs of <paramref name="count"/> elements each, the operands of
        ///     successive runs one outer stride apart: a whole 2-level iteration (a matrix against a row or a column,
        ///     an outer product, stepped or reversed rows) without an iterator.
        /// </summary>
        /// <param name="l">First left element of the first run.</param>
        /// <param name="lInner">Left byte stride inside a run (0: one value per run).</param>
        /// <param name="lOuter">Left byte stride between the starts of two successive runs.</param>
        /// <param name="r">First right element of the first run.</param>
        /// <param name="rInner">Right byte stride inside a run.</param>
        /// <param name="rOuter">Right byte stride between runs.</param>
        /// <param name="o">First output byte of the first run.</param>
        /// <param name="oInner">Output byte stride inside a run.</param>
        /// <param name="oOuter">Output byte stride between runs.</param>
        /// <param name="rows">Number of runs.</param>
        /// <param name="count">Elements per run.</param>
        /// <remarks>
        ///     <para><b>The output must not overlap either input.</b> Unlike <see cref="Run"/>, this method may compute
        ///     the last elements of a run twice; its caller hands it a fresh result.</para>
        ///     <para>When every run has one of the three in-place forms, the kernel and the stride tests are resolved
        ///     once and each run costs one kernel call (7.8 ns for 40 float64 values, against 23 ns through the
        ///     iterator's per-chunk driver); a matrix against one short row shares kernel calls between rows
        ///     (<see cref="RunRepeatedRow"/>). Runs that go backwards in the kernel dtype (reversed columns) are turned
        ///     around once: each costs one kernel call into a stack block and one reversed byte store. An operand whose
        ///     successive runs are adjacent elements (an F-ordered matrix) is staged a tile of several runs at a time
        ///     (<see cref="RunTiles"/>). Every other layout is one <see cref="Chunk"/> per run.</para>
        ///     <para>[SkipLocalsInit]: the result block of the backward form is written by the kernel before it is
        ///     read.</para>
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void RunRows(byte* l, long lInner, long lOuter, byte* r, long rInner, long rOuter, byte* o, long oInner, long oOuter,
            long rows, long count)
        {
            if (rows <= 0 || count <= 0)
                return;

            if (oInner == 1)
            {
                bool lRun = lInner == _lhsSize && !_convertLhs;
                bool rRun = rInner == _rhsSize && !_convertRhs;
                // One value per run needs no conversion when it has the kernel dtype; a value of another dtype
                // changes from run to run and is converted by the general path below.
                ComparisonKernel kernel = null;
                bool alignRight = false;
                if (lRun && rRun)
                    kernel = _full;
                else if (lRun && rInner == 0 && !_convertRhs)
                    kernel = _scalarRight;
                else if (lInner == 0 && !_convertLhs && rRun)
                {
                    kernel = _scalarLeft;
                    alignRight = true;      // the right operand is the array of this form
                }

                if (kernel is not null)
                {
                    // A matrix whose rows follow each other against ONE short row: several rows per kernel call.
                    if (RepeatsRow(lInner, lOuter, rInner, rOuter, oOuter, rows, count))
                    {
                        if (lOuter == 0)
                            RunRepeatedRow(l, r, rowIsLeft: true, o, rows, count);
                        else
                            RunRepeatedRow(r, l, rowIsLeft: false, o, rows, count);
                        return;
                    }

                    // Rows of one-byte elements that end in a partial vector: the kernel would compare those last
                    // 8 to 31 elements one at a time, which for a row of a few hundred bytes is half of its cost
                    // (400 bool: 31 ns a row). Instead the row's whole vectors are compared, and then its LAST
                    // full vector once more, overlapping the elements already done. That is safe only because the
                    // output is not an input (see the remarks): the overlapped results are computed twice from
                    // the same operands.
                    long lanes = s_vectorBytes;
                    if (_lhsSize == 1 && _rhsSize == 1 && lanes > 0 && count > lanes && count % lanes >= OverlapTailMinElements)
                    {
                        long whole = count - count % lanes, last = count - lanes;
                        for (long i = 0; i < rows; i++)
                        {
                            kernel(l, r, (bool*)o, null, null, null, 1, whole);
                            kernel(l + last * lInner, r + last * rInner, (bool*)(o + last), null, null, null, 1, lanes);
                            l += lOuter;
                            r += rOuter;
                            o += oOuter;
                        }
                        return;
                    }

                    for (long i = 0; i < rows; i++)
                    {
                        long peel = alignRight ? AlignmentPeel(r, _rhsSize, o, count) : AlignmentPeel(l, _lhsSize, o, count);
                        InPlace(kernel, l, lInner, r, rInner, o, count, peel);
                        l += lOuter;
                        r += rOuter;
                        o += oOuter;
                    }
                    return;
                }

                // Backward runs of the kernel dtype (and values that need no conversion): read each run forwards from
                // its last element, store its results reversed. Runs longer than one block go through Chunk, which
                // does the same block by block.
                bool lValue = lInner == 0, rValue = rInner == 0;
                if (count <= BlockBytes && _byteCopy is not null && IsBackwardForm(lInner, rInner, oInner)
                    && !(lValue && _convertLhs) && !(rValue && _convertRhs))
                {
                    if (!lValue)
                    {
                        l += (count - 1) * lInner;      // the run's last element is the first of the turned run
                        lInner = -lInner;
                    }
                    if (!rValue)
                    {
                        r += (count - 1) * rInner;
                        rInner = -rInner;
                    }
                    kernel = lValue ? _scalarLeft : rValue ? _scalarRight : _full;
                    byte* results = stackalloc byte[BlockBytes];
                    long srcStride = -1, dstStride = 1, extent = count;
                    for (long i = 0; i < rows; i++)
                    {
                        kernel(l, r, (bool*)results, null, null, null, 1, count);
                        _byteCopy(results + (count - 1), o, &srcStride, &dstStride, &extent, 1);
                        l += lOuter;
                        r += rOuter;
                        o += oOuter;
                    }
                    return;
                }

                // An operand whose successive ROWS are adjacent elements (an F-ordered matrix beside a C-ordered one):
                // staged a tile of several rows at a time instead of one strided row at a time.
                if (RunTiles(l, lInner, lOuter, r, rInner, rOuter, o, oOuter, rows, count))
                    return;
            }

            void** ptrs = stackalloc void*[3];
            long* strides = stackalloc long[3];
            strides[0] = lInner;
            strides[1] = rInner;
            strides[2] = oInner;
            for (long i = 0; i < rows; i++)
            {
                ptrs[0] = l;
                ptrs[1] = r;
                ptrs[2] = o;
                Chunk(ptrs, strides, count, null);
                l += lOuter;
                r += rOuter;
                o += oOuter;
            }
        }

        // ---------------------------------------------------------------------------------------------------------
        //  A matrix against one row
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Whether <see cref="RunRows"/> compares these rows through <see cref="RunRepeatedRow"/>: one operand is
        ///     the SAME short contiguous row for every run, the other operand's rows and the output's rows follow
        ///     each other in memory, and neither operand is converted.
        /// </summary>
        /// <param name="lInner">Left byte stride inside a run.</param>
        /// <param name="lOuter">Left byte stride between runs (0: the same run every time).</param>
        /// <param name="rInner">Right byte stride inside a run.</param>
        /// <param name="rOuter">Right byte stride between runs.</param>
        /// <param name="oOuter">Output byte stride between runs (the output's inner stride must be 1).</param>
        /// <param name="rows">Number of runs.</param>
        /// <param name="count">Elements per run.</param>
        /// <returns>True for the repeated-row form.</returns>
        /// <remarks>
        ///     The caller of <see cref="RunRows"/> uses it too: rows of fewer than
        ///     <see cref="DirectRowMinElements"/> elements are worth a call of <see cref="RunRows"/> only in this
        ///     form, where they share kernel calls.
        /// </remarks>
        internal bool RepeatsRow(long lInner, long lOuter, long rInner, long rOuter, long oOuter, long rows, long count)
        {
            if (RepeatedRowDisabled || rows < RepeatedRowMinRows || oOuter != count || _convertLhs || _convertRhs)
                return false;
            if (lInner != _lhsSize || rInner != _rhsSize || count * _kernelUnit > RepeatedRowMaxBytes)
                return false;
            return (lOuter == 0 && rOuter == count * _rhsSize) || (rOuter == 0 && lOuter == count * _lhsSize);
        }

        /// <summary>
        ///     Compares <paramref name="rows"/> rows of a matrix, which follow each other in memory, against ONE row:
        ///     the row is repeated in a stack block and several matrix rows are compared per kernel call.
        /// </summary>
        /// <param name="row">The row: <paramref name="count"/> contiguous elements of the kernel dtype.</param>
        /// <param name="matrix">First element of the matrix: <paramref name="rows"/> x <paramref name="count"/>
        ///     contiguous elements of the kernel dtype.</param>
        /// <param name="rowIsLeft">True when the row is the comparison's left operand.</param>
        /// <param name="o">First output byte: <paramref name="rows"/> x <paramref name="count"/> contiguous bytes.</param>
        /// <param name="rows">Number of rows (positive).</param>
        /// <param name="count">Elements per row (positive; a row is at most <see cref="RepeatedRowMaxBytes"/> bytes).</param>
        /// <remarks>
        ///     <para><b>Call it only after <see cref="RepeatsRow"/> answered true.</b> The block is
        ///     <see cref="BlockBytes"/> long; a row longer than that would be copied past its end. The limit of
        ///     <see cref="RepeatedRowMaxBytes"/> keeps every row a quarter of a block or less.</para>
        ///     <para>The block is filled by doubling (the row once, then the filled part again and again), so a row of
        ///     two elements costs 12 copies for a block of 8 KB, not 4,096.</para>
        ///     <para>[SkipLocalsInit]: every byte of the block that a kernel reads is written by those copies.</para>
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void RunRepeatedRow(byte* row, byte* matrix, bool rowIsLeft, byte* o, long rows, long count)
        {
            int rowItem = rowIsLeft ? _lhsSize : _rhsSize, matrixItem = rowIsLeft ? _rhsSize : _lhsSize;
            long rowBytes = count * rowItem;
            long copies = Math.Min(rows, BlockBytes / rowBytes);
            RepeatedRowRuns++;

            byte* scratch = stackalloc byte[BlockBytes + 64];
            byte* block = (byte*)(((nuint)scratch + 63) & ~(nuint)63);
            long total = copies * rowBytes, filled = rowBytes;
            Buffer.MemoryCopy(row, block, rowBytes, rowBytes);
            while (filled < total)
            {
                long step = Math.Min(filled, total - filled);
                Buffer.MemoryCopy(block, block + filled, step, step);
                filled += step;
            }

            ComparisonKernel kernel = _full;
            long matrixStep = copies * count * matrixItem, outStep = copies * count;
            for (long left = rows; left > 0; left -= copies)
            {
                long n = Math.Min(copies, left) * count;
                if (rowIsLeft)
                    kernel(block, matrix, (bool*)o, null, null, null, 1, n);
                else
                    kernel(matrix, block, (bool*)o, null, null, null, 1, n);
                matrix += matrixStep;
                o += outStep;
            }
        }

        // ---------------------------------------------------------------------------------------------------------
        //  Tiles: an operand whose successive rows are adjacent elements
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Whether an operand of a two-level iteration has its successive RUNS one element apart (in either
        ///     direction) while the elements of one run are further apart: the layout of an F-ordered matrix read in
        ///     C order. The <c>outer</c> elements of one column are then contiguous, which is what a tile reads.
        /// </summary>
        /// <param name="inner">Byte stride inside a run.</param>
        /// <param name="outer">Byte stride between runs.</param>
        /// <param name="size">Item size of the operand.</param>
        /// <returns>True for the tiled layout.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsTiled(long inner, long outer, int size)
            => (outer == size || outer == -size) && inner != 0 && inner != size && inner != -size;

        /// <summary>
        ///     Fewest rows of a tile of an operand whose rows span <see cref="TileTallSpanBytes"/> or more: 64 bytes
        ///     of each column, and between 16 and 32 rows (the measurements are at <see cref="TileTallSpanBytes"/>).
        /// </summary>
        /// <param name="size">Item size of the tiled operand.</param>
        /// <returns>16 for elements of 4 bytes or more, 32 for elements of 1 or 2 bytes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long TallTileRows(int size) => Math.Clamp(64 / size, 16, 32);

        /// <summary>
        ///     Runs the rows of <see cref="RunRows"/> tile by tile when an operand has the tiled layout
        ///     (<see cref="IsTiled"/>): the operand's part of each tile is copied into a stack block with contiguous
        ///     rows by <see cref="DirectILKernelGenerator.TransposeTile"/>, and the kernel compares the block.
        /// </summary>
        /// <param name="l">First left element of the first run.</param>
        /// <param name="lInner">Left byte stride inside a run.</param>
        /// <param name="lOuter">Left byte stride between runs.</param>
        /// <param name="r">First right element of the first run.</param>
        /// <param name="rInner">Right byte stride inside a run.</param>
        /// <param name="rOuter">Right byte stride between runs.</param>
        /// <param name="o">First output byte of the first run (the output's inner stride is 1).</param>
        /// <param name="oOuter">Output byte stride between runs.</param>
        /// <param name="rows">Number of runs.</param>
        /// <param name="count">Elements per run.</param>
        /// <returns>False, having read and written nothing, when no operand is tiled or the operand beside the tiled
        ///     one has a form this method does not serve (the caller's row-by-row loop then runs).</returns>
        /// <remarks>
        ///     <para><b>Why.</b> A row of a tiled operand is a run with a stride of one column. Staging it row by row
        ///     reads one element for each cache line it touches (and for each page, once a column is 4 KB long):
        ///     0.3 to 0.6 ns an element for a (250, 400) matrix, 4 to 8 ns for a (3162, 3162) one. A tile of B rows
        ///     reads B adjacent elements of each column instead.</para>
        ///     <para><b>The tile.</b> One tile is at most <see cref="BlockBytes"/> of the kernel dtype. It is a whole
        ///     row wide when <see cref="TileMinRows"/> rows of that width fit, and then as many rows high as fit;
        ///     otherwise the row is cut evenly into pieces that fit <see cref="TileMinRows"/> rows. Widths and heights
        ///     are multiples of the vector group of the transposition where the operand allows it. The LAST tile of
        ///     each direction is moved back to end on the operand's edge, so its first rows or columns are computed a
        ///     second time; that is correct because the output is not an input (the contract of
        ///     <see cref="RunRows"/>).</para>
        ///     <para><b>The other operand.</b> Rows of contiguous elements are read in place, or converted into the
        ///     other block when their dtype is not the kernel's; one value of the kernel dtype per row is read in
        ///     place; a second tiled operand gets its own block. Any other form declines.</para>
        ///     <para><b>Kernel calls.</b> One call for the whole tile when each operand's rows of the tile follow each
        ///     other in memory (a staged block always; rows read in place when the tile is a row wide and the rows are
        ///     adjacent) and the output's rows do; otherwise one call a row.</para>
        ///     <para>[SkipLocalsInit]: the blocks are written by the transposition or the conversion before a kernel
        ///     reads them.</para>
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private bool RunTiles(byte* l, long lInner, long lOuter, byte* r, long rInner, long rOuter, byte* o, long oOuter, long rows,
            long count)
        {
            bool lTiled = IsTiled(lInner, lOuter, _lhsSize), rTiled = IsTiled(rInner, rOuter, _rhsSize);
            if (!(lTiled || rTiled) || TilesDisabled || rows < 2 || rows * count < StagedMinElements)
                return false;

            // The operand beside a tiled one: contiguous rows (of any dtype), or one value of the kernel dtype a row.
            bool lValue = !lTiled && lInner == 0, rValue = !rTiled && rInner == 0;
            if (!lTiled && lInner != _lhsSize && !(lValue && !_convertLhs))
                return false;
            if (!rTiled && rInner != _rhsSize && !(rValue && !_convertRhs))
                return false;

            int unit = _kernelUnit;
            // A 16-byte element is moved one at a time whatever the route, so a tile of them only saves the calls
            // of the row-by-row loop: worth it for rows too short to be staged (measured 1.4x to 2.9x for rows of 3
            // to 40 complex128 elements), a loss for longer ones (0.93x at 400 elements a row). And a conversion
            // TO complex128 is slower through the house cast than through the per-element kernel (0.70x).
            if (unit == 16 && (count > StagedMinElements || _convertLhs || _convertRhs))
                return false;
            TileRuns++;

            // The vector group of the transposition (of both, when both operands are tiled), the memory one row of
            // a tiled operand spans, and the item size the tile's height is chosen for.
            int rowGroup = 1, columnGroup = 1, tiledSize = unit;
            long span = 0;
            if (lTiled)
            {
                rowGroup = DirectILKernelGenerator.TransposeTileRowGroup(_lhsSize);
                columnGroup = DirectILKernelGenerator.TransposeTileColumnGroup(_lhsSize);
                span = count * Math.Abs(lInner);
                tiledSize = _lhsSize;
            }
            if (rTiled)
            {
                rowGroup = Math.Max(rowGroup, DirectILKernelGenerator.TransposeTileRowGroup(_rhsSize));
                columnGroup = Math.Max(columnGroup, DirectILKernelGenerator.TransposeTileColumnGroup(_rhsSize));
                span = Math.Max(span, count * Math.Abs(rInner));
                tiledSize = Math.Min(tiledSize, _rhsSize);
            }

            // Width: the whole row when the fewest rows of a tile fit one block at that width, else an even cut of
            // the row.
            long minRows = Math.Max(rowGroup, span >= TileTallSpanBytes ? TallTileRows(tiledSize) : TileMinRows);
            long widest = Math.Max(columnGroup, BlockBytes / (minRows * unit) / columnGroup * columnGroup);
            long width = count;
            if (count > widest)
            {
                long pieces = (count + widest - 1) / widest;
                width = Math.Min(widest, ((count + pieces - 1) / pieces + columnGroup - 1) / columnGroup * columnGroup);
            }
            // Height: what the block leaves; when that is not every row, an even cut in multiples of the row group.
            long height = Math.Min(rows, BlockBytes / (width * unit));
            if (height < rows)
            {
                height = height / rowGroup * rowGroup;
                long tiles = (rows + height - 1) / height;
                height = Math.Min(height, ((rows + tiles - 1) / tiles + rowGroup - 1) / rowGroup * rowGroup);
            }

            byte* scratch = stackalloc byte[3 * BlockBytes + 64];
            scratch = (byte*)(((nuint)scratch + 63) & ~(nuint)63);
            byte* lBlock = scratch, rBlock = scratch + BlockBytes, temp = scratch + 2 * BlockBytes;

            ComparisonKernel kernel = lValue ? _scalarLeft : rValue ? _scalarRight : _full;
            // Whether an operand's rows of one tile follow each other in memory (see "Kernel calls" in the remarks).
            bool lJoined = lTiled || (lValue ? lOuter == 0 : _convertLhs || (width == count && lOuter == count * _lhsSize));
            bool rJoined = rTiled || (rValue ? rOuter == 0 : _convertRhs || (width == count && rOuter == count * _rhsSize));
            bool joined = lJoined && rJoined && oOuter == width;

            for (long row0 = 0; row0 < rows; row0 += height)
            {
                if (row0 > rows - height)
                    row0 = rows - height;
                for (long col0 = 0; col0 < count; col0 += width)
                {
                    if (col0 > count - width)
                        col0 = count - width;

                    byte* lp = TileOperand(l, lInner, lOuter, lTiled, _convertLhs, LhsType, _lhsKernelType, _lhsSize, row0, col0,
                        height, width, count, lBlock, temp, out long lStep);
                    byte* rp = TileOperand(r, rInner, rOuter, rTiled, _convertRhs, RhsType, _rhsKernelType, _rhsSize, row0, col0,
                        height, width, count, rBlock, temp, out long rStep);
                    byte* op = o + row0 * oOuter + col0;

                    if (joined)
                    {
                        kernel(lp, rp, (bool*)op, null, null, null, 1, height * width);
                        continue;
                    }
                    for (long b = 0; b < height; b++)
                    {
                        kernel(lp, rp, (bool*)op, null, null, null, 1, width);
                        lp += lStep;
                        rp += rStep;
                        op += oOuter;
                    }
                }
            }
            return true;
        }

        /// <summary>
        ///     Brings one operand's part of a tile into the form the kernel reads, and returns its first element.
        /// </summary>
        /// <param name="p">First element of the operand's first run.</param>
        /// <param name="inner">Byte stride inside a run (0: one value a run).</param>
        /// <param name="outer">Byte stride between runs.</param>
        /// <param name="tiled">The operand has the tiled layout (<see cref="IsTiled"/>).</param>
        /// <param name="convert">The operand's stored dtype is not the dtype the kernel reads.</param>
        /// <param name="type">Stored dtype.</param>
        /// <param name="kernelType">The dtype the kernel reads.</param>
        /// <param name="size">Item size of the stored dtype.</param>
        /// <param name="row0">First run of the tile.</param>
        /// <param name="col0">First element of the tile inside a run.</param>
        /// <param name="height">Runs in the tile.</param>
        /// <param name="width">Elements of a run in the tile.</param>
        /// <param name="count">Elements of a whole run.</param>
        /// <param name="block">This operand's block (at least height x width elements of the kernel dtype).</param>
        /// <param name="temp">Scratch of the same size, for a tiled operand that is also converted.</param>
        /// <param name="step">Receives the byte distance between two successive rows at the returned address.</param>
        /// <returns>The address the kernel reads the tile's first row from: the block, or the operand itself.</returns>
        /// <remarks>
        ///     A tiled operand whose runs go DOWNWARDS in memory (<paramref name="outer"/> negative) is read from the
        ///     tile's last run, which has the lowest address, and its rows are stored from the block's last row
        ///     upwards, so the block holds the runs in iteration order either way.
        /// </remarks>
        private byte* TileOperand(byte* p, long inner, long outer, bool tiled, bool convert, NPTypeCode type, NPTypeCode kernelType,
            int size, long row0, long col0, long height, long width, long count, byte* block, byte* temp, out long step)
        {
            byte* first = p + row0 * outer + col0 * inner;
            long blockRow = width * _kernelUnit;
            if (tiled)
            {
                byte* into = convert ? temp : block;
                long rowBytes = width * size;
                if (outer > 0)
                    DirectILKernelGenerator.TransposeTile(first, inner, into, rowBytes, height, width, size);
                else
                    DirectILKernelGenerator.TransposeTile(first + (height - 1) * outer, inner, into + (height - 1) * rowBytes, -rowBytes,
                        height, width, size);
                if (convert)
                    NDPolySeries.CopyInto(PolySeriesView.Raw(temp, height * width, type), height * width, kernelType, block);
                step = blockRow;
                return block;
            }

            if (inner == 0 || !convert)
            {
                // One value a run, or rows of the kernel dtype: read where they are.
                step = outer;
                return first;
            }

            // Contiguous rows of another dtype: converted into the block, in one call when they follow each other.
            if (width == count && outer == count * size)
            {
                NDPolySeries.CopyInto(PolySeriesView.Raw(first, height * width, type), height * width, kernelType, block);
            }
            else
            {
                for (long b = 0; b < height; b++)
                    NDPolySeries.CopyInto(PolySeriesView.Raw(first + b * outer, width, type), width, kernelType, block + b * blockRow);
            }
            step = blockRow;
            return block;
        }

        /// <summary>
        ///     A long chunk the kernels cannot read in place: both operands one value, a value that can be restated in
        ///     the array's dtype, or runs that are staged block by block.
        /// </summary>
        /// <param name="l">First left element.</param>
        /// <param name="ls">Left byte stride.</param>
        /// <param name="r">First right element.</param>
        /// <param name="rs">Right byte stride.</param>
        /// <param name="o">First output byte.</param>
        /// <param name="os">Output byte stride.</param>
        /// <param name="n">Element count (at least <see cref="StagedMinElements"/>).</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void Staged(byte* l, long ls, byte* r, long rs, byte* o, long os, long n)
        {
            bool lValue = ls == 0, rValue = rs == 0;
            if (lValue && rValue)
            {
                // Two fully broadcast operands: one comparison decides every element.
                ValueSlot a, b;
                byte* pa = l, pb = r;
                if (_convertLhs)
                {
                    _lhsValueCast(l, 1, 0, (byte*)&a);
                    pa = (byte*)&a;
                }
                if (_convertRhs)
                {
                    _rhsValueCast(r, 1, 0, (byte*)&b);
                    pb = (byte*)&b;
                }
                byte verdict;
                _full(pa, pb, (bool*)&verdict, null, null, null, 1, 1);
                Fill(o, os, n, verdict);
                return;
            }

            // One value against an array of another dtype: compare the array as it is stored when the value can be
            // restated exactly in the array's dtype.
            if (rValue && _adoptRight is not null && _adoptRight.TryRun(l, ls, r, o, os, n))
                return;
            if (lValue && _adoptLeft is not null && _adoptLeft.TryRun(r, rs, l, o, os, n))
                return;

            Blocks(l, ls, lValue, r, rs, rValue, o, os, n);
        }

        /// <summary>Writes <paramref name="value"/> (0 or 1) to <paramref name="n"/> output slots.</summary>
        /// <param name="o">First output byte.</param>
        /// <param name="os">Output byte stride.</param>
        /// <param name="n">Element count.</param>
        /// <param name="value">The byte to store.</param>
        private void Fill(byte* o, long os, long n, byte value)
        {
            if (os == 1)
                NativeMemory.Fill(o, (nuint)n, value);
            else
                _byteMove(o, os, &value, 0, n);   // source stride 0: one byte for every slot
        }

        /// <summary>
        ///     Brings <paramref name="m"/> elements of one input run into <paramref name="block"/> as a contiguous run
        ///     of the kernel dtype: at most one copy that removes the stride, then at most one conversion.
        /// </summary>
        /// <param name="src">First element of the run.</param>
        /// <param name="stride">Its byte stride (not 0; a value is never staged).</param>
        /// <param name="m">Element count (at most one block).</param>
        /// <param name="srcType">Stored dtype of the run.</param>
        /// <param name="copyType">The dtype the run is copied as (<see cref="_lhsCopyType"/>).</param>
        /// <param name="srcSize">Item size of the stored dtype.</param>
        /// <param name="kernelType">The dtype the kernels read.</param>
        /// <param name="move">The strided move kernel of the stored item size.</param>
        /// <param name="block">Receives the staged run.</param>
        /// <param name="temp">Scratch for the contiguous copy that precedes a conversion.</param>
        /// <remarks>
        ///     <para>The stride is removed by the house vector copy when the layout is one it reads without a gather
        ///     (a step of two elements, or a reversed run, of a dtype of at most 8 bytes: 0.04 to 0.2 ns an element),
        ///     and by the unrolled strided move for every other stride (0.2 ns; the house's generic strided copy takes
        ///     0.7 ns for a one- or two-byte dtype, and a gather of far-apart 4- or 8-byte elements 0.5).</para>
        ///     <para>The conversion is the house contiguous cast (vectorized from
        ///     <see cref="DirectILKernelGenerator.PolyHouseKernelThreshold"/> elements), always out of contiguous
        ///     memory, so its speed does not depend on the source's layout.</para>
        /// </remarks>
        private void Stage(byte* src, long stride, long m, NPTypeCode srcType, NPTypeCode copyType, int srcSize, NPTypeCode kernelType,
            StridedMoveKernel move, byte* block, byte* temp)
        {
            bool convert = srcType != kernelType;
            if (stride != srcSize)
            {
                byte* contiguous = convert ? temp : block;
                if (srcSize <= 8 && (stride == 2 * srcSize || stride == -srcSize))
                    NDPolySeries.CopyInto(PolySeriesView.RawStrided(src, m, stride, copyType), m, copyType, contiguous);
                else
                    move(contiguous, srcSize, src, stride, m);
                src = contiguous;
            }
            if (convert)
                NDPolySeries.CopyInto(PolySeriesView.Raw(src, m, srcType), m, kernelType, block);
        }

        /// <summary>
        ///     The staged loop: each block of the chunk is brought into the form a vector kernel reads, compared, and
        ///     its results stored.
        /// </summary>
        /// <param name="l">First left element.</param>
        /// <param name="ls">Left byte stride.</param>
        /// <param name="lValue">The left operand is one value.</param>
        /// <param name="r">First right element.</param>
        /// <param name="rs">Right byte stride.</param>
        /// <param name="rValue">The right operand is one value.</param>
        /// <param name="o">First output byte.</param>
        /// <param name="os">Output byte stride.</param>
        /// <param name="n">Element count.</param>
        /// <remarks>
        ///     [SkipLocalsInit]: the 32 KB of stack blocks are written by the staging copy or by the kernel before
        ///     anything reads them, and this method runs once per chunk (once per row of a strided 2-D operand):
        ///     zeroing the blocks on every call would cost more than comparing a short row.
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void Blocks(byte* l, long ls, bool lValue, byte* r, long rs, bool rValue, byte* o, long os, long n)
        {
            // One value of another dtype: converted once, outside the block loop.
            ValueSlot lSlot, rSlot;
            if (lValue && _convertLhs)
            {
                _lhsValueCast(l, 1, 0, (byte*)&lSlot);
                l = (byte*)&lSlot;
            }
            if (rValue && _convertRhs)
            {
                _rhsValueCast(r, 1, 0, (byte*)&rSlot);
                r = (byte*)&rSlot;
            }

            // Runs that go BACKWARDS in the kernel's own dtype are not staged: the chunk is turned around, so the
            // kernel reads each such run forwards from its last element and the results are stored backwards
            // instead. That reverses one byte an element, where staging would reverse 8 or 16 for each input. It
            // needs every input to be either such a run or one value (a forward run would have to be staged
            // reversed, which gains nothing).
            bool lBackward = !lValue && !_convertLhs && ls == -_lhsSize;
            bool rBackward = !rValue && !_convertRhs && rs == -_rhsSize;
            if ((lBackward || rBackward) && (lBackward || lValue) && (rBackward || rValue))
            {
                if (lBackward)
                {
                    l += (n - 1) * ls;
                    ls = -ls;
                }
                if (rBackward)
                {
                    r += (n - 1) * rs;
                    rs = -rs;
                }
                o += (n - 1) * os;
                os = -os;
            }

            bool lDirect = lValue || (!_convertLhs && ls == _lhsSize);
            bool rDirect = rValue || (!_convertRhs && rs == _rhsSize);
            ComparisonKernel kernel = lValue ? _scalarLeft : rValue ? _scalarRight : _full;

            if (os == 1 && lDirect && rDirect)
            {
                // The turned-around chunk is one of the in-place forms (its output ran backwards too).
                long peel = lValue ? AlignmentPeel(r, _rhsSize, o, n) : AlignmentPeel(l, _lhsSize, o, n);
                InPlace(kernel, l, ls, r, rs, o, n, peel);
                return;
            }

            // Two staged inputs share the block budget (see BlockBytes). The blocks start on a cache line, so the
            // kernel's loads from them never straddle one. The fourth block holds a run between the copy that
            // removes its stride and its conversion (one input at a time).
            long block = (lDirect || rDirect ? BlockBytes : BlockBytes / 2) / _kernelUnit;
            byte* scratch = stackalloc byte[4 * BlockBytes + 64];
            scratch = (byte*)(((nuint)scratch + 63) & ~(nuint)63);
            byte* lBlock = scratch, rBlock = scratch + BlockBytes, resultBlock = scratch + 2 * BlockBytes, temp = scratch + 3 * BlockBytes;

            for (long k = 0; k < n; k += block)
            {
                long m = Math.Min(block, n - k);

                byte* lp = l, rp = r;
                if (!lValue)
                {
                    lp = l + k * ls;
                    if (!lDirect)
                    {
                        Stage(lp, ls, m, LhsType, _lhsCopyType, _lhsSize, _lhsKernelType, _lhsMove, lBlock, temp);
                        lp = lBlock;
                    }
                }
                if (!rValue)
                {
                    rp = r + k * rs;
                    if (!rDirect)
                    {
                        Stage(rp, rs, m, RhsType, _rhsCopyType, _rhsSize, _rhsKernelType, _rhsMove, rBlock, temp);
                        rp = rBlock;
                    }
                }

                if (os == 1)
                {
                    kernel(lp, rp, (bool*)(o + k), null, null, null, 1, m);
                    continue;
                }

                kernel(lp, rp, (bool*)resultBlock, null, null, null, 1, m);
                byte* dst = o + k * os;
                if (os == -1 && _byteCopy is not null)
                {
                    // A reversed contiguous store: the SIMD byte-reverse copy, read from the block's last result
                    // into the lowest output address of this block.
                    long srcStride = -1, dstStride = 1, extent = m;
                    _byteCopy(resultBlock + (m - 1), dst - (m - 1), &srcStride, &dstStride, &extent, 1);
                }
                else
                {
                    _byteMove(dst, os, resultBlock, 1, m);
                }
            }
        }

        // ---------------------------------------------------------------------------------------------------------
        //  One value against an array of another dtype
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Restates ONE value in the dtype of the array it is compared with, so that the array is compared as it
        ///     is stored (in its own, narrower dtype) instead of being converted element by element to the wider
        ///     comparison dtype.
        /// </summary>
        /// <remarks>
        ///     <para>The comparison NumPy defines for the pair is <c>convert(a) op convert(v)</c> in the comparison
        ///     dtype T. Every array value <c>a</c> converts to T exactly (T is at least as wide as the array's dtype),
        ///     so the question for each element is the mathematical <c>a op t</c> with <c>t = convert(v)</c>. That
        ///     question has the same answer as <c>a op a'</c> for ONE array-dtype value <c>a'</c>, or it has the same
        ///     answer for every element:</para>
        ///     <list type="bullet">
        ///         <item>integer array, integer value: <c>a' = t</c> when t is inside the array dtype's range;
        ///             outside it, t is above or below every element.</item>
        ///         <item>integer array (at most 32 bits, so every element is exact in T), float value:
        ///             <c>a &gt; t</c> is <c>a &gt; floor(t)</c>, <c>a &gt;= t</c> is <c>a &gt;= ceil(t)</c>,
        ///             <c>a &lt; t</c> is <c>a &lt; ceil(t)</c>, <c>a &lt;= t</c> is <c>a &lt;= floor(t)</c>;
        ///             <c>a == t</c> can hold only for an integral t in range.</item>
        ///         <item>float16 / float32 array, wider value: <c>a &gt; t</c> is <c>a &gt; down(t)</c> and
        ///             <c>a &lt;= t</c> is <c>a &lt;= down(t)</c>, with down(t) the largest array-dtype value not above
        ///             t; <c>a &gt;= t</c> and <c>a &lt; t</c> use up(t), the smallest not below t; <c>a == t</c> can
        ///             hold only when t is an array-dtype value. A NaN element makes every form False (True for
        ///             <c>!=</c>) on both sides of each equivalence.</item>
        ///     </list>
        ///     <para>A NaN value decides the whole chunk (False; True for <c>!=</c>). The restatement is not a cast:
        ///     <c>float32_array &gt; 0.1</c> keeps comparing against the double 0.1 (through down(0.1)), where a cast
        ///     of the value to float32 would compare against a different number.</para>
        ///     <para>Pairs it does not serve keep the staged conversion: a 64-bit integer array against a float (its
        ///     elements are not exact in float64), complex, decimal and char operands.</para>
        /// </remarks>
        private sealed class ScalarAdoption
        {
            /// <summary>How the value is restated.</summary>
            private enum Form
            {
                /// <summary>Integer array, integer value.</summary>
                IntegerValue,
                /// <summary>Integer array of at most 32 bits, float value.</summary>
                FloatValueIntegerArray,
                /// <summary>float16 / float32 array, value compared in a wider float dtype.</summary>
                FloatArray,
            }

            /// <summary>Outcome of <see cref="Resolve"/>.</summary>
            private enum Verdict
            {
                /// <summary>The slot holds the value in the array's dtype; compare with the same operator.</summary>
                Restated,
                /// <summary>Every element compares False.</summary>
                AllFalse,
                /// <summary>Every element compares True.</summary>
                AllTrue,
            }

            /// <summary>The run that owns this adoption (its <see cref="Fill"/> writes a decided chunk).</summary>
            private readonly ComparisonRun _owner;

            /// <summary>The same operator over (array dtype, array dtype).</summary>
            private readonly ComparisonRun _sameDtype;

            /// <summary>How the value is restated.</summary>
            private readonly Form _form;

            /// <summary>The operator with the ARRAY on the left: the owner's operator, mirrored when the value is
            ///     the left operand (<c>v &lt; a</c> is <c>a &gt; v</c>).</summary>
            private readonly ComparisonOp _arrayFirstOp;

            /// <summary>The value is the left operand.</summary>
            private readonly bool _scalarOnLeft;

            /// <summary>Item size of the array dtype.</summary>
            private readonly int _arraySize;

            /// <summary>Item size of the value's dtype.</summary>
            private readonly int _scalarSize;

            /// <summary>The value's dtype is a signed integer.</summary>
            private readonly bool _scalarSigned;

            /// <summary>The value's dtype is bool (read by truth value).</summary>
            private readonly bool _scalarIsBool;

            /// <summary>Smallest value of an integer array dtype.</summary>
            private readonly long _min;

            /// <summary>Largest value of an integer array dtype.</summary>
            private readonly ulong _max;

            /// <summary>Value dtype → float64: the value as the comparison sees it (one rounding at most, the same
            ///     one the comparison dtype's conversion makes).</summary>
            private readonly PolyCastKernel _scalarToDouble;

            /// <summary>float64 → array dtype (round to nearest).</summary>
            private readonly PolyCastKernel _doubleToArray;

            /// <summary>Array dtype → float64 (exact).</summary>
            private readonly PolyCastKernel _arrayToDouble;

            /// <summary>Stores the resolved plan.</summary>
            /// <param name="owner">The run of the mixed pair.</param>
            /// <param name="sameDtype">The run of (array dtype, array dtype).</param>
            /// <param name="form">How the value is restated.</param>
            /// <param name="arrayType">The array's dtype.</param>
            /// <param name="scalarType">The value's dtype.</param>
            /// <param name="scalarOnLeft">The value is the left operand.</param>
            private ScalarAdoption(ComparisonRun owner, ComparisonRun sameDtype, Form form, NPTypeCode arrayType, NPTypeCode scalarType,
                bool scalarOnLeft)
            {
                _owner = owner;
                _sameDtype = sameDtype;
                _form = form;
                _scalarOnLeft = scalarOnLeft;
                _arrayFirstOp = scalarOnLeft ? DirectILKernelGenerator.MirrorComparison(owner.Op) : owner.Op;
                _arraySize = DirectILKernelGenerator.GetTypeSize(arrayType);
                _scalarSize = DirectILKernelGenerator.GetTypeSize(scalarType);
                _scalarSigned = NDExprTypeRules.IsSignedInteger(scalarType);
                _scalarIsBool = scalarType == NPTypeCode.Boolean;
                (_min, _max) = NDExprTypeRules.IntBounds(arrayType);
                if (form != Form.IntegerValue)
                    _scalarToDouble = DirectILKernelGenerator.GetPolyCastKernel(scalarType, NPTypeCode.Double);
                if (form == Form.FloatArray)
                {
                    _doubleToArray = DirectILKernelGenerator.GetPolyCastKernel(NPTypeCode.Double, arrayType);
                    _arrayToDouble = DirectILKernelGenerator.GetPolyCastKernel(arrayType, NPTypeCode.Double);
                }
            }

            /// <summary>
            ///     The adoption of a <paramref name="scalarType"/> value by an <paramref name="arrayType"/> array, or
            ///     null when the pair is not one the restatement is exact for (see the class remarks).
            /// </summary>
            /// <param name="owner">The run of the mixed pair.</param>
            /// <param name="arrayType">The array's stored dtype (it differs from <paramref name="kernelType"/>).</param>
            /// <param name="scalarType">The value's stored dtype.</param>
            /// <param name="kernelType">The dtype the pair is compared in.</param>
            /// <param name="scalarOnLeft">The value is the left operand.</param>
            /// <returns>The adoption, or null.</returns>
            /// <exception cref="NotSupportedException">A conversion kernel the plan needs does not exist (the owner's
            ///     constructor treats it as "run unavailable").</exception>
            public static ScalarAdoption TryCreate(ComparisonRun owner, NPTypeCode arrayType, NPTypeCode scalarType, NPTypeCode kernelType,
                bool scalarOnLeft)
            {
                // The restated value is written as raw little-endian bytes of the array dtype.
                if (!BitConverter.IsLittleEndian)
                    return null;

                bool arrayInt = IsIntegerLike(arrayType), scalarInt = IsIntegerLike(scalarType);
                bool scalarFloat = scalarType == NPTypeCode.Half || scalarType == NPTypeCode.Single || scalarType == NPTypeCode.Double;

                Form form;
                if (arrayInt && scalarInt)
                    form = Form.IntegerValue;
                else if (arrayInt && scalarFloat && DirectILKernelGenerator.GetTypeSize(arrayType) <= 4)
                    form = Form.FloatValueIntegerArray;
                else if ((arrayType == NPTypeCode.Half || arrayType == NPTypeCode.Single) && (scalarInt || scalarFloat)
                         && (kernelType == NPTypeCode.Single || kernelType == NPTypeCode.Double))
                    form = Form.FloatArray;
                else
                    return null;

                var sameDtype = Get(owner.Op, arrayType, arrayType);
                return sameDtype is null ? null : new ScalarAdoption(owner, sameDtype, form, arrayType, scalarType, scalarOnLeft);
            }

            /// <summary>Bool and the eight integer dtypes: the dtypes whose values are integers with a range.</summary>
            /// <param name="t">The dtype.</param>
            /// <returns>True for bool, int8 … int64 and uint8 … uint64 (not char: its promotion is NumSharp's own).</returns>
            private static bool IsIntegerLike(NPTypeCode t)
                => t == NPTypeCode.Boolean || (NDExprTypeRules.IsIntegerKind(t) && t != NPTypeCode.Char);

            /// <summary>
            ///     Compares the array against the value in the array's own dtype, or fills the chunk when the value
            ///     decides it.
            /// </summary>
            /// <param name="array">First array element.</param>
            /// <param name="arrayStride">Byte stride of the array operand.</param>
            /// <param name="scalar">The value (one element of the value dtype).</param>
            /// <param name="o">First output byte.</param>
            /// <param name="os">Output byte stride.</param>
            /// <param name="n">Element count.</param>
            /// <returns>Always true: every value of a served pair is restated or decides the chunk.</returns>
            public bool TryRun(byte* array, long arrayStride, byte* scalar, byte* o, long os, long n)
            {
                ValueSlot slot = default;
                switch (Resolve(scalar, (byte*)&slot))
                {
                    case Verdict.AllFalse:
                        _owner.Fill(o, os, n, 0);
                        return true;
                    case Verdict.AllTrue:
                        _owner.Fill(o, os, n, 1);
                        return true;
                }
                if (_scalarOnLeft)
                    _sameDtype.Run((byte*)&slot, 0, array, arrayStride, o, os, n);
                else
                    _sameDtype.Run(array, arrayStride, (byte*)&slot, 0, o, os, n);
                return true;
            }

            /// <summary>Restates the value into <paramref name="slot"/> or decides the chunk.</summary>
            /// <param name="scalar">The value, in its own dtype.</param>
            /// <param name="slot">Receives the value in the array's dtype (zeroed by the caller).</param>
            /// <returns>What the caller does next.</returns>
            private Verdict Resolve(byte* scalar, byte* slot)
            {
                if (_form == Form.IntegerValue)
                    return ResolveInteger(ReadInteger(scalar), slot);

                double t;
                _scalarToDouble(scalar, 1, 0, (byte*)&t);
                // A NaN value is unordered with every element: only != holds.
                if (double.IsNaN(t))
                    return _arrayFirstOp == ComparisonOp.NotEqual ? Verdict.AllTrue : Verdict.AllFalse;
                return _form == Form.FloatValueIntegerArray ? ResolveIntegerArray(t, slot) : ResolveFloatArray(t, slot);
            }

            /// <summary>The value of an integer (or bool) dtype as an exact integer.</summary>
            /// <param name="scalar">The value.</param>
            /// <returns>The integer; a bool reads as 0 or 1 whatever its byte.</returns>
            private Int128 ReadInteger(byte* scalar)
            {
                // The stored bytes, zero-extended; only the value's own bytes are read.
                ulong raw = 0;
                Buffer.MemoryCopy(scalar, &raw, sizeof(ulong), _scalarSize);
                if (_scalarIsBool)
                    return raw != 0 ? 1 : 0;
                if (_scalarSigned)
                {
                    int shift = 64 - 8 * _scalarSize;
                    return (long)(raw << shift) >> shift;   // sign-extend from the value's width
                }
                return raw;
            }

            /// <summary>Integer array, integer value.</summary>
            /// <param name="v">The value.</param>
            /// <param name="slot">Receives the value in the array's dtype.</param>
            /// <returns>The verdict.</returns>
            private Verdict ResolveInteger(Int128 v, byte* slot)
            {
                if (v > (Int128)_max)   // above every element
                    return _arrayFirstOp is ComparisonOp.Less or ComparisonOp.LessEqual or ComparisonOp.NotEqual ? Verdict.AllTrue : Verdict.AllFalse;
                if (v < (Int128)_min)   // below every element
                    return _arrayFirstOp is ComparisonOp.Greater or ComparisonOp.GreaterEqual or ComparisonOp.NotEqual ? Verdict.AllTrue : Verdict.AllFalse;
                // In range: the low bytes of the two's-complement value are the array dtype's value (0 or 1 for bool).
                *(ulong*)slot = unchecked((ulong)v);
                return Verdict.Restated;
            }

            /// <summary>Integer array of at most 32 bits, float value <paramref name="t"/> (not NaN).</summary>
            /// <param name="t">The value as the comparison sees it.</param>
            /// <param name="slot">Receives the integer threshold in the array's dtype.</param>
            /// <returns>The verdict.</returns>
            /// <remarks>The range bounds are exact as doubles (at most 32 bits), and so is every threshold that
            ///     passes the range tests, so the casts below lose nothing.</remarks>
            private Verdict ResolveIntegerArray(double t, byte* slot)
            {
                double lo = _min, hi = _max;
                double threshold;
                switch (_arrayFirstOp)
                {
                    case ComparisonOp.Greater:        // a > t   <=>  a > floor(t)
                        threshold = Math.Floor(t);
                        if (threshold >= hi) return Verdict.AllFalse;
                        if (threshold < lo) return Verdict.AllTrue;
                        break;
                    case ComparisonOp.LessEqual:      // a <= t  <=>  a <= floor(t)
                        threshold = Math.Floor(t);
                        if (threshold >= hi) return Verdict.AllTrue;
                        if (threshold < lo) return Verdict.AllFalse;
                        break;
                    case ComparisonOp.GreaterEqual:   // a >= t  <=>  a >= ceil(t)
                        threshold = Math.Ceiling(t);
                        if (threshold > hi) return Verdict.AllFalse;
                        if (threshold <= lo) return Verdict.AllTrue;
                        break;
                    case ComparisonOp.Less:           // a < t   <=>  a < ceil(t)
                        threshold = Math.Ceiling(t);
                        if (threshold > hi) return Verdict.AllTrue;
                        if (threshold <= lo) return Verdict.AllFalse;
                        break;
                    default:                          // == / != : only an integral t inside the range can be met
                        if (t != Math.Floor(t) || t < lo || t > hi)
                            return _arrayFirstOp == ComparisonOp.NotEqual ? Verdict.AllTrue : Verdict.AllFalse;
                        threshold = t;
                        break;
                }
                *(ulong*)slot = unchecked((ulong)(long)threshold);
                return Verdict.Restated;
            }

            /// <summary>float16 / float32 array, value <paramref name="t"/> (not NaN) of a wider float dtype.</summary>
            /// <param name="t">The value as the comparison sees it.</param>
            /// <param name="slot">Receives down(t), up(t) or t itself in the array's dtype.</param>
            /// <returns>The verdict.</returns>
            /// <remarks>
            ///     The conversion to the array dtype gives a neighbour of t (within one unit in the last place, also
            ///     when it rounds twice on its way to float16, and ±inf past the range); one step on the stored bit
            ///     pattern then reaches the neighbour on the side the operator needs.
            /// </remarks>
            private Verdict ResolveFloatArray(double t, byte* slot)
            {
                double nearest;
                _doubleToArray((byte*)&t, 1, 0, slot);
                _arrayToDouble(slot, 1, 0, (byte*)&nearest);
                switch (_arrayFirstOp)
                {
                    case ComparisonOp.Equal:
                    case ComparisonOp.NotEqual:
                        // An array value equals t only when t is itself an array-dtype value.
                        if (nearest != t)
                            return _arrayFirstOp == ComparisonOp.NotEqual ? Verdict.AllTrue : Verdict.AllFalse;
                        break;
                    case ComparisonOp.Greater:
                    case ComparisonOp.LessEqual:
                        if (nearest > t)      // need down(t): the largest array value not above t
                            StepTowardNegativeInfinity(slot);
                        break;
                    default:                  // >= and < need up(t): the smallest array value not below t
                        if (nearest < t)
                            StepTowardPositiveInfinity(slot);
                        break;
                }
                return Verdict.Restated;
            }

            /// <summary>
            ///     Replaces the IEEE value in <paramref name="slot"/> (array dtype) by the next one below it. Not
            ///     called for NaN or -inf.
            /// </summary>
            /// <param name="slot">The value, as its stored bits.</param>
            /// <remarks>
            ///     Works on the bit pattern, whatever the width: a positive value steps to the pattern below (+0 to the
            ///     smallest negative value), a negative one to the pattern above (its magnitude grows). The +0 case
            ///     keeps the step right for every finite input; <see cref="ResolveFloatArray"/> itself never asks for
            ///     it (a value below +0 rounds to -0 or lower, so the nearest value it steps from is not +0).
            /// </remarks>
            private void StepTowardNegativeInfinity(byte* slot)
            {
                ulong bits = 0;
                Buffer.MemoryCopy(slot, &bits, sizeof(ulong), _arraySize);
                ulong sign = 1UL << (8 * _arraySize - 1);
                bits = (bits & sign) == 0
                    ? (bits == 0 ? (sign | 1) : bits - 1)
                    : bits + 1;
                Buffer.MemoryCopy(&bits, slot, _arraySize, _arraySize);
            }

            /// <summary>
            ///     Replaces the IEEE value in <paramref name="slot"/> (array dtype) by the next one above it. Not
            ///     called for NaN or +inf.
            /// </summary>
            /// <param name="slot">The value, as its stored bits.</param>
            /// <remarks>The mirror of <see cref="StepTowardNegativeInfinity"/>: -0 steps to the smallest positive value
            ///     (a case <see cref="ResolveFloatArray"/> never asks for either: a value above -0 rounds to +0 or
            ///     higher).</remarks>
            private void StepTowardPositiveInfinity(byte* slot)
            {
                ulong bits = 0;
                Buffer.MemoryCopy(slot, &bits, sizeof(ulong), _arraySize);
                ulong sign = 1UL << (8 * _arraySize - 1);
                bits = (bits & sign) != 0
                    ? (bits == sign ? 1UL : bits - 1)
                    : bits + 1;
                Buffer.MemoryCopy(&bits, slot, _arraySize, _arraySize);
            }
        }
    }
}
