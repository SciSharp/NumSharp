using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.InnerLoop2D.Stationary.cs — the 2-D kernel's stationary-out mode
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// A reduction over the outer axis of a (rows, n) block — np.sum(m, axis=0), the reduce double
// loop of NumPy's iterator when the out does not move along the outer dimension — calls the
// add's element-wise loop once per row with operands {o, x[r], o}: o[i] = o[i] + x[r][i]. As a
// 2-D block that is the kernel's binary {CC} pattern with the out (and the first input, the
// same memory) at outer stride 0, and the plain block loads and stores the out row once per
// row: for a short row every vector add waits on the store of the previous row's add to the
// same address. The reduction's hand-written rows form (NumPyAddLoops.DoubleBinaryRows) keeps
// the out in registers instead, sixteen lanes at a time across all the rows; the PA2 prototype
// measured the plain block at 0.33-0.85x of it on (1000, 8) and (10000, 10) rows.
//
// THE MODE
// --------
// A stationary-out call — the first input IS the output (same pointer), both contiguous along
// the row and at outer stride 0, the second input a contiguous or reversed row — runs a STAGE:
//   for each strip of 8 vectors of columns: load the out strip into registers once, then for
//   every row strip = vectorBody(strip, x[row, strip]), then store the strip once;
//   then strips of 4 and of 1 vector (8 chains: the add's 4-5 cycle latency hidden even where
//   the rows sit in L1 - 4 chains measured 0.93x on a float32 broadcast row of 1000, the plain
//   block being throughput-bound there); then the columns left over: one masked vector held across the rows
//   (AVX2, 32- and 64-bit lanes, as the plain block's masked tail), else scalar strips of four
//   columns, then one, held in registers across the rows through the scalar body (the 1- and
//   2-byte lanes' short rows are all leftover columns at 16 / 32 lanes: one column per chain
//   lost 0.61-0.76x, a second row-by-row pass 0.80-0.89x).
// Each lane is still the rows' operations in order, through the op's own bodies and with the
// same vector / masked / scalar choice per element as the plain block, so every result byte is
// the plain block's. A reversed row (PA2a's reversed-run load) needs a host that reverses lanes.
//
// WHERE IT LIVES
// --------------
// In the stationary-out VARIANT of the 2-D kernel (Generate2DElementwiseKernel with
// withStationaryOut, which carries the reversed-run stages too), cached apart: a reduction asks
// for it (Compile2DStationaryOutKernel); the element-wise routes never produce an out at outer
// stride 0 and keep the plain kernel, whose IL stays unchanged (stage calls change how the JIT
// allocates a whole method's registers: DirectILKernelGenerator.InnerLoop.Reversed.cs).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// The stationary-out VARIANTS of the 2-D block kernels (<see cref="Generate2DElementwiseKernel"/> with
        /// <c>withStationaryOut</c> and <c>withReversedRuns</c>): the kernels a reduction's rows run through, keyed by the
        /// plain kernel's identity and cached apart so the plain kernel keeps its IL.
        /// </summary>
        internal static readonly ConcurrentDictionary<InnerLoopKernelKey, ND2DElementwiseKernel> _innerLoop2DStationaryCache = new();

        /// <summary>Look up a stationary-out 2-D kernel by its packed key without building a string.</summary>
        /// <param name="key">The plain kernel's packed key.</param>
        /// <param name="kernel">The variant, when cached.</param>
        /// <returns>True when the variant is cached.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryGet2DStationaryOutKernel(in InnerLoopKernelKey key, out ND2DElementwiseKernel kernel)
            => _innerLoop2DStationaryCache.TryGetValue(key, out kernel!);

        /// <summary>
        /// Serve the stationary-out 2-D kernel for <paramref name="key"/> from its cache, else compile and register it: a
        /// complete 2-D block kernel (every pattern the plain kernel runs, the reversed-run stages too) that runs a call of
        /// the reduction shape — operand 0 IS the output, both contiguous along the row and at outer stride 0, operand 1 a
        /// contiguous or reversed row — with the out row held in registers across the rows.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output (two inputs for the mode to apply).</param>
        /// <param name="scalarBody">Per-element body.</param>
        /// <param name="vectorBody">Per-vector body (null: a scalar-only kernel, without the mode).</param>
        /// <param name="key">The plain kernel's packed key.</param>
        /// <returns>The variant kernel.</returns>
        internal static ND2DElementwiseKernel Compile2DStationaryOutKernel(
            NPTypeCode[] operandTypes, Action<ILGenerator> scalarBody, Action<ILGenerator>? vectorBody, in InnerLoopKernelKey key)
        {
            if (_innerLoop2DStationaryCache.TryGetValue(key, out var cached))
                return cached;

            var kernel = Generate2DElementwiseKernel(operandTypes, scalarBody, vectorBody, key.ToCacheKey(),
                withReversedRuns: true, withStationaryOut: true);
            _innerLoop2DStationaryCache.TryAdd(key, kernel);
            return kernel;
        }

        /// <summary>
        /// The stationary-out dispatch, emitted first in the variant's SIMD route: when operand 0 is the output (the same
        /// pointer), both are contiguous along the row (inner byte stride <c>elemSize</c>) and stay put along the outer axis
        /// (outer byte stride 0), forward to the stage of operand 1's row — contiguous (<c>+elemSize</c>, any outer stride:
        /// a row of a matrix, or a broadcast row at 0) or reversed (<c>−elemSize</c>, when the host reverses lanes). Every
        /// other call continues at <paramref name="lblNotStationary"/>: the kernel's own dispatch, unchanged.
        /// </summary>
        /// <param name="il">The kernel's generator (its prologue loaded the locals).</param>
        /// <param name="operandTypes">Two inputs then the output (one shared SIMD dtype).</param>
        /// <param name="ptrLocals">Per-operand pointers at the block's first row.</param>
        /// <param name="innerStrideLocals">Per-operand inner byte strides.</param>
        /// <param name="outerStrideLocals">Per-operand outer byte strides.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages).</param>
        /// <param name="lblNotStationary">Where any other call goes.</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        /// <remarks>
        /// A dozen compares per call. The out-is-the-first-input test is what makes holding the out in registers right:
        /// row r's first operand is then row r-1's result, which the registers hold.
        /// </remarks>
        private static void Emit2DStationaryOutDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] innerStrideLocals,
            LocalBuilder[] outerStrideLocals, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string cacheKey,
            Label lblNotStationary, Label lblEnd)
        {
            long elemSize = GetTypeSize(operandTypes[2]);

            // Operand 0 is the output itself.
            il.Emit(OpCodes.Ldloc, ptrLocals[0]);
            il.Emit(OpCodes.Ldloc, ptrLocals[2]);
            il.Emit(OpCodes.Bne_Un, lblNotStationary);
            foreach (int op in new[] { 0, 2 })
            {
                il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, elemSize);
                il.Emit(OpCodes.Bne_Un, lblNotStationary);
                il.Emit(OpCodes.Ldloc, outerStrideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, lblNotStationary);
            }

            // Operand 1's row: contiguous, else reversed (where the host reverses lanes), else the kernel's own dispatch.
            var lblContiguousRow = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, innerStrideLocals[1]);
            il.Emit(OpCodes.Ldc_I8, elemSize);
            il.Emit(OpCodes.Beq, lblContiguousRow);
            if (CanReverseLanes(operandTypes[2]))
            {
                var lblReversedRow = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, innerStrideLocals[1]);
                il.Emit(OpCodes.Ldc_I8, -elemSize);
                il.Emit(OpCodes.Beq, lblReversedRow);
                il.Emit(OpCodes.Br, lblNotStationary);

                il.MarkLabel(lblReversedRow);
                EmitForwardToStage(il, Generate2DStationaryOutStage(operandTypes, reversedRow: true, vectorBody, scalarBody,
                    $"ND2DLoop_{Sanitize(cacheKey)}_so_r"), argCount: 5);
                il.Emit(OpCodes.Br, lblEnd);
            }
            else
            {
                il.Emit(OpCodes.Br, lblNotStationary);
            }

            il.MarkLabel(lblContiguousRow);
            EmitForwardToStage(il, Generate2DStationaryOutStage(operandTypes, reversedRow: false, vectorBody, scalarBody,
                $"ND2DLoop_{Sanitize(cacheKey)}_so_c"), argCount: 5);
            il.Emit(OpCodes.Br, lblEnd);
        }

        /// <summary>
        /// One stationary-out stage: the <see cref="ND2DElementwiseKernel"/> signature and
        /// <see cref="Emit2DStationaryOutBlock"/> for operand 1's row mode.
        /// </summary>
        /// <param name="operandTypes">Two inputs then the output (one shared SIMD dtype).</param>
        /// <param name="reversedRow">True: operand 1's row is reversed (inner byte stride −elemSize).</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage (compiled on its first call).</returns>
        private static DynamicMethod Generate2DStationaryOutStage(
            NPTypeCode[] operandTypes, bool reversedRow, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string name)
        {
            var dm = new DynamicMethod(
                name: name,
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();
            Emit2DStationaryOutBlock(il, operandTypes, reversedRow, vectorBody, scalarBody);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        /// Emit the stationary-out loops of one stage (the dispatch checked the shape): strips of eight, then four, then one
        /// vector of columns, each loaded from the out once, run through every row in order, stored once; then the columns
        /// left over — one masked vector held across the rows on a host with the plain block's masked tail (AVX2, 32- and
        /// 64-bit lanes), else scalar strips of four columns, then one, held across the rows through the scalar body.
        /// </summary>
        /// <param name="il">The stage's generator.</param>
        /// <param name="operandTypes">Two inputs then the output (one shared SIMD dtype).</param>
        /// <param name="reversedRow">True: operand 1's row is reversed — its group at column <c>c</c> is the V elements
        /// ending at <c>c</c>, loaded from below and reversed (InnerLoop.Reversed.cs), its masked tail with the reversed lane
        /// mask, its scalar tail at <c>x − c·e</c>.</param>
        /// <param name="vectorBody">The op's vector body: <c>[out strip, x strip] → [out strip]</c>.</param>
        /// <param name="scalarBody">The op's scalar body: <c>[out value, x value] → [out value]</c>.</param>
        /// <remarks>
        /// The vector / masked / scalar choice of every column is the plain block's (full vectors where a whole group
        /// fits, the masked vector or the scalar body for the row's last <c>n mod V</c> columns), so each lane sees the
        /// same operations in the same order as one call per row would give it.
        /// </remarks>
        private static void Emit2DStationaryOutBlock(
            ILGenerator il, NPTypeCode[] operandTypes, bool reversedRow, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody)
        {
            NPTypeCode dtype = operandTypes[2];
            int elemSize = GetTypeSize(dtype);
            long vectorCount = GetVectorCount(dtype);
            long lastLane = vectorCount - 1;
            var vecType = VectorMethodCache.V(VectorBits, GetSimdLaneType(dtype));
            bool masked = TryGetMaskSupport(dtype, out var ms) && ms.Lanes == vectorCount;

            // ---- Prologue: the out (operand 0 = operand 2), x and its row stride, the reversal control ----
            var locO = il.DeclareLocal(typeof(byte*));
            var locX = il.DeclareLocal(typeof(byte*));
            var locXOuter = il.DeclareLocal(typeof(long));
            EmitLdarg(il, Arg2DDataPtrs);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locO);
            EmitLdarg(il, Arg2DDataPtrs);
            il.Emit(OpCodes.Ldc_I4, IntPtr.Size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locX);
            EmitLdarg(il, Arg2DOuterStrides);
            il.Emit(OpCodes.Ldc_I4, sizeof(long));
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locXOuter);
            var locControl = reversedRow ? EmitReverseControl(il, dtype) : null;

            var locI = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));            // −i: a reversed row's index
            var locXr = il.DeclareLocal(typeof(byte*));          // the current row's address of column i
            var locRows = il.DeclareLocal(typeof(long));
            var acc = new LocalBuilder[8];
            for (int k = 0; k < 8; k++)
                acc[k] = il.DeclareLocal(vecType);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // ---- Strips of `width` vectors (8, 4, then 1): out strip in registers across every row ----
            foreach (int width in new[] { 8, 4, 1 })
            {
                long stripElems = width * vectorCount;
                var lblStrip = il.DefineLabel();
                var lblStripEnd = il.DefineLabel();
                var lblRow = il.DefineLabel();
                var lblRowEnd = il.DefineLabel();

                il.MarkLabel(lblStrip);
                // while (i + stripElems <= n)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, stripElems);
                il.Emit(OpCodes.Add);
                EmitLdarg(il, Arg2DInnerCount);
                il.Emit(OpCodes.Bgt, lblStripEnd);

                // acc[k] = load(out + (i + k·V)·e)
                for (int k = 0; k < width; k++)
                {
                    EmitRunElementAddr(il, locO, locI, k * vectorCount, elemSize);
                    EmitVectorLoad(il, dtype);
                    il.Emit(OpCodes.Stloc, acc[k]);
                }

                // xr = the first row's address of column i (a reversed row: of the group ending at column i)
                if (reversedRow)
                    EmitRunElementAddr(il, locX, locJ, -lastLane, elemSize);
                else
                    EmitRunElementAddr(il, locX, locI, 0, elemSize);
                il.Emit(OpCodes.Stloc, locXr);
                EmitLdarg(il, Arg2DOuterCount);
                il.Emit(OpCodes.Stloc, locRows);

                il.MarkLabel(lblRow);
                il.Emit(OpCodes.Ldloc, locRows);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Ble, lblRowEnd);
                for (int k = 0; k < width; k++)
                {
                    // acc[k] = body(acc[k], x strip k of this row)
                    il.Emit(OpCodes.Ldloc, acc[k]);
                    il.Emit(OpCodes.Ldloc, locXr);
                    long disp = (reversedRow ? -k : k) * vectorCount * elemSize;
                    if (disp != 0)
                    {
                        il.Emit(OpCodes.Ldc_I8, disp);
                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                    }
                    EmitVectorLoad(il, dtype);
                    if (reversedRow)
                        EmitReverseLanes(il, dtype, locControl);
                    vectorBody(il);
                    il.Emit(OpCodes.Stloc, acc[k]);
                }
                // xr += the row stride; rows--
                il.Emit(OpCodes.Ldloc, locXr);
                il.Emit(OpCodes.Ldloc, locXOuter);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locXr);
                il.Emit(OpCodes.Ldloc, locRows);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locRows);
                il.Emit(OpCodes.Br, lblRow);
                il.MarkLabel(lblRowEnd);

                // store the strip once
                for (int k = 0; k < width; k++)
                {
                    il.Emit(OpCodes.Ldloc, acc[k]);
                    EmitRunElementAddr(il, locO, locI, k * vectorCount, elemSize);
                    EmitVectorStore(il, dtype);
                }
                EmitStepIndices(il, locI, locJ, stripElems);
                il.Emit(OpCodes.Br, lblStrip);
                il.MarkLabel(lblStripEnd);
            }

            // ---- The columns left over (n − i < V) ----
            var lblDone = il.DefineLabel();
            if (masked)
            {
                // One masked vector run through every row: the plain block's masked tail per row, held in a register.
                var locTail = il.DeclareLocal(typeof(long));
                var locMask = il.DeclareLocal(vecType);
                var locLoadMask = locMask;
                EmitLdarg(il, Arg2DInnerCount);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locTail);
                il.Emit(OpCodes.Ldloc, locTail);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Ble, lblDone);
                EmitLaneMask(il, ms, locTail, reversed: false);
                il.Emit(OpCodes.Stloc, locMask);
                if (reversedRow)
                {
                    // The reversed row's last `tail` elements sit in the TOP lanes of the load that starts V-1 below them.
                    locLoadMask = il.DeclareLocal(vecType);
                    EmitLaneMask(il, ms, locTail, reversed: true);
                    il.Emit(OpCodes.Stloc, locLoadMask);
                }
                LocalBuilder? locFill = null;
                if (ms.One != null)
                {
                    // fill = AndNot(One, mask): integer masked-off lanes become 1, so a software vector divide cannot trap.
                    locFill = il.DeclareLocal(vecType);
                    il.EmitCall(OpCodes.Call, ms.One, null);
                    il.Emit(OpCodes.Ldloc, locMask);
                    il.EmitCall(OpCodes.Call, ms.AndNot!, null);
                    il.Emit(OpCodes.Stloc, locFill);
                }

                // acc = MaskLoad(out + i·e) (| fill)
                EmitRunElementAddr(il, locO, locI, 0, elemSize);
                il.Emit(OpCodes.Ldloc, locMask);
                il.EmitCall(OpCodes.Call, ms.MaskLoad, null);
                if (locFill != null)
                {
                    il.Emit(OpCodes.Ldloc, locFill);
                    il.EmitCall(OpCodes.Call, ms.Or!, null);
                }
                il.Emit(OpCodes.Stloc, acc[0]);

                if (reversedRow)
                    EmitRunElementAddr(il, locX, locJ, -lastLane, elemSize);
                else
                    EmitRunElementAddr(il, locX, locI, 0, elemSize);
                il.Emit(OpCodes.Stloc, locXr);
                EmitLdarg(il, Arg2DOuterCount);
                il.Emit(OpCodes.Stloc, locRows);

                var lblRow = il.DefineLabel();
                var lblRowEnd = il.DefineLabel();
                il.MarkLabel(lblRow);
                il.Emit(OpCodes.Ldloc, locRows);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Ble, lblRowEnd);
                il.Emit(OpCodes.Ldloc, acc[0]);
                il.Emit(OpCodes.Ldloc, locXr);
                il.Emit(OpCodes.Ldloc, locLoadMask);
                il.EmitCall(OpCodes.Call, ms.MaskLoad, null);
                if (reversedRow)
                    EmitReverseLanes(il, dtype, locControl);
                if (locFill != null)
                {
                    il.Emit(OpCodes.Ldloc, locFill);
                    il.EmitCall(OpCodes.Call, ms.Or!, null);
                }
                vectorBody(il);
                il.Emit(OpCodes.Stloc, acc[0]);
                il.Emit(OpCodes.Ldloc, locXr);
                il.Emit(OpCodes.Ldloc, locXOuter);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locXr);
                il.Emit(OpCodes.Ldloc, locRows);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locRows);
                il.Emit(OpCodes.Br, lblRow);
                il.MarkLabel(lblRowEnd);

                // MaskStore(out + i·e, mask, acc)
                EmitRunElementAddr(il, locO, locI, 0, elemSize);
                il.Emit(OpCodes.Ldloc, locMask);
                il.Emit(OpCodes.Ldloc, acc[0]);
                il.EmitCall(OpCodes.Call, ms.MaskStore, null);
            }
            else
            {
                // Scalar column strips over the leftover columns: four columns held in four registers across every row
                // (four independent chains, the rows read once), then single columns. Each lane is still its rows'
                // operations in order through the scalar body, as the plain block's scalar tail gives it. Measured
                // against the two plain-tail alternatives on int16 / uint8, whose short rows are all leftover columns at
                // 16 / 32 lanes: one column across all the rows (a dependent chain per column) lost 0.61-0.76x on rows of
                // 8 and 10; the rows walked again for the leftover columns lost 0.80-0.89x on rows of 33 (one leftover
                // column, a second pass over 2000 rows).
                var acs = new LocalBuilder[4];
                for (int k = 0; k < 4; k++)
                    acs[k] = il.DeclareLocal(GetClrType(dtype));

                foreach (int width in new[] { 4, 1 })
                {
                    var lblStrip = il.DefineLabel();
                    var lblStripEnd = il.DefineLabel();
                    var lblRow = il.DefineLabel();
                    var lblRowEnd = il.DefineLabel();

                    il.MarkLabel(lblStrip);
                    // while (i + width <= n)
                    il.Emit(OpCodes.Ldloc, locI);
                    il.Emit(OpCodes.Ldc_I8, (long)width);
                    il.Emit(OpCodes.Add);
                    EmitLdarg(il, Arg2DInnerCount);
                    il.Emit(OpCodes.Bgt, lblStripEnd);

                    // acs[k] = out[i + k]
                    for (int k = 0; k < width; k++)
                    {
                        EmitRunElementAddr(il, locO, locI, k, elemSize);
                        EmitLoadIndirect(il, dtype);
                        il.Emit(OpCodes.Stloc, acs[k]);
                    }

                    // xr = the first row's column i (x + i·e; a reversed row: x − i·e = x + j·e)
                    EmitRunElementAddr(il, locX, reversedRow ? locJ : locI, 0, elemSize);
                    il.Emit(OpCodes.Stloc, locXr);
                    EmitLdarg(il, Arg2DOuterCount);
                    il.Emit(OpCodes.Stloc, locRows);

                    il.MarkLabel(lblRow);
                    il.Emit(OpCodes.Ldloc, locRows);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Ble, lblRowEnd);
                    for (int k = 0; k < width; k++)
                    {
                        // acs[k] = body(acs[k], x_row[i + k]) — a reversed row's column i + k sits k elements BELOW xr
                        il.Emit(OpCodes.Ldloc, acs[k]);
                        il.Emit(OpCodes.Ldloc, locXr);
                        long disp = (reversedRow ? -k : k) * (long)elemSize;
                        if (disp != 0)
                        {
                            il.Emit(OpCodes.Ldc_I8, disp);
                            il.Emit(OpCodes.Conv_I);
                            il.Emit(OpCodes.Add);
                        }
                        EmitLoadIndirect(il, dtype);
                        scalarBody(il);
                        il.Emit(OpCodes.Stloc, acs[k]);
                    }
                    // xr += the row stride; rows--
                    il.Emit(OpCodes.Ldloc, locXr);
                    il.Emit(OpCodes.Ldloc, locXOuter);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, locXr);
                    il.Emit(OpCodes.Ldloc, locRows);
                    il.Emit(OpCodes.Ldc_I8, 1L);
                    il.Emit(OpCodes.Sub);
                    il.Emit(OpCodes.Stloc, locRows);
                    il.Emit(OpCodes.Br, lblRow);
                    il.MarkLabel(lblRowEnd);

                    // out[i + k] = acs[k]
                    for (int k = 0; k < width; k++)
                    {
                        EmitRunElementAddr(il, locO, locI, k, elemSize);
                        il.Emit(OpCodes.Ldloc, acs[k]);
                        EmitStoreIndirect(il, dtype);
                    }
                    EmitStepIndices(il, locI, locJ, width);
                    il.Emit(OpCodes.Br, lblStrip);
                    il.MarkLabel(lblStripEnd);
                }
            }
            il.MarkLabel(lblDone);
        }

        /// <summary>
        /// Push the masked tail's lane mask for <paramref name="locTail"/> live lanes: <c>GreaterThan(Create(tail),
        /// {0, 1, …})</c> — the bottom lanes — or, <paramref name="reversed"/>, <c>GreaterThan(Create(tail), {V−1, …, 0})</c>
        /// — the top lanes, where a reversed row's last elements sit before the reversal (the plain block builds the same
        /// two masks, <c>Emit2DSimdBlock</c>).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="ms">The dtype's mask support.</param>
        /// <param name="locTail">The live lane count (a long local, 1 .. V−1).</param>
        /// <param name="reversed">True for the top-lanes mask.</param>
        private static void EmitLaneMask(ILGenerator il, MaskSupport ms, LocalBuilder locTail, bool reversed)
        {
            il.Emit(OpCodes.Ldloc, locTail);
            if (ms.Is32) il.Emit(OpCodes.Conv_I4);
            il.EmitCall(OpCodes.Call, ms.MaskBroadcast, null);
            for (int k = 0; k < ms.Lanes; k++)
            {
                int lane = reversed ? ms.Lanes - 1 - k : k;
                if (ms.Is32) il.Emit(OpCodes.Ldc_I4, lane);
                else il.Emit(OpCodes.Ldc_I8, (long)lane);
            }
            il.EmitCall(OpCodes.Call, ms.MaskElements, null);
            il.EmitCall(OpCodes.Call, ms.MaskGreaterThan, null);
            if (ms.MaskAs != null)
                il.EmitCall(OpCodes.Call, ms.MaskAs, null);
        }
    }
}
