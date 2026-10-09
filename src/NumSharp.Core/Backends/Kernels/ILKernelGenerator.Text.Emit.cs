using System;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Text.Emit.cs — the IL of the text kernels (see ILKernelGenerator.Text.cs for the design)
// =============================================================================
//
// Every kernel has the NDInnerLoopFunc signature (void** dataptrs, long* strides, long count, void* auxdata) and the same
// frame: a prologue snapshots the operand pointers and byte strides into locals, the element loop runs `count` times,
// and each element's body is straight-line IL over the chunk plans (TextChunkPlan) of its byte ranges:
//
//   comparison, Equal / NotEqual (branch-free):
//       acc = 0;  for each prefix chunk c: acc |= A[c] ^ B[c];  for each tail chunk c of the longer L: acc |= L[c]
//       equal = acc == 0
//   comparison, Less / LessEqual (first difference):
//       for each prefix chunk c: m = lanes where A[c] != B[c]; if (m) { off = c + first(m); goto diff }
//       result = outcome decided by the longer operand's tail (Less: B's tail nonzero; LessEqual: A's tail all zero)
//     diff:
//       result = A.unit(off) < B.unit(off)          (unsigned; the units differ there)
//   truth:
//       acc = 0;  for each chunk c: acc |= X[c];  nonzero = acc != 0
//       (cast flavor: if (X.unit(0) > U+10FFFF && every later unit of X is 0) inv = 1 — the one-unit str CPython refuses)
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>The highest valid code point; NumPy's bool cast of a <c>U</c> unit above it raises SystemError.</summary>
        private const int MaxUnicodeCodePoint = 0x10FFFF;

        /// <summary><c>BitOperations.TrailingZeroCount(uint)</c>: the first differing lane of a vector mask.</summary>
        private static readonly MethodInfo s_tzcnt32 = typeof(BitOperations).GetMethod(nameof(BitOperations.TrailingZeroCount), new[] { typeof(uint) })
                                                       ?? throw new MissingMethodException(nameof(BitOperations), "TrailingZeroCount(uint)");

        /// <summary><c>BitOperations.TrailingZeroCount(ulong)</c>: the first differing bit of a scalar chunk's xor.</summary>
        private static readonly MethodInfo s_tzcnt64 = typeof(BitOperations).GetMethod(nameof(BitOperations.TrailingZeroCount), new[] { typeof(ulong) })
                                                       ?? throw new MissingMethodException(nameof(BitOperations), "TrailingZeroCount(ulong)");

        /// <summary>
        ///     Emits a comparison kernel. Elements up to <see cref="TextHoistMaxBytes"/> wide get the broadcast-pattern
        ///     dispatch: the kernel reads both input strides once per call and forwards a stride-0 pattern to its stage
        ///     (<see cref="GenerateTextCompareStage"/>); pattern 0 (no broadcast input) runs inline.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoopFunc GenerateTextCompareKernel(TextCompareKey key)
        {
            var dm = NewTextKernelMethod($"NDTextCompare_{key.Op}_u{key.Unit}_{key.WidthA}_{key.WidthB}_a{key.AIndex}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { key.AIndex, 1 - key.AIndex, 2 });

            if (Math.Max(key.WidthA, key.WidthB) <= TextHoistMaxBytes)
            {
                // The pattern of the call: bit 0 = A broadcast (stride 0), bit 1 = B broadcast. NDIter keeps an operand's
                // inner stride fixed within a call, so the pattern holds for every element of it.
                var inline = il.DefineLabel();
                var aStrided = il.DefineLabel();
                var onlyA = il.DefineLabel();
                var both = il.DefineLabel();
                var onlyB = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, frame.Strides[0]);
                il.Emit(OpCodes.Brtrue, aStrided);
                il.Emit(OpCodes.Ldloc, frame.Strides[1]);
                il.Emit(OpCodes.Brtrue, onlyA);
                il.Emit(OpCodes.Br, both);
                il.MarkLabel(aStrided);
                il.Emit(OpCodes.Ldloc, frame.Strides[1]);
                il.Emit(OpCodes.Brtrue, inline);
                il.Emit(OpCodes.Br, onlyB);

                // Each broadcast pattern is a stage of its own (compiled on its first call), so the inline loop's method
                // keeps the JIT's inline budget for its vector helpers.
                il.MarkLabel(onlyA);
                EmitForwardToTextStage(il, GenerateTextCompareStage(key, hoistA: true, hoistB: false));
                il.MarkLabel(onlyB);
                EmitForwardToTextStage(il, GenerateTextCompareStage(key, hoistA: false, hoistB: true));
                il.MarkLabel(both);
                EmitForwardToTextStage(il, GenerateTextCompareStage(key, hoistA: true, hoistB: true));
                il.MarkLabel(inline);
            }

            EmitTextCompareLoop(il, key, frame, hoistA: false, hoistB: false);
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>
        ///     One broadcast-pattern stage of a comparison kernel: the same signature and frame, with the broadcast
        ///     operand's prefix chunks loaded once before the loop and its tail's truth precomputed.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="hoistA">A has inner stride 0 in the calls this stage serves.</param>
        /// <param name="hoistB">B has inner stride 0.</param>
        /// <returns>The stage, ready to be the target of a <c>call</c>.</returns>
        private static DynamicMethod GenerateTextCompareStage(TextCompareKey key, bool hoistA, bool hoistB)
        {
            var dm = NewTextKernelMethod(
                $"NDTextCompare_{key.Op}_u{key.Unit}_{key.WidthA}_{key.WidthB}_a{key.AIndex}_bp{(hoistA ? 1 : 0) | (hoistB ? 2 : 0)}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { key.AIndex, 1 - key.AIndex, 2 });
            EmitTextCompareLoop(il, key, frame, hoistA, hoistB);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        ///     Emits a truth kernel: <c>out[i] = any unit of text[i] != 0</c>, plus (cast flavor) the lone-unit test whose
        ///     result is stored into <c>*(int*)auxdata</c> after the loop when an element failed it.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        /// <remarks>
        ///     <para>
        ///     <b>The cast flavor's rule.</b> NumPy's string-to-bool cast builds a Python <c>str</c> from each element with its
        ///     trailing NULs stripped (<c>PyArray_Scalar</c> → <c>PyUnicode_FromKindAndData</c>). CPython raises only for a
        ///     string of exactly ONE code unit above U+10FFFF (<c>unicode_char</c> → <c>PyUnicode_New(1, ch)</c>): a longer
        ///     string's maximum character comes from <c>find_max_char</c>, which caps it at U+10FFFF, so it builds without an
        ///     error (probed 2.4.2: <c>[0x110000, 'A']</c> casts to True, <c>[0x110000, 0]</c> raises SystemError). An element
        ///     fails exactly when its FIRST unit is above U+10FFFF and every later unit is zero.
        ///     </para>
        ///     <para>
        ///     <b>Cost.</b> The common path pays one scalar load, one unsigned compare and one well-predicted branch per
        ///     element; only a first unit above U+10FFFF reads the rest of the element again.
        ///     </para>
        /// </remarks>
        private static NDInnerLoopFunc GenerateTextTruthKernel(TextTruthKey key)
        {
            var dm = NewTextKernelMethod($"NDTextTruth_u{key.Unit}_{key.Width}{(key.CheckCodePoints ? "_cast" : "")}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { 0, 1 });
            var plan = PlanTextRange(0, key.Width, key.Unit);
            // The truth is a plain OR of the units, so the vector lane type is irrelevant: byte lanes.
            var lane = typeof(byte);

            // The cast flavor's flag spans the whole call: NumPy raises at the first failing element, and the caller raises
            // once after the loop (the partial output is discarded). An empty element (width 0) is '' and never fails.
            LocalBuilder invalid = null;
            TextChunkPlan rest = default;
            if (key.CheckCodePoints && key.Width > 0)
            {
                invalid = il.DeclareLocal(typeof(int));
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, invalid);
                // The units after the first: all zero is what makes a first unit above U+10FFFF a one-unit string.
                rest = PlanTextRange(4, key.Width - 4, 4);
            }

            var acc = new TextAccumulators(il);
            acc.Declare(plan.Kind, lane);
            EmitTextElementLoop(il, frame, advance: new[] { true, true }, body: () =>
            {
                acc.ZeroAll();
                ForEachTextChunk(il, plan, site =>
                {
                    acc.LoadFor(plan.Kind);
                    EmitTextChunkLoad(il, frame.Pointers[0], site, plan.Kind, lane);
                    EmitTextOr(il, plan.Kind, lane);
                    acc.StoreFor(plan.Kind);
                });

                // out = (any chunk nonzero)
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                acc.PushAllZero();
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Ceq);
                il.Emit(OpCodes.Stind_I1);

                if (invalid is not null)
                    EmitTextLoneUnitCheck(il, frame.Pointers[0], rest, invalid);
            });

            if (invalid is not null)
            {
                // *(int*)auxdata = 1 when an element of the call is one unit above U+10FFFF.
                var clean = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, invalid);
                il.Emit(OpCodes.Brfalse, clean);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Stind_I4);
                il.MarkLabel(clean);
            }

            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>
        ///     Emits the cast flavor's per-element test: sets <paramref name="invalid"/> to 1 when the element's first unit is
        ///     above U+10FFFF and every later unit is zero (the element is a one-unit Python <c>str</c> CPython cannot build).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local (a <c>U</c> element at least one unit wide).</param>
        /// <param name="rest">The plan of the units after the first (empty for a one-unit element).</param>
        /// <param name="invalid">The int flag local.</param>
        private static void EmitTextLoneUnitCheck(ILGenerator il, LocalBuilder pointer, TextChunkPlan rest, LocalBuilder invalid)
        {
            var next = il.DefineLabel();
            // The first unit, read unaligned (a U view can sit at any byte), compared unsigned.
            il.Emit(OpCodes.Ldloc, pointer);
            il.Emit(OpCodes.Unaligned, (byte)1);
            il.Emit(OpCodes.Ldind_U4);
            il.Emit(OpCodes.Ldc_I4, MaxUnicodeCodePoint);
            il.Emit(OpCodes.Ble_Un, next);
            // Rare path: a nonzero unit after the first makes the string longer than one unit, which CPython builds.
            if (rest.Count > 0)
            {
                EmitTextRangeNonzero(il, pointer, rest, typeof(byte));
                il.Emit(OpCodes.Brtrue, next);
            }
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Stloc, invalid);
            il.MarkLabel(next);
        }

        // =============================================================================================================
        // The comparison loop
        // =============================================================================================================

        /// <summary>
        ///     Emits the comparison element loop over a frame whose pointers are (A, B, out): hoists the broadcast operands'
        ///     chunks, then for each element computes the bool result and stores it.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="key">The kernel identity.</param>
        /// <param name="frame">The frame (pointers and strides of A, B, out).</param>
        /// <param name="hoistA">A is broadcast in this loop: its chunks are read once, its pointer never advances.</param>
        /// <param name="hoistB">B is broadcast in this loop.</param>
        private static void EmitTextCompareLoop(ILGenerator il, TextCompareKey key, TextFrame frame, bool hoistA, bool hoistB)
        {
            int wA = key.WidthA, wB = key.WidthB;
            var prefix = PlanTextRange(0, Math.Min(wA, wB), key.Unit);
            var tail = PlanTextRange(Math.Min(wA, wB), Math.Abs(wA - wB), key.Unit);
            bool ordering = key.Op is TextCompareOp.Less or TextCompareOp.LessEqual;
            // The first-difference search reads unit lanes so a lane index IS a unit index; equality only xors and ORs.
            var lane = ordering && key.Unit == 4 ? typeof(uint) : typeof(byte);

            var a = new TextOperand(frame.Pointers[0]);
            var b = new TextOperand(frame.Pointers[1]);
            if (hoistA)
                a.Hoist(il, prefix, lane, wA > wB ? tail : default);
            if (hoistB)
                b.Hoist(il, prefix, lane, wB > wA ? tail : default);

            var accumulators = new TextAccumulators(il);
            accumulators.Declare(prefix.Kind, lane);
            if (tail.Count > 0)
                accumulators.Declare(tail.Kind, lane);
            var result = il.DeclareLocal(typeof(int));

            EmitTextElementLoop(il, frame, advance: new[] { !hoistA, !hoistB, true }, body: () =>
            {
                if (ordering)
                    EmitTextOrderBody(il, key, a, b, prefix, tail, lane, result);
                else
                {
                    EmitTextEqualBody(il, a, b, wA >= wB ? a : b, prefix, tail, lane, accumulators);
                    if (key.Op == TextCompareOp.NotEqual)
                    {
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                    }
                    il.Emit(OpCodes.Stloc, result);
                }

                il.Emit(OpCodes.Ldloc, frame.Pointers[2]);
                il.Emit(OpCodes.Ldloc, result);
                il.Emit(OpCodes.Stind_I1);
            });
        }

        /// <summary>
        ///     Emits the Equal test of one element pair and leaves it on the stack (int 1 = equal): the prefix chunks
        ///     xor-ed and OR-ed into the accumulators, the longer operand's tail OR-ed in (or its precomputed truth when it
        ///     is hoisted), then one all-zero test.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="a">Operand A.</param>
        /// <param name="b">Operand B.</param>
        /// <param name="longer">The operand that owns the tail (A when the widths are equal; the tail is empty then).</param>
        /// <param name="prefix">The common-prefix plan.</param>
        /// <param name="tail">The longer operand's tail plan.</param>
        /// <param name="lane">The vector lane type.</param>
        /// <param name="acc">The accumulators (declared for both plans' kinds).</param>
        private static void EmitTextEqualBody(ILGenerator il, TextOperand a, TextOperand b, TextOperand longer,
                                              TextChunkPlan prefix, TextChunkPlan tail, Type lane, TextAccumulators acc)
        {
            acc.ZeroAll();
            ForEachTextChunk(il, prefix, site =>
            {
                acc.LoadFor(prefix.Kind);
                a.PushPrefixChunk(il, site, prefix.Kind, lane);
                b.PushPrefixChunk(il, site, prefix.Kind, lane);
                EmitTextXor(il, prefix.Kind, lane);
                EmitTextOr(il, prefix.Kind, lane);
                acc.StoreFor(prefix.Kind);
            });
            bool tailHoisted = longer.TailNonzero is not null;
            if (tail.Count > 0 && !tailHoisted)
            {
                ForEachTextChunk(il, tail, site =>
                {
                    acc.LoadFor(tail.Kind);
                    EmitTextChunkLoad(il, longer.Pointer, site, tail.Kind, lane);
                    EmitTextOr(il, tail.Kind, lane);
                    acc.StoreFor(tail.Kind);
                });
            }

            acc.PushAllZero();
            if (tail.Count > 0 && tailHoisted)
            {
                // equal = prefix equal AND the broadcast longer operand's tail is all NULs (computed once per call).
                il.Emit(OpCodes.Ldloc, longer.TailNonzero);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Ceq);
                il.Emit(OpCodes.And);
            }
        }

        /// <summary>
        ///     Emits the Less / LessEqual test of one element pair into <paramref name="result"/>: the first differing unit
        ///     of the prefix decides (unsigned), else the longer operand's tail does.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="key">The kernel identity (op, unit, widths).</param>
        /// <param name="a">Operand A.</param>
        /// <param name="b">Operand B.</param>
        /// <param name="prefix">The common-prefix plan.</param>
        /// <param name="tail">The longer operand's tail plan.</param>
        /// <param name="lane">The vector lane type (uint for U, so a lane index is a unit index).</param>
        /// <param name="result">The int local receiving 1 / 0.</param>
        private static void EmitTextOrderBody(ILGenerator il, TextCompareKey key, TextOperand a, TextOperand b,
                                              TextChunkPlan prefix, TextChunkPlan tail, Type lane, LocalBuilder result)
        {
            var diff = il.DefineLabel();
            var done = il.DefineLabel();
            var offset = il.DeclareLocal(typeof(int));
            LocalBuilder mask = prefix.Count == 0 ? null : il.DeclareLocal(IsVectorChunk(prefix.Kind) ? typeof(uint) : typeof(ulong));

            ForEachTextChunk(il, prefix, site =>
            {
                var next = il.DefineLabel();
                a.PushPrefixChunk(il, site, prefix.Kind, lane);
                b.PushPrefixChunk(il, site, prefix.Kind, lane);
                if (IsVectorChunk(prefix.Kind))
                {
                    // mask = the lanes that differ: ~ExtractMostSignificantBits(Equals(a, b)) over the lanes that exist.
                    int bits = TextChunkBytes(prefix.Kind) * 8;
                    int lanes = TextChunkBytes(prefix.Kind) / (lane == typeof(uint) ? 4 : 1);
                    il.Emit(OpCodes.Call, VectorMethodCache.Equals(bits, lane));
                    il.Emit(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(bits, lane));
                    il.Emit(OpCodes.Ldc_I4, lanes == 32 ? -1 : (1 << lanes) - 1);
                    il.Emit(OpCodes.Xor);
                    il.Emit(OpCodes.Dup);
                    il.Emit(OpCodes.Stloc, mask);
                    il.Emit(OpCodes.Brfalse, next);
                    // offset = chunk offset + first differing lane * lane size (a unit for U, a byte for S).
                    PushTextSiteOffset(il, site);
                    il.Emit(OpCodes.Ldloc, mask);
                    il.Emit(OpCodes.Call, s_tzcnt32);
                    if (lane == typeof(uint))
                    {
                        il.Emit(OpCodes.Ldc_I4_2);
                        il.Emit(OpCodes.Shl);
                    }
                }
                else
                {
                    // mask = a ^ b; the first differing BYTE is its lowest set bit / 8 (little-endian element order),
                    // rounded down to its code unit for U.
                    il.Emit(OpCodes.Xor);
                    il.Emit(OpCodes.Dup);
                    il.Emit(OpCodes.Stloc, mask);
                    il.Emit(OpCodes.Brfalse, next);
                    PushTextSiteOffset(il, site);
                    il.Emit(OpCodes.Ldloc, mask);
                    il.Emit(OpCodes.Call, s_tzcnt64);
                    il.Emit(OpCodes.Ldc_I4_3);
                    il.Emit(OpCodes.Shr_Un);
                    if (key.Unit == 4)
                    {
                        il.Emit(OpCodes.Ldc_I4, -4);
                        il.Emit(OpCodes.And);
                    }
                }
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, offset);
                il.Emit(OpCodes.Br, diff);
                il.MarkLabel(next);
            });

            // The prefixes are equal: the zero-padded tail decides.
            //   Less:      A < B  iff B is the longer one and its tail holds a nonzero unit.
            //   LessEqual: A <= B unless A is the longer one and its tail holds a nonzero unit.
            int wA = key.WidthA, wB = key.WidthB;
            if (key.Op == TextCompareOp.Less)
            {
                if (wB > wA)
                    EmitTextTailNonzero(il, b, tail, lane);
                else
                    il.Emit(OpCodes.Ldc_I4_0);
            }
            else
            {
                if (wA > wB)
                {
                    EmitTextTailNonzero(il, a, tail, lane);
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                }
                else
                    il.Emit(OpCodes.Ldc_I4_1);
            }
            il.Emit(OpCodes.Stloc, result);
            il.Emit(OpCodes.Br, done);

            // The units at `offset` differ; the smaller (unsigned) belongs to the smaller string.
            il.MarkLabel(diff);
            EmitTextUnitLoad(il, a.Pointer, offset, key.Unit);
            EmitTextUnitLoad(il, b.Pointer, offset, key.Unit);
            il.Emit(OpCodes.Clt_Un);
            il.Emit(OpCodes.Stloc, result);
            il.MarkLabel(done);
        }

        /// <summary>
        ///     Pushes (int 1 / 0) whether the operand's tail holds a nonzero unit: its precomputed local when hoisted, else an
        ///     OR over the tail's chunks.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="x">The longer operand.</param>
        /// <param name="tail">Its tail plan (non-empty).</param>
        /// <param name="lane">The vector lane type.</param>
        private static void EmitTextTailNonzero(ILGenerator il, TextOperand x, TextChunkPlan tail, Type lane)
        {
            if (x.TailNonzero is not null)
            {
                il.Emit(OpCodes.Ldloc, x.TailNonzero);
                return;
            }
            EmitTextRangeNonzero(il, x.Pointer, tail, lane);
        }

        /// <summary>
        ///     Pushes (int 1 / 0) whether any byte of the planned range of the element at <paramref name="pointer"/> is
        ///     nonzero (fresh accumulators, so it can run anywhere in a body).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local.</param>
        /// <param name="plan">The range plan.</param>
        /// <param name="lane">The vector lane type.</param>
        private static void EmitTextRangeNonzero(ILGenerator il, LocalBuilder pointer, TextChunkPlan plan, Type lane)
        {
            if (plan.Count == 0)
            {
                il.Emit(OpCodes.Ldc_I4_0);
                return;
            }
            var acc = new TextAccumulators(il);
            acc.Declare(plan.Kind, lane);
            acc.ZeroAll();
            ForEachTextChunk(il, plan, site =>
            {
                acc.LoadFor(plan.Kind);
                EmitTextChunkLoad(il, pointer, site, plan.Kind, lane);
                EmitTextOr(il, plan.Kind, lane);
                acc.StoreFor(plan.Kind);
            });
            acc.PushAllZero();
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ceq);
        }

        // =============================================================================================================
        // Frame, loop and chunk primitives
        // =============================================================================================================

        /// <summary>A kernel's snapshot of its operands: one byte pointer and one byte stride local per operand used.</summary>
        private sealed class TextFrame
        {
            /// <summary>The operand pointers, in the order the kernel uses them.</summary>
            internal readonly LocalBuilder[] Pointers;

            /// <summary>The operands' inner byte strides, same order.</summary>
            internal readonly LocalBuilder[] Strides;

            /// <summary>The element counter.</summary>
            internal readonly LocalBuilder Index;

            /// <summary>Builds a frame from its locals.</summary>
            /// <param name="pointers">The pointer locals.</param>
            /// <param name="strides">The stride locals.</param>
            /// <param name="index">The counter local.</param>
            internal TextFrame(LocalBuilder[] pointers, LocalBuilder[] strides, LocalBuilder index)
            {
                Pointers = pointers;
                Strides = strides;
                Index = index;
            }
        }

        /// <summary>A new kernel or stage method with the <see cref="NDInnerLoopFunc"/> signature.</summary>
        /// <param name="name">The method name (shows in JIT diagnostics).</param>
        /// <returns>The method, its IL still to be emitted.</returns>
        private static DynamicMethod NewTextKernelMethod(string name)
            => new DynamicMethod(name, typeof(void), new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                                 typeof(ILKernelGenerator), skipVisibility: true);

        /// <summary>
        ///     Emits the prologue: <c>p[k] = dataptrs[slots[k]]</c> and <c>s[k] = strides[slots[k]]</c> for every used slot,
        ///     snapshotted once so the loop works on locals (the operand order inside the kernel is the caller's choice —
        ///     a swapped comparison reads slot 1 as A).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="slots">The operand slot of each kernel operand.</param>
        /// <returns>The frame.</returns>
        private static TextFrame EmitTextFrame(ILGenerator il, int[] slots)
        {
            var pointers = new LocalBuilder[slots.Length];
            var strides = new LocalBuilder[slots.Length];
            for (int k = 0; k < slots.Length; k++)
            {
                pointers[k] = il.DeclareLocal(typeof(byte*));
                strides[k] = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Ldarg_0);
                if (slots[k] != 0)
                {
                    il.Emit(OpCodes.Ldc_I4, slots[k] * IntPtr.Size);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I);
                il.Emit(OpCodes.Stloc, pointers[k]);
                il.Emit(OpCodes.Ldarg_1);
                if (slots[k] != 0)
                {
                    il.Emit(OpCodes.Ldc_I4, slots[k] * sizeof(long));
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, strides[k]);
            }
            return new TextFrame(pointers, strides, il.DeclareLocal(typeof(long)));
        }

        /// <summary>
        ///     Emits <c>for (i = 0; i &lt; count; i++) { body; p[k] += s[k] (for every advancing operand) }</c>.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame.</param>
        /// <param name="advance">Per frame operand: whether its pointer advances (a hoisted broadcast operand does not —
        ///     its stride is 0 and its chunks live in locals).</param>
        /// <param name="body">Emits one element's work (must leave the stack empty).</param>
        private static void EmitTextElementLoop(ILGenerator il, TextFrame frame, bool[] advance, Action body)
        {
            var top = il.DefineLabel();
            var end = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, frame.Index);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, frame.Index);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, end);

            body();

            for (int k = 0; k < frame.Pointers.Length; k++)
            {
                if (!advance[k])
                    continue;
                il.Emit(OpCodes.Ldloc, frame.Pointers[k]);
                il.Emit(OpCodes.Ldloc, frame.Strides[k]);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, frame.Pointers[k]);
            }
            il.Emit(OpCodes.Ldloc, frame.Index);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, frame.Index);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(end);
        }

        /// <summary>Emits the forward of the kernel's four arguments to a stage, then <c>ret</c>.</summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="stage">The stage (same signature).</param>
        private static void EmitForwardToTextStage(ILGenerator il, DynamicMethod stage)
        {
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, stage);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>Where one chunk of a plan is: a constant offset (unrolled) or an int local (the loop form).</summary>
        private readonly struct TextChunkSite
        {
            /// <summary>The constant offset (valid when <see cref="VarOffset"/> is null).</summary>
            internal readonly int ConstOffset;

            /// <summary>The loop's offset local, or null for a constant site.</summary>
            internal readonly LocalBuilder VarOffset;

            /// <summary>The chunk's index in its plan when unrolled; -1 inside the loop form (no hoisted local for it).</summary>
            internal readonly int Index;

            /// <summary>Builds a site.</summary>
            /// <param name="constOffset">The constant offset.</param>
            /// <param name="varOffset">The offset local, or null.</param>
            /// <param name="index">The unrolled chunk index, or -1.</param>
            internal TextChunkSite(int constOffset, LocalBuilder varOffset, int index)
            {
                ConstOffset = constOffset;
                VarOffset = varOffset;
                Index = index;
            }
        }

        /// <summary>
        ///     Emits <paramref name="body"/> once per chunk of <paramref name="plan"/>: unrolled with constant offsets when the
        ///     plan is short (<see cref="TextChunkPlan.Unrolled"/>), else a loop over the full chunks (an int offset local,
        ///     constant trip count) followed by the overlapping chunk at its constant offset.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="plan">The plan.</param>
        /// <param name="body">Emits one chunk's work at a site; may branch out of the sequence.</param>
        private static void ForEachTextChunk(ILGenerator il, TextChunkPlan plan, Action<TextChunkSite> body)
        {
            if (plan.Count == 0)
                return;
            if (plan.Unrolled)
            {
                for (int i = 0; i < plan.Count; i++)
                    body(new TextChunkSite(plan.Offset(i), null, i));
                return;
            }

            var offset = il.DeclareLocal(typeof(int));
            var top = il.DefineLabel();
            var end = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I4, plan.Start);
            il.Emit(OpCodes.Stloc, offset);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, offset);
            il.Emit(OpCodes.Ldc_I4, plan.Start + plan.FullCount * plan.Size);
            il.Emit(OpCodes.Bge, end);
            body(new TextChunkSite(0, offset, -1));
            il.Emit(OpCodes.Ldloc, offset);
            il.Emit(OpCodes.Ldc_I4, plan.Size);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, offset);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(end);
            if (plan.HasOverlap)
                body(new TextChunkSite(plan.Start + plan.Length - plan.Size, null, -1));
        }

        /// <summary>Pushes a site's offset (int).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="site">The site.</param>
        private static void PushTextSiteOffset(ILGenerator il, TextChunkSite site)
        {
            if (site.VarOffset is not null)
                il.Emit(OpCodes.Ldloc, site.VarOffset);
            else
                il.Emit(OpCodes.Ldc_I4, site.ConstOffset);
        }

        /// <summary>
        ///     Pushes the value of one chunk of the element at <paramref name="pointer"/>: a Vector{256,128}&lt;lane&gt; for a
        ///     vector kind, else the bytes zero-extended to a ulong. Every read is unaligned-safe (an <c>S</c> element sits
        ///     at any byte, a <c>U</c> view at any 4-byte boundary or worse).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local.</param>
        /// <param name="site">The chunk's offset.</param>
        /// <param name="kind">The chunk kind.</param>
        /// <param name="lane">The vector lane type (ignored for scalar kinds).</param>
        private static void EmitTextChunkLoad(ILGenerator il, LocalBuilder pointer, TextChunkSite site, TextChunkKind kind, Type lane)
        {
            il.Emit(OpCodes.Ldloc, pointer);
            if (site.VarOffset is not null)
            {
                il.Emit(OpCodes.Ldloc, site.VarOffset);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            else if (site.ConstOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I4, site.ConstOffset);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            switch (kind)
            {
                case TextChunkKind.Vector256:
                    il.Emit(OpCodes.Call, VectorMethodCache.Load(256, lane));
                    break;
                case TextChunkKind.Vector128:
                    il.Emit(OpCodes.Call, VectorMethodCache.Load(128, lane));
                    break;
                case TextChunkKind.UInt64:
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Ldind_I8);
                    break;
                case TextChunkKind.UInt32:
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Ldind_U4);
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case TextChunkKind.UInt16:
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Ldind_U2);
                    il.Emit(OpCodes.Conv_U8);
                    break;
                default:
                    il.Emit(OpCodes.Ldind_U1);
                    il.Emit(OpCodes.Conv_U8);
                    break;
            }
        }

        /// <summary>Pushes the code unit at <c>pointer + offset</c> as an int (an unsigned byte for <c>S</c>, a uint's bits for <c>U</c>).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="pointer">The element pointer local.</param>
        /// <param name="offset">The int offset local (a unit boundary).</param>
        /// <param name="unit">1 or 4.</param>
        private static void EmitTextUnitLoad(ILGenerator il, LocalBuilder pointer, LocalBuilder offset, int unit)
        {
            il.Emit(OpCodes.Ldloc, pointer);
            il.Emit(OpCodes.Ldloc, offset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            if (unit == 4)
            {
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_U4);
            }
            else
                il.Emit(OpCodes.Ldind_U1);
        }

        /// <summary>Emits the xor of the two chunk values on the stack.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="kind">Their kind.</param>
        /// <param name="lane">The vector lane type.</param>
        private static void EmitTextXor(ILGenerator il, TextChunkKind kind, Type lane)
        {
            if (IsVectorChunk(kind))
                il.Emit(OpCodes.Call, VectorMethodCache.Operator(TextChunkBytes(kind) * 8, lane, "op_ExclusiveOr"));
            else
                il.Emit(OpCodes.Xor);
        }

        /// <summary>Emits the OR of the two chunk values on the stack.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="kind">Their kind.</param>
        /// <param name="lane">The vector lane type.</param>
        private static void EmitTextOr(ILGenerator il, TextChunkKind kind, Type lane)
        {
            if (IsVectorChunk(kind))
                il.Emit(OpCodes.Call, VectorMethodCache.Operator(TextChunkBytes(kind) * 8, lane, "op_BitwiseOr"));
            else
                il.Emit(OpCodes.Or);
        }

        /// <summary>
        ///     The OR accumulators of an element body: at most one per value class (Vector256, Vector128, ulong), declared
        ///     for the plans the body walks and zeroed at the start of every element.
        /// </summary>
        private sealed class TextAccumulators
        {
            /// <summary>The generator the locals belong to.</summary>
            private readonly ILGenerator _il;

            /// <summary>The Vector256 accumulator, or null.</summary>
            private LocalBuilder _v256;

            /// <summary>The Vector128 accumulator, or null.</summary>
            private LocalBuilder _v128;

            /// <summary>The ulong accumulator, or null.</summary>
            private LocalBuilder _scalar;

            /// <summary>The vector lane type of the declared vector accumulators.</summary>
            private Type _lane = typeof(byte);

            /// <summary>Creates an empty set.</summary>
            /// <param name="il">The generator.</param>
            internal TextAccumulators(ILGenerator il) => _il = il;

            /// <summary>Declares the accumulator a chunk kind feeds (idempotent).</summary>
            /// <param name="kind">The kind.</param>
            /// <param name="lane">The vector lane type (the same for every vector accumulator of one body).</param>
            internal void Declare(TextChunkKind kind, Type lane)
            {
                _lane = lane;
                switch (kind)
                {
                    case TextChunkKind.Vector256:
                        _v256 ??= _il.DeclareLocal(VectorMethodCache.V(256, lane));
                        break;
                    case TextChunkKind.Vector128:
                        _v128 ??= _il.DeclareLocal(VectorMethodCache.V(128, lane));
                        break;
                    default:
                        _scalar ??= _il.DeclareLocal(typeof(ulong));
                        break;
                }
            }

            /// <summary>Zeroes every declared accumulator (the start of an element).</summary>
            internal void ZeroAll()
            {
                if (_v256 is not null)
                {
                    _il.Emit(OpCodes.Call, VectorMethodCache.Zero(256, _lane));
                    _il.Emit(OpCodes.Stloc, _v256);
                }
                if (_v128 is not null)
                {
                    _il.Emit(OpCodes.Call, VectorMethodCache.Zero(128, _lane));
                    _il.Emit(OpCodes.Stloc, _v128);
                }
                if (_scalar is not null)
                {
                    _il.Emit(OpCodes.Ldc_I8, 0L);
                    _il.Emit(OpCodes.Stloc, _scalar);
                }
            }

            /// <summary>Pushes the accumulator a chunk kind feeds.</summary>
            /// <param name="kind">The kind.</param>
            internal void LoadFor(TextChunkKind kind) => _il.Emit(OpCodes.Ldloc, For(kind));

            /// <summary>Stores the stack top into the accumulator a chunk kind feeds.</summary>
            /// <param name="kind">The kind.</param>
            internal void StoreFor(TextChunkKind kind) => _il.Emit(OpCodes.Stloc, For(kind));

            /// <summary>The accumulator local of a kind.</summary>
            /// <param name="kind">The kind.</param>
            /// <returns>The local (declared).</returns>
            /// <exception cref="InvalidOperationException">No accumulator was declared for the kind's value class — an
            ///     emitter bug (a body walked a plan it did not declare), caught at kernel generation.</exception>
            private LocalBuilder For(TextChunkKind kind) => kind switch
            {
                TextChunkKind.Vector256 => _v256,
                TextChunkKind.Vector128 => _v128,
                _ => _scalar,
            } ?? throw new InvalidOperationException($"No {kind} accumulator was declared.");

            /// <summary>Pushes (int 1 / 0) whether every declared accumulator is zero; 1 when none is declared.</summary>
            internal void PushAllZero()
            {
                bool any = false;
                if (_v256 is not null)
                {
                    _il.Emit(OpCodes.Ldloc, _v256);
                    _il.Emit(OpCodes.Call, VectorMethodCache.Zero(256, _lane));
                    _il.Emit(OpCodes.Call, VectorMethodCache.EqualsAll(256, _lane));
                    any = true;
                }
                if (_v128 is not null)
                {
                    _il.Emit(OpCodes.Ldloc, _v128);
                    _il.Emit(OpCodes.Call, VectorMethodCache.Zero(128, _lane));
                    _il.Emit(OpCodes.Call, VectorMethodCache.EqualsAll(128, _lane));
                    if (any)
                        _il.Emit(OpCodes.And);
                    any = true;
                }
                if (_scalar is not null)
                {
                    _il.Emit(OpCodes.Ldloc, _scalar);
                    _il.Emit(OpCodes.Ldc_I8, 0L);
                    _il.Emit(OpCodes.Ceq);
                    if (any)
                        _il.Emit(OpCodes.And);
                    any = true;
                }
                if (!any)
                    _il.Emit(OpCodes.Ldc_I4_1);
            }
        }

        /// <summary>
        ///     One comparison operand inside a loop: its pointer local and, when it is broadcast, its hoisted prefix chunks
        ///     and the precomputed truth of its tail.
        /// </summary>
        private sealed class TextOperand
        {
            /// <summary>The element pointer local.</summary>
            internal readonly LocalBuilder Pointer;

            /// <summary>The prefix chunk values read once before the loop (by unrolled chunk index), or null.</summary>
            private LocalBuilder[] _hoisted;

            /// <summary>The int local holding whether this (longer, broadcast) operand's tail is nonzero, or null.</summary>
            internal LocalBuilder TailNonzero { get; private set; }

            /// <summary>Wraps an operand pointer.</summary>
            /// <param name="pointer">The pointer local.</param>
            internal TextOperand(LocalBuilder pointer) => Pointer = pointer;

            /// <summary>
            ///     Emits the hoisting of a broadcast operand (before the element loop): every unrolled prefix chunk into a
            ///     local, and, when it owns a tail, the tail's nonzero test.
            /// </summary>
            /// <param name="il">The generator.</param>
            /// <param name="prefix">The common-prefix plan (unrolled for every hoistable width).</param>
            /// <param name="lane">The vector lane type of the body.</param>
            /// <param name="tail">This operand's tail plan, or a default (empty) plan when it does not own the tail.</param>
            internal void Hoist(ILGenerator il, TextChunkPlan prefix, Type lane, TextChunkPlan tail)
            {
                if (prefix.Unrolled && prefix.Count > 0)
                {
                    _hoisted = new LocalBuilder[prefix.Count];
                    var type = IsVectorChunk(prefix.Kind) ? VectorMethodCache.V(TextChunkBytes(prefix.Kind) * 8, lane) : typeof(ulong);
                    for (int i = 0; i < prefix.Count; i++)
                    {
                        _hoisted[i] = il.DeclareLocal(type);
                        EmitTextChunkLoad(il, Pointer, new TextChunkSite(prefix.Offset(i), null, i), prefix.Kind, lane);
                        il.Emit(OpCodes.Stloc, _hoisted[i]);
                    }
                }
                if (tail.Count > 0)
                {
                    TailNonzero = il.DeclareLocal(typeof(int));
                    EmitTextRangeNonzero(il, Pointer, tail, lane);
                    il.Emit(OpCodes.Stloc, TailNonzero);
                }
            }

            /// <summary>Pushes a prefix chunk: its hoisted local when there is one for the site, else a fresh read.</summary>
            /// <param name="il">The generator.</param>
            /// <param name="site">The chunk site.</param>
            /// <param name="kind">The prefix chunk kind.</param>
            /// <param name="lane">The vector lane type.</param>
            internal void PushPrefixChunk(ILGenerator il, TextChunkSite site, TextChunkKind kind, Type lane)
            {
                if (_hoisted is not null && site.Index >= 0)
                    il.Emit(OpCodes.Ldloc, _hoisted[site.Index]);
                else
                    EmitTextChunkLoad(il, Pointer, site, kind, lane);
            }
        }
    }
}
