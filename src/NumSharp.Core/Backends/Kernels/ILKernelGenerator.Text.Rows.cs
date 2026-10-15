using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Text.Rows.cs — the rows and masked forms of the text kernels (strings phase 2, P2.A)
// =============================================================================
//
// THE ROWS FORM (docs/plans/strings-phase2.md §3.2 D16)
//   An NDInnerLoop2DFunc over the same operands as the 1-D kernel: the kernel runs its broadcast-pattern dispatch ONCE per
//   call, then loops `outerCount` rows, each row the 1-D element loop over `innerCount` elements from the row's start
//   pointers, then every row pointer advances by its operand's outer stride. NDIterRef.ForEachRows hands it whole blocks
//   of rows, so a narrow-row iteration (`m[:, :3] == "abc"`, a column-broadcast comparand, a truth array of a column slice)
//   pays the kernel prologue and the iterator step once per block instead of once per row (discovery §2.5: 19.3 µs ->
//   ~2.4 µs for a (1000, 3) slice). A broadcast operand (inner stride 0) is hoisted at every ROW start: a column-broadcast
//   comparand holds a different element per row, and a scalar pays one chunk load per row.
//
// THE MASKED FORM (D17)
//   The comparison over (A, B, out, mask): every element is computed and blended into `out` under its mask byte,
//   `out = mask ? result : out`, branch-free (EmitTextMaskedStore). It replaces one kernel call per mask run (the
//   ARRAYMASK route of NDIterRef.ForEach) with one pass, the mask read as an ordinary operand: 426 µs -> ~173-200 µs for
//   a random two-thirds mask at 100K. Both a 1-D form (ForEach) and a rows form (ForEachRows) exist.
//
// IDENTITY
//   Every form computes exactly the 1-D kernel's elements: the element bodies are the same emitters
//   (EmitTextCompareLoop / EmitTextTruthLoop), only the loop around them differs. The tests compare the forms with the 1-D
//   kernels and the managed reference.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>The rows comparison kernels, keyed by the comparison identity and whether the form is masked.</summary>
        private static readonly ConcurrentDictionary<(TextCompareKey Key, bool Masked), NDInnerLoop2DFunc> s_textCompareRowsKernels = new();

        /// <summary>The 1-D masked comparison kernels, keyed by the comparison identity.</summary>
        private static readonly ConcurrentDictionary<TextCompareKey, NDInnerLoopFunc> s_textCompareMaskedKernels = new();

        /// <summary>The rows truth kernels, keyed like the 1-D truth kernels.</summary>
        private static readonly ConcurrentDictionary<TextTruthKey, NDInnerLoop2DFunc> s_textTruthRowsKernels = new();

        /// <summary>
        ///     The rows form of a comparison kernel (see the file header): operands <c>dataptrs[aIndex]</c> = A,
        ///     <c>dataptrs[1 - aIndex]</c> = B, <c>dataptrs[2]</c> = the bool output and, when <paramref name="masked"/>,
        ///     <c>dataptrs[3]</c> = the bool mask the results are blended under.
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's element width in bytes.</param>
        /// <param name="widthB">B's element width in bytes.</param>
        /// <param name="aIndex">The operand slot A is read from (0 or 1).</param>
        /// <param name="masked">True for the masked form (four operands, <c>out = mask ? result : out</c>).</param>
        /// <param name="rstrip">True for NumPy's <c>rstrip=True</c> loop (see <see cref="GetTextCompareKernel"/>).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller keeps its 1-D route).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, a width that is negative or not a
        ///     whole number of units, or an operand index other than 0 or 1.</exception>
        internal static NDInnerLoop2DFunc GetTextCompareRowsKernel(TextCompareOp op, int unit, int widthA, int widthB, int aIndex,
                                                                    bool masked, bool rstrip = false)
        {
            var key = ValidatedTextCompareKey(op, unit, widthA, widthB, aIndex, rstrip);
            if (!TextKernelsAvailable)
                return null;
            if (s_textCompareRowsKernels.TryGetValue((key, masked), out var cached))
                return cached;
            try
            {
                return s_textCompareRowsKernels.GetOrAdd((key, masked), GenerateTextCompareRowsKernel(key, masked));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextCompareRowsKernel({key}, {masked}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The 1-D masked comparison kernel: the operands of <see cref="GetTextCompareKernel"/> plus
        ///     <c>dataptrs[3]</c> = the bool mask; every element is computed and blended into the output,
        ///     <c>out = mask ? result : out</c>. Driven by <see cref="NDIterRef.ForEach"/> over an iteration whose mask is an
        ///     ordinary read-only operand (not ARRAYMASK).
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's element width in bytes.</param>
        /// <param name="widthB">B's element width in bytes.</param>
        /// <param name="aIndex">The operand slot A is read from (0 or 1).</param>
        /// <param name="rstrip">True for NumPy's <c>rstrip=True</c> loop (see <see cref="GetTextCompareKernel"/>).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException">As <see cref="GetTextCompareRowsKernel"/>.</exception>
        internal static NDInnerLoopFunc GetTextCompareMaskedKernel(TextCompareOp op, int unit, int widthA, int widthB, int aIndex,
                                                                   bool rstrip = false)
        {
            var key = ValidatedTextCompareKey(op, unit, widthA, widthB, aIndex, rstrip);
            if (!TextKernelsAvailable)
                return null;
            if (s_textCompareMaskedKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textCompareMaskedKernels.GetOrAdd(key, GenerateTextCompareMaskedKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextCompareMaskedKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The rows form of a truth kernel: operands <c>dataptrs[0]</c> = text and <c>dataptrs[1]</c> = the bool output;
        ///     the cast flavor's flag (one unit above U+10FFFF somewhere in the block) is stored into <c>*(int*)auxdata</c>
        ///     after the last row, as the 1-D kernel stores it after its loop.
        /// </summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <param name="checkCodePoints">The cast flavor for <c>U</c> (see <see cref="GetTextTruthKernel"/>).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, or a width that is negative or not a
        ///     whole number of units.</exception>
        internal static NDInnerLoop2DFunc GetTextTruthRowsKernel(int unit, int width, bool checkCodePoints)
        {
            ValidateTextShape(unit, width);
            if (!TextKernelsAvailable)
                return null;
            var key = new TextTruthKey(unit, width, checkCodePoints && unit == 4);
            if (s_textTruthRowsKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textTruthRowsKernels.GetOrAdd(key, GenerateTextTruthRowsKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextTruthRowsKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>Validates a comparison's shape (as <see cref="GetTextCompareKernel"/> does) and builds its key.</summary>
        /// <param name="op">The comparison.</param>
        /// <param name="unit">The code-unit size.</param>
        /// <param name="widthA">A's width.</param>
        /// <param name="widthB">B's width.</param>
        /// <param name="aIndex">A's operand slot.</param>
        /// <param name="rstrip">True for NumPy's <c>rstrip=True</c> loop.</param>
        /// <returns>The key.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An invalid unit, width or operand slot.</exception>
        private static TextCompareKey ValidatedTextCompareKey(TextCompareOp op, int unit, int widthA, int widthB, int aIndex,
                                                              bool rstrip = false)
        {
            ValidateTextShape(unit, widthA);
            ValidateTextShape(unit, widthB);
            if (aIndex is not (0 or 1))
                throw new ArgumentOutOfRangeException(nameof(aIndex), aIndex, "A comparison reads its operands from slots 0 and 1.");
            return new TextCompareKey(op, unit, widthA, widthB, aIndex, rstrip);
        }

        // =============================================================================================================
        // Generation
        // =============================================================================================================

        /// <summary>
        ///     Emits a rows comparison kernel: the broadcast-pattern dispatch on the two inner strides (once per call), each
        ///     pattern a rows stage of its own; pattern 0 runs inline.
        /// </summary>
        /// <param name="key">The comparison identity.</param>
        /// <param name="masked">Whether the form blends under a mask (operand 3).</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoop2DFunc GenerateTextCompareRowsKernel(TextCompareKey key, bool masked)
        {
            var dm = NewTextRowsKernelMethod($"{TextCompareName(key)}_rows{(masked ? "_m" : "")}");
            var il = dm.GetILGenerator();
            int[] slots = TextCompareSlots(key, masked);
            var frame = EmitTextFrame(il, slots);
            if (Math.Max(key.WidthA, key.WidthB) <= TextHoistMaxBytes)
                EmitTextComparePatternDispatch(il, frame, (hoistA, hoistB) =>
                    EmitForwardToTextRowsStage(il, GenerateTextCompareRowsStage(key, masked, hoistA, hoistB)));
            var rows = EmitTextRowsFrame(il, frame, slots);
            EmitTextRowLoop(il, frame, rows, () =>
                EmitTextCompareLoop(il, key, frame, hoistA: false, hoistB: false, masked ? frame.Pointers[3] : null));
            il.Emit(OpCodes.Ret);
            return (NDInnerLoop2DFunc)dm.CreateDelegate(typeof(NDInnerLoop2DFunc));
        }

        /// <summary>
        ///     One broadcast-pattern stage of a rows comparison kernel: the rows signature and frame, the hoisting of the
        ///     broadcast operand(s) at every row start.
        /// </summary>
        /// <param name="key">The comparison identity.</param>
        /// <param name="masked">Whether the form blends under a mask.</param>
        /// <param name="hoistA">A has inner stride 0 in the calls this stage serves.</param>
        /// <param name="hoistB">B has inner stride 0.</param>
        /// <returns>The stage, ready to be the target of a <c>call</c>.</returns>
        private static DynamicMethod GenerateTextCompareRowsStage(TextCompareKey key, bool masked, bool hoistA, bool hoistB)
        {
            var dm = NewTextRowsKernelMethod(
                $"{TextCompareName(key)}_rows{(masked ? "_m" : "")}_bp{(hoistA ? 1 : 0) | (hoistB ? 2 : 0)}");
            var il = dm.GetILGenerator();
            int[] slots = TextCompareSlots(key, masked);
            var frame = EmitTextFrame(il, slots);
            var rows = EmitTextRowsFrame(il, frame, slots);
            EmitTextRowLoop(il, frame, rows, () =>
                EmitTextCompareLoop(il, key, frame, hoistA, hoistB, masked ? frame.Pointers[3] : null));
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>Emits the 1-D masked comparison kernel: the pattern dispatch, each pattern a 1-D masked stage.</summary>
        /// <param name="key">The comparison identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoopFunc GenerateTextCompareMaskedKernel(TextCompareKey key)
        {
            var dm = NewTextKernelMethod($"{TextCompareName(key)}_m");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, TextCompareSlots(key, masked: true));
            if (Math.Max(key.WidthA, key.WidthB) <= TextHoistMaxBytes)
                EmitTextComparePatternDispatch(il, frame, (hoistA, hoistB) =>
                    EmitForwardToTextStage(il, GenerateTextCompareMaskedStage(key, hoistA, hoistB)));
            EmitTextCompareLoop(il, key, frame, hoistA: false, hoistB: false, frame.Pointers[3]);
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>One broadcast-pattern stage of the 1-D masked comparison kernel.</summary>
        /// <param name="key">The comparison identity.</param>
        /// <param name="hoistA">A has inner stride 0 in the calls this stage serves.</param>
        /// <param name="hoistB">B has inner stride 0.</param>
        /// <returns>The stage, ready to be the target of a <c>call</c>.</returns>
        private static DynamicMethod GenerateTextCompareMaskedStage(TextCompareKey key, bool hoistA, bool hoistB)
        {
            var dm = NewTextKernelMethod($"{TextCompareName(key)}_m_bp{(hoistA ? 1 : 0) | (hoistB ? 2 : 0)}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, TextCompareSlots(key, masked: true));
            EmitTextCompareLoop(il, key, frame, hoistA, hoistB, frame.Pointers[3]);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>Emits a rows truth kernel: the 1-D truth loop inside a row loop, the cast flavor's flag stored once.</summary>
        /// <param name="key">The truth identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoop2DFunc GenerateTextTruthRowsKernel(TextTruthKey key)
        {
            var dm = NewTextRowsKernelMethod($"NDTextTruth_u{key.Unit}_{key.Width}{(key.CheckCodePoints ? "_cast" : "")}_rows");
            var il = dm.GetILGenerator();
            int[] slots = { 0, 1 };
            var frame = EmitTextFrame(il, slots);
            var invalid = DeclareTextTruthFlag(il, key);
            var rows = EmitTextRowsFrame(il, frame, slots);
            EmitTextRowLoop(il, frame, rows, () => EmitTextTruthLoop(il, key, frame, invalid));
            // The rows form's auxdata is argument 5.
            EmitTextTruthFlagStore(il, invalid, auxArgument: 5);
            il.Emit(OpCodes.Ret);
            return (NDInnerLoop2DFunc)dm.CreateDelegate(typeof(NDInnerLoop2DFunc));
        }

        // =============================================================================================================
        // Shared pieces of the forms
        // =============================================================================================================

        /// <summary>
        ///     The method-name stem of a comparison kernel (shows in JIT diagnostics such as <c>DOTNET_JitDisasm</c>); the
        ///     rstrip flavor ends in <c>_rs</c>, so <c>DOTNET_JitDisasm="NDTextCompare_*_rs*"</c> selects only it.
        /// </summary>
        /// <param name="key">The comparison identity.</param>
        /// <returns>The stem.</returns>
        private static string TextCompareName(TextCompareKey key)
            => $"NDTextCompare_{key.Op}_u{key.Unit}_{key.WidthA}_{key.WidthB}_a{key.AIndex}{(key.Rstrip ? "_rs" : "")}";

        /// <summary>The operand slots of a comparison kernel's frame: A, B, out, and the mask for the masked form.</summary>
        /// <param name="key">The comparison identity (its <c>AIndex</c> decides which slot is A).</param>
        /// <param name="masked">Whether the frame includes the mask (slot 3).</param>
        /// <returns>The slots, in frame order.</returns>
        private static int[] TextCompareSlots(TextCompareKey key, bool masked)
            => masked ? new[] { key.AIndex, 1 - key.AIndex, 2, 3 } : new[] { key.AIndex, 1 - key.AIndex, 2 };

        /// <summary>
        ///     Emits the broadcast-pattern dispatch of a comparison kernel: the two inner strides (frame operands 0 = A and
        ///     1 = B) select pattern 1 (A broadcast), 2 (B broadcast) or 3 (both), each forwarded by
        ///     <paramref name="emitForward"/>; pattern 0 falls through to the IL after the dispatch. NDIter keeps an
        ///     operand's inner stride fixed within a call, so the pattern holds for every element (and row) of it.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="frame">The kernel's frame (its strides are the inner strides).</param>
        /// <param name="emitForward">Emits the forward to the stage of (hoistA, hoistB), ending in <c>ret</c>.</param>
        private static void EmitTextComparePatternDispatch(ILGenerator il, TextFrame frame, Action<bool, bool> emitForward)
        {
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

            il.MarkLabel(onlyA);
            emitForward(true, false);
            il.MarkLabel(onlyB);
            emitForward(false, true);
            il.MarkLabel(both);
            emitForward(true, true);
            il.MarkLabel(inline);
        }

        /// <summary>A new rows kernel or stage method with the <see cref="NDInnerLoop2DFunc"/> signature.</summary>
        /// <param name="name">The method name.</param>
        /// <returns>The method, its IL still to be emitted.</returns>
        private static DynamicMethod NewTextRowsKernelMethod(string name)
            => new DynamicMethod(name, typeof(void),
                                 new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long), typeof(void*) },
                                 typeof(ILKernelGenerator), skipVisibility: true);

        /// <summary>Emits the forward of a rows kernel's six arguments to a rows stage, then <c>ret</c>.</summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="stage">The stage (rows signature).</param>
        private static void EmitForwardToTextRowsStage(ILGenerator il, DynamicMethod stage)
        {
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Call, stage);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>A rows kernel's row state: each operand's row start pointer and outer stride, and the row counter.</summary>
        private sealed class TextRowsFrame
        {
            /// <summary>Per frame operand, the pointer to the current row's first element.</summary>
            internal readonly LocalBuilder[] Bases;

            /// <summary>Per frame operand, the byte distance between two rows.</summary>
            internal readonly LocalBuilder[] OuterStrides;

            /// <summary>The row counter.</summary>
            internal readonly LocalBuilder Row;

            /// <summary>Builds a rows frame from its locals.</summary>
            /// <param name="bases">The row start pointer locals.</param>
            /// <param name="outerStrides">The outer stride locals.</param>
            /// <param name="row">The row counter local.</param>
            internal TextRowsFrame(LocalBuilder[] bases, LocalBuilder[] outerStrides, LocalBuilder row)
            {
                Bases = bases;
                OuterStrides = outerStrides;
                Row = row;
            }
        }

        /// <summary>
        ///     Emits the rows prologue: each row start pointer begins at the frame pointer the 1-D prologue read from
        ///     <c>dataptrs</c>, and each outer stride is read from argument 3 at the operand's slot.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame (its pointers already hold the block's first element of each operand).</param>
        /// <param name="slots">The operand slot of each frame operand (the slots <see cref="EmitTextFrame"/> used).</param>
        /// <returns>The rows frame.</returns>
        private static TextRowsFrame EmitTextRowsFrame(ILGenerator il, TextFrame frame, int[] slots)
        {
            var bases = new LocalBuilder[slots.Length];
            var outer = new LocalBuilder[slots.Length];
            for (int k = 0; k < slots.Length; k++)
            {
                bases[k] = il.DeclareLocal(typeof(byte*));
                il.Emit(OpCodes.Ldloc, frame.Pointers[k]);
                il.Emit(OpCodes.Stloc, bases[k]);
                outer[k] = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Ldarg_3);
                if (slots[k] != 0)
                {
                    il.Emit(OpCodes.Ldc_I4, slots[k] * sizeof(long));
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, outer[k]);
            }
            return new TextRowsFrame(bases, outer, il.DeclareLocal(typeof(long)));
        }

        /// <summary>
        ///     Emits <c>for (row = 0; row &lt; outerCount; row++) { p[k] = base[k]; body; base[k] += outer[k] }</c>: each
        ///     row restarts the frame pointers at its first elements, so <paramref name="body"/> is a 1-D element loop over
        ///     <c>innerCount</c> (argument 2 in both signatures).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame whose pointers the body walks.</param>
        /// <param name="rows">The rows frame.</param>
        /// <param name="body">Emits one row's work (hoisting and the element loop; must leave the stack empty).</param>
        private static void EmitTextRowLoop(ILGenerator il, TextFrame frame, TextRowsFrame rows, Action body)
        {
            var top = il.DefineLabel();
            var end = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, rows.Row);
            il.MarkLabel(top);
            il.Emit(OpCodes.Ldloc, rows.Row);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, end);

            for (int k = 0; k < rows.Bases.Length; k++)
            {
                il.Emit(OpCodes.Ldloc, rows.Bases[k]);
                il.Emit(OpCodes.Stloc, frame.Pointers[k]);
            }

            body();

            for (int k = 0; k < rows.Bases.Length; k++)
            {
                il.Emit(OpCodes.Ldloc, rows.Bases[k]);
                il.Emit(OpCodes.Ldloc, rows.OuterStrides[k]);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, rows.Bases[k]);
            }
            il.Emit(OpCodes.Ldloc, rows.Row);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, rows.Row);
            il.Emit(OpCodes.Br, top);
            il.MarkLabel(end);
        }
    }
}
