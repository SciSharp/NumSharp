using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

// =============================================================================
// ILKernelGenerator.Text.Methods.cs — the IL shell of the numpy.strings loops (strings phase 2, P2.5)
// =============================================================================
//
// RESPONSIBILITY
//   The NDIter inner loops (NDInnerLoopFunc, and the rows / masked forms of strings plan D16 / D17) of NumPy's string
//   ufuncs that compute WITH the text — str_len, find / rfind / count, startswith / endswith, the strip loops, _slice,
//   the pad loops, _zfill, _replace, _expandtabs and _partition_index — and of the wrappers' size statements. The
//   element body is a C# method of NDTextElements (S / U, generic over the code unit) or of NDVStringMethods (the
//   StringDType loops, the V* operations); this shell owns everything around it as emit-time constants.
//
// THE SHELL (strings plan D11, "helper call vs inline IL")
//   One DynamicMethod per TextMethodKey: the operation, the code-unit size and every text operand's element width. The
//   kernel snapshots its operand pointers and strides, then runs the element loop, calling the body with the widths as
//   LITERALS (ldc.i4) and the int64 operands loaded from their elements. The bodies carry AggressiveInlining, so the JIT
//   inlines them into the dynamic method and folds every width test: a <U8 haystack's length is one vector compare,
//   with no width loop and no per-call width argument. A body that needs NumPy's per-call state (a size accumulator)
//   receives the kernel's auxdata pointer.
//
// FORMS
//   * 1-D (NDInnerLoopFunc): the element loop over `count` elements at the operands' byte strides (0 for a broadcast
//     operand, negative for a reversed view).
//   * rows (NDInnerLoop2DFunc, D16): the same element loop inside a row loop, for NDIterRef.ForEachRows' blocks of narrow
//     rows (one kernel call per block instead of per row).
//   * masked (NDInnerLoopFunc with a trailing bool mask operand, D17): every element computed, the result blended into
//     the output under its mask byte without a branch — for the ops with a numeric output (str_len, whose public ufunc
//     takes where=).
//
// CONTRACT
//   dataptrs / strides (BYTES) per operand in the operation's operand order (see TextMethodOp), any strides. The text
//   operands are read exactly within their elements (NDTextElements' contract); an output element is written exactly
//   within its width.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The loops the IL shell of <see cref="ILKernelGenerator.GetTextMethodKernel"/> emits. Each member lists its
    ///     operands in kernel slot order; "text" operands are fixed-width elements whose widths are part of the key
    ///     (<see cref="TextMethodKey.W0"/> … <see cref="TextMethodKey.W4"/>, in the order the operands appear), "i64" operands
    ///     are int64 elements, and "aux" is the kernel's auxdata (an int64 accumulator array).
    /// </summary>
    internal enum TextMethodOp : byte
    {
        /// <summary><c>str_len</c>: (text, i64 out).</summary>
        Length,

        /// <summary><c>find</c>: (text hay, text sub, i64 start, i64 end, i64 out).</summary>
        Find,

        /// <summary><c>rfind</c>: the operands of <see cref="Find"/>.</summary>
        RFind,

        /// <summary><c>count</c>: the operands of <see cref="Find"/>.</summary>
        Count,

        /// <summary><c>startswith</c>: (text, text prefix, i64 start, i64 end, bool out).</summary>
        StartsWith,

        /// <summary><c>endswith</c>: the operands of <see cref="StartsWith"/>.</summary>
        EndsWith,

        /// <summary>The whitespace strip loops: (text, text out of the input's width); <see cref="TextMethodKey.Extra"/> is
        ///     the strip type (<see cref="NDTextElements.LeftStrip"/> …).</summary>
        StripWhitespace,

        /// <summary>The <c>chars</c> strip loops: (text, text chars, text out of the input's width); Extra = strip type.</summary>
        StripChars,

        /// <summary><c>_slice</c>: (text, i64 start, i64 stop, i64 step, text out of the input's width).</summary>
        Slice,

        /// <summary>The pad wrappers' size statement: (text, i64 width), aux = the maximum.</summary>
        PadSize,

        /// <summary><c>_center</c> / <c>_ljust</c> / <c>_rjust</c>: (text, i64 width, text fill, text out); Extra = the
        ///     justification (<see cref="NDTextElements.Center"/> …); the out width is <see cref="TextMethodKey.W2"/>.</summary>
        Pad,

        /// <summary><c>_zfill</c>: (text, i64 width, text out); the out width is <see cref="TextMethodKey.W1"/>.</summary>
        ZFill,

        /// <summary>The <c>replace</c> wrapper's size statement: (text, text old, text new, i64 count), aux = the maximum.</summary>
        ReplaceSize,

        /// <summary><c>_replace</c> with the per-element limit: (text, text old, text new, i64 count, text out); the out
        ///     width is <see cref="TextMethodKey.W3"/>.</summary>
        Replace,

        /// <summary><c>_expandtabs_length</c> as a size statement: (text, i64 tabsize), aux = [maximum, overflow].</summary>
        ExpandTabsSize,

        /// <summary><c>_expandtabs</c>: (text, i64 tabsize, text out); the out width is <see cref="TextMethodKey.W1"/>.</summary>
        ExpandTabs,

        /// <summary>The <c>partition</c> wrappers' size statements: (text, text sep), aux = five accumulators; Extra = 1 for
        ///     <c>rpartition</c>.</summary>
        PartitionSize,

        /// <summary><c>_partition_index</c> / <c>_rpartition_index</c>: (text, text sep, text o1, text o2, text o3); the
        ///     out widths are <see cref="TextMethodKey.W2"/> … <see cref="TextMethodKey.W4"/>; Extra = 1 for the back form.</summary>
        Partition,

        /// <summary>The predicates <c>isalpha</c> … <c>isnumeric</c>: (text, bool out); Extra = the predicate
        ///     (<see cref="NDTextElements.IsAlphaKind"/> …); aux = the call's escalation word (an int64, 0 at the start).</summary>
        IsKind,

        /// <summary>The case maps <c>upper</c> / <c>lower</c> / <c>swapcase</c> / <c>capitalize</c> / <c>title</c>: (text,
        ///     text out of the input's width); Extra = the map (<see cref="NDTextElements.UpperMap"/> …); aux = the call's
        ///     <see cref="NDTextElements.CaseState"/>.</summary>
        CaseMap,

        // The StringDType loops (NDVStringMethods). Their "text" operands are 16-byte cells (every width in the key is
        // the cell size), aux is the call's VStringMethodAux, and the unit is 1 (UTF-8).

        /// <summary><c>str_len</c> over <c>StringDType</c>: (cell, i64 out).</summary>
        VLength,

        /// <summary><c>find</c> / <c>index</c> over <c>StringDType</c>: (cell, cell sub, i64 start, i64 end, i64 out);
        ///     Extra = 1 for <c>index</c> (a miss is recorded as an error).</summary>
        VFind,

        /// <summary><c>rfind</c> / <c>rindex</c> over <c>StringDType</c>: the operands of <see cref="VFind"/>.</summary>
        VRFind,

        /// <summary><c>count</c> over <c>StringDType</c>: the operands of <see cref="VFind"/>.</summary>
        VCount,

        /// <summary><c>startswith</c> over <c>StringDType</c>: (cell, cell prefix, i64 start, i64 end, bool out).</summary>
        VStartsWith,

        /// <summary><c>endswith</c> over <c>StringDType</c>: the operands of <see cref="VStartsWith"/>.</summary>
        VEndsWith,

        /// <summary>The whitespace strip loops over <c>StringDType</c>: (cell, cell out); Extra = the strip type.</summary>
        VStripWhitespace,

        /// <summary>The <c>chars</c> strip loops over <c>StringDType</c>: (cell, cell chars, cell out); Extra = strip type.</summary>
        VStripChars,

        /// <summary><c>_slice</c> over <c>StringDType</c>: (cell, i64 start, i64 stop, i64 step, cell out).</summary>
        VSlice,

        /// <summary><c>_center</c> / <c>_ljust</c> / <c>_rjust</c> over <c>StringDType</c>: (cell, i64 width, cell fill,
        ///     cell out); Extra = the justification.</summary>
        VPad,

        /// <summary><c>_zfill</c> over <c>StringDType</c>: (cell, i64 width, cell out).</summary>
        VZFill,

        /// <summary><c>_replace</c> over <c>StringDType</c>: (cell, cell old, cell new, i64 count, cell out).</summary>
        VReplace,

        /// <summary><c>_expandtabs</c> over <c>StringDType</c>: (cell, i64 tabsize, cell out).</summary>
        VExpandTabs,

        /// <summary><c>_partition</c> / <c>_rpartition</c> over <c>StringDType</c>: (cell, cell sep, cell o1, cell o2,
        ///     cell o3); Extra = 1 for the back form.</summary>
        VPartition,

        /// <summary>The predicates over <c>StringDType</c>: (cell, bool out); Extra = the predicate.</summary>
        VIsKind,

        /// <summary>The case maps over <c>StringDType</c>: (cell, cell out); Extra = the map.</summary>
        VCaseMap,
    }

    /// <summary>
    ///     The identity of a <see cref="TextMethodOp"/> kernel: everything its IL depends on (the operation, the code-unit
    ///     size, the text operands' element widths in operand order, and the operation's flag).
    /// </summary>
    /// <param name="Op">The operation.</param>
    /// <param name="Unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
    /// <param name="W0">The first text operand's element width in bytes.</param>
    /// <param name="W1">The second text operand's width (0 when the operation has fewer text operands).</param>
    /// <param name="W2">The third text operand's width.</param>
    /// <param name="W3">The fourth text operand's width.</param>
    /// <param name="W4">The fifth text operand's width.</param>
    /// <param name="Extra">The operation's flag (strip type, justification, partition direction).</param>
    internal readonly record struct TextMethodKey(TextMethodOp Op, int Unit, int W0, int W1 = 0, int W2 = 0, int W3 = 0, int W4 = 0,
                                                  int Extra = 0);

    public static partial class ILKernelGenerator
    {
        /// <summary>The 1-D <see cref="TextMethodOp"/> kernels, by identity.</summary>
        private static readonly ConcurrentDictionary<TextMethodKey, NDInnerLoopFunc> s_textMethodKernels = new();

        /// <summary>The rows <see cref="TextMethodOp"/> kernels, by identity.</summary>
        private static readonly ConcurrentDictionary<TextMethodKey, NDInnerLoop2DFunc> s_textMethodRowsKernels = new();

        /// <summary>The masked 1-D <see cref="TextMethodOp"/> kernels (numeric outputs only), by identity.</summary>
        private static readonly ConcurrentDictionary<TextMethodKey, NDInnerLoopFunc> s_textMethodMaskedKernels = new();

        /// <summary>How one argument of an element body is loaded.</summary>
        private enum TextArgKind : byte
        {
            /// <summary>The frame operand's element pointer (<c>byte*</c>).</summary>
            Pointer,

            /// <summary>The frame operand's element read as an int64.</summary>
            Int64,

            /// <summary>An int constant.</summary>
            Constant,

            /// <summary>The kernel's auxdata pointer.</summary>
            Aux,
        }

        /// <summary>One argument of an element body.</summary>
        /// <param name="Kind">How it is loaded.</param>
        /// <param name="Value">The frame operand (<see cref="TextArgKind.Pointer"/>, <see cref="TextArgKind.Int64"/>) or the
        ///     constant (<see cref="TextArgKind.Constant"/>).</param>
        private readonly record struct TextArg(TextArgKind Kind, int Value);

        /// <summary>Where an element body's return value goes.</summary>
        private enum TextStore : byte
        {
            /// <summary>The body returns nothing (it writes its own output element, or folds into the auxdata).</summary>
            None,

            /// <summary>The body's long is stored at the frame operand <see cref="TextMethodPlan.StoreOperand"/>.</summary>
            Int64,

            /// <summary>The body's byte (0 / 1) is stored at the frame operand <see cref="TextMethodPlan.StoreOperand"/>.</summary>
            Bool,
        }

        /// <summary>The element body of one kernel and how its arguments and result travel.</summary>
        private sealed class TextMethodPlan
        {
            /// <summary>The closed generic body (<see cref="NDTextElements"/>, instantiated over byte / uint).</summary>
            internal MethodInfo Body;

            /// <summary>The body's arguments, in its parameter order.</summary>
            internal TextArg[] Args;

            /// <summary>How the body's result is stored.</summary>
            internal TextStore Store;

            /// <summary>The frame operand the result is stored at.</summary>
            internal int StoreOperand;

            /// <summary>The number of kernel operands (the frame's slots 0 … Operands - 1).</summary>
            internal int Operands;
        }

        /// <summary>
        ///     The 1-D kernel of one <see cref="TextMethodKey"/>, generated on first use and cached.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller takes
        ///     <see cref="NDTextLoop.ManagedKernel"/>).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A unit size other than 1 or 4, or a width that is negative or not a
        ///     whole number of units.</exception>
        internal static NDInnerLoopFunc GetTextMethodKernel(TextMethodKey key)
        {
            ValidateTextMethodKey(key);
            if (!TextKernelsAvailable)
                return null;
            if (s_textMethodKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textMethodKernels.GetOrAdd(key, GenerateTextMethodKernel(key, masked: false));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextMethodKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The rows kernel (strings plan D16) of one <see cref="TextMethodKey"/>: the element loop inside a row loop for
        ///     <see cref="NDIterRef.ForEachRows"/>.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller keeps the 1-D form).</returns>
        /// <exception cref="ArgumentOutOfRangeException">An invalid unit or width.</exception>
        internal static NDInnerLoop2DFunc GetTextMethodRowsKernel(TextMethodKey key)
        {
            ValidateTextMethodKey(key);
            if (!TextKernelsAvailable)
                return null;
            if (s_textMethodRowsKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textMethodRowsKernels.GetOrAdd(key, GenerateTextMethodRowsKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextMethodRowsKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The masked 1-D kernel (strings plan D17) of an operation with a numeric output: the operation's operands plus a
        ///     trailing bool mask, every element computed and blended into the output under its mask byte.
        /// </summary>
        /// <param name="key">The kernel identity (an operation whose result is an int64 or bool element).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An invalid unit or width.</exception>
        /// <exception cref="ArgumentException">The operation writes text (its loop keeps NDIter's per-run masked calls).</exception>
        internal static NDInnerLoopFunc GetTextMethodMaskedKernel(TextMethodKey key)
        {
            ValidateTextMethodKey(key);
            if (!TextKernelsAvailable)
                return null;
            if (s_textMethodMaskedKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_textMethodMaskedKernels.GetOrAdd(key, GenerateTextMethodKernel(key, masked: true));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTextMethodMaskedKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>Rejects a key no fixed-width descriptor can produce.</summary>
        /// <param name="key">The key.</param>
        /// <exception cref="ArgumentOutOfRangeException">The unit is not 1 or 4, or a width is negative or not a multiple of
        ///     the unit.</exception>
        private static void ValidateTextMethodKey(TextMethodKey key)
        {
            ValidateTextShape(key.Unit, key.W0);
            ValidateTextShape(key.Unit, key.W1);
            ValidateTextShape(key.Unit, key.W2);
            ValidateTextShape(key.Unit, key.W3);
            ValidateTextShape(key.Unit, key.W4);
        }

        /// <summary>
        ///     The element body and argument plan of one key: which <see cref="NDTextElements"/> method each operation calls,
        ///     with which constants and operands, and where its result goes.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The plan.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An unknown operation.</exception>
        private static TextMethodPlan PlanTextMethod(TextMethodKey key)
        {
            var unit = key.Unit == 4 ? typeof(uint) : typeof(byte);
            static TextArg P(int operand) => new(TextArgKind.Pointer, operand);
            static TextArg I(int operand) => new(TextArgKind.Int64, operand);
            static TextArg C(int value) => new(TextArgKind.Constant, value);
            var aux = new TextArg(TextArgKind.Aux, 0);
            var plan = new TextMethodPlan();
            string name;
            switch (key.Op)
            {
                case TextMethodOp.Length:
                    name = nameof(NDTextElements.Units);
                    plan.Args = new[] { P(0), C(key.W0) };
                    plan.Store = TextStore.Int64;
                    plan.StoreOperand = 1;
                    plan.Operands = 2;
                    break;
                case TextMethodOp.Find:
                case TextMethodOp.RFind:
                case TextMethodOp.Count:
                    name = key.Op switch
                    {
                        TextMethodOp.Find => nameof(NDTextElements.Find),
                        TextMethodOp.RFind => nameof(NDTextElements.RFind),
                        _ => nameof(NDTextElements.Count),
                    };
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), I(2), I(3) };
                    plan.Store = TextStore.Int64;
                    plan.StoreOperand = 4;
                    plan.Operands = 5;
                    break;
                case TextMethodOp.StartsWith:
                case TextMethodOp.EndsWith:
                    name = nameof(NDTextElements.TailMatch);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), I(2), I(3), C(key.Op == TextMethodOp.EndsWith ? 1 : 0) };
                    plan.Store = TextStore.Bool;
                    plan.StoreOperand = 4;
                    plan.Operands = 5;
                    break;
                case TextMethodOp.StripWhitespace:
                    name = nameof(NDTextElements.StripWhitespace);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.Extra) };
                    plan.Operands = 2;
                    break;
                case TextMethodOp.StripChars:
                    name = nameof(NDTextElements.StripChars);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), P(2), C(key.Extra) };
                    plan.Operands = 3;
                    break;
                case TextMethodOp.Slice:
                    name = nameof(NDTextElements.Slice);
                    plan.Args = new[] { P(0), C(key.W0), I(1), I(2), I(3), P(4) };
                    plan.Operands = 5;
                    break;
                case TextMethodOp.PadSize:
                    name = nameof(NDTextElements.PadSize);
                    plan.Args = new[] { P(0), C(key.W0), I(1), aux };
                    plan.Operands = 2;
                    break;
                case TextMethodOp.Pad:
                    name = nameof(NDTextElements.Pad);
                    plan.Args = new[] { P(0), C(key.W0), I(1), P(2), P(3), C(key.W2), C(key.Extra) };
                    plan.Operands = 4;
                    break;
                case TextMethodOp.ZFill:
                    name = nameof(NDTextElements.ZFill);
                    plan.Args = new[] { P(0), C(key.W0), I(1), P(2), C(key.W1) };
                    plan.Operands = 3;
                    break;
                case TextMethodOp.ReplaceSize:
                    name = nameof(NDTextElements.ReplaceSize);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), P(2), C(key.W2), I(3), aux };
                    plan.Operands = 4;
                    break;
                case TextMethodOp.Replace:
                    name = nameof(NDTextElements.ReplaceCounted);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), P(2), C(key.W2), I(3), P(4), C(key.W3) };
                    plan.Operands = 5;
                    break;
                case TextMethodOp.ExpandTabsSize:
                    name = nameof(NDTextElements.ExpandTabsSize);
                    plan.Args = new[] { P(0), C(key.W0), I(1), aux };
                    plan.Operands = 2;
                    break;
                case TextMethodOp.ExpandTabs:
                    name = nameof(NDTextElements.ExpandTabs);
                    plan.Args = new[] { P(0), C(key.W0), I(1), P(2), C(key.W1) };
                    plan.Operands = 3;
                    break;
                case TextMethodOp.PartitionSize:
                    name = nameof(NDTextElements.PartitionSize);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), C(key.Extra), aux };
                    plan.Operands = 2;
                    break;
                case TextMethodOp.Partition:
                    name = nameof(NDTextElements.Partition);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.W1), C(key.Extra), P(2), C(key.W2), P(3), C(key.W3), P(4), C(key.W4) };
                    plan.Operands = 5;
                    break;
                case TextMethodOp.IsKind:
                    // The predicate is a literal, so the body's switches on it fold; the auxdata is the call's
                    // NDTextElements.KindState (escalation word and flags table).
                    name = nameof(NDTextElements.IsKind);
                    plan.Args = new[] { P(0), C(key.W0), C(key.Extra), aux };
                    plan.Store = TextStore.Bool;
                    plan.StoreOperand = 1;
                    plan.Operands = 2;
                    break;
                case TextMethodOp.CaseMap:
                    // The map is a literal, so the body's branches on it fold; the body writes its own output element.
                    name = nameof(NDTextElements.CaseMap);
                    plan.Args = new[] { P(0), C(key.W0), P(1), C(key.Extra), aux };
                    plan.Operands = 2;
                    break;
                default:
                    return PlanVStringMethod(key, plan);
            }
            var open = typeof(NDTextElements).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)
                       ?? throw new MissingMethodException(nameof(NDTextElements), name);
            plan.Body = open.MakeGenericMethod(unit);
            return plan;
        }

        /// <summary>
        ///     The plan of a <c>StringDType</c> operation: the <see cref="NDVStringMethods"/> body (not generic: the cells are
        ///     UTF-8 whatever the widths say), its operands, and the auxdata (<see cref="VStringMethodAux"/>) as its last
        ///     argument.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="plan">The plan to fill.</param>
        /// <returns><paramref name="plan"/>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An unknown operation.</exception>
        private static TextMethodPlan PlanVStringMethod(TextMethodKey key, TextMethodPlan plan)
        {
            static TextArg P(int operand) => new(TextArgKind.Pointer, operand);
            static TextArg I(int operand) => new(TextArgKind.Int64, operand);
            static TextArg C(int value) => new(TextArgKind.Constant, value);
            var aux = new TextArg(TextArgKind.Aux, 0);
            string name;
            switch (key.Op)
            {
                case TextMethodOp.VLength:
                    name = nameof(NDVStringMethods.VLength);
                    plan.Args = new[] { P(0), aux };
                    plan.Store = TextStore.Int64;
                    plan.StoreOperand = 1;
                    plan.Operands = 2;
                    break;
                case TextMethodOp.VFind:
                case TextMethodOp.VRFind:
                case TextMethodOp.VCount:
                    name = nameof(NDVStringMethods.VFindLike);
                    int mode = key.Op == TextMethodOp.VFind ? 0 : key.Op == TextMethodOp.VRFind ? 1 : 2;
                    plan.Args = new[] { P(0), P(1), I(2), I(3), C(mode), C(key.Extra), aux };
                    plan.Store = TextStore.Int64;
                    plan.StoreOperand = 4;
                    plan.Operands = 5;
                    break;
                case TextMethodOp.VStartsWith:
                case TextMethodOp.VEndsWith:
                    name = nameof(NDVStringMethods.VTailMatch);
                    plan.Args = new[] { P(0), P(1), I(2), I(3), C(key.Op == TextMethodOp.VEndsWith ? 1 : 0), aux };
                    plan.Store = TextStore.Bool;
                    plan.StoreOperand = 4;
                    plan.Operands = 5;
                    break;
                case TextMethodOp.VStripWhitespace:
                    name = nameof(NDVStringMethods.VStripWhitespace);
                    plan.Args = new[] { P(0), P(1), C(key.Extra), aux };
                    plan.Operands = 2;
                    break;
                case TextMethodOp.VStripChars:
                    name = nameof(NDVStringMethods.VStripChars);
                    plan.Args = new[] { P(0), P(1), P(2), C(key.Extra), aux };
                    plan.Operands = 3;
                    break;
                case TextMethodOp.VSlice:
                    name = nameof(NDVStringMethods.VSlice);
                    plan.Args = new[] { P(0), I(1), I(2), I(3), P(4), aux };
                    plan.Operands = 5;
                    break;
                case TextMethodOp.VPad:
                    name = nameof(NDVStringMethods.VPad);
                    plan.Args = new[] { P(0), I(1), P(2), P(3), C(key.Extra), aux };
                    plan.Operands = 4;
                    break;
                case TextMethodOp.VZFill:
                    name = nameof(NDVStringMethods.VZFill);
                    plan.Args = new[] { P(0), I(1), P(2), aux };
                    plan.Operands = 3;
                    break;
                case TextMethodOp.VReplace:
                    name = nameof(NDVStringMethods.VReplace);
                    plan.Args = new[] { P(0), P(1), P(2), I(3), P(4), aux };
                    plan.Operands = 5;
                    break;
                case TextMethodOp.VExpandTabs:
                    name = nameof(NDVStringMethods.VExpandTabs);
                    plan.Args = new[] { P(0), I(1), P(2), aux };
                    plan.Operands = 3;
                    break;
                case TextMethodOp.VPartition:
                    name = nameof(NDVStringMethods.VPartition);
                    plan.Args = new[] { P(0), P(1), P(2), P(3), P(4), C(key.Extra), aux };
                    plan.Operands = 5;
                    break;
                case TextMethodOp.VIsKind:
                    name = nameof(NDVStringMethods.VIsKind);
                    plan.Args = new[] { P(0), C(key.Extra), aux };
                    plan.Store = TextStore.Bool;
                    plan.StoreOperand = 1;
                    plan.Operands = 2;
                    break;
                case TextMethodOp.VCaseMap:
                    name = nameof(NDVStringMethods.VCaseMap);
                    plan.Args = new[] { P(0), P(1), C(key.Extra), aux };
                    plan.Operands = 2;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(key), key.Op, "Not a text method operation.");
            }
            plan.Body = typeof(NDVStringMethods).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)
                        ?? throw new MissingMethodException(nameof(NDVStringMethods), name);
            return plan;
        }

        /// <summary>The method-name stem of a key's kernels (shows in JIT diagnostics such as <c>DOTNET_JitDisasm</c>).</summary>
        /// <param name="key">The key.</param>
        /// <returns>The stem.</returns>
        private static string TextMethodName(TextMethodKey key)
            => $"NDTextMethod_{key.Op}_u{key.Unit}_{key.W0}_{key.W1}_{key.W2}_{key.W3}_{key.W4}_x{key.Extra}";

        /// <summary>Emits a 1-D kernel (or its masked form, whose mask is the operand after the plan's operands).</summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="masked">True for the masked form.</param>
        /// <returns>The compiled kernel.</returns>
        /// <exception cref="ArgumentException"><paramref name="masked"/> for an operation without a numeric output.</exception>
        private static NDInnerLoopFunc GenerateTextMethodKernel(TextMethodKey key, bool masked)
        {
            var plan = PlanTextMethod(key);
            if (masked && plan.Store == TextStore.None)
                throw new ArgumentException($"{key.Op} writes text: its masked loop is NDIter's per-run call.", nameof(masked));
            var dm = NewTextKernelMethod(TextMethodName(key) + (masked ? "_m" : ""));
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, Slots(plan.Operands + (masked ? 1 : 0)));
            var prefetch = EmitTextPrefetchDelta(il, frame, key.W0);
            EmitTextElementLoop(il, frame, AllTrue(frame.Pointers.Length), () =>
            {
                EmitTextPrefetch(il, frame, prefetch);
                EmitTextMethodBody(il, plan, frame, auxArgument: 3, mask: masked ? frame.Pointers[plan.Operands] : null);
            });
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>The narrowest text element (bytes) whose 1-D loop prefetches its text operand
        ///     (<see cref="EmitTextPrefetchDelta"/>).</summary>
        private const int TextPrefetchMinWidth = 32;

        /// <summary>How far ahead (bytes, in the direction of travel) the 1-D loop prefetches its text operand.</summary>
        private const int TextPrefetchBytes = 4096;

        /// <summary><c>Sse.Prefetch0(void*)</c>, or null where the host has no SSE (the loop then emits no prefetch).</summary>
        private static readonly MethodInfo s_textPrefetch0 =
            Sse.IsSupported ? typeof(Sse).GetMethod(nameof(Sse.Prefetch0), new[] { typeof(void*) }) : null;

        /// <summary>
        ///     Emits the prologue of the text operand's software prefetch: a local holding the byte offset, from the current
        ///     element, of the element <see cref="TextPrefetchBytes"/> ahead in the direction of travel
        ///     (<c>stride * (TextPrefetchBytes / width)</c>), or nothing when the loop should not prefetch.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame (operand 0 is the text the operation reads: every plan's first argument).</param>
        /// <param name="width">Operand 0's element width in bytes (the key's <c>W0</c>).</param>
        /// <returns>The offset local, or null: elements narrower than <see cref="TextPrefetchMinWidth"/> bytes, or a host
        ///     without SSE.</returns>
        /// <remarks>
        ///     A text loop over elements of 32 bytes or more is limited by memory latency, not by the hardware prefetcher's
        ///     stream detection: <c>str_len</c> of 1M <c>&lt;U24</c> elements (96 MB) took 8.3 ms without a prefetch and
        ///     5.9 ms prefetching 4 KB ahead (a bare read of the 96 MB takes 5.2 ms on that core), and the same loop over 100K
        ///     elements (in L3) 0.88 against 0.44 ms. Narrower elements lose 3–7 % to the extra instruction (their loop
        ///     body is short and the hardware keeps up), so they emit none. The offset follows the stride: a reversed view
        ///     prefetches downward, a broadcast operand (stride 0) its own element, a stepped one further ahead; a prefetch
        ///     never faults, so an address past the operand's end is harmless.
        /// </remarks>
        private static LocalBuilder EmitTextPrefetchDelta(ILGenerator il, TextFrame frame, int width)
        {
            if (s_textPrefetch0 is null || width < TextPrefetchMinWidth)
                return null;
            var delta = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldloc, frame.Strides[0]);
            il.Emit(OpCodes.Ldc_I8, (long)(TextPrefetchBytes / width));
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, delta);
            return delta;
        }

        /// <summary>Emits <c>Sse.Prefetch0(p0 + delta)</c> for one element, or nothing when <paramref name="delta"/> is null.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame.</param>
        /// <param name="delta">The offset local of <see cref="EmitTextPrefetchDelta"/>, or null.</param>
        private static void EmitTextPrefetch(ILGenerator il, TextFrame frame, LocalBuilder delta)
        {
            if (delta is null)
                return;
            il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
            il.Emit(OpCodes.Ldloc, delta);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, s_textPrefetch0);
        }

        /// <summary>Emits a rows kernel: the 1-D element loop inside the row loop of <see cref="EmitTextRowLoop"/>.</summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoop2DFunc GenerateTextMethodRowsKernel(TextMethodKey key)
        {
            var plan = PlanTextMethod(key);
            var dm = NewTextRowsKernelMethod(TextMethodName(key) + "_rows");
            var il = dm.GetILGenerator();
            int[] slots = Slots(plan.Operands);
            var frame = EmitTextFrame(il, slots);
            var rows = EmitTextRowsFrame(il, frame, slots);
            EmitTextRowLoop(il, frame, rows, () =>
                EmitTextElementLoop(il, frame, AllTrue(frame.Pointers.Length),
                    // The rows signature carries its auxdata as argument 5.
                    () => EmitTextMethodBody(il, plan, frame, auxArgument: 5, mask: null)));
            il.Emit(OpCodes.Ret);
            return (NDInnerLoop2DFunc)dm.CreateDelegate(typeof(NDInnerLoop2DFunc));
        }

        /// <summary>
        ///     Emits one element: the body's arguments (operand pointers, int64 operand values, width literals, the auxdata),
        ///     the call, and the result store — blended under the mask byte for the masked form.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="plan">The element plan.</param>
        /// <param name="frame">The frame.</param>
        /// <param name="auxArgument">The method argument that holds the auxdata (3 for the 1-D signature, 5 for rows).</param>
        /// <param name="mask">The mask pointer of the masked form, or null.</param>
        /// <remarks>
        ///     The masked store is <c>out = old ^ ((old ^ result) &amp; -mask)</c> (mask 0 or 1): no branch on the mask, which
        ///     a random mask would mispredict, and masked-off elements keep their bits (NumPy's READWRITE <c>out=</c>).
        /// </remarks>
        private static void EmitTextMethodBody(ILGenerator il, TextMethodPlan plan, TextFrame frame, int auxArgument, LocalBuilder mask)
        {
            LocalBuilder result = null;
            if (plan.Store != TextStore.None)
            {
                result = il.DeclareLocal(plan.Store == TextStore.Int64 ? typeof(long) : typeof(int));
            }
            foreach (var arg in plan.Args)
            {
                switch (arg.Kind)
                {
                    case TextArgKind.Pointer:
                        il.Emit(OpCodes.Ldloc, frame.Pointers[arg.Value]);
                        break;
                    case TextArgKind.Int64:
                        il.Emit(OpCodes.Ldloc, frame.Pointers[arg.Value]);
                        il.Emit(OpCodes.Unaligned, (byte)1);
                        il.Emit(OpCodes.Ldind_I8);
                        break;
                    case TextArgKind.Constant:
                        il.Emit(OpCodes.Ldc_I4, arg.Value);
                        break;
                    default:
                        il.Emit(OpCodes.Ldarg_S, (byte)auxArgument);
                        break;
                }
            }
            il.Emit(OpCodes.Call, plan.Body);
            if (plan.Store == TextStore.None)
                return;
            // The Units body returns an int: widen it to the int64 output.
            if (plan.Store == TextStore.Int64 && plan.Body.ReturnType == typeof(int))
                il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Stloc, result);

            var output = frame.Pointers[plan.StoreOperand];
            if (mask is null)
            {
                il.Emit(OpCodes.Ldloc, output);
                il.Emit(OpCodes.Ldloc, result);
                if (plan.Store == TextStore.Int64)
                {
                    il.Emit(OpCodes.Unaligned, (byte)1);
                    il.Emit(OpCodes.Stind_I8);
                }
                else
                    il.Emit(OpCodes.Stind_I1);
                return;
            }

            // Branch-free blend: out = old ^ ((old ^ result) & -(mask != 0)).
            il.Emit(OpCodes.Ldloc, output);
            il.Emit(OpCodes.Ldloc, output);
            if (plan.Store == TextStore.Int64)
            {
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Ldloc, result);
                il.Emit(OpCodes.Xor);
                EmitMaskBit(il, mask);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Neg);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Xor);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Stind_I8);
            }
            else
            {
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Ldloc, result);
                il.Emit(OpCodes.Xor);
                EmitMaskBit(il, mask);
                il.Emit(OpCodes.Neg);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Xor);
                il.Emit(OpCodes.Stind_I1);
            }
        }

        /// <summary>Pushes the mask byte normalized to 0 or 1 (int).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="mask">The mask pointer local.</param>
        private static void EmitMaskBit(ILGenerator il, LocalBuilder mask)
        {
            il.Emit(OpCodes.Ldloc, mask);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Cgt_Un);
        }

        /// <summary>The slot list 0 … <paramref name="n"/> - 1.</summary>
        /// <param name="n">The number of operands.</param>
        /// <returns>The slots.</returns>
        private static int[] Slots(int n)
        {
            var slots = new int[n];
            for (int i = 0; i < n; i++)
                slots[i] = i;
            return slots;
        }

        /// <summary>An advance-every-operand flag array.</summary>
        /// <param name="n">The number of operands.</param>
        /// <returns>The flags.</returns>
        private static bool[] AllTrue(int n)
        {
            var flags = new bool[n];
            for (int i = 0; i < n; i++)
                flags[i] = true;
            return flags;
        }
    }
}
