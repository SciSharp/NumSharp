using System;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDTextLoop.cs — the N-operand driver of the numpy.strings loops (strings phase 2, P2.A's unary text driver, P2.5)
// =============================================================================
//
// RESPONSIBILITY
//   Runs a TextMethodOp kernel (ILKernelGenerator.Text.Methods.cs, or its managed twin) over one or more input operands
//   and one or more output operands the caller already allocated — NumPy's ufunc call after type resolution: the
//   operands are in the loop's descriptors (text in its own descriptor, the integer parameters as int64), the outputs
//   exist. It also runs the SIZE passes of the computed-width wrappers (no output; the kernel folds into its auxdata).
//
// ROUTES (the comparison driver's, generalized to N operands; strings plan Appendix D)
//   * NumPy's trivial loop (try_trivial_single_output_loop): no where= mask, every input one element (stride 0) or of
//     the outputs' dimensions in the outputs' shared contiguous order (any 1-D input at its own byte stride for a 0-d /
//     1-D output), and no caller output that may share memory with an input — one kernel call covers everything.
//   * NDIter: EXTERNAL_LOOP over all operands (COPY_IF_OVERLAP for a caller output, REFS_OK for StringDType operands).
//     An unmasked iteration of several rows takes the rows kernel (D16, one call per block of rows); a where= mask is
//     the ARRAYMASK operand (one kernel call per mask run), or the masked kernel (D17) when the caller asks for it.
//
// BROADCAST ERRORS
//   NumPy's ufunc layer names every operand's shape: "operands could not be broadcast together with shapes (2,) (3,) ()
//   () " (the trailing space is NumPy's). BroadcastDims raises that text (IncorrectShapeException, the house type for
//   NumPy's ValueError of a broadcast failure).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The N-operand driver of the <c>numpy.strings</c> loops: NumPy's trivial loop or an NDIter pass over operands the
    ///     caller resolved and allocated (see the file header), and the size passes of the computed-width functions.
    /// </summary>
    internal static unsafe class NDTextLoop
    {
        /// <summary>
        ///     Test switch: forces every call through NDIter's 1-D route (no trivial loop, no rows kernel), so the tests can
        ///     compare the routes. Process-wide; set it only from a test that restores it.
        /// </summary>
        internal static bool ForceIterator;

        /// <summary>Test counter: the calls that took the rows kernel.</summary>
        internal static long RowsRuns;

        /// <summary>
        ///     The broadcast dimensions of the operands (null entries skipped), raising NumPy's ufunc text when they do not
        ///     broadcast.
        /// </summary>
        /// <param name="operands">The operands, in the order NumPy's message lists them.</param>
        /// <returns>The broadcast dimensions (empty for all-0-d operands).</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast: <c>operands could not be broadcast
        ///     together with shapes (2,) (3,) </c>.</exception>
        internal static long[] BroadcastDims(params NDArray[] operands)
        {
            int nd = 0;
            foreach (var op in operands)
                if (op is not null && op.ndim > nd)
                    nd = op.ndim;
            var dims = new long[nd];
            for (int d = 0; d < nd; d++)
                dims[d] = 1;
            foreach (var op in operands)
            {
                if (op is null)
                    continue;
                var od = op.Shape.dimensions ?? Array.Empty<long>();
                int lead = nd - od.Length;
                for (int k = 0; k < od.Length; k++)
                {
                    long v = od[k];
                    if (v == 1)
                        continue;
                    ref long cur = ref dims[lead + k];
                    if (cur == 1)
                        cur = v;
                    else if (cur != v)
                        throw BroadcastError(operands);
                }
            }
            return dims;
        }

        /// <summary>NumPy's ufunc broadcast error naming every operand's shape (each followed by a space).</summary>
        /// <param name="operands">The operands (null entries skipped).</param>
        /// <returns>The exception (not thrown).</returns>
        internal static IncorrectShapeException BroadcastError(NDArray[] operands)
        {
            var sb = new StringBuilder("operands could not be broadcast together with shapes ");
            foreach (var op in operands)
            {
                if (op is null)
                    continue;
                sb.Append(op.Shape.ToPythonTuple()).Append(' ');
            }
            return new IncorrectShapeException(sb.ToString());
        }

        /// <summary>The product of a dimension list (1 for 0-d).</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The element count.</returns>
        internal static long Size(long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            return n;
        }

        /// <summary>
        ///     Runs a kernel over inputs and pre-allocated outputs (NumPy's ufunc loop after resolution): the trivial loop
        ///     when it applies, else NDIter (rows kernel for an unmasked multi-row iteration, ARRAYMASK runs or the masked
        ///     kernel under <paramref name="where"/>).
        /// </summary>
        /// <param name="key">The kernel identity (its widths must match the operands' descriptors).</param>
        /// <param name="operands">The inputs, then the outputs, in the kernel's slot order.</param>
        /// <param name="nin">The number of inputs.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="aux">The kernel's auxdata, or null.</param>
        /// <param name="callerOut">True when an output is the caller's <c>out=</c> (it may share memory with an input:
        ///     the trivial loop declines, NDIter copies an overlapping input first).</param>
        /// <param name="maskedBlend">True to run a mask through the masked kernel (D17: one pass, every element computed
        ///     and blended) instead of NDIter's per-run calls; only for an operation with a numeric output.</param>
        /// <exception cref="Exception">The kernel's own exceptions (none for the fixed-width bodies).</exception>
        internal static void Execute(TextMethodKey key, NDArray[] operands, int nin, NDArray where, void* aux, bool callerOut,
                                     bool maskedBlend = false)
        {
            var first = operands[nin];
            if (first.size == 0)
                return;
            var kernel = ILKernelGenerator.GetTextMethodKernel(key) ?? ManagedKernel(key, masked: false);
            if (where is null && !ForceIterator && TryTrivial(kernel, operands, nin, aux, callerOut))
                return;

            int nop = operands.Length;
            bool anyVString = false;
            foreach (var op in operands)
                anyVString |= op.Storage.IsVString;
            var flags = NDIterGlobalFlags.EXTERNAL_LOOP;
            if (callerOut)
                flags |= NDIterGlobalFlags.COPY_IF_OVERLAP;
            if (anyVString)
                flags |= NDIterGlobalFlags.REFS_OK;

            if (where is not null && maskedBlend)
            {
                // The mask as an ordinary read-only operand after the outputs; the masked kernel blends every result.
                var masked = ILKernelGenerator.GetTextMethodMaskedKernel(key) ?? ManagedKernel(key, masked: true);
                var mops = new NDArray[nop + 1];
                Array.Copy(operands, mops, nop);
                mops[nop] = where;
                var mflags = OperandFlags(nop + 1, nin, outputFlags: NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.NO_BROADCAST);
                mflags[nop] = NDIterPerOpFlags.READONLY;
                using var miter = NDIterRef.AdvancedNew(nop + 1, mops, flags, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                    mflags, opDescrs: new DType[nop + 1]);
                miter.ForEach(masked, aux);
                return;
            }

            NDArray[] ops = operands;
            NDIterPerOpFlags[] opFlags;
            if (where is null)
                opFlags = OperandFlags(nop, nin, outputFlags: NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST);
            else
            {
                // NumPy's masked ufunc loop: the outputs READWRITE and WRITEMASKED by the trailing ARRAYMASK operand.
                ops = new NDArray[nop + 1];
                Array.Copy(operands, ops, nop);
                ops[nop] = where;
                opFlags = OperandFlags(nop + 1, nin,
                    outputFlags: NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEMASKED | NDIterPerOpFlags.NO_BROADCAST);
                opFlags[nop] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK;
            }
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, flags, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags, opDescrs: new DType[ops.Length]);
            if (where is null && !ForceIterator && iter.CanRunRows)
            {
                var rows = ILKernelGenerator.GetTextMethodRowsKernel(key);
                if (rows is not null)
                {
                    iter.ForEachRows(rows, aux);
                    RowsRuns++;
                    return;
                }
            }
            iter.ForEach(kernel, aux);
        }

        /// <summary>
        ///     Runs a size kernel over the broadcast of its inputs alone (the wrappers' sizing statements: the kernel folds
        ///     every element into the accumulators at <paramref name="aux"/>).
        /// </summary>
        /// <param name="key">The size kernel's identity.</param>
        /// <param name="inputs">The inputs (they broadcast; the caller checked).</param>
        /// <param name="aux">The accumulator array.</param>
        internal static void ExecuteInputs(TextMethodKey key, NDArray[] inputs, void* aux)
        {
            var dims = BroadcastDims(inputs);
            if (Size(dims) == 0)
                return;
            var kernel = ILKernelGenerator.GetTextMethodKernel(key) ?? ManagedKernel(key, masked: false);
            if (!ForceIterator && TryTrivialInputs(kernel, inputs, dims, aux))
                return;
            bool anyVString = false;
            foreach (var op in inputs)
                anyVString |= op.Storage.IsVString;
            var flags = NDIterGlobalFlags.EXTERNAL_LOOP | (anyVString ? NDIterGlobalFlags.REFS_OK : 0);
            var opFlags = new NDIterPerOpFlags[inputs.Length];
            for (int i = 0; i < opFlags.Length; i++)
                opFlags[i] = NDIterPerOpFlags.READONLY;
            using var iter = NDIterRef.AdvancedNew(inputs.Length, inputs, flags, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                opFlags, opDescrs: new DType[inputs.Length]);
            if (!ForceIterator && iter.CanRunRows)
            {
                var rows = ILKernelGenerator.GetTextMethodRowsKernel(key);
                if (rows is not null)
                {
                    iter.ForEachRows(rows, aux);
                    RowsRuns++;
                    return;
                }
            }
            iter.ForEach(kernel, aux);
        }

        /// <summary>The per-operand flags of an iteration: the inputs read-only, the outputs <paramref name="outputFlags"/>.</summary>
        /// <param name="nop">The operand count.</param>
        /// <param name="nin">The input count.</param>
        /// <param name="outputFlags">The outputs' flags.</param>
        /// <returns>The flags array.</returns>
        private static NDIterPerOpFlags[] OperandFlags(int nop, int nin, NDIterPerOpFlags outputFlags)
        {
            var flags = new NDIterPerOpFlags[nop];
            for (int i = 0; i < nop; i++)
                flags[i] = i < nin ? NDIterPerOpFlags.READONLY : outputFlags;
            return flags;
        }

        // =============================================================================================================
        // NumPy's trivial loop
        // =============================================================================================================

        /// <summary>
        ///     NumPy's trivial loop over inputs and outputs: one kernel call when every input is one element or walks the
        ///     outputs' linear order and no caller output may share memory with an input.
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="operands">The inputs, then the outputs (all outputs of one shape and layout).</param>
        /// <param name="nin">The input count.</param>
        /// <param name="aux">The auxdata.</param>
        /// <param name="callerOut">True when an output is the caller's.</param>
        /// <returns>True when the call ran here.</returns>
        private static bool TryTrivial(NDInnerLoopFunc kernel, NDArray[] operands, int nin, void* aux, bool callerOut)
        {
            int nop = operands.Length;
            var result = operands[nin];
            var rs = result.Shape;
            void** ptrs = stackalloc void*[nop];
            long* strides = stackalloc long[nop];
            if (rs.NDim <= 1)
            {
                for (int i = 0; i < nop; i++)
                {
                    if (!Linear1D(operands[i], rs))
                        return false;
                    strides[i] = Stride1D(operands[i]);
                }
            }
            else
            {
                bool c = rs.IsContiguous, f = rs.IsFContiguous;
                if (!c && !f)
                    return false;
                for (int i = 0; i < nop; i++)
                {
                    if (!LinearOperand(operands[i], rs, ref c, ref f))
                        return false;
                    // An output is never one element of a larger result (it never broadcasts).
                    if (i >= nin && operands[i].size != result.size)
                        return false;
                }
                if (!(c || f))
                    return false;
                for (int i = 0; i < nop; i++)
                    strides[i] = operands[i].size == 1 ? 0 : NDIterRef.ElementBytes(operands[i]);
            }
            if (callerOut)
                for (int o = nin; o < nop; o++)
                    for (int i = 0; i < nin; i++)
                        if (NDMemOverlap.SolveMayShareMemory(operands[o], operands[i], maxWork: 0) != MemOverlap.No)
                            return false;

            for (int i = 0; i < nop; i++)
                ptrs[i] = ElementPointer(operands[i]);
            kernel(ptrs, strides, result.size, aux);
            for (int i = 0; i < nop; i++)
                GC.KeepAlive(operands[i]);
            return true;
        }

        /// <summary>
        ///     The trivial loop of a size pass (no output): every input one element or of the broadcast dimensions in one
        ///     shared contiguous order (any 1-D input at its own stride for a 0-d / 1-D broadcast).
        /// </summary>
        /// <param name="kernel">The size kernel.</param>
        /// <param name="inputs">The inputs.</param>
        /// <param name="dims">Their broadcast dimensions.</param>
        /// <param name="aux">The accumulators.</param>
        /// <returns>True when the pass ran here.</returns>
        private static bool TryTrivialInputs(NDInnerLoopFunc kernel, NDArray[] inputs, long[] dims, void* aux)
        {
            int n = inputs.Length;
            void** ptrs = stackalloc void*[n];
            long* strides = stackalloc long[n];
            long size = Size(dims);
            if (dims.Length <= 1)
            {
                for (int i = 0; i < n; i++)
                {
                    var x = inputs[i];
                    if (!(x.size == 1 || x.ndim == 1 && dims.Length == 1 && x.Shape.dimensions[0] == dims[0]))
                        return false;
                    strides[i] = Stride1D(inputs[i]);
                }
            }
            else
            {
                bool c = true, f = true;
                for (int i = 0; i < n; i++)
                {
                    var x = inputs[i];
                    if (x.size == 1)
                        continue;
                    if (!SameDimensions(x.Shape.dimensions, dims))
                        return false;
                    c &= x.Shape.IsContiguous;
                    f &= x.Shape.IsFContiguous;
                }
                if (!(c || f))
                    return false;
                for (int i = 0; i < n; i++)
                    strides[i] = inputs[i].size == 1 ? 0 : NDIterRef.ElementBytes(inputs[i]);
            }
            for (int i = 0; i < n; i++)
                ptrs[i] = ElementPointer(inputs[i]);
            kernel(ptrs, strides, size, aux);
            for (int i = 0; i < n; i++)
                GC.KeepAlive(inputs[i]);
            return true;
        }

        /// <summary>
        ///     Whether an operand walks linearly next to a multi-dimensional output: one element always, otherwise the
        ///     output's dimensions in an order every operand so far shares.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="rs">The output's shape.</param>
        /// <param name="c">In / out: C order still shared.</param>
        /// <param name="f">In / out: F order still shared.</param>
        /// <returns>False when the operand needs the iterator.</returns>
        private static bool LinearOperand(NDArray x, Shape rs, ref bool c, ref bool f)
        {
            if (x.size == 1)
                return true;
            var xs = x.Shape;
            if (!SameDimensions(xs.dimensions, rs.dimensions))
                return false;
            c &= xs.IsContiguous;
            f &= xs.IsFContiguous;
            return c || f;
        }

        /// <summary>Whether an operand walks linearly next to a 0-d / 1-D output (one element, or 1-D of its length).</summary>
        /// <param name="x">The operand.</param>
        /// <param name="rs">The output's shape.</param>
        /// <returns>False when the operand needs the iterator.</returns>
        private static bool Linear1D(NDArray x, Shape rs)
            => x.size == 1 || x.ndim == 1 && rs.NDim == 1 && x.Shape.dimensions[0] == rs.dimensions[0];

        /// <summary>The byte stride a 1-D linear operand walks with: 0 for one element, else its own (negative when reversed).</summary>
        /// <param name="x">The operand.</param>
        /// <returns>The byte stride.</returns>
        private static long Stride1D(NDArray x) => x.size == 1 || x.ndim == 0 ? 0 : x.Shape.strides[0] * (long)NDIterRef.ElementBytes(x);

        /// <summary>Whether two dimension lists are equal (null reads as 0-d).</summary>
        /// <param name="a">The first list.</param>
        /// <param name="b">The second list.</param>
        /// <returns>True when equal.</returns>
        private static bool SameDimensions(long[] a, long[] b)
        {
            a ??= Array.Empty<long>();
            b ??= Array.Empty<long>();
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>The address of an array's first element (<c>Address + offset × itemsize</c>, never the storage base).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The element pointer.</returns>
        private static byte* ElementPointer(NDArray a) => (byte*)a.Address + a.Shape.offset * (long)NDIterRef.ElementBytes(a);

        // =============================================================================================================
        // The managed kernels (hosts without dynamic code; the IL kernels are tested against them)
        // =============================================================================================================

        /// <summary>
        ///     A <see cref="TextMethodOp"/> kernel as a managed loop over the same <see cref="NDTextElements"/> bodies, with
        ///     run-time widths: the route of hosts without dynamic code, and the reference the IL kernels are compared with.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="masked">True for the masked form (a numeric-output operation; the mask is the operand after the
        ///     outputs).</param>
        /// <returns>The kernel.</returns>
        internal static NDInnerLoopFunc ManagedKernel(TextMethodKey key, bool masked)
            => key.Op >= TextMethodOp.VLength ? ManagedVKernel(key)
                : key.Unit == 4 ? ManagedKernel<uint>(key, masked) : ManagedKernel<byte>(key, masked);

        /// <summary>
        ///     A <c>StringDType</c> kernel as a managed loop over the <see cref="NDVStringMethods"/> bodies (the same calls the
        ///     IL shell's plan makes; the auxdata is the call's <see cref="VStringMethodAux"/>). No masked form: a
        ///     <c>StringDType</c> call with a mask takes NDIter's per-run calls.
        /// </summary>
        /// <param name="key">The kernel identity (a V* operation).</param>
        /// <returns>The kernel.</returns>
        private static NDInnerLoopFunc ManagedVKernel(TextMethodKey key)
            => (dataptrs, strides, count, auxdata) =>
            {
                int nop = OperandCount(key.Op);
                byte** p = stackalloc byte*[nop];
                for (int k = 0; k < nop; k++)
                    p[k] = (byte*)dataptrs[k];
                var aux = (VStringMethodAux*)auxdata;
                for (long i = 0; i < count; i++)
                {
                    ManagedVElement(key, p, aux);
                    for (int k = 0; k < nop; k++)
                        p[k] += strides[k];
                }
            };

        /// <summary>One element of <see cref="ManagedVKernel"/>: the body the IL shell's <c>StringDType</c> plan calls.</summary>
        /// <param name="key">The kernel identity.</param>
        /// <param name="p">The operand element pointers.</param>
        /// <param name="aux">The call's state.</param>
        /// <exception cref="ArgumentOutOfRangeException">Not a <c>StringDType</c> operation.</exception>
        private static void ManagedVElement(TextMethodKey key, byte** p, VStringMethodAux* aux)
        {
            switch (key.Op)
            {
                case TextMethodOp.VLength:
                    *(long*)p[1] = NDVStringMethods.VLength(p[0], aux);
                    break;
                case TextMethodOp.VFind:
                case TextMethodOp.VRFind:
                case TextMethodOp.VCount:
                    int mode = key.Op == TextMethodOp.VFind ? 0 : key.Op == TextMethodOp.VRFind ? 1 : 2;
                    *(long*)p[4] = NDVStringMethods.VFindLike(p[0], p[1], *(long*)p[2], *(long*)p[3], mode, key.Extra, aux);
                    break;
                case TextMethodOp.VStartsWith:
                case TextMethodOp.VEndsWith:
                    *p[4] = NDVStringMethods.VTailMatch(p[0], p[1], *(long*)p[2], *(long*)p[3], key.Op == TextMethodOp.VEndsWith ? 1 : 0, aux);
                    break;
                case TextMethodOp.VStripWhitespace:
                    NDVStringMethods.VStripWhitespace(p[0], p[1], key.Extra, aux);
                    break;
                case TextMethodOp.VStripChars:
                    NDVStringMethods.VStripChars(p[0], p[1], p[2], key.Extra, aux);
                    break;
                case TextMethodOp.VSlice:
                    NDVStringMethods.VSlice(p[0], *(long*)p[1], *(long*)p[2], *(long*)p[3], p[4], aux);
                    break;
                case TextMethodOp.VPad:
                    NDVStringMethods.VPad(p[0], *(long*)p[1], p[2], p[3], key.Extra, aux);
                    break;
                case TextMethodOp.VZFill:
                    NDVStringMethods.VZFill(p[0], *(long*)p[1], p[2], aux);
                    break;
                case TextMethodOp.VReplace:
                    NDVStringMethods.VReplace(p[0], p[1], p[2], *(long*)p[3], p[4], aux);
                    break;
                case TextMethodOp.VExpandTabs:
                    NDVStringMethods.VExpandTabs(p[0], *(long*)p[1], p[2], aux);
                    break;
                case TextMethodOp.VPartition:
                    NDVStringMethods.VPartition(p[0], p[1], p[2], p[3], p[4], key.Extra, aux);
                    break;
                case TextMethodOp.VIsKind:
                    *p[1] = NDVStringMethods.VIsKind(p[0], key.Extra, aux);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(key), key.Op, "Not a StringDType text method operation.");
            }
        }

        /// <summary>The managed kernel for one unit type (see <see cref="ManagedKernel(TextMethodKey, bool)"/>).</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="key">The kernel identity.</param>
        /// <param name="masked">True for the masked form.</param>
        /// <returns>The kernel.</returns>
        private static NDInnerLoopFunc ManagedKernel<TUnit>(TextMethodKey key, bool masked) where TUnit : unmanaged, IEquatable<TUnit>
            => (dataptrs, strides, count, auxdata) =>
            {
                int nop = OperandCount(key.Op) + (masked ? 1 : 0);
                byte** p = stackalloc byte*[nop];
                for (int k = 0; k < nop; k++)
                    p[k] = (byte*)dataptrs[k];
                long* acc = (long*)auxdata;
                for (long i = 0; i < count; i++)
                {
                    ManagedElement<TUnit>(key, p, acc, masked);
                    for (int k = 0; k < nop; k++)
                        p[k] += strides[k];
                }
            };

        /// <summary>The number of kernel operands of an operation (its inputs and outputs).</summary>
        /// <param name="op">The operation.</param>
        /// <returns>The operand count.</returns>
        internal static int OperandCount(TextMethodOp op) => op switch
        {
            TextMethodOp.Length or TextMethodOp.StripWhitespace or TextMethodOp.PadSize or TextMethodOp.ExpandTabsSize
                or TextMethodOp.PartitionSize or TextMethodOp.IsKind or TextMethodOp.VLength or TextMethodOp.VStripWhitespace
                or TextMethodOp.VIsKind => 2,
            TextMethodOp.StripChars or TextMethodOp.ZFill or TextMethodOp.ExpandTabs
                or TextMethodOp.VStripChars or TextMethodOp.VZFill or TextMethodOp.VExpandTabs => 3,
            TextMethodOp.Pad or TextMethodOp.ReplaceSize or TextMethodOp.VPad => 4,
            _ => 5,
        };

        /// <summary>One element of <see cref="ManagedKernel{TUnit}"/>: the body the IL shell's plan calls, with run-time widths.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="key">The kernel identity.</param>
        /// <param name="p">The operand element pointers.</param>
        /// <param name="acc">The auxdata as accumulators.</param>
        /// <param name="masked">True for the masked form (the mask pointer follows the operation's operands).</param>
        private static void ManagedElement<TUnit>(TextMethodKey key, byte** p, long* acc, bool masked) where TUnit : unmanaged, IEquatable<TUnit>
        {
            switch (key.Op)
            {
                case TextMethodOp.Length:
                    StoreLong(p[1], NDTextElements.Units<TUnit>(p[0], key.W0), masked ? p[2] : null);
                    break;
                case TextMethodOp.Find:
                    StoreLong(p[4], NDTextElements.Find<TUnit>(p[0], key.W0, p[1], key.W1, *(long*)p[2], *(long*)p[3]), masked ? p[5] : null);
                    break;
                case TextMethodOp.RFind:
                    StoreLong(p[4], NDTextElements.RFind<TUnit>(p[0], key.W0, p[1], key.W1, *(long*)p[2], *(long*)p[3]), masked ? p[5] : null);
                    break;
                case TextMethodOp.Count:
                    StoreLong(p[4], NDTextElements.Count<TUnit>(p[0], key.W0, p[1], key.W1, *(long*)p[2], *(long*)p[3]), masked ? p[5] : null);
                    break;
                case TextMethodOp.StartsWith:
                case TextMethodOp.EndsWith:
                {
                    byte r = NDTextElements.TailMatch<TUnit>(p[0], key.W0, p[1], key.W1, *(long*)p[2], *(long*)p[3], key.Op == TextMethodOp.EndsWith);
                    if (!masked || *p[5] != 0)
                        *p[4] = r;
                    break;
                }
                case TextMethodOp.StripWhitespace:
                    NDTextElements.StripWhitespace<TUnit>(p[0], key.W0, p[1], key.Extra);
                    break;
                case TextMethodOp.StripChars:
                    NDTextElements.StripChars<TUnit>(p[0], key.W0, p[1], key.W1, p[2], key.Extra);
                    break;
                case TextMethodOp.Slice:
                    NDTextElements.Slice<TUnit>(p[0], key.W0, *(long*)p[1], *(long*)p[2], *(long*)p[3], p[4]);
                    break;
                case TextMethodOp.PadSize:
                    NDTextElements.PadSize<TUnit>(p[0], key.W0, *(long*)p[1], acc);
                    break;
                case TextMethodOp.Pad:
                    NDTextElements.Pad<TUnit>(p[0], key.W0, *(long*)p[1], p[2], p[3], key.W2, key.Extra);
                    break;
                case TextMethodOp.ZFill:
                    NDTextElements.ZFill<TUnit>(p[0], key.W0, *(long*)p[1], p[2], key.W1);
                    break;
                case TextMethodOp.ReplaceSize:
                    NDTextElements.ReplaceSize<TUnit>(p[0], key.W0, p[1], key.W1, p[2], key.W2, *(long*)p[3], acc);
                    break;
                case TextMethodOp.Replace:
                    NDTextElements.ReplaceCounted<TUnit>(p[0], key.W0, p[1], key.W1, p[2], key.W2, *(long*)p[3], p[4], key.W3);
                    break;
                case TextMethodOp.ExpandTabsSize:
                    NDTextElements.ExpandTabsSize<TUnit>(p[0], key.W0, *(long*)p[1], acc);
                    break;
                case TextMethodOp.ExpandTabs:
                    NDTextElements.ExpandTabs<TUnit>(p[0], key.W0, *(long*)p[1], p[2], key.W1);
                    break;
                case TextMethodOp.PartitionSize:
                    NDTextElements.PartitionSize<TUnit>(p[0], key.W0, p[1], key.W1, key.Extra != 0, acc);
                    break;
                case TextMethodOp.Partition:
                    NDTextElements.Partition<TUnit>(p[0], key.W0, p[1], key.W1, key.Extra != 0, p[2], key.W2, p[3], key.W3, p[4], key.W4);
                    break;
                case TextMethodOp.IsKind:
                {
                    // The auxdata is the call's KindState (escalation word and flags table), as in the IL shell.
                    bool r = NDTextElements.IsKind<TUnit>(p[0], key.W0, key.Extra, (NDTextElements.KindState*)acc);
                    if (!masked || *p[2] != 0)
                        *p[1] = r ? (byte)1 : (byte)0;
                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(key), key.Op, "Not a text method operation.");
            }
        }

        /// <summary>Stores an int64 result, or under a mask byte only where the mask is set.</summary>
        /// <param name="output">The output element.</param>
        /// <param name="value">The result.</param>
        /// <param name="mask">The mask element, or null.</param>
        private static void StoreLong(byte* output, long value, byte* mask)
        {
            if (mask is null || *mask != 0)
                *(long*)output = value;
        }
    }
}
