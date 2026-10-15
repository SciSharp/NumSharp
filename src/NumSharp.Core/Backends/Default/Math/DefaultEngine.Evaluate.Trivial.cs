using System;
using NumSharp.Backends.Iteration;

// =============================================================================
// DefaultEngine.Evaluate.Trivial.cs — the trivial-loop dispatch of np.evaluate's elementwise pass
// (plan docs/plans/ndexpr-evaluate.md, "Perf review 2026-09-23", lever 1)
// =============================================================================
//
// NumPy's ufunc layer never builds an iterator for a call whose operands are "trivially iterable":
// try_trivial_single_output_loop (numpy/_core/src/umath/ufunc_object.c) calls the strided inner loop ONCE
// over the whole operation when every non-0-d operand has exactly the operation's shape and is either 1-D
// (any stride) or contiguous in ONE shared memory order (C or F), 0-d inputs ride a zero stride, and a
// provided output overlaps no input except exactly (PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK).
//
// np.evaluate always built an NDIter for its fused pass. For such calls the iterator's whole job is to
// discover what the host already knows — the operation is one flat loop — and it costs ~130-150 ns per
// call (construction + ForEach + dispose, measured at n = 8), about as much as the result allocation and
// thirty times the fused kernel itself (4 ns for a*b+c at n = 8). This file is NumPy's shortcut: the host
// classifies the operands with O(1) flag reads and calls program.Kernel directly.
//
// Byte-identical to the NDIter pass BY CONSTRUCTION, not by tuning: for operands contiguous in one shared
// order NDIter coalesces the operation to ONE inner loop and calls the SAME kernel with the SAME data
// pointers, count and byte strides — this path makes exactly that call. The accepted 1-D strided / 0-d
// zero-stride shapes are the calls NDIter would chunk differently at most, and the fused kernels are
// vector==scalar by construction (the fuzz corpus and NDEvaluateVectorTests pin it), so any chunking
// yields the same bytes.
//
// Everything outside the accepted shapes returns null and runs the NDIter pass UNCHANGED — including
// every call that must raise (broadcast mismatches, a stretched or read-only out, a bad out cast): the
// gate never throws, it only declines. Declined: where= (masked driver), a dtype= request or a dtype-
// mismatched out (buffered cast operand), any broadcast between non-0-d operands, an N-d operand that is
// not contiguous or disagrees with the shared order, and a provided out whose memory overlaps an input
// anywhere except exactly (NDIter's COPY_IF_OVERLAP copies that input; the trivial loop has no buffer).
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Upper bound on the operand count (iterator inputs + the output) for which the trivial loop
        /// stack-allocates its pointer and stride tables. A wider fan-in simply runs the NDIter pass (which
        /// has no operand limit in NumSharp), so a pathological tree cannot overflow the stack here.
        /// </summary>
        private const int EvaluateTrivialMaxOperands = 64;

        /// <summary>No contiguous memory order has been fixed yet by the operands classified so far.</summary>
        private const int TrivialOrderAny = 0;

        /// <summary>The N-d operands classified so far are row-major (C) contiguous.</summary>
        private const int TrivialOrderC = 1;

        /// <summary>The N-d operands classified so far are column-major (F) contiguous.</summary>
        private const int TrivialOrderF = 2;

        /// <summary>
        /// NumPy's trivial loop for the fused elementwise pass: when the call is trivially iterable (see the
        /// file header), run <see cref="NDExprProgram.Kernel"/> ONCE over the whole operation — no NDIter —
        /// and return the result; otherwise return null so the caller runs the NDIter pass.
        /// </summary>
        /// <remarks>
        /// Must be called after the caller validated the out cast (the gate relies on a matching out dtype
        /// but never re-validates) and only when there is no <c>where=</c> mask. A fresh result gets exactly
        /// the layout the NDIter pass would allocate (C, or F when <paramref name="order"/> resolves to F via
        /// <see cref="ResolveEvalOrder"/> — the strict-F input heuristic included), and the call is declined
        /// when that layout disagrees with the operands' shared order, so a declined call allocates nothing.
        /// A 0-d parameter aliasing the output keeps its original value because
        /// <see cref="NDExprProgram.PackParams"/> reads it before the kernel runs, exactly as on the NDIter
        /// path.
        /// </remarks>
        /// <param name="program">The elementwise program; its <see cref="NDExprProgram.Kernel"/> is non-null.</param>
        /// <param name="inputs">Every input of the call in input order, hoisted parameters included (their elements are packed into the aux block).</param>
        /// <param name="ops">The iterator operands — <paramref name="inputs"/> minus the hoisted parameters (<see cref="NDExprProgram.IteratorOperands"/>).</param>
        /// <param name="out">The caller's output array, or null for a fresh result.</param>
        /// <param name="resolvedType">The tree's NumPy result dtype — the dtype the kernel writes.</param>
        /// <param name="targetType">The dtype the result must carry (an explicit <c>dtype=</c>, else <paramref name="resolvedType"/>); any mismatch declines, because the output then needs a buffered cast.</param>
        /// <param name="order">The API-validated <c>order=</c> char of the call (the default is resolved like <c>'K'</c>).</param>
        /// <returns>The result — <paramref name="out"/> itself or a fresh array — or null when the call is not trivially iterable.</returns>
        private unsafe NDArray TryEvaluateTrivialLoop(NDExprProgram program, NDArray[] inputs, NDArray[] ops,
            NDArray @out, NPTypeCode resolvedType, NPTypeCode targetType, char order)
        {
            if (NDExpr.DisableTrivialLoop)
                return null;

            // The kernel writes resolvedType straight into the output. A dtype= request or a dtype-mismatched
            // out turns the output into a BUFFERED cast operand, which only the iterator provides. A read-only
            // out is left to the NDIter pass so the call raises exactly what it always raised (the kernel would
            // otherwise write through the pointer and bypass the guarded setters).
            if (targetType != resolvedType)
                return null;
            if (@out is not null && (@out.typecode != resolvedType || !@out.Shape.IsWriteable))
                return null;

            int nIn = ops.Length;
            int nOps = nIn + 1;
            if (nOps > EvaluateTrivialMaxOperands)
                return null;

            // The operation shape is the first non-0-d operand's, inputs first and then the out (NumPy's rule):
            // a 0-d input joins by a zero stride, so an all-0-d input list with an N-d out iterates the out.
            Shape opShape = Shape.NewScalar();
            for (int j = 0; j < nIn; j++)
            {
                if (ops[j].Shape.NDim != 0)
                {
                    opShape = ops[j].Shape;
                    break;
                }
            }

            if (opShape.NDim == 0 && @out is not null && @out.Shape.NDim != 0)
                opShape = @out.Shape;

            int opNdim = opShape.NDim;
            long count = opShape.size;
            int sharedOrder = TrivialOrderAny;
            long* byteStrides = stackalloc long[nOps];

            for (int j = 0; j < nIn; j++)
            {
                var s = ops[j].Shape;
                if (s.NDim == 0)
                {
                    // A 0-d iterator operand (a program that hoists nothing — e.g. every input 0-d, or a
                    // positional Compile handle) is read at every element: NumPy's zero fixed stride.
                    byteStrides[j] = 0;
                    continue;
                }

                // Any real broadcast between N-d operands (a stretched axis, a rank difference) is the
                // iterator's job — and its error texts' when the shapes are incompatible.
                if (!SameDims(s, opShape))
                    return null;
                if (!TryClassifyTrivialOperand(s, ops[j].typecode.SizeOf(), ref sharedOrder, out byteStrides[j]))
                    return null;
            }

            int outItemSize = resolvedType.SizeOf();
            NDArray target;
            if (@out is not null)
            {
                var s = @out.Shape;
                if (s.NDim == 0)
                {
                    // A 0-d out receives exactly one element; against an N-d operation the NDIter pass
                    // raises NumPy's "non-broadcastable output operand" text, so decline and let it.
                    if (opNdim != 0)
                        return null;
                    byteStrides[nIn] = 0;
                }
                else
                {
                    // The out is never stretched (NO_BROADCAST); a different-dims out is either a
                    // broadcast of the inputs into it or an error — both the iterator's business.
                    if (!SameDims(s, opShape))
                        return null;
                    if (!TryClassifyTrivialOperand(s, outItemSize, ref sharedOrder, out byteStrides[nIn]))
                        return null;
                }

                // COPY_IF_OVERLAP: the NDIter pass copies any input whose memory the out overlaps other than
                // EXACTLY (same start, same stride, same element size — the in-place `out=a` case, where every
                // element is read before the same element is written). The trivial loop has no copy buffer.
                if (!TrivialOutOverlapIsSafe(ops, byteStrides, @out, byteStrides[nIn], outItemSize, count))
                    return null;

                target = @out;
            }
            else
            {
                // A fresh result gets exactly the layout the NDIter pass allocates (ResolveEvalOrder over the
                // strict-F heuristic), built without a dims/strides clone when the operation's source shape is
                // already canonical. Its classification joins the shared order BEFORE the allocation, so a
                // layout conflict (order='F' over C-contiguous 2-D inputs) declines without allocating.
                bool wantF = ResolveEvalOrder(order, AreAllInputsStrictFContig(ops, opShape));
                Shape targetShape = opNdim == 0 ? Shape.NewScalar() : CanonicalResultShape(opShape, wantF);
                if (opNdim == 0)
                    byteStrides[nIn] = 0;
                else if (!TryClassifyTrivialOperand(targetShape, outItemSize, ref sharedOrder, out byteStrides[nIn]))
                    return null;

                target = new NDArray(targetType, targetShape, false);
            }

            // The call is served by the trivial loop from here on; counted before the empty return so the
            // engagement of an empty call is observable too.
            NDExpr.TrivialLoopRuns++;

            // An empty operation writes nothing — the NDIter pass returns the (allocated or provided) target
            // before touching the kernel, and so does this one.
            if (count == 0)
                return target;

            // The data pointers: each operand's logical element 0 (base address + the view's element offset),
            // the output last — the kernel's [iterator operands…, output] contract, identical to the operand
            // array the NDIter pass hands MultiNew.
            void** dataptrs = stackalloc void*[nOps];
            for (int j = 0; j < nIn; j++)
                dataptrs[j] = (byte*)ops[j].Address + ops[j].Shape.offset * ops[j].typecode.SizeOf();
            dataptrs[nIn] = (byte*)target.Address + target.Shape.offset * outItemSize;

            // Parameters: packed into the aux block BEFORE the pass, like the NDIter path (a parameter that
            // aliases the output must keep its original value). stackalloc lives until the method returns.
            byte* aux = null;
            if (program.ParamCount > 0)
            {
                byte* buf = stackalloc byte[NDExprParamPlan.SlotBytes * program.ParamCount];
                program.PackParams(inputs, buf);
                aux = buf;
            }

            program.Kernel(dataptrs, byteStrides, count, aux);
            return target;
        }

        /// <summary>
        /// Classify one non-0-d operand (whose dims already equal the operation's) for the trivial loop and
        /// produce its byte stride over the flattened operation: a 1-D operand walks its own stride (any sign —
        /// NumPy's trivial loop takes 1-D strides as they are), an N-d operand must be C- or F-contiguous and
        /// agree with the order the operands classified so far fixed (an operand contiguous in BOTH orders —
        /// at most one non-unit axis — fixes none), and then walks one element at a time.
        /// </summary>
        /// <param name="shape">The operand's shape.</param>
        /// <param name="itemSize">The operand's element size in bytes.</param>
        /// <param name="sharedOrder">The order fixed so far (<see cref="TrivialOrderAny"/>/<see cref="TrivialOrderC"/>/<see cref="TrivialOrderF"/>); updated when this operand fixes it.</param>
        /// <param name="byteStride">Receives the operand's byte stride over the flattened operation.</param>
        /// <returns>False when the operand cannot join a single flat loop (non-contiguous or broadcast N-d, or a conflicting order).</returns>
        private static bool TryClassifyTrivialOperand(in Shape shape, int itemSize, ref int sharedOrder, out long byteStride)
        {
            if (shape.NDim == 1)
            {
                // Shape.strides are ELEMENT strides; the kernel contract takes BYTE strides. A 1-D stride-0
                // input (a broadcast vector) is a legal zero stride here exactly as in NumPy — the NDIter pass
                // would hand the kernel the very same stride.
                byteStride = shape.strides[0] * itemSize;
                return true;
            }

            byteStride = itemSize;
            if (shape.IsBroadcasted)
                return false;

            bool c = shape.IsContiguous;
            bool f = shape.IsFContiguous;
            if (!c && !f)
                return false;
            if (c && f)
                return true;

            int mine = c ? TrivialOrderC : TrivialOrderF;
            if (sharedOrder == TrivialOrderAny)
            {
                sharedOrder = mine;
                return true;
            }

            return sharedOrder == mine;
        }

        /// <summary>
        /// Whether a provided output may be written by the trivial loop without the copy NDIter's
        /// COPY_IF_OVERLAP would make: true when no input's byte extent intersects the output's, or every
        /// intersecting input aliases it EXACTLY (same first byte, same byte stride, same element size), so each
        /// output element is written only after the same element of that input was read. Any other overlap —
        /// a shifted self-view, an element-size mismatch, a 0-d input inside the output — returns false and the
        /// NDIter pass copies the input first.
        /// </summary>
        /// <param name="ops">The iterator operands.</param>
        /// <param name="byteStrides">The classified byte strides, inputs first (the output's slot is not read here).</param>
        /// <param name="out">The provided output.</param>
        /// <param name="outByteStride">The output's classified byte stride.</param>
        /// <param name="outItemSize">The output's element size in bytes.</param>
        /// <param name="count">The operation's element count (≥ 0).</param>
        /// <returns>True when running the kernel in place is equivalent to running it on non-overlapping memory.</returns>
        private static unsafe bool TrivialOutOverlapIsSafe(NDArray[] ops, long* byteStrides, NDArray @out,
            long outByteStride, int outItemSize, long count)
        {
            if (count == 0)
                return true;

            byte* outStart = (byte*)@out.Address + @out.Shape.offset * outItemSize;
            TrivialByteExtent(outStart, outByteStride, outItemSize, count, out byte* outLo, out byte* outHi);

            for (int j = 0; j < ops.Length; j++)
            {
                int itemSize = ops[j].typecode.SizeOf();
                byte* start = (byte*)ops[j].Address + ops[j].Shape.offset * itemSize;
                TrivialByteExtent(start, byteStrides[j], itemSize, count, out byte* lo, out byte* hi);
                if (lo >= outHi || outLo >= hi)
                    continue; // disjoint

                if (start != outStart || byteStrides[j] != outByteStride || itemSize != outItemSize)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// The half-open byte range <c>[lo, hi)</c> a flat walk of <paramref name="count"/> elements touches from
        /// <paramref name="start"/> with <paramref name="byteStride"/> (negative strides walk down; a zero stride
        /// touches one element).
        /// </summary>
        /// <param name="start">The first element's address.</param>
        /// <param name="byteStride">The byte step between consecutive elements.</param>
        /// <param name="itemSize">The element size in bytes.</param>
        /// <param name="count">The number of elements walked (≥ 1).</param>
        /// <param name="lo">Receives the lowest byte touched.</param>
        /// <param name="hi">Receives one past the highest byte touched.</param>
        private static unsafe void TrivialByteExtent(byte* start, long byteStride, int itemSize, long count,
            out byte* lo, out byte* hi)
        {
            long span = (count - 1) * byteStride;
            lo = span < 0 ? start + span : start;
            hi = (span > 0 ? start + span : start) + itemSize;
        }
    }
}

namespace NumSharp.Backends.Iteration
{
    public abstract partial class NDExpr
    {
        /// <summary>
        /// Test / diagnostics hook: when set on the current thread, np.evaluate SKIPS the trivial-loop dispatch
        /// (DefaultEngine.Evaluate.Trivial.cs) and runs every elementwise pass through NDIter, as it always did.
        /// It exists so a test can evaluate the same call BOTH ways and prove the results byte-identical — the
        /// whole safety claim of the shortcut. Thread-static so a parallel test never perturbs another's run.
        /// </summary>
        [System.ThreadStatic] internal static bool DisableTrivialLoop;

        /// <summary>
        /// Test / diagnostics hook: incremented (on the current thread) each time an elementwise pass is computed
        /// by the trivial loop. A test resets it, evaluates, and asserts whether it moved — proving the shortcut
        /// ENGAGED where it applies (a silent fallback would still be correct and would hide a dead fast path) and
        /// DECLINED where it must not run (overlap, casts, broadcasts). Thread-static like
        /// <see cref="DisableTrivialLoop"/>.
        /// </summary>
        [System.ThreadStatic] internal static int TrivialLoopRuns;
    }
}
