using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The string loops that WRITE text (strings phase 2, P2.3): NumPy's <c>string_add</c> (<c>np.add</c> over two
    ///     strings of one class), <c>string_multiply</c> (<c>np.multiply</c> of a string and an integer, into an <c>out=</c>)
    ///     and <c>str_len</c> (<c>np.strings.str_len</c>), run by NDIter or NumPy's trivial loop over the IL text kernels of
    ///     <c>ILKernelGenerator.Text.Concat.cs</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Where it sits.</b> <see cref="NDStringUfuncs.BinaryGuard"/> runs every check NumPy makes before the loop —
    ///     <c>out=</c> / <c>where=</c> parsing, the operand classes, the result descriptor (<c>width(a) + width(b)</c>, or
    ///     <c>out=</c>'s for the repeat), the output cast into <c>out=</c> (same_kind) and the broadcast — and hands a call
    ///     that computes to <see cref="Add"/> / <see cref="Multiply"/>. Nothing here re-checks those.
    ///     </para>
    ///     <para>
    ///     <b>The loop's output and <c>out=</c>.</b> NumPy's loop always writes the LOOP's descriptor: the concatenation's
    ///     <c>width(a) + width(b)</c> in the inputs' class, the repeat's <c>out=</c> descriptor read in the text's class. An
    ///     <c>out=</c> of exactly that descriptor is written in place; any other (narrower, wider, or <c>U</c> for an
    ///     <c>S</c> loop) receives the loop's result through NumPy's output cast — computed here into a temporary of the loop
    ///     descriptor and copied in by <see cref="np.copyto(NDArray, NDArray, string, NDArray)"/> under the mask, which
    ///     truncates, zero-pads, or ASCII-decodes exactly as the cast does (and raises its UnicodeDecodeError for a byte
    ///     ≥ 0x80 only in an element the mask selects).
    ///     </para>
    ///     <para>
    ///     <b>The result.</b> A fresh result is allocated in NumPy's ufunc output layout (<see cref="Shape.NpyIterOutputShape"/>
    ///     over the inputs and the mask: an F-ordered pair gives an F result). A provided <c>out=</c> is returned itself.
    ///     A <c>where=</c> mask leaves the masked-off elements of <c>out=</c> untouched; those of a fresh result are zero (a
    ///     <c>U</c> allocation is zero-initialized in NumPy too, <c>NPY_NEEDS_INIT</c>).
    ///     </para>
    ///     <para>
    ///     <b>Routes.</b> As the comparisons: NumPy's trivial loop (one kernel call) when every operand walks one linear
    ///     order and there is no mask and no possible overlap with <c>out=</c>; otherwise an NDIter pass with
    ///     <c>EXTERNAL_LOOP</c> and <c>COPY_IF_OVERLAP</c>.
    ///     </para>
    /// </remarks>
    internal static class NDStringArithmetic
    {
        /// <summary>NumPy's OverflowError text of the repeat loop (<c>string_multiply_*_loop</c>).</summary>
        internal const string RepeatOverflowMessage = "Overflow detected in string multiply";

        /// <summary>The per-operand flags of an unmasked binary call: the inputs read, the output written, never broadcast.</summary>
        private static readonly NDIterPerOpFlags[] s_flags =
        {
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
        };

        /// <summary>
        ///     The per-operand flags of a masked binary call: the output READWRITE and WRITEMASKED by the trailing ARRAYMASK
        ///     operand (NumPy's masked loop; <see cref="NDIterRef.ForEach"/> runs one kernel call per mask run).
        /// </summary>
        private static readonly NDIterPerOpFlags[] s_maskedFlags =
        {
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEMASKED | NDIterPerOpFlags.NO_BROADCAST |
            NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK,
        };

        /// <summary>The per-operand flags of the unary length pass: the text read, the int64 output written.</summary>
        private static readonly NDIterPerOpFlags[] s_unaryFlags =
        {
            NDIterPerOpFlags.READONLY,
            NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST,
        };

        // =============================================================================================================
        // add
        // =============================================================================================================

        /// <summary>
        ///     Computes <c>np.add(x1, x2, out=out, where=where)</c> over two string arrays of one class (NumPy's
        ///     <c>string_add</c>): each element is <c>x1</c>'s text then <c>x2</c>'s, at width <c>width(x1) + width(x2)</c>.
        /// </summary>
        /// <param name="x1">The first operand (<c>S</c> or <c>U</c>).</param>
        /// <param name="x2">The second operand, of the same class.</param>
        /// <param name="out">The <c>out=</c> operand, or null (its cast and shape were checked by the guard).</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <returns><paramref name="out"/> itself, or a fresh array of the concatenation's descriptor.</returns>
        /// <exception cref="ArgumentException">The operands are not two strings of one class (a caller bug).</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> concatenation cast into a <c>U</c> <paramref name="out"/>
        ///     meets a byte ≥ 0x80 in a selected element (NumPy's ASCII decode).</exception>
        internal static unsafe NDArray Add(NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            if (!NDStringUfuncs.ComparesText(x1.dtype, x2.dtype))
                throw new ArgumentException($"'add' concatenates two strings of one class, not {x1.dtype} and {x2.dtype}.");
            var loop = NDStringUfuncs.Concatenation(x1.dtype, x2.dtype);
            if (@out is not null && @out.dtype != loop)
                return CastInto(@out, where, () => Add(x1, x2, null, null));

            int unit = x1.dtype.kind == 'U' ? 4 : 1;
            int w1 = x1.dtype.itemsize, w2 = x2.dtype.itemsize;
            var kernel = ILKernelGenerator.GetTextConcatKernel(unit, w1, w2) ?? ManagedConcatKernel(unit, w1, w2);
            return RunBinary(kernel, x1, x2, @out, where, loop, null);
        }

        // =============================================================================================================
        // multiply
        // =============================================================================================================

        /// <summary>
        ///     Computes <c>np.multiply(x1, x2, out=out, where=where)</c> for a string and an integer operand (NumPy's
        ///     <c>string_multiply</c>): the text repeated, cut at <c>out=</c>'s width.
        /// </summary>
        /// <param name="x1">The first operand (the text, or the integer count).</param>
        /// <param name="x2">The second operand (the other one).</param>
        /// <param name="out">The required <c>out=</c> operand (its cast and shape were checked by the guard).</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ArgumentException">The operands are not one string and one integer array, or there is no
        ///     <paramref name="out"/> (caller bugs).</exception>
        /// <exception cref="OverflowException">An element's <c>reps * len</c> exceeds 2^63 - 1 (NumPy's OverflowError,
        ///     raised after the loop; that element is left untouched, every other one written).</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> repeat cast into a <c>U</c> <paramref name="out"/> meets a
        ///     byte ≥ 0x80 in a selected element.</exception>
        internal static unsafe NDArray Multiply(NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            bool textFirst = NDStringUfuncs.IsString(x1);
            var text = textFirst ? x1 : x2;
            var count = textFirst ? x2 : x1;
            if (@out is null || !NDStringUfuncs.IsString(text) || !count.typecode.IsInteger())
                throw new ArgumentException("A string repeat multiplies one string array by one integer array into out=.");
            var meta = text.dtype.Meta;
            // NumPy's loop writes out='s descriptor read in the text's class (an S loop into a U out computes S{chars}).
            var loop = @out.dtype.Meta == meta ? @out.dtype : DTypePromotion.CastDescrToDType(@out.dtype, meta);
            if (@out.dtype != loop)
                return CastInto(@out, where, () =>
                {
                    // The temporary has the inputs' broadcast shape (copyto broadcasts it into out=).
                    var dims = NDStringUfuncs.BroadcastDims(x1, x2, null, null);
                    var temp = new NDArray(loop, dims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(dims), true);
                    try
                    {
                        return Multiply(x1, x2, temp, null);
                    }
                    catch
                    {
                        temp.Dispose();
                        throw;
                    }
                });

            // NumPy's promoter casts the integer operand to int64 (a uint64 count past 2^63 wraps to a negative count, which
            // repeats nothing).
            var reps = count.typecode == NPTypeCode.Int64 ? count : count.astype(np.int64);
            try
            {
                int unit = text.dtype.kind == 'U' ? 4 : 1;
                int textIndex = textFirst ? 0 : 1;
                var kernel = ILKernelGenerator.GetTextRepeatKernel(unit, text.dtype.itemsize, @out.dtype.itemsize, textIndex)
                             ?? ManagedRepeatKernel(unit, text.dtype.itemsize, @out.dtype.itemsize, textIndex);
                int overflow = 0;
                // RunBinary returns @out itself (a provided output is never replaced), so the result is the caller's array.
                var result = textFirst
                    ? RunBinary(kernel, text, reps, @out, where, loop, &overflow)
                    : RunBinary(kernel, reps, text, @out, where, loop, &overflow);
                // NumPy raises after the whole loop: every other element is written, an overflowing one is left as it was.
                if (overflow != 0)
                    throw new OverflowException(RepeatOverflowMessage);
                return result;
            }
            finally
            {
                if (!ReferenceEquals(reps, count))
                    reps.Dispose();
            }
        }

        // =============================================================================================================
        // str_len
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>str_len</c> loop: the length of every element (the index of its last nonzero code unit plus one —
        ///     trailing NULs never count, embedded ones do) as int64, in NumPy's ufunc output layout.
        /// </summary>
        /// <param name="text">A string array (<c>S</c> or <c>U</c>, any layout, 0-d included).</param>
        /// <returns>A fresh int64 array of <paramref name="text"/>'s dimensions.</returns>
        /// <exception cref="ArgumentException"><paramref name="text"/> is not a string array (a caller bug).</exception>
        internal static unsafe NDArray Length(NDArray text)
        {
            if (!NDStringUfuncs.IsString(text))
                throw new ArgumentException($"str_len reads a string array, not {text?.dtype}.", nameof(text));
            int unit = text.dtype.kind == 'U' ? 4 : 1;
            int width = text.dtype.itemsize;
            var kernel = ILKernelGenerator.GetTextLengthKernel(unit, width) ?? ManagedLengthKernel(unit, width);
            var ts = text.Shape;
            var shape = ts.NDim == 0 ? Shape.NewScalar()
                : Shape.NpyIterOutputShape((long[])ts.dimensions.Clone(), new[] { ts });
            var result = new NDArray(NPTypeCode.Int64, shape, false);
            try
            {
                if (result.size == 0)
                    return result;
                var rs = result.Shape;
                // Linear walk: both C-contiguous or both F-contiguous (a 0-d / one-element pair is both).
                if (ts.IsContiguous && rs.IsContiguous || ts.IsFContiguous && rs.IsFContiguous)
                {
                    void** ptrs = stackalloc void*[2];
                    long* strides = stackalloc long[2];
                    ptrs[0] = ElementPointer(text);
                    ptrs[1] = ElementPointer(result);
                    strides[0] = width;
                    strides[1] = sizeof(long);
                    kernel(ptrs, strides, result.size, null);
                    GC.KeepAlive(text);
                    GC.KeepAlive(result);
                }
                else
                {
                    using var iter = NDIterRef.AdvancedNew(2, new[] { text, result }, NDIterGlobalFlags.EXTERNAL_LOOP,
                        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, s_unaryFlags, opDescrs: new DType[2]);
                    iter.ForEach(kernel);
                }
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        // =============================================================================================================
        // The shared driver
        // =============================================================================================================

        /// <summary>
        ///     Runs a binary text kernel over <c>(x1, x2) → result</c>: allocates a fresh result of <paramref name="loop"/> in
        ///     NumPy's output layout when <paramref name="out"/> is null, then takes NumPy's trivial loop or an NDIter pass.
        /// </summary>
        /// <param name="kernel">The kernel (operand slots 0, 1 = the inputs, 2 = the output).</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="out">The output to write (of descriptor <paramref name="loop"/>), or null for a fresh one.</param>
        /// <param name="where">The mask, or null.</param>
        /// <param name="loop">The loop's output descriptor.</param>
        /// <param name="auxdata">The kernel's auxdata (the repeat's overflow flag), or null.</param>
        /// <returns><paramref name="out"/>, or the fresh result.</returns>
        private static unsafe NDArray RunBinary(NDInnerLoopFunc kernel, NDArray x1, NDArray x2, NDArray @out, NDArray where,
                                                DType loop, int* auxdata)
        {
            var dims = NDStringUfuncs.BroadcastDims(x1, x2, @out, where);
            NDArray result = @out;
            if (result is null)
            {
                var shape = ResultShape(dims, x1, x2, where);
                // Every element of an unmasked result is written by the kernel; a masked one keeps NumPy's zero-initialized
                // U (and S) elements where the mask is false.
                result = where is null ? NDArray.ForOverwrite(loop, shape) : new NDArray(loop, shape, true);
            }
            try
            {
                if (result.size == 0)
                    return result;
                if (!TryTrivial(kernel, x1, x2, result, where, @out is not null, auxdata))
                    RunIterator(kernel, x1, x2, result, where, auxdata);
                return result;
            }
            catch
            {
                // A fresh result never escapes a failed call; a caller's out= stays the caller's.
                if (@out is null)
                    result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's output cast for a loop whose descriptor is not <paramref name="out"/>'s: the loop's result (computed by
        ///     <paramref name="compute"/> into a fresh temporary) copied into <paramref name="out"/> under the mask with the
        ///     same_kind rule the guard already checked — truncation, zero padding, or the ASCII decode of S into U.
        /// </summary>
        /// <param name="out">The caller's output.</param>
        /// <param name="where">The mask, or null.</param>
        /// <param name="compute">Computes the unmasked loop result as a fresh array (owned here).</param>
        /// <returns><paramref name="out"/>.</returns>
        private static NDArray CastInto(NDArray @out, NDArray where, Func<NDArray> compute)
        {
            using var temp = compute();
            np.copyto(@out, temp, "same_kind", where);
            return @out;
        }

        /// <summary>
        ///     The layout of a fresh result: NumPy's NpyIter output allocation over the two inputs and the mask (each votes
        ///     with the axes it really strides; C order wins ties).
        /// </summary>
        /// <param name="dims">The broadcast dimensions (adopted by the returned shape — a fresh array).</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>A dense, offset-0 shape of <paramref name="dims"/>.</returns>
        private static Shape ResultShape(long[] dims, NDArray x1, NDArray x2, NDArray where)
        {
            if (dims.Length == 0)
                return Shape.NewScalar();
            if (where is null)
                return Shape.NpyIterOutputShape(dims, new[] { x1.Shape, x2.Shape });
            return Shape.NpyIterOutputShape(dims, new[] { x1.Shape, x2.Shape, where.Shape });
        }

        /// <summary>
        ///     NumPy's trivial loop: one kernel call covers the whole result when there is no mask and each input is one
        ///     element (stride 0) or walks the result's linear order — the result's dimensions in a shared contiguous order,
        ///     or, for a 0-d / 1-D result, any 1-D input at its own byte stride (<c>PyArray_TRIVIALLY_ITERABLE</c>).
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="result">The output (fresh or <c>out=</c>).</param>
        /// <param name="where">The mask, or null.</param>
        /// <param name="callerOut">True when <paramref name="result"/> is the caller's <c>out=</c>, which may overlap an
        ///     input (such a call takes the iterator, whose COPY_IF_OVERLAP makes it safe).</param>
        /// <param name="auxdata">The kernel's auxdata, or null.</param>
        /// <returns>True when the call was computed here.</returns>
        private static unsafe bool TryTrivial(NDInnerLoopFunc kernel, NDArray x1, NDArray x2, NDArray result, NDArray where,
                                              bool callerOut, int* auxdata)
        {
            if (where is not null)
                return false;
            var rs = result.Shape;
            long outBytes = NDIterRef.ElementBytes(result);
            void** ptrs = stackalloc void*[3];
            long* strides = stackalloc long[3];
            if (rs.NDim <= 1)
            {
                if (!Linear1D(x1, rs) || !Linear1D(x2, rs))
                    return false;
                strides[0] = Stride1D(x1);
                strides[1] = Stride1D(x2);
                strides[2] = rs.NDim == 0 || result.size == 1 ? 0 : rs.strides[0] * outBytes;
            }
            else
            {
                bool c = rs.IsContiguous, f = rs.IsFContiguous;
                if (!c && !f)
                    return false;
                if (!LinearOperand(x1, rs, ref c, ref f) || !LinearOperand(x2, rs, ref c, ref f) || !(c || f))
                    return false;
                strides[0] = x1.size == 1 ? 0 : NDIterRef.ElementBytes(x1);
                strides[1] = x2.size == 1 ? 0 : NDIterRef.ElementBytes(x2);
                strides[2] = outBytes;
            }
            if (callerOut && (MayOverlap(result, x1) || MayOverlap(result, x2)))
                return false;

            ptrs[0] = ElementPointer(x1);
            ptrs[1] = ElementPointer(x2);
            ptrs[2] = ElementPointer(result);
            kernel(ptrs, strides, result.size, auxdata);
            GC.KeepAlive(x1);
            GC.KeepAlive(x2);
            GC.KeepAlive(result);
            return true;
        }

        /// <summary>
        ///     Whether an input can be walked linearly next to the result: one element always (stride 0), otherwise the
        ///     result's dimensions in one of the orders every operand so far shares.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The result's shape.</param>
        /// <param name="c">In: C order still shared; out: and by this input.</param>
        /// <param name="f">In / out: the same for F order.</param>
        /// <returns>False when the input needs the iterator.</returns>
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

        /// <summary>Whether an input can be walked linearly next to a 0-d or 1-D result (one element, or 1-D of its length).</summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The result's shape.</param>
        /// <returns>False when the input needs the iterator.</returns>
        private static bool Linear1D(NDArray x, Shape rs)
            => x.size == 1 || x.ndim == 1 && rs.NDim == 1 && x.Shape.dimensions[0] == rs.dimensions[0];

        /// <summary>The byte stride a <see cref="Linear1D"/> input walks with: 0 for one element, else its own.</summary>
        /// <param name="x">The input.</param>
        /// <returns>The byte stride (negative for a reversed view).</returns>
        private static long Stride1D(NDArray x) => x.size == 1 ? 0 : x.Shape.strides[0] * NDIterRef.ElementBytes(x);

        /// <summary>Whether two dimension lists are equal.</summary>
        /// <param name="a">The first (null for 0-d).</param>
        /// <param name="b">The second (null for 0-d).</param>
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

        /// <summary>Whether two arrays' byte ranges may intersect (NumPy's <c>MAY_SHARE_BOUNDS</c>).</summary>
        /// <param name="a">The output.</param>
        /// <param name="b">An input.</param>
        /// <returns>True when the extents intersect.</returns>
        private static bool MayOverlap(NDArray a, NDArray b) => NDMemOverlap.SolveMayShareMemory(a, b, maxWork: 0) != MemOverlap.No;

        /// <summary>The address of an array's first element.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The element pointer.</returns>
        private static unsafe byte* ElementPointer(NDArray a) => (byte*)a.Address + a.Shape.offset * NDIterRef.ElementBytes(a);

        /// <summary>
        ///     The NDIter pass: <c>EXTERNAL_LOOP</c>, <c>COPY_IF_OVERLAP</c>, every operand with its own descriptor (no cast:
        ///     the output already has the loop's descriptor, the repeat counts are int64).
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="result">The output.</param>
        /// <param name="where">The mask, or null.</param>
        /// <param name="auxdata">The kernel's auxdata, or null.</param>
        private static unsafe void RunIterator(NDInnerLoopFunc kernel, NDArray x1, NDArray x2, NDArray result, NDArray where,
                                               int* auxdata)
        {
            var flags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            var ops = where is null ? new[] { x1, x2, result } : new[] { x1, x2, result, where };
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, flags, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                where is null ? s_flags : s_maskedFlags, opDescrs: new DType[ops.Length]);
            iter.ForEach(kernel, auxdata);
        }

        // =============================================================================================================
        // The managed kernels (hosts without dynamic code; the IL kernels are tested against them)
        // =============================================================================================================

        /// <summary>The concatenation as a managed <see cref="NDInnerLoopFunc"/>: NumPy's <c>string_add</c> per element.</summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's width in bytes.</param>
        /// <param name="widthB">B's width in bytes (the output is <c>widthA + widthB</c>).</param>
        /// <returns>The kernel.</returns>
        internal static unsafe NDInnerLoopFunc ManagedConcatKernel(int unit, int widthA, int widthB)
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* pa = (byte*)dataptrs[0], pb = (byte*)dataptrs[1], po = (byte*)dataptrs[2];
                long sa = strides[0], sb = strides[1], so = strides[2];
                int outWidth = widthA + widthB;
                for (long i = 0; i < count; i++, pa += sa, pb += sb, po += so)
                {
                    int la = ILKernelGenerator.LastNonzeroEnd(pa, widthA, unit);
                    int lb = ILKernelGenerator.LastNonzeroEnd(pb, widthB, unit);
                    Buffer.MemoryCopy(pa, po, la, la);
                    Buffer.MemoryCopy(pb, po + la, lb, lb);
                    new Span<byte>(po + la + lb, outWidth - la - lb).Clear();
                }
            };

        /// <summary>The length as a managed <see cref="NDInnerLoopFunc"/>: <c>str_len</c> per element, int64 units.</summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The element width in bytes.</param>
        /// <returns>The kernel.</returns>
        internal static unsafe NDInnerLoopFunc ManagedLengthKernel(int unit, int width)
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* p = (byte*)dataptrs[0], po = (byte*)dataptrs[1];
                long sp = strides[0], so = strides[1];
                for (long i = 0; i < count; i++, p += sp, po += so)
                    Unsafe.WriteUnaligned(po, (long)(ILKernelGenerator.LastNonzeroEnd(p, width, unit) / unit));
            };

        /// <summary>The repetition as a managed <see cref="NDInnerLoopFunc"/>: <c>string_multiply</c> per element.</summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="width">The text width in bytes.</param>
        /// <param name="outWidth">The output width in bytes.</param>
        /// <param name="textIndex">The text operand's slot (0 or 1; the int64 counts are the other).</param>
        /// <returns>The kernel (it stores 1 into <c>*(int*)auxdata</c> when an element overflows).</returns>
        internal static unsafe NDInnerLoopFunc ManagedRepeatKernel(int unit, int width, int outWidth, int textIndex)
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* pt = (byte*)dataptrs[textIndex], pr = (byte*)dataptrs[1 - textIndex], po = (byte*)dataptrs[2];
                long st = strides[textIndex], sr = strides[1 - textIndex], so = strides[2];
                int overflow = 0;
                for (long i = 0; i < count; i++, pt += st, pr += sr, po += so)
                    overflow |= ILKernelGenerator.RepeatElement(pt, width, unit, Unsafe.ReadUnaligned<long>(pr), po, outWidth);
                if (overflow != 0)
                    *(int*)auxdata = 1;
            };
    }
}
