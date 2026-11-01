using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// Default.Reduction.Add.Axes.cs — NumPy's add.reduce over an axis SET
// =============================================================================
//
// np.sum and np.mean take one axis or every axis in NumSharp, but NumPy's add.reduce takes any SET of axes, and
// NumPy's Python-level functions use that form: np.average's `a.mean(axis=(0, 2))` and
// `wgt.sum(axis=axis_tuple, dtype=result_dtype)`. A tuple is NOT a sequence of single-axis reductions. NumPy reduces
// the whole set with ONE iterator, and that iterator's call schedule decides the bits:
//   * the axes are sorted by |stride| and coalesced (reduced axes that sit next to each other in memory merge);
//   * the innermost iteration axis decides the loop form: a REDUCED one runs the reduce form, `*out += pairwise_sum(run)`
//     per inner-loop call; a KEPT one runs the elementwise form, `out[i] = out[i] + x[i]`, every reduced element added in
//     iteration order;
//   * a cast operand is buffered, and the fills (np.getbufsize() elements) cut the pairwise calls;
//   * the result starts as the loop's identity (PyUFunc_ReduceWrapper assigns +0.0 before the loop), so even the empty
//     tuple adds: `np.add.reduce(x, axis=())` is 0 + x element by element (a -0.0 becomes +0.0).
// So `a.mean(axis=(0, 1))` of a C-ordered (2, 3, 4) array adds the six elements of each output one after the other (its
// innermost axis is kept), where two single-axis means average three elements, round, then average two; and the mean
// over the empty tuple divides a float64 copy by 1, where returning the operand kept its dtype and its memory.
//
// NumPyReduceIterator is the port of that iterator (np.max's keyword path runs it); this file hands it NumPy's add
// loops: DOUBLE / FLOAT add (the reduce form through the exact pairwise folds the flat engine uses, the elementwise form
// with NumPy's NaN operand order, a vector at a time over a contiguous call), HALF add (float32 arithmetic, the running
// result narrowed back after each call), CDOUBLE add (the complex pairwise fold, per part), and decimal (NumSharp's own:
// a sequential decimal sum). np.sum runs the same iterator for an operand with a reversed or a broadcast axis (ReduceAdd),
// whose order the axis kernels and the flat engine do not reproduce.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>add.reduce(arr, axis=axes, dtype=loop, keepdims=keepdims)</c> over an axis SET, bit for bit: one
        ///     axis is np.sum's own axis route, every axis its flat route, and any other set (two or more axes, or none)
        ///     NumPy's reduction iterator over all of them at once (see the file header).
        /// </summary>
        /// <param name="arr">The numeric operand (any layout; text is the caller's: np.average answers it first).</param>
        /// <param name="axes">The reduced axes, normalized and distinct (NumPy's <c>normalize_axis_tuple</c> ran), in any
        ///     order; empty for none.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="loop">The loop dtype: every element is cast to it before it is added. float16, float32, float64,
        ///     complex128 or decimal for a set of two axes or more, or none (np.average's loops); any loop np.sum takes for
        ///     one axis or every axis.</param>
        /// <returns>The sum in <paramref name="loop"/>: a fresh array laid out as NumPy's iterator allocates it (the input's
        ///     axis order), owning its data; a read-only 0-d scalar (NumPy's scalar) when every axis is reduced without
        ///     keepdims.</returns>
        /// <exception cref="NotSupportedException">A set of two axes or more, or none, in a loop other than the five above
        ///     (an integer, bool or char loop: none of np.average's statements reaches one).</exception>
        /// <exception cref="OverflowException">A decimal sum leaves decimal's range.</exception>
        /// <remarks>
        ///     NumPy's statement order matters only through the iterator: the order of <paramref name="axes"/> never does
        ///     (<c>(0, 1)</c> and <c>(1, 0)</c> are one set). An empty operand gives the identity in every output (the
        ///     iterator makes no call), so a mean over an empty set of elements divides 0 by 0.
        /// </remarks>
        internal NDArray ReduceAddOverAxes(NDArray arr, int[] axes, bool keepdims, NPTypeCode loop)
        {
            // One axis and every axis are add.reduce's forms np.sum already has, with NumPy's schedules (np.sum along an
            // axis, the flat sum); a 0-d operand's only tuple, (), is every axis too.
            if (axes.Length == 1)
                return arr.TensorEngine.ReduceAdd(arr, axes[0], keepdims, loop);
            if (axes.Length == arr.ndim)
                return arr.TensorEngine.ReduceAdd(arr, null, keepdims, loop);

            var flags = new bool[arr.ndim];
            foreach (int ax in axes)
                flags[ax] = true;
            return ReduceAddThroughIterator(arr, flags, keepdims, loop);
        }

        /// <summary>
        ///     NumPy's <c>_methods._mean</c> over an axis set: <c>umr_sum(arr, axis, dtype)</c> through
        ///     <see cref="ReduceAddOverAxes"/> (float64 for bool / integers / char, float32 for float16, the input's own
        ///     dtype otherwise), then the division by the count of the reduced elements, exactly as the single-axis mean
        ///     divides (<see cref="FinishMeanLikeNumPy"/>).
        /// </summary>
        /// <param name="arr">The numeric operand.</param>
        /// <param name="axes">The reduced axes, normalized and distinct, in any order; empty for none (the count is then 1:
        ///     a float copy of every element).</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The mean: float64 for bool / integers / char, float16 for float16 (summed in float32), the input's own
        ///     dtype otherwise; a fresh array in NumPy's layout, or a read-only 0-d scalar when every axis is reduced
        ///     without keepdims. A decimal operand whose reduced axes hold no element gives a float64 NaN in every output,
        ///     as np.mean's single-axis route answers decimal (decimal has no NaN for the 0 / 0).</returns>
        /// <exception cref="NotSupportedException">A text operand (np.average answers text before it reaches here), or a
        ///     set of two axes or more in a loop <see cref="ReduceAddOverAxes"/> has no add loop for.</exception>
        /// <exception cref="OverflowException">A decimal sum leaves decimal's range.</exception>
        internal NDArray MeanOverAxes(NDArray arr, int[] axes, bool keepdims)
        {
            // _mean's loop: dtype is None, so float64 for an integer or bool operand and float32 for float16
            // (is_float16_result); every other dtype sums in its own loop.
            NPTypeCode inT = arr.GetTypeCode;
            bool float16Result = inT == NPTypeCode.Half;
            NPTypeCode loop = float16Result ? NPTypeCode.Single : inT.GetComputingType();

            // rcount = _count_reduce_items(arr, axis): the product of the reduced extents (1 for the empty tuple).
            long count = 1;
            foreach (int ax in axes)
                count *= arr.Shape.dimensions[ax];

            // Decimal (no NumPy dtype) cannot hold the NaN of a mean over no element, and its division by 0 raises:
            // answer as np.mean answers a decimal mean over an empty axis (NaNOfMeanType: float64 NaN in every output).
            if (count == 0 && loop == NPTypeCode.Decimal)
                return DecimalMeanOfNoElement(arr, axes, keepdims);

            var ret = ReduceAddOverAxes(arr, axes, keepdims, loop);
            return FinishMeanLikeNumPy(ret, count, loop, float16Result);
        }

        /// <summary>
        ///     The mean of a decimal operand over axes that hold no element, as np.mean's single-axis route answers it: a
        ///     float64 NaN (.NET's <see cref="double.NaN"/>, the value that route fills) in every output, the reduced axes
        ///     dropped or kept with length 1.
        /// </summary>
        /// <param name="arr">The decimal operand (empty along at least one reduced axis).</param>
        /// <param name="axes">The reduced axes, normalized and distinct.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>A fresh float64 array of NaN, C-ordered; a read-only 0-d scalar when every axis is reduced without
        ///     keepdims.</returns>
        private unsafe NDArray DecimalMeanOfNoElement(NDArray arr, int[] axes, bool keepdims)
        {
            var dims = new long[keepdims ? arr.ndim : arr.ndim - axes.Length];
            for (int d = 0, k = 0; d < arr.ndim; d++)
            {
                bool reduced = Array.IndexOf(axes, d) >= 0;
                if (!reduced)
                    dims[k++] = arr.Shape.dimensions[d];
                else if (keepdims)
                    dims[k++] = 1;
            }

            var nan = NewArray(NPTypeCode.Double, dims.Length == 0 ? Shape.Scalar : Shape.Adopt(dims), false);
            // The allocation is uninitialized: every element is written here, the buffer is dense (a fresh C array).
            double* p = (double*)nan.Address;
            for (long i = 0; i < nan.size; i++)
                p[i] = double.NaN;
            return dims.Length == 0 ? nan.MarkReductionScalar() : nan;
        }

        /// <summary>
        ///     add.reduce over any set of axes through <see cref="NumPyReduceIterator"/>: the result allocated in the
        ///     iteration order and set to the identity (+0, the add loop's), then NumPy's add loop over every inner-loop
        ///     call the iterator makes. The axis-set reduction runs it for two axes or more, or none; np.sum (ReduceAdd) for
        ///     one axis or every axis of an operand with a reversed or a broadcast axis.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="flags">One flag per axis: true where it is reduced.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="loop">The loop dtype (float16, float32, float64, complex128 or decimal).</param>
        /// <returns>The fresh sum, owning its data; a read-only 0-d scalar (NumPy's scalar) when every axis is reduced
        ///     without keepdims.</returns>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not one of the five loops.</exception>
        /// <exception cref="OverflowException">A decimal sum leaves decimal's range.</exception>
        private unsafe NDArray ReduceAddThroughIterator(NDArray arr, bool[] flags, bool keepdims, NPTypeCode loop)
        {
            // The loop must exist before anything is allocated: the caller never sees a half-built result.
            if (loop != NPTypeCode.Double && loop != NPTypeCode.Single && loop != NPTypeCode.Half
                && loop != NPTypeCode.Complex && loop != NPTypeCode.Decimal)
                throw new NotSupportedException(
                    $"add.reduce over an axis tuple has no {loop} loop here (np.average reaches float16, float32, float64, complex128 and decimal).");

            // NumPy's buffer size (np.getbufsize(), thread-local) cuts a cast input's pairwise calls into fills.
            var it = NumPyReduceIterator.Build(arr.Shape, flags, keepdims, null, null, false, arr.GetTypeCode != loop,
                NDIterBufferManager.CurrentBufferSize);

            // npyiter_new_temp_array's layout, zero-filled: PyUFunc_ReduceWrapper assigns the identity (+0.0 for an inexact
            // loop, 0 for decimal) to the result before the first call, so every output, an empty reduction's included,
            // starts at it.
            var result = NewArray(loop, it.ResultDims.Length == 0 ? Shape.Scalar : Shape.Adopt(it.ResultDims, it.ResultStrides), true);
            try
            {
                byte* o = (byte*)result.Address + result.Shape.offset * result.dtypesize;
                byte* x = (byte*)arr.Address + arr.Shape.offset * arr.dtypesize;
                NPTypeCode xT = arr.GetTypeCode;
                switch (loop)
                {
                    case NPTypeCode.Double: it.Run(o, loop, x, xT, null, loop, 0, NumPyAddLoops.Double()); break;
                    case NPTypeCode.Single: it.Run(o, loop, x, xT, null, loop, 0, NumPyAddLoops.Single()); break;
                    case NPTypeCode.Half: it.Run(o, loop, x, xT, null, loop, 0, NumPyAddLoops.Half()); break;
                    case NPTypeCode.Complex: it.Run(o, loop, x, xT, null, loop, 0, NumPyAddLoops.Complex()); break;
                    default: it.Run(o, loop, x, xT, null, loop, 0, NumPyAddLoops.Decimal()); break;
                }
            }
            catch
            {
                // A decimal overflow mid-walk: give the half-computed result's buffer back before the exception leaves.
                result.Dispose();
                throw;
            }

            // A reduction over every axis without keepdims (np.sum's flat sum of a broadcast operand) is NumPy's scalar;
            // anything with a dimension left is an array.
            return result.ndim == 0 ? result.MarkReductionScalar() : result;
        }
    }

    /// <summary>
    ///     NumPy's add loops in the two call forms of <see cref="NumPyReduceIterator.Loop{T}"/>: the reduce form (the out
    ///     stride is 0: <c>*out += pairwise_sum(run)</c>, IS_BINARY_REDUCE) and the elementwise form
    ///     (<c>out[i] = out[i] + x[i]</c>).
    /// </summary>
    /// <remarks>
    ///     The reduce forms are the flat add.reduce engine's statements (<c>NumPyAddReduce</c>'s accumulate steps) through
    ///     the same exact pairwise folds; the elementwise forms carry NumPy's NaN operand order for array add on win-amd64
    ///     (the SECOND operand's NaN when both are NaN, <c>DirectILKernelGenerator.NaNOperandOrder.cs</c>), written out
    ///     because RyuJIT may swap a commutative add's operands.
    /// </remarks>
    internal static unsafe class NumPyAddLoops
    {
        /// <summary>DOUBLE_add's two forms, and the several-rows form of the elementwise one.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<double> Double() => new(&DoubleReduce, &DoubleBinary, null, null, &DoubleBinaryRows);

        /// <summary>FLOAT_add's two forms (float32 arithmetic), and the several-rows form of the elementwise one.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<float> Single() => new(&SingleReduce, &SingleBinary, null, null, &SingleBinaryRows);

        /// <summary>HALF_add's two forms over raw float16 bits (float32 arithmetic, narrowed back after each call).</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<ushort> Half() => new(&HalfReduce, &HalfBinary);

        /// <summary>CDOUBLE_add's two forms (per part).</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<Complex> Complex() => new(&ComplexReduce, &ComplexBinary);

        /// <summary>NumSharp's decimal add (no NumPy dtype): sequential decimal sums.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<decimal> Decimal() => new(&DecimalReduce, &DecimalBinary);

        /// <summary>DOUBLE_add's reduce form: <c>*acc += DOUBLE_pairwise_sum(run)</c>, through the SIMD fold when the host
        ///     has one, else the scalar port (the same bits).</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="stride">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <remarks>A stride-0 run (a broadcast axis innermost) is <paramref name="n"/> copies of one value: see
        ///     <see cref="PairwiseOfCopies(double, long)"/>.</remarks>
        private static double DoubleReduce(double acc, double* x, long stride, long n)
        {
            if (stride == 0)
                return AddRunSum(acc, PairwiseOfCopies(*x, n));
            var fold = ILKernelGenerator.TryGetPairwiseFoldDouble();
            double s = fold != null ? fold(x, n, stride) : ILKernelGenerator.PairwiseFold(x, n, stride);
            return AddRunSum(acc, s);
        }

        /// <summary>
        ///     The last statement of DOUBLE_add's reduce form, <c>*out += pairwise_sum(run)</c>, as NumPy's win-amd64 build
        ///     computes it: MSVC adds the out's value to the run's sum (<c>addsd sum, [out]</c>), so when both are NaN the
        ///     RUN's NaN survives (probed 2.4.2: <c>np.sum([p2], initial=p1)</c> keeps p2's payload).
        /// </summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="s">The run's pairwise sum.</param>
        /// <returns><c>acc + s</c>, the run's NaN (quieted) when both are NaN.</returns>
        /// <remarks>Explicit because RyuJIT may swap a commutative add's operands. The out's value is NaN only after an
        ///     earlier call produced one, or with a NaN <c>initial=</c>; a running sum that starts at the identity never is.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double AddRunSum(double acc, double s)
        {
            double r = acc + s;
            if (double.IsNaN(r) && double.IsNaN(s))
                r = s + s;
            return r;
        }

        /// <summary>FLOAT_add's last reduce statement (see <see cref="AddRunSum(double, double)"/>): the run's NaN survives
        ///     when both are NaN.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="s">The run's pairwise sum (float32).</param>
        /// <returns><c>acc + s</c> in float32, the run's NaN when both are NaN.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float AddRunSum(float acc, float s)
        {
            float r = acc + s;
            if (float.IsNaN(r) && float.IsNaN(s))
                r = s + s;
            return r;
        }

        /// <summary>
        ///     NumPy's <c>DOUBLE_pairwise_sum</c> of <paramref name="n"/> copies of <paramref name="v"/> (a stride-0 run: NumPy
        ///     reads one address <paramref name="n"/> times), computed with NumPy's own operations on the value. Every
        ///     element is <paramref name="v"/>, so the eight accumulators of a block are equal (each starts at
        ///     <paramref name="v"/> and adds it once per further group of eight), their combine is NumPy's expression over
        ///     equal operands, and the rest of the block and the short runs add <paramref name="v"/> in sequence; a run past
        ///     the 128-element block splits where NumPy's recursion splits it. The bits are NumPy's (the same IEEE
        ///     additions in the same order), at a few additions per eight elements instead of one load each.
        /// </summary>
        /// <param name="v">The repeated value.</param>
        /// <param name="n">The copy count.</param>
        /// <returns>The pairwise sum.</returns>
        /// <remarks>Internal, not private: its test holds it to the exact fold over a buffer of copies.</remarks>
        internal static double PairwiseOfCopies(double v, long n)
        {
            if (n < 8)
            {
                // NumPy starts a short run at -0.0, so a run of -0.0 sums to -0.0.
                double res = -0.0;
                for (long i = 0; i < n; i++)
                    res += v;
                return res;
            }
            if (n <= 128)
            {
                // r[0..7] = a[0..7], then r[j] += a[i + j] for each further whole group of eight.
                double r = v;
                for (long g = 1; g < n / 8; g++)
                    r += v;
                double res = ((r + r) + (r + r)) + ((r + r) + (r + r));
                for (long i = 0; i < n % 8; i++)
                    res += v;
                return res;
            }
            long n2 = n / 2;
            n2 -= n2 % 8;
            return PairwiseOfCopies(v, n2) + PairwiseOfCopies(v, n - n2);
        }

        /// <summary>DOUBLE_add's elementwise form: <c>o[i] = o[i] + x[i]</c>, with NumPy's NaN operand rules.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <remarks>
        ///     The out (NumPy's first operand and its output) is always a fresh, positive-strided result here. A contiguous
        ///     out runs a vector loop for a contiguous, a broadcast (stride 0) or a reversed (stride -1) input; any other
        ///     input takes the scalar statement. The NaN of a lane where both operands are NaN is NumPy's: the SECOND
        ///     operand's (the input's) in every call of two arrays and in every short call, and in the vector body of a long
        ///     call over a broadcast input the BROADCAST operand's, in its tail the ARRAY operand's (NumPy's scalar-broadcast
        ///     loop; <c>DirectILKernelGenerator.NaNOperandOrder.cs</c>). Every non-NaN result is the plain sum either way.
        /// </remarks>
        private static void DoubleBinary(double* o, long os, double* x, long xs, long n)
        {
            if (os == 1 && Vector256.IsHardwareAccelerated)
            {
                if (xs == 1)
                {
                    DoubleBinaryContiguous(o, x, n);
                    return;
                }
                if (xs == 0)
                {
                    DoubleBinaryBroadcast(o, *x, n);
                    return;
                }
                if (xs == -1 && Avx2.IsSupported)
                {
                    DoubleBinaryReversed(o, x, n);
                    return;
                }
            }
            for (long i = 0; i < n; i++)
            {
                double a = o[i * os], b = x[i * xs];
                double r = a + b;
                // NumPy's add loop keeps the SECOND operand's NaN: when both are NaN return b's, quieted.
                if (double.IsNaN(r) && double.IsNaN(b))
                    r = b + b;
                o[i * os] = r;
            }
        }

        /// <summary><see cref="DoubleBinary"/> for a contiguous out and input: two vectors a step, the second operand's NaN
        ///     written as <c>(a where b is a number, else +0) + b</c>, which leaves no NaN choice to the instruction.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="x">The input run.</param>
        /// <param name="n">The element count.</param>
        private static void DoubleBinaryContiguous(double* o, double* x, long n)
        {
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var b0 = Vector256.Load(x + i);
                var b1 = Vector256.Load(x + i + 4);
                var a0 = Vector256.Load(o + i);
                var a1 = Vector256.Load(o + i + 4);
                Vector256.Store((a0 & Vector256.Equals(b0, b0)) + b0, o + i);
                Vector256.Store((a1 & Vector256.Equals(b1, b1)) + b1, o + i + 4);
            }
            for (; i + 4 <= n; i += 4)
            {
                var b = Vector256.Load(x + i);
                Vector256.Store((Vector256.Load(o + i) & Vector256.Equals(b, b)) + b, o + i);
            }
            for (; i < n; i++)
            {
                double a = o[i], b = x[i];
                o[i] = double.IsNaN(b) ? b + b : a + b;
            }
        }

        /// <summary><see cref="DoubleBinary"/> for a contiguous out and one broadcast value: NumPy's scalar-broadcast loop
        ///     for a call longer than two vectors (its body keeps the broadcast value's NaN, its tail the out's), the plain
        ///     loop for a shorter one (the second operand's NaN).</summary>
        /// <param name="o">The out run.</param>
        /// <param name="v">The broadcast input value.</param>
        /// <param name="n">The element count.</param>
        private static void DoubleBinaryBroadcast(double* o, double v, long n)
        {
            long i = 0;
            if (n > 8)
            {
                // The body [0, n - n % 8): the broadcast value's NaN wherever it is NaN (then every lane takes it).
                long body = n - n % 8;
                var vb = Vector256.Create(v);
                var keep = Vector256.Equals(vb, vb);
                for (; i < body; i += 4)
                    Vector256.Store((Vector256.Load(o + i) & keep) + vb, o + i);
                // The tail: the out's (the array operand's) NaN when both are NaN.
                for (; i < n; i++)
                {
                    double a = o[i];
                    o[i] = double.IsNaN(a) ? a + a : a + v;
                }
                return;
            }
            for (; i < n; i++)
            {
                double a = o[i];
                o[i] = double.IsNaN(v) ? v + v : a + v;
            }
        }

        /// <summary><see cref="DoubleBinary"/> for a contiguous out and a reversed input (stride -1): four input elements
        ///     loaded from below and reversed in the register, the second operand's NaN as in the contiguous loop.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="x">The input's first element (the highest address of the run).</param>
        /// <param name="n">The element count.</param>
        private static void DoubleBinaryReversed(double* o, double* x, long n)
        {
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                // x[-i - 3] .. x[-i] in memory order, then reversed: the logical elements i .. i + 3.
                var b = Avx2.Permute4x64(Vector256.Load(x - i - 3), 0b00_01_10_11);
                Vector256.Store((Vector256.Load(o + i) & Vector256.Equals(b, b)) + b, o + i);
            }
            for (; i < n; i++)
            {
                double a = o[i], b = x[-i];
                o[i] = double.IsNaN(b) ? b + b : a + b;
            }
        }

        /// <summary>
        ///     DOUBLE_add's elementwise form over several consecutive calls into one out row (the reduce double loop's rows
        ///     when the out does not move along the outer dimension): <c>o[i] = (((o[i] + x0[i]) + x1[i]) + ...)</c>, the
        ///     rows in order, each addition the one call of that row would make.
        /// </summary>
        /// <param name="o">The out row.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The first input row.</param>
        /// <param name="xs">The input stride within a row, in elements.</param>
        /// <param name="xRowStride">The distance from one input row to the next, in elements (0: a broadcast row).</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The elements per row.</param>
        /// <remarks>
        ///     Contiguous rows keep sixteen lanes of the out in four registers across the rows, so one row's additions do
        ///     not wait on each other and the out is loaded and stored once per block instead of once per row; each lane is
        ///     still the rows' additions in order with the second operand's NaN, so the bits are the per-row calls'. A
        ///     broadcast ROW (<paramref name="xRowStride"/> 0, every row the same memory) is contiguous and takes the same
        ///     loop. An input that is not contiguous WITHIN its row — one broadcast value (<paramref name="xs"/> 0, whose
        ///     NaN rule depends on the call's length), a reversed or a stepped row — is one <see cref="DoubleBinary"/>
        ///     call per row.
        /// </remarks>
        private static void DoubleBinaryRows(double* o, long os, double* x, long xs, long xRowStride, long rows, long n)
        {
            if (os == 1 && xs == 1 && Vector256.IsHardwareAccelerated)
            {
                long i = 0;
                for (; i + 16 <= n; i += 16)
                {
                    var a0 = Vector256.Load(o + i);
                    var a1 = Vector256.Load(o + i + 4);
                    var a2 = Vector256.Load(o + i + 8);
                    var a3 = Vector256.Load(o + i + 12);
                    double* xr = x + i;
                    for (long r = 0; r < rows; r++, xr += xRowStride)
                    {
                        var b0 = Vector256.Load(xr);
                        var b1 = Vector256.Load(xr + 4);
                        var b2 = Vector256.Load(xr + 8);
                        var b3 = Vector256.Load(xr + 12);
                        a0 = (a0 & Vector256.Equals(b0, b0)) + b0;
                        a1 = (a1 & Vector256.Equals(b1, b1)) + b1;
                        a2 = (a2 & Vector256.Equals(b2, b2)) + b2;
                        a3 = (a3 & Vector256.Equals(b3, b3)) + b3;
                    }
                    Vector256.Store(a0, o + i);
                    Vector256.Store(a1, o + i + 4);
                    Vector256.Store(a2, o + i + 8);
                    Vector256.Store(a3, o + i + 12);
                }
                for (; i + 4 <= n; i += 4)
                {
                    var a = Vector256.Load(o + i);
                    double* xr = x + i;
                    for (long r = 0; r < rows; r++, xr += xRowStride)
                    {
                        var b = Vector256.Load(xr);
                        a = (a & Vector256.Equals(b, b)) + b;
                    }
                    Vector256.Store(a, o + i);
                }
                for (; i < n; i++)
                {
                    double a = o[i];
                    double* xr = x + i;
                    for (long r = 0; r < rows; r++, xr += xRowStride)
                    {
                        double b = *xr;
                        a = double.IsNaN(b) ? b + b : a + b;
                    }
                    o[i] = a;
                }
                return;
            }
            for (long r = 0; r < rows; r++)
                DoubleBinary(o, os, x + r * xRowStride, xs, n);
        }

        /// <summary>FLOAT_add's reduce form: <c>*acc += FLOAT_pairwise_sum(run)</c> in float32.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="stride">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <remarks>A stride-0 run is <paramref name="n"/> copies of one value: see
        ///     <see cref="PairwiseOfCopies(float, long)"/>.</remarks>
        private static float SingleReduce(float acc, float* x, long stride, long n)
        {
            if (stride == 0)
                return AddRunSum(acc, PairwiseOfCopies(*x, n));
            var fold = ILKernelGenerator.TryGetPairwiseFoldSingle();
            float s = fold != null ? fold(x, n, stride) : ILKernelGenerator.PairwiseFold(x, n, stride);
            return AddRunSum(acc, s);
        }

        /// <summary>NumPy's <c>FLOAT_pairwise_sum</c> of <paramref name="n"/> copies of <paramref name="v"/> (see
        ///     <see cref="PairwiseOfCopies(double, long)"/>: the same operations in float32 arithmetic).</summary>
        /// <param name="v">The repeated value.</param>
        /// <param name="n">The copy count.</param>
        /// <returns>The pairwise sum.</returns>
        /// <remarks>Internal, not private: its test holds it to the exact fold over a buffer of copies.</remarks>
        internal static float PairwiseOfCopies(float v, long n)
        {
            if (n < 8)
            {
                float res = -0.0f;
                for (long i = 0; i < n; i++)
                    res += v;
                return res;
            }
            if (n <= 128)
            {
                float r = v;
                for (long g = 1; g < n / 8; g++)
                    r += v;
                float res = ((r + r) + (r + r)) + ((r + r) + (r + r));
                for (long i = 0; i < n % 8; i++)
                    res += v;
                return res;
            }
            long n2 = n / 2;
            n2 -= n2 % 8;
            return PairwiseOfCopies(v, n2) + PairwiseOfCopies(v, n - n2);
        }

        /// <summary>FLOAT_add's elementwise form (float32), with NumPy's NaN operand rules (see
        ///     <see cref="DoubleBinary"/>: the same specializations, eight lanes a vector).</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void SingleBinary(float* o, long os, float* x, long xs, long n)
        {
            if (os == 1 && Vector256.IsHardwareAccelerated)
            {
                if (xs == 1)
                {
                    SingleBinaryContiguous(o, x, n);
                    return;
                }
                if (xs == 0)
                {
                    SingleBinaryBroadcast(o, *x, n);
                    return;
                }
                if (xs == -1 && Avx2.IsSupported)
                {
                    SingleBinaryReversed(o, x, n);
                    return;
                }
            }
            for (long i = 0; i < n; i++)
            {
                float a = o[i * os], b = x[i * xs];
                float r = a + b;
                if (float.IsNaN(r) && float.IsNaN(b))
                    r = b + b;
                o[i * os] = r;
            }
        }

        /// <summary><see cref="SingleBinary"/> for a contiguous out and input (see <see cref="DoubleBinaryContiguous"/>).</summary>
        /// <param name="o">The out run.</param>
        /// <param name="x">The input run.</param>
        /// <param name="n">The element count.</param>
        private static void SingleBinaryContiguous(float* o, float* x, long n)
        {
            long i = 0;
            for (; i + 16 <= n; i += 16)
            {
                var b0 = Vector256.Load(x + i);
                var b1 = Vector256.Load(x + i + 8);
                var a0 = Vector256.Load(o + i);
                var a1 = Vector256.Load(o + i + 8);
                Vector256.Store((a0 & Vector256.Equals(b0, b0)) + b0, o + i);
                Vector256.Store((a1 & Vector256.Equals(b1, b1)) + b1, o + i + 8);
            }
            for (; i + 8 <= n; i += 8)
            {
                var b = Vector256.Load(x + i);
                Vector256.Store((Vector256.Load(o + i) & Vector256.Equals(b, b)) + b, o + i);
            }
            for (; i < n; i++)
            {
                float a = o[i], b = x[i];
                o[i] = float.IsNaN(b) ? b + b : a + b;
            }
        }

        /// <summary><see cref="SingleBinary"/> for a contiguous out and one broadcast value: NumPy's scalar-broadcast loop
        ///     past two vectors (16 lanes), the plain loop for a shorter call (see <see cref="DoubleBinaryBroadcast"/>).</summary>
        /// <param name="o">The out run.</param>
        /// <param name="v">The broadcast input value.</param>
        /// <param name="n">The element count.</param>
        private static void SingleBinaryBroadcast(float* o, float v, long n)
        {
            long i = 0;
            if (n > 16)
            {
                long body = n - n % 16;
                var vb = Vector256.Create(v);
                var keep = Vector256.Equals(vb, vb);
                for (; i < body; i += 8)
                    Vector256.Store((Vector256.Load(o + i) & keep) + vb, o + i);
                for (; i < n; i++)
                {
                    float a = o[i];
                    o[i] = float.IsNaN(a) ? a + a : a + v;
                }
                return;
            }
            for (; i < n; i++)
            {
                float a = o[i];
                o[i] = float.IsNaN(v) ? v + v : a + v;
            }
        }

        /// <summary><see cref="SingleBinary"/> for a contiguous out and a reversed input (see
        ///     <see cref="DoubleBinaryReversed"/>; eight lanes reversed by one permute).</summary>
        /// <param name="o">The out run.</param>
        /// <param name="x">The input's first element (the highest address of the run).</param>
        /// <param name="n">The element count.</param>
        private static void SingleBinaryReversed(float* o, float* x, long n)
        {
            long i = 0;
            var reverse = Vector256.Create(7, 6, 5, 4, 3, 2, 1, 0);
            for (; i + 8 <= n; i += 8)
            {
                var b = Avx2.PermuteVar8x32(Vector256.Load(x - i - 7), reverse);
                Vector256.Store((Vector256.Load(o + i) & Vector256.Equals(b, b)) + b, o + i);
            }
            for (; i < n; i++)
            {
                float a = o[i], b = x[-i];
                o[i] = float.IsNaN(b) ? b + b : a + b;
            }
        }

        /// <summary>FLOAT_add's elementwise form over several consecutive calls into one out row (see
        ///     <see cref="DoubleBinaryRows"/>: the same blocking, eight lanes a register, thirty-two a step).</summary>
        /// <param name="o">The out row.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The first input row.</param>
        /// <param name="xs">The input stride within a row, in elements.</param>
        /// <param name="xRowStride">The distance from one input row to the next, in elements (0: a broadcast row).</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The elements per row.</param>
        private static void SingleBinaryRows(float* o, long os, float* x, long xs, long xRowStride, long rows, long n)
        {
            if (os == 1 && xs == 1 && Vector256.IsHardwareAccelerated)
            {
                long i = 0;
                for (; i + 32 <= n; i += 32)
                {
                    var a0 = Vector256.Load(o + i);
                    var a1 = Vector256.Load(o + i + 8);
                    var a2 = Vector256.Load(o + i + 16);
                    var a3 = Vector256.Load(o + i + 24);
                    float* xr = x + i;
                    for (long r = 0; r < rows; r++, xr += xRowStride)
                    {
                        var b0 = Vector256.Load(xr);
                        var b1 = Vector256.Load(xr + 8);
                        var b2 = Vector256.Load(xr + 16);
                        var b3 = Vector256.Load(xr + 24);
                        a0 = (a0 & Vector256.Equals(b0, b0)) + b0;
                        a1 = (a1 & Vector256.Equals(b1, b1)) + b1;
                        a2 = (a2 & Vector256.Equals(b2, b2)) + b2;
                        a3 = (a3 & Vector256.Equals(b3, b3)) + b3;
                    }
                    Vector256.Store(a0, o + i);
                    Vector256.Store(a1, o + i + 8);
                    Vector256.Store(a2, o + i + 16);
                    Vector256.Store(a3, o + i + 24);
                }
                for (; i + 8 <= n; i += 8)
                {
                    var a = Vector256.Load(o + i);
                    float* xr = x + i;
                    for (long r = 0; r < rows; r++, xr += xRowStride)
                    {
                        var b = Vector256.Load(xr);
                        a = (a & Vector256.Equals(b, b)) + b;
                    }
                    Vector256.Store(a, o + i);
                }
                for (; i < n; i++)
                {
                    float a = o[i];
                    float* xr = x + i;
                    for (long r = 0; r < rows; r++, xr += xRowStride)
                    {
                        float b = *xr;
                        a = float.IsNaN(b) ? b + b : a + b;
                    }
                    o[i] = a;
                }
                return;
            }
            for (long r = 0; r < rows; r++)
                SingleBinary(o, os, x + r * xRowStride, xs, n);
        }

        /// <summary>HALF_add's reduce form: <c>io1 = half_to_float(*acc); io1 += HALF_pairwise_sum(run); *acc =
        ///     float_to_half(io1)</c> (the pairwise sum in float32, the running result narrowed back after every call).</summary>
        /// <param name="acc">The out's running value (float16 bits).</param>
        /// <param name="x">The run's first element (float16 bits).</param>
        /// <param name="stride">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value (float16 bits).</returns>
        private static ushort HalfReduce(ushort acc, ushort* x, long stride, long n)
        {
            float s = ILKernelGenerator.PairwiseFoldHalf(x, n, stride);
            float io1 = DirectILKernelGenerator.HalfToFloatScalarExact(acc) + s;
            return DirectILKernelGenerator.SingleToHalfBits(io1);
        }

        /// <summary>HALF_add's elementwise form: <c>float_to_half(half_to_float(o[i]) + half_to_float(x[i]))</c>, keeping
        ///     <c>x[i]</c>'s NaN when both are NaN (float16 keeps the second operand's NaN on every path).</summary>
        /// <param name="o">The out run (float16 bits).</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run (float16 bits).</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <remarks>
        ///     A contiguous out (the reduce double loop's usual out row) runs the house SIMD widen-compute-narrow kernel in
        ///     place (<see cref="DirectILKernelGenerator.HalfArithRun"/>: NumPy's HALF loop eight lanes at a time with the
        ///     same NaN rule, bit for bit; a strided input is gathered into a stack block first). Measured on a (100, 1000)
        ///     broadcast row summed along axis 0 (100,000 element adds): the scalar loop 705 µs (7 ns an element), the
        ///     kernel 139 µs, NumPy's own scalar HALF_add 908 µs. A strided out keeps the scalar loop, the one form that
        ///     kernel does not take.
        /// </remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private static void HalfBinary(ushort* o, long os, ushort* x, long xs, long n)
        {
            // HalfArithRun takes runs of one element or more (it reads a broadcast operand's value before its loop).
            if (os == 1 && n > 0)
            {
                // The gather blocks for a strided input (only the right one is used: the left operand, the out itself,
                // is read in place). Each is written before it is read, so it is never zeroed.
                ushort* stageA = stackalloc ushort[DirectILKernelGenerator.HalfArithGatherBlock];
                ushort* stageB = stackalloc ushort[DirectILKernelGenerator.HalfArithGatherBlock];
                DirectILKernelGenerator.HalfArithRun(o, 1, x, xs, o, n, BinaryOp.Add, stageA, stageB);
                return;
            }
            for (long i = 0; i < n; i++)
            {
                float a = DirectILKernelGenerator.HalfToFloatScalarExact(o[i * os]);
                float b = DirectILKernelGenerator.HalfToFloatScalarExact(x[i * xs]);
                float r = a + b;
                if (float.IsNaN(r) && float.IsNaN(b))
                    r = b + b;
                o[i * os] = DirectILKernelGenerator.SingleToHalfBits(r);
            }
        }

        /// <summary>CDOUBLE_add's reduce form: <c>*or += rr; *oi += ri</c> with <c>(rr, ri)</c> from CDOUBLE_pairwise_sum
        ///     (four accumulators per part over the interleaved doubles).</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first complex.</param>
        /// <param name="stride">The run's stride in complexes.</param>
        /// <param name="n">The run's complex count.</param>
        /// <returns>The out's new value.</returns>
        private static Complex ComplexReduce(Complex acc, Complex* x, long stride, long n)
        {
            NumPyAddReduce.FoldComplex((byte*)x, n, stride, out double rr, out double ri);
            return new Complex(acc.Real + rr, acc.Imaginary + ri);
        }

        /// <summary>CDOUBLE_add's elementwise form, per part, keeping <c>x[i]</c>'s NaN in a part where both are NaN.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in complexes.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in complexes.</param>
        /// <param name="n">The complex count.</param>
        /// <remarks>A contiguous call is <paramref name="n"/> * 2 contiguous doubles added part by part, the same statement
        ///     per part: it runs as <see cref="DoubleBinary"/>'s vector loop.</remarks>
        private static void ComplexBinary(Complex* o, long os, Complex* x, long xs, long n)
        {
            if (os == 1 && xs == 1)
            {
                DoubleBinary((double*)o, 1, (double*)x, 1, 2 * n);
                return;
            }
            for (long i = 0; i < n; i++)
            {
                double* a = (double*)(o + i * os);
                double* b = (double*)(x + i * xs);
                double re = a[0] + b[0], im = a[1] + b[1];
                if (double.IsNaN(re) && double.IsNaN(b[0]))
                    re = b[0] + b[0];
                if (double.IsNaN(im) && double.IsNaN(b[1]))
                    im = b[1] + b[1];
                a[0] = re;
                a[1] = im;
            }
        }

        /// <summary>The decimal reduce form: the run added to the running value one element after the other.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="stride">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <exception cref="OverflowException">The sum leaves decimal's range.</exception>
        private static decimal DecimalReduce(decimal acc, decimal* x, long stride, long n)
        {
            for (long i = 0; i < n; i++)
                acc += x[i * stride];
            return acc;
        }

        /// <summary>The decimal elementwise form: <c>o[i] = o[i] + x[i]</c>.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <exception cref="OverflowException">A sum leaves decimal's range.</exception>
        private static void DecimalBinary(decimal* o, long os, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] += x[i * xs];
        }
    }
}
