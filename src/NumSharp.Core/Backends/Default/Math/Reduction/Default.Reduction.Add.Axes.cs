using System;
using System.Numerics;
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
// with NumPy's NaN operand order), HALF add (float32 arithmetic, the running result narrowed back after each call),
// CDOUBLE add (the complex pairwise fold, per part), and decimal (NumSharp's own: a sequential decimal sum).
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
        internal static NDArray ReduceAddOverAxes(NDArray arr, int[] axes, bool keepdims, NPTypeCode loop)
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
        internal static NDArray MeanOverAxes(NDArray arr, int[] axes, bool keepdims)
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
        private static unsafe NDArray DecimalMeanOfNoElement(NDArray arr, int[] axes, bool keepdims)
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

            var nan = new NDArray(NPTypeCode.Double, dims.Length == 0 ? Shape.Scalar : Shape.Adopt(dims), false);
            // The allocation is uninitialized: every element is written here, the buffer is dense (a fresh C array).
            double* p = (double*)nan.Address;
            for (long i = 0; i < nan.size; i++)
                p[i] = double.NaN;
            return dims.Length == 0 ? nan.MarkReductionScalar() : nan;
        }

        /// <summary>
        ///     add.reduce over a set of axes that is neither one axis nor every axis, through
        ///     <see cref="NumPyReduceIterator"/>: the result allocated in the iteration order and set to the identity
        ///     (+0, the add loop's), then NumPy's add loop over every inner-loop call the iterator makes.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="flags">One flag per axis: true where it is reduced.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="loop">The loop dtype (float16, float32, float64, complex128 or decimal).</param>
        /// <returns>The fresh sum (never 0-d: a kept axis remains).</returns>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not one of the five loops.</exception>
        /// <exception cref="OverflowException">A decimal sum leaves decimal's range.</exception>
        private static unsafe NDArray ReduceAddThroughIterator(NDArray arr, bool[] flags, bool keepdims, NPTypeCode loop)
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
            var result = new NDArray(loop, it.ResultDims.Length == 0 ? Shape.Scalar : Shape.Adopt(it.ResultDims, it.ResultStrides), true);
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

            // Every axis kept, or a kept axis remaining: an array; a 0-d result (not reachable from ReduceAddOverAxes,
            // which sends every-axis sets to the flat route) would be NumPy's scalar.
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
        /// <summary>DOUBLE_add's two forms.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<double> Double() => new(&DoubleReduce, &DoubleBinary);

        /// <summary>FLOAT_add's two forms (float32 arithmetic).</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<float> Single() => new(&SingleReduce, &SingleBinary);

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
        private static double DoubleReduce(double acc, double* x, long stride, long n)
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldDouble();
            double s = fold != null ? fold(x, n, stride) : ILKernelGenerator.PairwiseFold(x, n, stride);
            return acc + s;
        }

        /// <summary>DOUBLE_add's elementwise form: <c>o[i] = o[i] + x[i]</c>, keeping <c>x[i]</c>'s NaN when both are NaN.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void DoubleBinary(double* o, long os, double* x, long xs, long n)
        {
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

        /// <summary>FLOAT_add's reduce form: <c>*acc += FLOAT_pairwise_sum(run)</c> in float32.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="stride">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        private static float SingleReduce(float acc, float* x, long stride, long n)
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldSingle();
            float s = fold != null ? fold(x, n, stride) : ILKernelGenerator.PairwiseFold(x, n, stride);
            return acc + s;
        }

        /// <summary>FLOAT_add's elementwise form (float32), keeping <c>x[i]</c>'s NaN when both are NaN.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void SingleBinary(float* o, long os, float* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
            {
                float a = o[i * os], b = x[i * xs];
                float r = a + b;
                if (float.IsNaN(r) && float.IsNaN(b))
                    r = b + b;
                o[i * os] = r;
            }
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
        private static void HalfBinary(ushort* o, long os, ushort* x, long xs, long n)
        {
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
        private static void ComplexBinary(Complex* o, long os, Complex* x, long xs, long n)
        {
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
