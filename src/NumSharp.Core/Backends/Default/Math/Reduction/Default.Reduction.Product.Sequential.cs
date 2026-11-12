using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    // ── np.prod of float32 / float64 in NumPy's order, at speed ────────────────────────────────────────────────────────
    //
    // NumPy's FLOAT / DOUBLE_multiply reduce is sequential: every output starts at the identity 1 and multiplies its
    // elements one after the other (`acc *= *src`; multiply has no pairwise form), so a product's last bits, and whether an
    // overflow meets a zero first (inf * 0 is NaN), follow that order. NumSharp used to multiply in SIMD lanes (the owner's
    // D24 / NL3, "Keep SIMD lanes"); the owner reversed that on 2026-11-12: every route is NumPy's order now.
    //
    // NumPy's reduction iterator (AddMulNumPy) gives that order for any layout, but pays a fixed cost per call and an
    // inner-loop call per row. The two layouts that carry nearly every product get kernels of their own, whose arithmetic
    // is the same sequence per output:
    //   * a contiguous array over every axis: one chain over memory (NumPy's iterator coalesces it to one run in memory
    //     order, C or F), at NumPy's own speed: a dependent multiply per element, which no reordering may shorten;
    //   * a C- or F-contiguous array along one axis, viewed in memory order as (outer, K, inner) (an F array is the C
    //     array of its reversed axes): rows (inner == 1, NumPy's reduce form, the first NaN of a row kept) run eight
    //     independent chains at once, so the multiplier's latency is spent on eight rows; outer axes (inner > 1, NumPy's
    //     elementwise form, the second operand's NaN kept) multiply the running row of outputs by each input row with
    //     vectors.
    // A chain multiplies without testing for NaN: a NaN anywhere propagates to the end, and a run that ends in NaN is
    // computed again with NumPy's exact loop (NumPyMultiplyLoops.ReduceRun), which keeps the NaN NumPy keeps. Every other
    // layout (strided, reversed, broadcast) and every cast loop takes AddMulNumPy.
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Whether a float32 / float64 product of <paramref name="arr"/> takes the exact kernels of this file: the loop is
        ///     the input's own dtype (no cast), the array holds elements, and it is C- or F-contiguous (an F array along an
        ///     axis is the C array of its reversed axes, see <see cref="SequentialFloatProductAlong"/>).
        /// </summary>
        /// <param name="arr">The input.</param>
        /// <param name="axis">The reduced axis, or null for every axis.</param>
        /// <param name="loop">The product's loop dtype.</param>
        /// <returns>True when <see cref="SequentialFloatProductFlat"/> / <see cref="SequentialFloatProductAlong"/> serve the
        ///     call.</returns>
        private static bool SequentialFloatProductApplies(NDArray arr, int? axis, NPTypeCode loop)
        {
            if (loop != NPTypeCode.Single && loop != NPTypeCode.Double)
                return false;
            if (arr.GetTypeCode != loop)
                return false;
            var shape = arr.Shape;
            if (shape.IsEmpty || shape.size == 0)
                return false;
            return shape.IsContiguous || shape.IsFContiguous;
        }

        /// <summary>
        ///     The product of every element of a contiguous float32 / float64 array in NumPy's order: one sequential chain
        ///     over memory from the identity 1.
        /// </summary>
        /// <param name="arr">The input (see <see cref="SequentialFloatProductApplies"/>).</param>
        /// <returns>The product, boxed as the array's element type (the form <c>prod_elementwise_il</c> returns).</returns>
        private unsafe object SequentialFloatProductFlat(NDArray arr)
        {
            byte* p = (byte*)arr.Address + arr.Shape.offset * arr.dtypesize;
            long n = arr.size;
            return arr.GetTypeCode == NPTypeCode.Double
                ? ChainProduct((double*)p, n)
                : (object)ChainProduct((float*)p, n);
        }

        /// <summary>
        ///     The product along one axis of a C- or F-contiguous float32 / float64 array in NumPy's order, allocated and
        ///     returned as <c>ExecuteAxisReduction</c> does (NumPy's reduction-output layout, keepdims' kept axis, a 0-d
        ///     result as NumPy's read-only scalar).
        /// </summary>
        /// <param name="arr">The input (see <see cref="SequentialFloatProductApplies"/>).</param>
        /// <param name="axis">The reduced axis, normalized, of a length of 2 or more.</param>
        /// <param name="keepdims">Keep the reduced axis with length 1.</param>
        /// <param name="outputType">The loop dtype (the input's).</param>
        /// <returns>The products.</returns>
        /// <remarks>
        ///     An F-contiguous array is the C-contiguous array of its reversed axes, and NumPy's iterator sees it that way:
        ///     its memory-contiguous axis (the first) is the inner loop, so reducing it is the reduce form and reducing any
        ///     other axis the elementwise form, as for the C array. Its result is F-ordered (AllocateReductionResult), whose
        ///     memory is the C order of the reversed output dimensions: the kernels fill it from the reversed view's
        ///     (outer, K, inner) unchanged.
        /// </remarks>
        private unsafe NDArray SequentialFloatProductAlong(NDArray arr, int axis, bool keepdims, NPTypeCode outputType)
        {
            var shape = arr.Shape;
            var dims = shape.dimensions;
            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis)
                    outputDims[od++] = dims[d];
            // A C-contiguous input gives a C-contiguous result and an F-contiguous one an F-contiguous result
            // (AllocateReductionResult's NumPy layout), which the kernels fill in memory order.
            var result = AllocateReductionResult(outputType, outputDims, shape, axis);

            // The (outer, K, inner) view in MEMORY order: the axes as they are for a C array, reversed for an F one (a 1-D
            // array is both, and either reading is the same).
            bool fOrder = !shape.IsContiguous;
            long outer = 1, inner = 1;
            for (int d = 0; d < arr.ndim; d++)
            {
                if (d == axis)
                    continue;
                bool before = fOrder ? d > axis : d < axis;
                if (before) outer *= dims[d];
                else inner *= dims[d];
            }
            long k = dims[axis];
            byte* input = (byte*)arr.Address + shape.offset * arr.dtypesize;
            if (outputType == NPTypeCode.Double)
                AxisProduct((double*)input, (double*)result.Address, outer, k, inner);
            else
                AxisProduct((float*)input, (float*)result.Address, outer, k, inner);

            if (keepdims)
                ExpandReducedAxis(result, arr.Shape, axis);
            return result.PyArray_Return();
        }

        /// <summary>
        ///     One sequential product of <paramref name="n"/> contiguous elements from the identity 1 (NumPy's reduce form).
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="p">The first element.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The product; a NaN one is NumPy's (recomputed with its loop).</returns>
        /// <remarks>The loop has no NaN test: a NaN at any step makes the final value NaN, and only then is the run
        ///     computed again by NumPy's loop, whose first NaN is the answer (the plain loop's NaN could be another one:
        ///     RyuJIT may swap a multiply's operands).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe T ChainProduct<T>(T* p, long n) where T : unmanaged, IFloatingPointIeee754<T>
        {
            T acc = T.One;
            for (long i = 0; i < n; i++)
                acc *= p[i];
            return T.IsNaN(acc) ? NumPyMultiplyLoops.ReduceRun(T.One, p, 1, n) : acc;
        }

        /// <summary>
        ///     The products along the middle axis of a C-contiguous (outer, k, inner) array into a C-contiguous (outer, inner)
        ///     result, each output multiplied in increasing order of the reduced index from the identity 1.
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="a">The input's first element.</param>
        /// <param name="r">The result's first element.</param>
        /// <param name="outer">The product of the dimensions before the axis.</param>
        /// <param name="k">The reduced axis's length.</param>
        /// <param name="inner">The product of the dimensions after the axis.</param>
        /// <remarks>
        ///     inner == 1 is NumPy's reduce form (the reduced axis is the innermost run): <see cref="RowProducts{T}"/>.
        ///     Otherwise NumPy's iterator runs the elementwise form, <c>out[j] = out[j] * in[i][j]</c> over each input row,
        ///     whose NaN rule (the second operand's NaN when both are NaN) <see cref="NumPyMultiplyLoops.BinaryRun{T}"/>
        ///     implements with vectors.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisProduct<T>(T* a, T* r, long outer, long k, long inner)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (inner == 1)
            {
                RowProducts(a, r, outer, k);
                return;
            }
            for (long o = 0; o < outer; o++)
            {
                T* row = r + o * inner;
                // The identity first, as NumPy's reduction fills its output before the loop (1 * x is x, NaN quieted).
                for (long j = 0; j < inner; j++)
                    row[j] = T.One;
                T* block = a + o * k * inner;
                for (long i = 0; i < k; i++)
                    NumPyMultiplyLoops.BinaryRun(row, 1, block + i * inner, 1, inner);
            }
        }

        /// <summary>
        ///     The sequential product of each of <paramref name="rows"/> contiguous rows of <paramref name="k"/> elements
        ///     (NumPy's reduce form, one row per output), eight rows' chains in flight at once.
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="a">The first row's first element.</param>
        /// <param name="r">The results, one per row.</param>
        /// <param name="rows">The row count.</param>
        /// <param name="k">The row length.</param>
        /// <remarks>
        ///     Each row's multiplies are the same sequence as its own chain (only the instruction stream interleaves eight
        ///     of them), so its bits are NumPy's; a row that ends in NaN is computed again with NumPy's loop. Eight
        ///     independent chains keep the multiplier busy through its latency, which one chain at a time cannot.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RowProducts<T>(T* a, T* r, long rows, long k)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            long row = 0;
            for (; row + 8 <= rows; row += 8)
            {
                T* p0 = a + row * k;
                T* p1 = p0 + k, p2 = p1 + k, p3 = p2 + k, p4 = p3 + k, p5 = p4 + k, p6 = p5 + k, p7 = p6 + k;
                T c0 = T.One, c1 = T.One, c2 = T.One, c3 = T.One, c4 = T.One, c5 = T.One, c6 = T.One, c7 = T.One;
                for (long i = 0; i < k; i++)
                {
                    c0 *= p0[i]; c1 *= p1[i]; c2 *= p2[i]; c3 *= p3[i];
                    c4 *= p4[i]; c5 *= p5[i]; c6 *= p6[i]; c7 *= p7[i];
                }
                r[row] = T.IsNaN(c0) ? NumPyMultiplyLoops.ReduceRun(T.One, p0, 1, k) : c0;
                r[row + 1] = T.IsNaN(c1) ? NumPyMultiplyLoops.ReduceRun(T.One, p1, 1, k) : c1;
                r[row + 2] = T.IsNaN(c2) ? NumPyMultiplyLoops.ReduceRun(T.One, p2, 1, k) : c2;
                r[row + 3] = T.IsNaN(c3) ? NumPyMultiplyLoops.ReduceRun(T.One, p3, 1, k) : c3;
                r[row + 4] = T.IsNaN(c4) ? NumPyMultiplyLoops.ReduceRun(T.One, p4, 1, k) : c4;
                r[row + 5] = T.IsNaN(c5) ? NumPyMultiplyLoops.ReduceRun(T.One, p5, 1, k) : c5;
                r[row + 6] = T.IsNaN(c6) ? NumPyMultiplyLoops.ReduceRun(T.One, p6, 1, k) : c6;
                r[row + 7] = T.IsNaN(c7) ? NumPyMultiplyLoops.ReduceRun(T.One, p7, 1, k) : c7;
            }
            for (; row < rows; row++)
                r[row] = ChainProduct(a + row * k, k);
        }
    }
}
