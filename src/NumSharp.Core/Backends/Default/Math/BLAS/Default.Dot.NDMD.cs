using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// N-D x M-D dot product: sum product over last axis of lhs and second-to-last axis of rhs.
        /// dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        /// </summary>
        /// <remarks>
        /// This replaces the original ~15,880 line implementation that used nested switch statements
        /// for every combination of dimension counts. The new implementation uses dynamic iteration
        /// with optional SIMD optimization for contiguous float/double arrays.
        /// </remarks>
        public static NDArray DotNDMD(NDArray lhs, NDArray rhs)
        {
#if MINIMAL
            return null;
#else
            // Validate contracting dimension
            long contractDim = lhs.Shape[-1];
            if (contractDim != rhs.Shape[-2])
                throw new IncorrectShapeException(
                    $"shapes {lhs.Shape.ToPythonTuple()} and {rhs.Shape.ToPythonTuple()} not aligned: {contractDim} (dim {lhs.ndim - 1}) != {rhs.Shape[-2]} (dim {rhs.ndim - 2})");

            // Compute output shape:
            // lhs shape without last dim + rhs shape without second-to-last dim
            long[] lshape = lhs.Shape.dimensions.RemoveAt(lhs.ndim - 1);  // All but last
            long[] rshape = rhs.Shape.dimensions.RemoveAt(rhs.ndim - 2);  // All but second-to-last

            var retShape = new long[lshape.Length + rshape.Length];
            Array.Copy(lshape, 0, retShape, 0, lshape.Length);
            Array.Copy(rshape, 0, retShape, lshape.Length, rshape.Length);

            var resultType = np._FindCommonArrayType(lhs.typecode, rhs.typecode);
            var result = new NDArray(resultType, Shape.Adopt(retShape));

            // Try SIMD fast path for contiguous same-type float/double
            if (TryDotNDMDSimd(lhs, rhs, result, lshape, rshape, contractDim))
                return result;

            // NumPy's N-D route (PyArray_MatrixProduct2): both operands converted to the result dtype (PyArray_FromAny),
            // then the dtype's dot function once per output element over the contracted axes (DotNDMDStrided).
            NDArray ap1 = lhs.astype(result.dtype, false), ap2 = rhs.astype(result.dtype, false);
            try
            {
                DotNDMDStrided(ap1, ap2, result, lshape, rshape, contractDim);
            }
            finally
            {
                // astype(copy: false) is the operand itself when it already has the result dtype; a converted copy is
                // this call's intermediate, released here because the engine may run outside any NDScope.
                if (!ReferenceEquals(ap1, lhs))
                    ap1.Dispose();
                if (!ReferenceEquals(ap2, rhs))
                    ap2.Dispose();
            }
            return result;
#endif
        }

        /// <summary>
        /// SIMD-optimized N-D dot product for contiguous float/double arrays.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe bool TryDotNDMDSimd(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            // Require contiguous arrays and same type
            if (!lhs.Shape.IsContiguous || !rhs.Shape.IsContiguous || !result.Shape.IsContiguous)
                return false;

            var typeCode = result.typecode;
            if (lhs.typecode != typeCode || rhs.typecode != typeCode)
                return false;

            switch (typeCode)
            {
                case NPTypeCode.Single:
                    DotNDMDSimdFloat(lhs, rhs, result, lshape, rshape, K);
                    return true;

                case NPTypeCode.Double:
                    DotNDMDSimdDouble(lhs, rhs, result, lshape, rshape, K);
                    return true;

                default:
                    return false;
            }
        }

        /// <summary>
        /// SIMD float implementation of N-D dot product.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DotNDMDSimdFloat(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            float* lhsPtr = (float*)lhs.Address;
            float* rhsPtr = (float*)rhs.Address;
            float* resPtr = (float*)result.Address;

            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            // Pre-compute strides for iteration
            long lhsInnerStride = lhs.Shape.strides[lhsNdim - 1];  // Stride along contracting dim in lhs
            long rhsContractStride = rhs.Shape.strides[rhsNdim - 2];  // Stride along contracting dim in rhs
            long rhsInnerStride = rhs.Shape.strides[rhsNdim - 1];  // Stride along last dim in rhs

            // Total elements to compute
            long totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            long totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            // Compute lhs strides (for multi-index iteration)
            long[] lhsIterStrides = ComputeIterStrides64(lshape);
            long[] rhsIterStrides = ComputeIterStrides64(rshape);

            // For each position in lhs (excluding contract dim)
            for (long li = 0; li < totalLhs; li++)
            {
                // Compute lhs base offset: position in lhs without last dim
                long lhsBase = ComputeBaseOffset64(li, lhsIterStrides, lhs.Shape.strides, lshape.Length);

                // For each position in rhs (excluding contract dim)
                for (long ri = 0; ri < totalRhs; ri++)
                {
                    // Compute rhs base offset: we need to skip the contract dimension
                    long rhsBase = ComputeRhsBaseOffset64(ri, rhsIterStrides, rhs.Shape.strides, rshape, rhsNdim);

                    // Compute dot product along contracting dimension
                    float sum = DotProductFloat(
                        lhsPtr + lhsBase, lhsInnerStride,
                        rhsPtr + rhsBase, rhsContractStride,
                        K);

                    // Store result
                    resPtr[li * totalRhs + ri] = sum;
                }
            }
        }

        /// <summary>
        /// SIMD double implementation of N-D dot product.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DotNDMDSimdDouble(NDArray lhs, NDArray rhs, NDArray result,
            long[] lshape, long[] rshape, long K)
        {
            double* lhsPtr = (double*)lhs.Address;
            double* rhsPtr = (double*)rhs.Address;
            double* resPtr = (double*)result.Address;

            int lhsNdim = lhs.ndim;
            int rhsNdim = rhs.ndim;

            long lhsInnerStride = lhs.Shape.strides[lhsNdim - 1];
            long rhsContractStride = rhs.Shape.strides[rhsNdim - 2];
            long rhsInnerStride = rhs.Shape.strides[rhsNdim - 1];

            long totalLhs = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalLhs *= lshape[i];

            long totalRhs = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalRhs *= rshape[i];

            long[] lhsIterStrides = ComputeIterStrides64(lshape);
            long[] rhsIterStrides = ComputeIterStrides64(rshape);

            for (long li = 0; li < totalLhs; li++)
            {
                long lhsBase = ComputeBaseOffset64(li, lhsIterStrides, lhs.Shape.strides, lshape.Length);

                for (long ri = 0; ri < totalRhs; ri++)
                {
                    long rhsBase = ComputeRhsBaseOffset64(ri, rhsIterStrides, rhs.Shape.strides, rshape, rhsNdim);

                    double sum = DotProductDouble(
                        lhsPtr + lhsBase, lhsInnerStride,
                        rhsPtr + rhsBase, rhsContractStride,
                        K);

                    resPtr[li * totalRhs + ri] = sum;
                }
            }
        }

        /// <summary>
        /// Compute iteration strides for multi-index decomposition (64-bit).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long[] ComputeIterStrides64(long[] shape)
        {
            var strides = new long[shape.Length];
            long stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        /// <summary>
        /// Compute base offset for lhs array from linear index (64-bit).
        /// Maps linear index over lshape to offset in lhs storage (using lhs strides, not contract dim).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long ComputeBaseOffset64(long linearIdx, long[] iterStrides, long[] arrayStrides, int ndim)
        {
            long offset = 0;
            for (int d = 0; d < ndim; d++)
            {
                long idx = linearIdx / iterStrides[d];
                linearIdx %= iterStrides[d];
                offset += idx * arrayStrides[d];
            }
            return offset;
        }

        /// <summary>
        /// Compute base offset for rhs array from linear index (64-bit).
        /// rshape excludes the contracting dimension (second-to-last in original rhs).
        /// We need to map indices back, skipping the contract dim.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long ComputeRhsBaseOffset64(long linearIdx, long[] iterStrides, long[] arrayStrides, long[] rshape, int rhsNdim)
        {
            long offset = 0;
            int contractDimIdx = rhsNdim - 2;  // The dimension we're contracting over

            int rshapeIdx = 0;
            for (int d = 0; d < rhsNdim; d++)
            {
                if (d == contractDimIdx)
                    continue;  // Skip contract dimension

                long idx = linearIdx / iterStrides[rshapeIdx];
                linearIdx %= iterStrides[rshapeIdx];
                offset += idx * arrayStrides[d];
                rshapeIdx++;
            }
            return offset;
        }

        /// <summary>
        /// SIMD dot product for float with arbitrary strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotProductFloat(float* a, long strideA, float* b, long strideB, long n)
        {
            float sum = 0;

            // Fast path: both contiguous (stride=1)
            if (strideA == 1 && strideB == 1)
            {
                long i = 0;

                // SIMD loop
                if (Vector256.IsHardwareAccelerated && n >= 8)
                {
                    var vsum = Vector256<float>.Zero;
                    for (; i <= n - 8; i += 8)
                    {
                        var va = Vector256.Load(a + i);
                        var vb = Vector256.Load(b + i);
                        vsum = Vector256.Add(vsum, Vector256.Multiply(va, vb));
                    }
                    sum = Vector256.Sum(vsum);
                }

                // Scalar remainder
                for (; i < n; i++)
                    sum += a[i] * b[i];
            }
            else
            {
                // Strided access
                for (long i = 0; i < n; i++)
                    sum += a[i * strideA] * b[i * strideB];
            }

            return sum;
        }

        /// <summary>
        ///     NumPy's <c>DOUBLE_dot</c> over two strided runs. Blasable strides (<c>blas_stride</c>: positive; the 64-bit
        ///     CBLAS integers of scipy-openblas64 take any positive element stride and need no chunking) go to
        ///     <c>cblas_ddot</c> and the result is added into a zeroed double; any other stride (negative, or a broadcast
        ///     zero) runs DOUBLE_dot's own sequential loop.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="strideA">Its stride (elements).</param>
        /// <param name="b">The second operand.</param>
        /// <param name="strideB">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        /// <remarks>
        ///     scipy-openblas' ddot, the build NumPy 2.4.2 ships: with two unit strides, its vector kernel over the first
        ///     <c>n &amp; -16</c> elements and the rest added in order, so below 16 terms a plain sequential sum (exact here);
        ///     the vector kernel's own order only the backend reproduces, so from 16 terms on the four-lane sum below is
        ///     bounded, not NumPy's bits. With any other blasable stride, its strided loop: groups of four products
        ///     <c>m1..m4</c> added as <c>t1 += m1 + m3; t2 += m2 + m4</c>, the remainder into <c>t1</c>, and
        ///     <c>t1 + t2</c> - probed against NumPy 2.4.2 (a unit-stride run against a stride-3 one, 3 to 100 terms): 84 of
        ///     84 results, where a sequential sum matched 29. The former code summed every strided pair sequentially.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double DotProductDouble(double* a, long strideA, double* b, long strideB, long n)
        {
            if (strideA > 0 && strideB > 0)
            {
                double dot = strideA == 1 && strideB == 1 ? UnitStrideDdot(a, b, n) : StridedDdot(a, strideA, b, strideB, n);
                // DOUBLE_dot: `double sum = 0.; sum += cblas_ddot(...)` over its one chunk.
                double sum = 0.0;
                sum += dot;
                return sum;
            }

            // DOUBLE_dot without cblas: sum += ip1 * ip2 from +0, the pointers stepped by their strides (ip1 += is1).
            double plain = 0;
            for (long i = 0; i < n; i++, a += strideA, b += strideB)
                plain += *a * *b;
            return plain;
        }

        /// <summary>scipy-openblas' unit-stride ddot: sequential below its vector block of 16 terms (NumPy's bits), the
        ///     former four-lane SIMD sum from there on (bounded).</summary>
        /// <param name="a">The first operand (unit stride).</param>
        /// <param name="b">The second operand (unit stride).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        private static unsafe double UnitStrideDdot(double* a, double* b, long n)
        {
            double sum = 0;
            long i = 0;
            if (n >= 16 && Vector256.IsHardwareAccelerated)
            {
                var vsum = Vector256<double>.Zero;
                for (; i <= n - 4; i += 4)
                {
                    var va = Vector256.Load(a + i);
                    var vb = Vector256.Load(b + i);
                    vsum = Vector256.Add(vsum, Vector256.Multiply(va, vb));
                }
                sum = Vector256.Sum(vsum);
            }

            for (; i < n; i++)
                sum += a[i] * b[i];
            return sum;
        }

        /// <summary>scipy-openblas' strided ddot loop: four products a group, <c>t1 += m1 + m3; t2 += m2 + m4</c>, the
        ///     remainder into <c>t1</c>, then <c>t1 + t2</c> (see <see cref="DotProductDouble"/>).</summary>
        /// <param name="a">The first operand (NumPy's ip1, cblas' x).</param>
        /// <param name="sa">Its stride (elements, positive).</param>
        /// <param name="b">The second operand (ip2, cblas' y).</param>
        /// <param name="sb">Its stride (elements, positive).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        /// <remarks>RyuJIT never contracts a multiply and an add into a fused multiply-add, and neither did the kernel NumPy
        ///     loads (the probe matched the separate roundings, tails included).</remarks>
        private static unsafe double StridedDdot(double* a, long sa, double* b, long sb, long n)
        {
            double t1 = 0.0, t2 = 0.0;
            // The kernel's own pointer stepping: the group's four elements at 0..3 strides, then both pointers moved on.
            long sa2 = 2 * sa, sa3 = 3 * sa, sa4 = 4 * sa, sb2 = 2 * sb, sb3 = 3 * sb, sb4 = 4 * sb;
            long i = 0;
            long n1 = n & -4;
            for (; i < n1; i += 4, a += sa4, b += sb4)
            {
                double m1 = b[0] * a[0];
                double m2 = b[sb] * a[sa];
                double m3 = b[sb2] * a[sa2];
                double m4 = b[sb3] * a[sa3];
                t1 += m1 + m3;
                t2 += m2 + m4;
            }

            for (; i < n; i++, a += sa, b += sb)
                t1 += *b * *a;
            return t1 + t2;
        }

        /// <summary>
        ///     NumPy's N-D route of <c>PyArray_MatrixProduct2</c> for the operands <see cref="TryDotNDMDSimd"/> does not take:
        ///     the dtype's dot function (<c>PyArray_ArrFuncs.dotfunc</c>) once per output element, over the contracted
        ///     axes' strides - the last axis of <paramref name="ap1"/> and the second-to-last of <paramref name="ap2"/> - into
        ///     the C-ordered result, in the order NumPy's two all-but-axis iterators visit them.
        /// </summary>
        /// <param name="ap1">The first operand, already of the result dtype (NumPy's <c>PyArray_FromAny</c>).</param>
        /// <param name="ap2">The second operand, already of the result dtype.</param>
        /// <param name="result">The fresh C-contiguous result.</param>
        /// <param name="lshape"><paramref name="ap1"/>'s dims without the last.</param>
        /// <param name="rshape"><paramref name="ap2"/>'s dims without the second-to-last.</param>
        /// <param name="K">The contracted length.</param>
        /// <exception cref="NotSupportedException">A dtype with no dot function.</exception>
        /// <remarks>
        ///     The dot functions (<c>arraytypes.c.src</c>, and scipy-openblas behind the BLAS ones):
        ///     <list type="bullet">
        ///         <item>bool: <c>BOOL_dot</c>, true at the first pair of true elements;</item>
        ///         <item>the integers: the products summed in a C <c>long</c> / <c>long long</c> and the sum truncated to
        ///         the dtype, which is the sum modulo 2^bits: summed here in the dtype's own wrapping arithmetic, the same
        ///         value (the former route summed in double, so a sum past 2^53, or a uint64 product, lost its bits);</item>
        ///         <item>float16: <c>HALF_dot</c>, a sequential float32 sum of float32 products rounded once
        ///         (<see cref="NDHalfDot"/>);</item>
        ///         <item>float64: <c>DOUBLE_dot</c> (<see cref="DotProductDouble"/>);</item>
        ///         <item>complex128: <c>CDOUBLE_dot</c> (<see cref="ComplexDot"/>; the former route summed the real parts
        ///         only);</item>
        ///         <item>float32: not NumPy's - FLOAT_dot's blasable strides reach cblas_sdot, whose strided kernel no
        ///         managed model was found to reproduce (seven candidate orders probed against NumPy 2.4.2, none exact), so
        ///         the former double sum of the products stands; the backend gives NumPy's bits;</item>
        ///         <item>decimal (no NumPy analog): an exact decimal sum.</item>
        ///     </list>
        /// </remarks>
        private static unsafe void DotNDMDStrided(NDArray ap1, NDArray ap2, NDArray result, long[] lshape, long[] rshape,
            long K)
        {
            switch (result.typecode)
            {
                case NPTypeCode.Boolean: Walk<bool>(ap1, ap2, result, lshape, rshape, K, &BoolDot); return;
                case NPTypeCode.Byte: Walk<byte>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<byte>); return;
                case NPTypeCode.SByte: Walk<sbyte>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<sbyte>); return;
                case NPTypeCode.Int16: Walk<short>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<short>); return;
                case NPTypeCode.UInt16: Walk<ushort>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<ushort>); return;
                case NPTypeCode.Int32: Walk<int>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<int>); return;
                case NPTypeCode.UInt32: Walk<uint>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<uint>); return;
                case NPTypeCode.Int64: Walk<long>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<long>); return;
                case NPTypeCode.UInt64: Walk<ulong>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<ulong>); return;
                case NPTypeCode.Char: Walk<char>(ap1, ap2, result, lshape, rshape, K, &IntegerDot<char>); return;
                case NPTypeCode.Half: Walk<Half>(ap1, ap2, result, lshape, rshape, K, &HalfDot); return;
                case NPTypeCode.Single: Walk<float>(ap1, ap2, result, lshape, rshape, K, &SingleDot); return;
                case NPTypeCode.Double: Walk<double>(ap1, ap2, result, lshape, rshape, K, &DotProductDouble); return;
                case NPTypeCode.Decimal: Walk<decimal>(ap1, ap2, result, lshape, rshape, K, &DecimalDot); return;
                case NPTypeCode.Complex: Walk<Complex>(ap1, ap2, result, lshape, rshape, K, &ComplexDot); return;
                default: throw new NotSupportedException($"np.dot of {result.typecode} operands");
            }
        }

        /// <summary>
        ///     The output walk of <see cref="DotNDMDStrided"/>: for each position of <paramref name="ap1"/> but its last axis
        ///     (C order), each position of <paramref name="ap2"/> but its second-to-last (C order), one dot over the contracted
        ///     axes written to the next element of the C-ordered result - NumPy's <c>op += os</c> under its two iterators.
        /// </summary>
        /// <typeparam name="T">The element type of the result dtype.</typeparam>
        /// <param name="ap1">The first operand (of <typeparamref name="T"/>).</param>
        /// <param name="ap2">The second operand (of <typeparamref name="T"/>).</param>
        /// <param name="result">The fresh C-contiguous result (of <typeparamref name="T"/>).</param>
        /// <param name="lshape"><paramref name="ap1"/>'s dims without the last.</param>
        /// <param name="rshape"><paramref name="ap2"/>'s dims without the second-to-last.</param>
        /// <param name="K">The contracted length.</param>
        /// <param name="dot">The dtype's dot function: first operand, its stride, second operand, its stride, length (strides
        ///     in elements).</param>
        /// <remarks>
        ///     The operands are addressed from their storage plus the view's offset, so any layout - strided, reversed,
        ///     broadcast - is read through its own strides; the arrays are the caller's parameters and locals for the whole
        ///     walk, so the raw pointers stay valid. Each operand's position moves like NumPy's <c>PyArray_ITER_NEXT</c>: a
        ///     coordinate odometer over the iterated axes that adds a stride per step and takes back a whole axis on a carry,
        ///     so an output costs one dot call and no division.
        /// </remarks>
        private static unsafe void Walk<T>(NDArray ap1, NDArray ap2, NDArray result, long[] lshape, long[] rshape, long K,
            delegate*<T*, long, T*, long, long, T> dot) where T : unmanaged
        {
            int lnd = ap1.ndim, rnd = ap2.ndim;
            long[] lst = ap1.Shape.strides, rst = ap2.Shape.strides;
            T* lp = (T*)ap1.Storage.Address + ap1.Shape.offset;
            T* rp = (T*)ap2.Storage.Address + ap2.Shape.offset;
            T* op = (T*)result.Storage.Address;
            long ls = lst[lnd - 1];
            long rs = rst[rnd - 2];

            long totalL = 1;
            for (int i = 0; i < lshape.Length; i++)
                totalL *= lshape[i];
            long totalR = 1;
            for (int i = 0; i < rshape.Length; i++)
                totalR *= rshape[i];
            if (totalL == 0 || totalR == 0)
                return;   // an empty result: nothing to write

            // The strides of the iterated axes in C order, as PyArray_IterAllButAxis leaves them: ap1's axes but the last
            // (lshape), ap2's axes but the second-to-last (rshape). Ranks past 16 take heap arrays (NumSharp has no rank
            // limit); the coordinates start at zero and are reset at every pass of the inner walk.
            int lk = lshape.Length, rk = rshape.Length;
            Span<long> lStride = lk <= 16 ? stackalloc long[16] : new long[lk];
            Span<long> rStride = rk <= 16 ? stackalloc long[16] : new long[rk];
            Span<long> lc = lk <= 16 ? stackalloc long[16] : new long[lk];
            Span<long> rc = rk <= 16 ? stackalloc long[16] : new long[rk];
            for (int d = 0; d < lk; d++)
                lStride[d] = lst[d];
            for (int d = 0, j = 0; d < rnd; d++)
                if (d != rnd - 2)
                    rStride[j++] = rst[d];
            lc.Clear();

            long lOff = 0;
            for (long li = 0; li < totalL; li++)
            {
                rc.Clear();
                long rOff = 0;
                for (long ri = 0; ri < totalR; ri++)
                {
                    *op++ = dot(lp + lOff, ls, rp + rOff, rs, K);
                    for (int d = rk - 1; d >= 0; d--)
                    {
                        rOff += rStride[d];
                        if (++rc[d] < rshape[d])
                            break;
                        rOff -= rStride[d] * rshape[d];
                        rc[d] = 0;
                    }
                }

                for (int d = lk - 1; d >= 0; d--)
                {
                    lOff += lStride[d];
                    if (++lc[d] < lshape[d])
                        break;
                    lOff -= lStride[d] * lshape[d];
                    lc[d] = 0;
                }
            }
        }

        /// <summary>NumPy's <c>BOOL_dot</c>: true at the first index where both elements are true.</summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>Whether any pair is (true, true).</returns>
        private static unsafe bool BoolDot(bool* a, long sa, bool* b, long sb, long n)
        {
            for (long i = 0; i < n; i++, a += sa, b += sb)
                if (*a && *b)
                    return true;
            return false;
        }

        /// <summary>
        ///     NumPy's integer dot functions (<c>BYTE_dot</c> … <c>ULONGLONG_dot</c>): the products summed in a C
        ///     <c>long</c> (<c>long long</c> for the 64-bit types) and the sum cast to the dtype. Integer arithmetic wraps,
        ///     so that is the exact sum of the products modulo 2^bits of the dtype, whatever the width of the accumulator:
        ///     summed here in <typeparamref name="T"/>'s own wrapping arithmetic.
        /// </summary>
        /// <typeparam name="T">The integer element type (char: NumSharp's unsigned 16-bit dtype).</typeparam>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The sum of the products, wrapped to <typeparamref name="T"/>.</returns>
        private static unsafe T IntegerDot<T>(T* a, long sa, T* b, long sb, long n) where T : unmanaged, IBinaryInteger<T>
        {
            // Four independent sums, so consecutive multiply-adds do not wait on one accumulator. Wrapping addition is
            // associative and commutative (arithmetic modulo 2^bits), so any grouping gives NumPy's exact value.
            T s0 = T.Zero, s1 = T.Zero, s2 = T.Zero, s3 = T.Zero;
            long sa2 = 2 * sa, sa3 = 3 * sa, sa4 = 4 * sa, sb2 = 2 * sb, sb3 = 3 * sb, sb4 = 4 * sb;
            long i = 0;
            for (; i + 4 <= n; i += 4, a += sa4, b += sb4)
            {
                s0 = unchecked(s0 + a[0] * b[0]);
                s1 = unchecked(s1 + a[sa] * b[sb]);
                s2 = unchecked(s2 + a[sa2] * b[sb2]);
                s3 = unchecked(s3 + a[sa3] * b[sb3]);
            }

            for (; i < n; i++, a += sa, b += sb)
                s0 = unchecked(s0 + *a * *b);
            return unchecked(s0 + s1 + (s2 + s3));
        }

        /// <summary>NumPy's <c>HALF_dot</c>: float32 products summed in float32, rounded once to float16
        ///     (<see cref="NDHalfDot.Sum"/>, which keeps HALF_dot's NaN operand order).</summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        private static unsafe Half HalfDot(Half* a, long sa, Half* b, long sb, long n) => (Half)NDHalfDot.Sum(a, sa, b, sb, n);

        /// <summary>
        ///     The float32 dot of the strided route: the products and their sum in double, rounded once to float32 - NumSharp's
        ///     former arithmetic, kept because NumPy's FLOAT_dot reaches cblas_sdot and no managed model of its strided
        ///     kernel was found exact (seven candidate orders probed). Bounded, not NumPy's bits; the backend gives them.
        /// </summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        private static unsafe float SingleDot(float* a, long sa, float* b, long sb, long n)
        {
            double sum = 0;
            for (long i = 0; i < n; i++, a += sa, b += sb)
                sum += (double)*a * *b;
            return (float)sum;
        }

        /// <summary>The decimal dot (no NumPy analog): an exact decimal sum of the products.</summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand.</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        /// <exception cref="OverflowException">The sum leaves decimal's range.</exception>
        private static unsafe decimal DecimalDot(decimal* a, long sa, decimal* b, long sb, long n)
        {
            decimal s = 0m;
            for (long i = 0; i < n; i++, a += sa, b += sb)
                s += *a * *b;
            return s;
        }

        /// <summary>
        ///     NumPy's <c>CDOUBLE_dot</c>. Blasable strides (positive; see <see cref="DotProductDouble"/>) go to
        ///     <c>cblas_zdotu_sub</c>: scipy-openblas' zdotu keeps four separate double sums (<c>Σ ar·br</c>, <c>Σ ai·bi</c>,
        ///     <c>Σ ar·bi</c>, <c>Σ ai·br</c>) - at any length when a stride is not 1, below its vector block of eight when
        ///     both are - and returns <c>re + im·I</c>, which CDOUBLE_dot adds into a zeroed pair
        ///     (<see cref="SlidingDot.ZdotuResult(double, double)"/>). Any other stride runs CDOUBLE_dot's own loop.
        /// </summary>
        /// <param name="a">The first operand.</param><param name="sa">Its stride (elements).</param>
        /// <param name="b">The second operand (not conjugated).</param><param name="sb">Its stride (elements).</param>
        /// <param name="n">The length.</param>
        /// <returns>The dot product.</returns>
        /// <remarks>Probed against NumPy 2.4.2: a strided pair of 3 to 100 terms matches the four sums on every result (84 of
        ///     84), the naive per-term complex product on 6 of them. Two unit strides past seven terms meet zdotu's vector
        ///     kernel, whose order only the backend reproduces: the four sums are bounded there.</remarks>
        private static unsafe Complex ComplexDot(Complex* a, long sa, Complex* b, long sb, long n)
        {
            if (sa > 0 && sb > 0)
            {
                double d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (long i = 0; i < n; i++, a += sa, b += sb)
                {
                    var x = *a;
                    var y = *b;
                    d0 += x.Real * y.Real;
                    d1 += x.Imaginary * y.Imaginary;
                    d2 += x.Real * y.Imaginary;
                    d3 += x.Imaginary * y.Real;
                }
                return SlidingDot.ZdotuResult(d0 - d1, d2 + d3);
            }

            // CDOUBLE_dot without cblas: sumr += ar*br - ai*bi; sumi += ar*bi + ai*br, from +0.
            double sumr = 0.0, sumi = 0.0;
            for (long i = 0; i < n; i++, a += sa, b += sb)
            {
                var x = *a;
                var y = *b;
                sumr += x.Real * y.Real - x.Imaginary * y.Imaginary;
                sumi += x.Real * y.Imaginary + x.Imaginary * y.Real;
            }
            return new Complex(sumr, sumi);
        }
    }
}
