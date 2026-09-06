using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

// =============================================================================
// Stride-native generic GEMM for all 12 NumSharp dtypes.
// =============================================================================
//
// Every dtype goes through the same stride-aware code path: direct pointer
// arithmetic with Shape.strides absorbs transposes, slicing, and offsets
// without ever materializing a contiguous copy. Float and Double flow through
// the SIMD kernel in SimdMatMul; everything else goes through the INumber<T>
// generic kernel below.
//
// Layout:
//   same-type : MatMulStridedSame<T>  — JIT-specialized per T via INumber<T>.
//               Branches once on bStride1 == 1 to give the compiler a
//               "contig-B" inner loop it can auto-vectorize.
//   mixed-type: MatMulStridedMixed<TResult> — accumulates in double using
//               typed pointer reads (no GetValue(coords)). Used when the
//               operand dtypes differ from the result dtype.
//   bool      : MatMulStridedBool — OR of ANDs; short-circuits when aik=false.
//
// All paths handle Shape.offset on the base pointer, so sliced views with
// non-zero offset work natively.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Stride-native entry point. Reads strides and offset from each
        /// array's Shape, then dispatches on (sameType, dtype) to the
        /// specialized kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedGeneric(NDArray left, NDArray right, NDArray result, long M, long K, long N)
        {
            var lShape = left.Shape;
            var rShape = right.Shape;
            long aStride0 = lShape.strides[0];
            long aStride1 = lShape.strides[1];
            long bStride0 = rShape.strides[0];
            long bStride1 = rShape.strides[1];

            bool sameType = left.typecode == result.typecode && right.typecode == result.typecode;
            if (sameType)
                MatMulStridedSameDispatch(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
            else
                MatMulStridedMixedDispatch(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
        }

        // =====================================================================
        // Same-type path: T : INumber<T> (except bool)
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedSameDispatch(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            switch (result.typecode)
            {
                case NPTypeCode.Boolean:
                    RunBool(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Byte:
                    RunSame<byte>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.SByte:
                    RunSame<sbyte>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Int16:
                    RunSame<short>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.UInt16:
                    RunSame<ushort>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Int32:
                    RunSame<int>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.UInt32:
                    RunSame<uint>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Int64:
                    RunSame<long>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.UInt64:
                    RunSame<ulong>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Char:
                    RunSame<char>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Half:
                    // NOT RunSame<Half>: NumPy accumulates half products in FLOAT32 and rounds
                    // once at the end (matmul_inner_noblas: `float sum = 0; ... *(op) =
                    // npy_float_to_half(sum)`). Half is the one dtype where that matters and the
                    // one float dtype NumPy does NOT send to cblas (USEBLAS=0 in matmul.c.src),
                    // so this loop IS the reference implementation and can be matched exactly.
                    RunHalf(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Single:
                    // Usually handled by the SIMD path in TryMatMulSimd — this
                    // branch covers the rare fall-through (ILKernel disabled etc.).
                    RunSame<float>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Double:
                    RunSame<double>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Decimal:
                    RunSame<decimal>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Complex:
                    // Complex doesn't implement INumber<Complex> (no total ordering), so use a dedicated kernel.
                    RunComplex(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                default:
                    throw new NotSupportedException($"MatMul not supported for type {result.typecode}");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RunSame<T>(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
            where T : unmanaged, INumber<T>
        {
            T* a = (T*)left.Address   + left.Shape.offset;
            T* b = (T*)right.Address  + right.Shape.offset;
            T* c = (T*)result.Address + result.Shape.offset;
            new UnmanagedSpan<T>(c, M * N).Clear();
            MatMulStridedSame(a, aStride0, aStride1, b, bStride0, bStride1, c, M, N, K);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RunBool(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            bool* a = (bool*)left.Address   + left.Shape.offset;
            bool* b = (bool*)right.Address  + right.Shape.offset;
            bool* c = (bool*)result.Address + result.Shape.offset;
            new UnmanagedSpan<bool>(c, M * N).Clear();
            MatMulStridedBool(a, aStride0, aStride1, b, bStride0, bStride1, c, M, N, K);
        }

        /// <summary>
        /// Half GEMM with a FLOAT32 accumulator — the exact shape of NumPy's half loop.
        /// Accumulating in Half instead does not merely lose a ulp, it SATURATES: half's spacing
        /// above 2048 is 2, so <c>2048 + 1 == 2048</c> and <c>ones(4096)·ones(4096)</c> stalls at
        /// 2048 where NumPy returns 4096.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RunHalf(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            Half* a = (Half*)left.Address   + left.Shape.offset;
            Half* b = (Half*)right.Address  + right.Shape.offset;
            Half* c = (Half*)result.Address + result.Shape.offset;
            new UnmanagedSpan<Half>(c, M * N).Clear();
            if (M == 0 || N == 0 || K == 0)
                return;                       // k == 0 is the empty sum: the cleared C is the answer
            MatMulStridedHalf(a, aStride0, aStride1, b, bStride0, bStride1, c, M, N, K);
        }

        /// <summary>
        /// Stride-native Half GEMM. Every product is formed in float32 — exact, since the product
        /// of two halves needs at most 22 significand bits and float32 has 24 — and accumulated in
        /// float32, with a single narrowing back to Half per output cell.
        /// <para>
        /// The accumulator is a ROW (like <see cref="MatMulStridedMixed{TResult}"/>'s), not the
        /// scalar NumPy uses, purely to keep the cache-friendly j-inner loop. The bits are
        /// identical either way: for a given output cell the additions still run over k in
        /// ascending order, which is all that fixes the result.
        /// </para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedHalf(
            Half* A, long aStride0, long aStride1,
            Half* B, long bStride0, long bStride1,
            Half* C, long M, long N, long K)
        {
            var accBuf = new float[N];
            fixed (float* accBase = accBuf)
            {
                float* acc = accBase;
                for (long i = 0; i < M; i++)
                {
                    new UnmanagedSpan<float>(acc, N).Clear();
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        float aik = (float)A[aRowBase + k * aStride1];
                        long bRowBase = k * bStride0;
                        if (bStride1 == 1)
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * (float)B[bRowBase + j];
                        }
                        else
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * (float)B[bRowBase + j * bStride1];
                        }
                    }

                    Half* cRow = C + i * N;
                    for (long j = 0; j < N; j++)
                        cRow[j] = (Half)acc[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void RunComplex(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            Complex* a = (Complex*)left.Address   + left.Shape.offset;
            Complex* b = (Complex*)right.Address  + right.Shape.offset;
            Complex* c = (Complex*)result.Address + result.Shape.offset;
            new UnmanagedSpan<Complex>(c, M * N).Clear();
            MatMulStridedComplex(a, aStride0, aStride1, b, bStride0, bStride1, c, M, N, K);
        }

        /// <summary>
        /// Stride-native same-type Complex GEMM. Mirrors MatMulStridedSame but uses
        /// Complex's built-in arithmetic operators (no INumber&lt;Complex&gt; in .NET).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedComplex(
            Complex* A, long aStride0, long aStride1,
            Complex* B, long bStride0, long bStride1,
            Complex* C, long M, long N, long K)
        {
            if (bStride1 == 1)
            {
                for (long i = 0; i < M; i++)
                {
                    Complex* cRow = C + i * N;
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        Complex aik = A[aRowBase + k * aStride1];
                        Complex* bRow = B + k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] += aik * bRow[j];
                    }
                }
            }
            else
            {
                for (long i = 0; i < M; i++)
                {
                    Complex* cRow = C + i * N;
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        Complex aik = A[aRowBase + k * aStride1];
                        long bRowBase = k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] += aik * B[bRowBase + j * bStride1];
                    }
                }
            }
        }

        /// <summary>
        /// Stride-native same-type GEMM. Two JIT-specialized loops:
        ///   bStride1 == 1 → the inner loop reads a contiguous B row, which
        ///   the JIT can auto-vectorize for primitive T.
        ///   bStride1 != 1 → fully-scalar strided access (TransB case).
        /// C is row-major contiguous, already zeroed by the caller.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedSame<T>(
            T* A, long aStride0, long aStride1,
            T* B, long bStride0, long bStride1,
            T* C, long M, long N, long K)
            where T : unmanaged, INumber<T>
        {
            if (bStride1 == 1)
            {
                for (long i = 0; i < M; i++)
                {
                    T* cRow = C + i * N;
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        T aik = A[aRowBase + k * aStride1];
                        T* bRow = B + k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] += aik * bRow[j];
                    }
                }
            }
            else
            {
                for (long i = 0; i < M; i++)
                {
                    T* cRow = C + i * N;
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        T aik = A[aRowBase + k * aStride1];
                        long bRowBase = k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] += aik * B[bRowBase + j * bStride1];
                    }
                }
            }
        }

        /// <summary>
        /// Stride-native bool matmul. NumPy semantics:
        ///   C[i,j] = OR over k of (A[i,k] AND B[k,j]).
        /// Short-circuits when A[i,k] is false (common enough to matter).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedBool(
            bool* A, long aStride0, long aStride1,
            bool* B, long bStride0, long bStride1,
            bool* C, long M, long N, long K)
        {
            if (bStride1 == 1)
            {
                for (long i = 0; i < M; i++)
                {
                    bool* cRow = C + i * N;
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        if (!A[aRowBase + k * aStride1]) continue;
                        bool* bRow = B + k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] |= bRow[j];
                    }
                }
            }
            else
            {
                for (long i = 0; i < M; i++)
                {
                    bool* cRow = C + i * N;
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        if (!A[aRowBase + k * aStride1]) continue;
                        long bRowBase = k * bStride0;
                        for (long j = 0; j < N; j++)
                            cRow[j] |= B[bRowBase + j * bStride1];
                    }
                }
            }
        }

        // =====================================================================
        // Mixed-type path — typed reads + double accumulator.
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedMixedDispatch(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            switch (result.typecode)
            {
                case NPTypeCode.Boolean:
                    MatMulStridedMixed<bool>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Byte:
                    MatMulStridedMixed<byte>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.SByte:
                    MatMulStridedMixed<sbyte>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Int16:
                    MatMulStridedMixed<short>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.UInt16:
                    MatMulStridedMixed<ushort>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Int32:
                    MatMulStridedMixed<int>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.UInt32:
                    MatMulStridedMixed<uint>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Int64:
                    MatMulStridedMixed<long>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.UInt64:
                    MatMulStridedMixed<ulong>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Char:
                    MatMulStridedMixed<char>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Half:
                    // Half needs a FLOAT32 accumulator, not the double one — see RunHalf. Reached
                    // when the operand dtypes differ but still promote to float16, i.e. half against
                    // bool / int8 / uint8 (every value of those is exact in half, so reading them
                    // through float loses nothing NumPy's own input cast would have kept).
                    MatMulStridedMixedHalf(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Single:
                    MatMulStridedMixed<float>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Double:
                    MatMulStridedMixed<double>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Decimal:
                    MatMulStridedMixed<decimal>(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                case NPTypeCode.Complex:
                    // Complex needs a Complex accumulator, not double. Use the dedicated path.
                    MatMulStridedMixedComplex(left, right, result, aStride0, aStride1, bStride0, bStride1, M, N, K);
                    break;
                default:
                    throw new NotSupportedException($"MatMul not supported for type {result.typecode}");
            }
        }

        /// <summary>
        /// Mixed-type stride-native matmul. Accumulator is double (NumPy's
        /// promotion rule for cross-type matmul). Reads operands via typed
        /// pointer arithmetic — no GetValue(coords) boxing.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedMixed<TResult>(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
            where TResult : unmanaged
        {
            TResult* c = (TResult*)result.Address + result.Shape.offset;
            void* aBase = (byte*)left.Address  + left.Shape.offset  * left.dtypesize;
            void* bBase = (byte*)right.Address + right.Shape.offset * right.dtypesize;
            var aTc = left.typecode;
            var bTc = right.typecode;

            new UnmanagedSpan<TResult>(c, M * N).Clear();

            // Single-row double accumulator, reused per i.
            var accBuf = new double[N];
            fixed (double* accBase = accBuf)
            {
                double* acc = accBase;
                for (long i = 0; i < M; i++)
                {
                    new UnmanagedSpan<double>(acc, N).Clear();
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        double aik = ReadAsDouble(aBase, aTc, aRowBase + k * aStride1);
                        long bRowBase = k * bStride0;
                        if (bStride1 == 1)
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * ReadAsDouble(bBase, bTc, bRowBase + j);
                        }
                        else
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * ReadAsDouble(bBase, bTc, bRowBase + j * bStride1);
                        }
                    }

                    TResult* cRow = c + i * N;
                    for (long j = 0; j < N; j++)
                        cRow[j] = Converts.ChangeType<TResult>(acc[j]);
                }
            }
        }

        /// <summary>
        /// Mixed-type matmul whose RESULT is Half. Identical to
        /// <see cref="MatMulStridedMixed{TResult}"/> but accumulates in float32 rather than double,
        /// because that is the width NumPy's half loop uses.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedMixedHalf(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            Half* c = (Half*)result.Address + result.Shape.offset;
            void* aBase = (byte*)left.Address  + left.Shape.offset  * left.dtypesize;
            void* bBase = (byte*)right.Address + right.Shape.offset * right.dtypesize;
            var aTc = left.typecode;
            var bTc = right.typecode;

            new UnmanagedSpan<Half>(c, M * N).Clear();
            if (M == 0 || N == 0 || K == 0)
                return;

            var accBuf = new float[N];
            fixed (float* accBase = accBuf)
            {
                float* acc = accBase;
                for (long i = 0; i < M; i++)
                {
                    new UnmanagedSpan<float>(acc, N).Clear();
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        float aik = (float)ReadAsDouble(aBase, aTc, aRowBase + k * aStride1);
                        long bRowBase = k * bStride0;
                        if (bStride1 == 1)
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * (float)ReadAsDouble(bBase, bTc, bRowBase + j);
                        }
                        else
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * (float)ReadAsDouble(bBase, bTc, bRowBase + j * bStride1);
                        }
                    }

                    Half* cRow = c + i * N;
                    for (long j = 0; j < N; j++)
                        cRow[j] = (Half)acc[j];
                }
            }
        }

        /// <summary>
        /// Reads element at <paramref name="idx"/> from a typed buffer, returns
        /// as double. JIT eliminates the non-matching branches per call site
        /// when <paramref name="tc"/> is enregistered.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe double ReadAsDouble(void* basePtr, NPTypeCode tc, long idx)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return ((bool*)basePtr)[idx] ? 1.0 : 0.0;
                case NPTypeCode.Byte:    return ((byte*)basePtr)[idx];
                case NPTypeCode.SByte:   return ((sbyte*)basePtr)[idx];
                case NPTypeCode.Int16:   return ((short*)basePtr)[idx];
                case NPTypeCode.UInt16:  return ((ushort*)basePtr)[idx];
                case NPTypeCode.Int32:   return ((int*)basePtr)[idx];
                case NPTypeCode.UInt32:  return ((uint*)basePtr)[idx];
                case NPTypeCode.Int64:   return ((long*)basePtr)[idx];
                case NPTypeCode.UInt64:  return ((ulong*)basePtr)[idx];
                case NPTypeCode.Char:    return ((char*)basePtr)[idx];
                case NPTypeCode.Half:    return (double)((Half*)basePtr)[idx];
                case NPTypeCode.Single:  return ((float*)basePtr)[idx];
                case NPTypeCode.Double:  return ((double*)basePtr)[idx];
                case NPTypeCode.Decimal: return (double)((decimal*)basePtr)[idx];
                case NPTypeCode.Complex: return ((Complex*)basePtr)[idx].Real;
                default: throw new NotSupportedException($"Unsupported type {tc}");
            }
        }

        /// <summary>
        /// Reads an element and returns it as Complex. Used by the Complex mixed-type matmul
        /// kernel to preserve imaginary components.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe Complex ReadAsComplex(void* basePtr, NPTypeCode tc, long idx)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return new Complex(((bool*)basePtr)[idx] ? 1.0 : 0.0, 0);
                case NPTypeCode.Byte:    return new Complex(((byte*)basePtr)[idx], 0);
                case NPTypeCode.SByte:   return new Complex(((sbyte*)basePtr)[idx], 0);
                case NPTypeCode.Int16:   return new Complex(((short*)basePtr)[idx], 0);
                case NPTypeCode.UInt16:  return new Complex(((ushort*)basePtr)[idx], 0);
                case NPTypeCode.Int32:   return new Complex(((int*)basePtr)[idx], 0);
                case NPTypeCode.UInt32:  return new Complex(((uint*)basePtr)[idx], 0);
                case NPTypeCode.Int64:   return new Complex(((long*)basePtr)[idx], 0);
                case NPTypeCode.UInt64:  return new Complex(((ulong*)basePtr)[idx], 0);
                case NPTypeCode.Char:    return new Complex(((char*)basePtr)[idx], 0);
                case NPTypeCode.Half:    return new Complex((double)((Half*)basePtr)[idx], 0);
                case NPTypeCode.Single:  return new Complex(((float*)basePtr)[idx], 0);
                case NPTypeCode.Double:  return new Complex(((double*)basePtr)[idx], 0);
                case NPTypeCode.Decimal: return new Complex((double)((decimal*)basePtr)[idx], 0);
                case NPTypeCode.Complex: return ((Complex*)basePtr)[idx];
                default: throw new NotSupportedException($"Unsupported type {tc}");
            }
        }

        /// <summary>
        /// Complex-specific mixed-type matmul. Uses Complex accumulator so the imaginary
        /// component is preserved — matches NumPy's complex matmul semantics.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatMulStridedMixedComplex(
            NDArray left, NDArray right, NDArray result,
            long aStride0, long aStride1, long bStride0, long bStride1,
            long M, long N, long K)
        {
            Complex* c = (Complex*)result.Address + result.Shape.offset;
            void* aBase = (byte*)left.Address  + left.Shape.offset  * left.dtypesize;
            void* bBase = (byte*)right.Address + right.Shape.offset * right.dtypesize;
            var aTc = left.typecode;
            var bTc = right.typecode;

            new UnmanagedSpan<Complex>(c, M * N).Clear();

            var accBuf = new Complex[N];
            fixed (Complex* accBase = accBuf)
            {
                Complex* acc = accBase;
                for (long i = 0; i < M; i++)
                {
                    new UnmanagedSpan<Complex>(acc, N).Clear();
                    long aRowBase = i * aStride0;
                    for (long k = 0; k < K; k++)
                    {
                        Complex aik = ReadAsComplex(aBase, aTc, aRowBase + k * aStride1);
                        long bRowBase = k * bStride0;
                        if (bStride1 == 1)
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * ReadAsComplex(bBase, bTc, bRowBase + j);
                        }
                        else
                        {
                            for (long j = 0; j < N; j++)
                                acc[j] += aik * ReadAsComplex(bBase, bTc, bRowBase + j * bStride1);
                        }
                    }

                    Complex* cRow = c + i * N;
                    for (long j = 0; j < N; j++)
                        cRow[j] = acc[j];
                }
            }
        }
    }
}
