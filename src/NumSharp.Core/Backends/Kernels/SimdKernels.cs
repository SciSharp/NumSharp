using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// SIMD-optimized binary operation kernels.
    /// Each method implements all execution paths (FULL, SCALAR, CHUNK, GENERAL)
    /// and dispatches internally based on stride analysis.
    /// </summary>
    public static partial class SimdKernels
    {
        #region Int32 Add

        /// <summary>
        /// SIMD-optimized addition for Int32 arrays.
        /// </summary>
        public static unsafe void Add_Int32(
            int* lhs, int* rhs, int* result,
            int* lhsStrides, int* rhsStrides, int* shape,
            int ndim, int totalSize)
        {
            var path = StrideDetector.Classify<int>(lhsStrides, rhsStrides, shape, ndim);

            switch (path)
            {
                case ExecutionPath.SimdFull:
                    SimdFull_Add_Int32(lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarRight:
                    SimdScalarRight_Add_Int32(lhs, *rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarLeft:
                    SimdScalarLeft_Add_Int32(*lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdChunk:
                    SimdChunk_Add_Int32(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim);
                    break;
                default:
                    General_Add_Int32(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim, totalSize);
                    break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdFull_Add_Int32(int* lhs, int* rhs, int* result, int totalSize)
        {
            int i = 0;
            int vectorEnd = totalSize - Vector256<int>.Count;

            for (; i <= vectorEnd; i += Vector256<int>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(vl + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + rhs[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarRight_Add_Int32(int* lhs, int scalar, int* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<int>.Count;

            for (; i <= vectorEnd; i += Vector256<int>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                Vector256.Store(vl + scalarVec, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + scalar;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarLeft_Add_Int32(int scalar, int* rhs, int* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<int>.Count;

            for (; i <= vectorEnd; i += Vector256<int>.Count)
            {
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(scalarVec + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = scalar + rhs[i];
        }

        private static unsafe void SimdChunk_Add_Int32(
            int* lhs, int* rhs, int* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim)
        {
            int innerSize = shape[ndim - 1];
            int outerSize = 1;
            for (int d = 0; d < ndim - 1; d++)
                outerSize *= shape[d];

            int lhsInner = lhsStrides[ndim - 1];
            int rhsInner = rhsStrides[ndim - 1];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                int idx = outer;
                for (int d = ndim - 2; d >= 0; d--)
                {
                    int coord = idx % shape[d];
                    idx /= shape[d];
                    lhsOffset += coord * lhsStrides[d];
                    rhsOffset += coord * rhsStrides[d];
                }

                int* lhsRow = lhs + lhsOffset;
                int* rhsRow = rhs + rhsOffset;
                int* resultRow = result + outer * innerSize;

                if (lhsInner == 1 && rhsInner == 1)
                    SimdFull_Add_Int32(lhsRow, rhsRow, resultRow, innerSize);
                else if (rhsInner == 0)
                    SimdScalarRight_Add_Int32(lhsRow, *rhsRow, resultRow, innerSize);
                else if (lhsInner == 0)
                    SimdScalarLeft_Add_Int32(*lhsRow, rhsRow, resultRow, innerSize);
                else
                {
                    for (int i = 0; i < innerSize; i++)
                        resultRow[i] = lhsRow[i * lhsInner] + rhsRow[i * rhsInner];
                }
            }
        }

        private static unsafe void General_Add_Int32(
            int* lhs, int* rhs, int* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize)
        {
            Span<int> coords = stackalloc int[ndim];

            for (int i = 0; i < totalSize; i++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    lhsOffset += coords[d] * lhsStrides[d];
                    rhsOffset += coords[d] * rhsStrides[d];
                }

                result[i] = lhs[lhsOffset] + rhs[rhsOffset];

                for (int d = ndim - 1; d >= 0; d--)
                {
                    if (++coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        #endregion

        #region Double Add

        /// <summary>
        /// SIMD-optimized addition for Double arrays.
        /// </summary>
        public static unsafe void Add_Double(
            double* lhs, double* rhs, double* result,
            int* lhsStrides, int* rhsStrides, int* shape,
            int ndim, int totalSize)
        {
            var path = StrideDetector.Classify<double>(lhsStrides, rhsStrides, shape, ndim);

            switch (path)
            {
                case ExecutionPath.SimdFull:
                    SimdFull_Add_Double(lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarRight:
                    SimdScalarRight_Add_Double(lhs, *rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarLeft:
                    SimdScalarLeft_Add_Double(*lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdChunk:
                    SimdChunk_Add_Double(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim);
                    break;
                default:
                    General_Add_Double(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim, totalSize);
                    break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdFull_Add_Double(double* lhs, double* rhs, double* result, int totalSize)
        {
            int i = 0;
            int vectorEnd = totalSize - Vector256<double>.Count;

            for (; i <= vectorEnd; i += Vector256<double>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(vl + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + rhs[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarRight_Add_Double(double* lhs, double scalar, double* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<double>.Count;

            for (; i <= vectorEnd; i += Vector256<double>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                Vector256.Store(vl + scalarVec, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + scalar;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarLeft_Add_Double(double scalar, double* rhs, double* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<double>.Count;

            for (; i <= vectorEnd; i += Vector256<double>.Count)
            {
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(scalarVec + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = scalar + rhs[i];
        }

        private static unsafe void SimdChunk_Add_Double(
            double* lhs, double* rhs, double* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim)
        {
            int innerSize = shape[ndim - 1];
            int outerSize = 1;
            for (int d = 0; d < ndim - 1; d++)
                outerSize *= shape[d];

            int lhsInner = lhsStrides[ndim - 1];
            int rhsInner = rhsStrides[ndim - 1];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                int idx = outer;
                for (int d = ndim - 2; d >= 0; d--)
                {
                    int coord = idx % shape[d];
                    idx /= shape[d];
                    lhsOffset += coord * lhsStrides[d];
                    rhsOffset += coord * rhsStrides[d];
                }

                double* lhsRow = lhs + lhsOffset;
                double* rhsRow = rhs + rhsOffset;
                double* resultRow = result + outer * innerSize;

                if (lhsInner == 1 && rhsInner == 1)
                    SimdFull_Add_Double(lhsRow, rhsRow, resultRow, innerSize);
                else if (rhsInner == 0)
                    SimdScalarRight_Add_Double(lhsRow, *rhsRow, resultRow, innerSize);
                else if (lhsInner == 0)
                    SimdScalarLeft_Add_Double(*lhsRow, rhsRow, resultRow, innerSize);
                else
                {
                    for (int i = 0; i < innerSize; i++)
                        resultRow[i] = lhsRow[i * lhsInner] + rhsRow[i * rhsInner];
                }
            }
        }

        private static unsafe void General_Add_Double(
            double* lhs, double* rhs, double* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize)
        {
            Span<int> coords = stackalloc int[ndim];

            for (int i = 0; i < totalSize; i++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    lhsOffset += coords[d] * lhsStrides[d];
                    rhsOffset += coords[d] * rhsStrides[d];
                }

                result[i] = lhs[lhsOffset] + rhs[rhsOffset];

                for (int d = ndim - 1; d >= 0; d--)
                {
                    if (++coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        #endregion

        #region Single Add

        /// <summary>
        /// SIMD-optimized addition for Single (float) arrays.
        /// </summary>
        public static unsafe void Add_Single(
            float* lhs, float* rhs, float* result,
            int* lhsStrides, int* rhsStrides, int* shape,
            int ndim, int totalSize)
        {
            var path = StrideDetector.Classify<float>(lhsStrides, rhsStrides, shape, ndim);

            switch (path)
            {
                case ExecutionPath.SimdFull:
                    SimdFull_Add_Single(lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarRight:
                    SimdScalarRight_Add_Single(lhs, *rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarLeft:
                    SimdScalarLeft_Add_Single(*lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdChunk:
                    SimdChunk_Add_Single(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim);
                    break;
                default:
                    General_Add_Single(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim, totalSize);
                    break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdFull_Add_Single(float* lhs, float* rhs, float* result, int totalSize)
        {
            int i = 0;
            int vectorEnd = totalSize - Vector256<float>.Count;

            for (; i <= vectorEnd; i += Vector256<float>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(vl + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + rhs[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarRight_Add_Single(float* lhs, float scalar, float* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<float>.Count;

            for (; i <= vectorEnd; i += Vector256<float>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                Vector256.Store(vl + scalarVec, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + scalar;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarLeft_Add_Single(float scalar, float* rhs, float* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<float>.Count;

            for (; i <= vectorEnd; i += Vector256<float>.Count)
            {
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(scalarVec + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = scalar + rhs[i];
        }

        private static unsafe void SimdChunk_Add_Single(
            float* lhs, float* rhs, float* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim)
        {
            int innerSize = shape[ndim - 1];
            int outerSize = 1;
            for (int d = 0; d < ndim - 1; d++)
                outerSize *= shape[d];

            int lhsInner = lhsStrides[ndim - 1];
            int rhsInner = rhsStrides[ndim - 1];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                int idx = outer;
                for (int d = ndim - 2; d >= 0; d--)
                {
                    int coord = idx % shape[d];
                    idx /= shape[d];
                    lhsOffset += coord * lhsStrides[d];
                    rhsOffset += coord * rhsStrides[d];
                }

                float* lhsRow = lhs + lhsOffset;
                float* rhsRow = rhs + rhsOffset;
                float* resultRow = result + outer * innerSize;

                if (lhsInner == 1 && rhsInner == 1)
                    SimdFull_Add_Single(lhsRow, rhsRow, resultRow, innerSize);
                else if (rhsInner == 0)
                    SimdScalarRight_Add_Single(lhsRow, *rhsRow, resultRow, innerSize);
                else if (lhsInner == 0)
                    SimdScalarLeft_Add_Single(*lhsRow, rhsRow, resultRow, innerSize);
                else
                {
                    for (int i = 0; i < innerSize; i++)
                        resultRow[i] = lhsRow[i * lhsInner] + rhsRow[i * rhsInner];
                }
            }
        }

        private static unsafe void General_Add_Single(
            float* lhs, float* rhs, float* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize)
        {
            Span<int> coords = stackalloc int[ndim];

            for (int i = 0; i < totalSize; i++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    lhsOffset += coords[d] * lhsStrides[d];
                    rhsOffset += coords[d] * rhsStrides[d];
                }

                result[i] = lhs[lhsOffset] + rhs[rhsOffset];

                for (int d = ndim - 1; d >= 0; d--)
                {
                    if (++coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        #endregion

        #region Int64 Add

        /// <summary>
        /// SIMD-optimized addition for Int64 arrays.
        /// </summary>
        public static unsafe void Add_Int64(
            long* lhs, long* rhs, long* result,
            int* lhsStrides, int* rhsStrides, int* shape,
            int ndim, int totalSize)
        {
            var path = StrideDetector.Classify<long>(lhsStrides, rhsStrides, shape, ndim);

            switch (path)
            {
                case ExecutionPath.SimdFull:
                    SimdFull_Add_Int64(lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarRight:
                    SimdScalarRight_Add_Int64(lhs, *rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdScalarLeft:
                    SimdScalarLeft_Add_Int64(*lhs, rhs, result, totalSize);
                    break;
                case ExecutionPath.SimdChunk:
                    SimdChunk_Add_Int64(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim);
                    break;
                default:
                    General_Add_Int64(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim, totalSize);
                    break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdFull_Add_Int64(long* lhs, long* rhs, long* result, int totalSize)
        {
            int i = 0;
            int vectorEnd = totalSize - Vector256<long>.Count;

            for (; i <= vectorEnd; i += Vector256<long>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(vl + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + rhs[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarRight_Add_Int64(long* lhs, long scalar, long* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<long>.Count;

            for (; i <= vectorEnd; i += Vector256<long>.Count)
            {
                var vl = Vector256.Load(lhs + i);
                Vector256.Store(vl + scalarVec, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = lhs[i] + scalar;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SimdScalarLeft_Add_Int64(long scalar, long* rhs, long* result, int totalSize)
        {
            var scalarVec = Vector256.Create(scalar);
            int i = 0;
            int vectorEnd = totalSize - Vector256<long>.Count;

            for (; i <= vectorEnd; i += Vector256<long>.Count)
            {
                var vr = Vector256.Load(rhs + i);
                Vector256.Store(scalarVec + vr, result + i);
            }

            for (; i < totalSize; i++)
                result[i] = scalar + rhs[i];
        }

        private static unsafe void SimdChunk_Add_Int64(
            long* lhs, long* rhs, long* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim)
        {
            int innerSize = shape[ndim - 1];
            int outerSize = 1;
            for (int d = 0; d < ndim - 1; d++)
                outerSize *= shape[d];

            int lhsInner = lhsStrides[ndim - 1];
            int rhsInner = rhsStrides[ndim - 1];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                int idx = outer;
                for (int d = ndim - 2; d >= 0; d--)
                {
                    int coord = idx % shape[d];
                    idx /= shape[d];
                    lhsOffset += coord * lhsStrides[d];
                    rhsOffset += coord * rhsStrides[d];
                }

                long* lhsRow = lhs + lhsOffset;
                long* rhsRow = rhs + rhsOffset;
                long* resultRow = result + outer * innerSize;

                if (lhsInner == 1 && rhsInner == 1)
                    SimdFull_Add_Int64(lhsRow, rhsRow, resultRow, innerSize);
                else if (rhsInner == 0)
                    SimdScalarRight_Add_Int64(lhsRow, *rhsRow, resultRow, innerSize);
                else if (lhsInner == 0)
                    SimdScalarLeft_Add_Int64(*lhsRow, rhsRow, resultRow, innerSize);
                else
                {
                    for (int i = 0; i < innerSize; i++)
                        resultRow[i] = lhsRow[i * lhsInner] + rhsRow[i * rhsInner];
                }
            }
        }

        private static unsafe void General_Add_Int64(
            long* lhs, long* rhs, long* result,
            int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize)
        {
            Span<int> coords = stackalloc int[ndim];

            for (int i = 0; i < totalSize; i++)
            {
                int lhsOffset = 0, rhsOffset = 0;
                for (int d = 0; d < ndim; d++)
                {
                    lhsOffset += coords[d] * lhsStrides[d];
                    rhsOffset += coords[d] * rhsStrides[d];
                }

                result[i] = lhs[lhsOffset] + rhs[rhsOffset];

                for (int d = ndim - 1; d >= 0; d--)
                {
                    if (++coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        #endregion
    }
}
