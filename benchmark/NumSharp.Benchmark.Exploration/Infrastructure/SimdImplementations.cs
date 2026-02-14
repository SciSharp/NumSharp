using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Benchmark.Exploration.Infrastructure;

/// <summary>
/// SIMD implementations for benchmarking different strategies.
/// These are raw implementations without NumSharp overhead for isolated testing.
/// </summary>
public static unsafe class SimdImplementations
{
    #region SIMD-FULL: Contiguous arrays, same shape

    /// <summary>
    /// Add two contiguous float64 arrays using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddFull_Float64(double* lhs, double* rhs, double* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<double>.Count)
        {
            int vectorCount = count - Vector256<double>.Count + 1;
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var va = Avx.LoadVector256(lhs + i);
                var vb = Avx.LoadVector256(rhs + i);
                var vr = Avx.Add(va, vb);
                Avx.Store(result + i, vr);
            }
        }

        // Scalar tail
        for (; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Add two contiguous float32 arrays using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddFull_Float32(float* lhs, float* rhs, float* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<float>.Count)
        {
            int vectorCount = count - Vector256<float>.Count + 1;
            for (; i < vectorCount; i += Vector256<float>.Count)
            {
                var va = Avx.LoadVector256(lhs + i);
                var vb = Avx.LoadVector256(rhs + i);
                var vr = Avx.Add(va, vb);
                Avx.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Add two contiguous int32 arrays using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddFull_Int32(int* lhs, int* rhs, int* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<int>.Count)
        {
            int vectorCount = count - Vector256<int>.Count + 1;
            for (; i < vectorCount; i += Vector256<int>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vb = Avx2.LoadVector256(rhs + i);
                var vr = Avx2.Add(va, vb);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Add two contiguous int64 arrays using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddFull_Int64(long* lhs, long* rhs, long* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<long>.Count)
        {
            int vectorCount = count - Vector256<long>.Count + 1;
            for (; i < vectorCount; i += Vector256<long>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vb = Avx2.LoadVector256(rhs + i);
                var vr = Avx2.Add(va, vb);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Add two contiguous int16 arrays using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddFull_Int16(short* lhs, short* rhs, short* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<short>.Count)
        {
            int vectorCount = count - Vector256<short>.Count + 1;
            for (; i < vectorCount; i += Vector256<short>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vb = Avx2.LoadVector256(rhs + i);
                var vr = Avx2.Add(va, vb);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = (short)(lhs[i] + rhs[i]);
        }
    }

    /// <summary>
    /// Add two contiguous byte arrays using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddFull_Byte(byte* lhs, byte* rhs, byte* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<byte>.Count)
        {
            int vectorCount = count - Vector256<byte>.Count + 1;
            for (; i < vectorCount; i += Vector256<byte>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vb = Avx2.LoadVector256(rhs + i);
                var vr = Avx2.Add(va, vb);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = (byte)(lhs[i] + rhs[i]);
        }
    }

    #endregion

    #region SIMD-SCALAR: One operand is a scalar

    /// <summary>
    /// Add a scalar to a contiguous float64 array using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar_Float64(double* lhs, double scalar, double* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<double>.Count)
        {
            var vs = Vector256.Create(scalar);
            int vectorCount = count - Vector256<double>.Count + 1;
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var va = Avx.LoadVector256(lhs + i);
                var vr = Avx.Add(va, vs);
                Avx.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + scalar;
        }
    }

    /// <summary>
    /// Add a scalar to a contiguous float32 array using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar_Float32(float* lhs, float scalar, float* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<float>.Count)
        {
            var vs = Vector256.Create(scalar);
            int vectorCount = count - Vector256<float>.Count + 1;
            for (; i < vectorCount; i += Vector256<float>.Count)
            {
                var va = Avx.LoadVector256(lhs + i);
                var vr = Avx.Add(va, vs);
                Avx.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + scalar;
        }
    }

    /// <summary>
    /// Add a scalar to a contiguous int32 array using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar_Int32(int* lhs, int scalar, int* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<int>.Count)
        {
            var vs = Vector256.Create(scalar);
            int vectorCount = count - Vector256<int>.Count + 1;
            for (; i < vectorCount; i += Vector256<int>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vr = Avx2.Add(va, vs);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + scalar;
        }
    }

    /// <summary>
    /// Add a scalar to a contiguous int64 array using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar_Int64(long* lhs, long scalar, long* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<long>.Count)
        {
            var vs = Vector256.Create(scalar);
            int vectorCount = count - Vector256<long>.Count + 1;
            for (; i < vectorCount; i += Vector256<long>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vr = Avx2.Add(va, vs);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = lhs[i] + scalar;
        }
    }

    /// <summary>
    /// Add a scalar to a contiguous int16 array using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar_Int16(short* lhs, short scalar, short* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<short>.Count)
        {
            var vs = Vector256.Create(scalar);
            int vectorCount = count - Vector256<short>.Count + 1;
            for (; i < vectorCount; i += Vector256<short>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vr = Avx2.Add(va, vs);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = (short)(lhs[i] + scalar);
        }
    }

    /// <summary>
    /// Add a scalar to a contiguous byte array using Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScalar_Byte(byte* lhs, byte scalar, byte* result, int count)
    {
        int i = 0;

        if (Avx2.IsSupported && count >= Vector256<byte>.Count)
        {
            var vs = Vector256.Create(scalar);
            int vectorCount = count - Vector256<byte>.Count + 1;
            for (; i < vectorCount; i += Vector256<byte>.Count)
            {
                var va = Avx2.LoadVector256(lhs + i);
                var vr = Avx2.Add(va, vs);
                Avx2.Store(result + i, vr);
            }
        }

        for (; i < count; i++)
        {
            result[i] = (byte)(lhs[i] + scalar);
        }
    }

    #endregion

    #region SIMD-CHUNK: Row broadcast (inner dimension contiguous)

    /// <summary>
    /// Add a row vector to each row of a matrix using SIMD on inner dimension.
    /// matrix[M, N] + row[N] = result[M, N]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddRowBroadcast_Float64(double* matrix, double* row, double* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var matrixRow = matrix + r * cols;
            var resultRow = result + r * cols;
            AddFull_Float64(matrixRow, row, resultRow, cols);
        }
    }

    /// <summary>
    /// Add a row vector to each row of a matrix using SIMD on inner dimension.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddRowBroadcast_Float32(float* matrix, float* row, float* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var matrixRow = matrix + r * cols;
            var resultRow = result + r * cols;
            AddFull_Float32(matrixRow, row, resultRow, cols);
        }
    }

    /// <summary>
    /// Add a row vector to each row of a matrix using SIMD on inner dimension.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddRowBroadcast_Int32(int* matrix, int* row, int* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var matrixRow = matrix + r * cols;
            var resultRow = result + r * cols;
            AddFull_Int32(matrixRow, row, resultRow, cols);
        }
    }

    #endregion

    #region Scalar loops (baseline)

    /// <summary>
    /// Scalar add baseline for float64.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddScalarLoop_Float64(double* lhs, double* rhs, double* result, int count)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Scalar add baseline for float32.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddScalarLoop_Float32(float* lhs, float* rhs, float* result, int count)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Scalar add baseline for int32.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddScalarLoop_Int32(int* lhs, int* rhs, int* result, int count)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Scalar add baseline for int64.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddScalarLoop_Int64(long* lhs, long* rhs, long* result, int count)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = lhs[i] + rhs[i];
        }
    }

    /// <summary>
    /// Scalar add baseline for int16.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddScalarLoop_Int16(short* lhs, short* rhs, short* result, int count)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = (short)(lhs[i] + rhs[i]);
        }
    }

    /// <summary>
    /// Scalar add baseline for byte.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddScalarLoop_Byte(byte* lhs, byte* rhs, byte* result, int count)
    {
        for (int i = 0; i < count; i++)
        {
            result[i] = (byte)(lhs[i] + rhs[i]);
        }
    }

    /// <summary>
    /// Row broadcast baseline - scalar inner loop.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddRowBroadcastScalar_Float64(double* matrix, double* row, double* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result[r * cols + c] = matrix[r * cols + c] + row[c];
            }
        }
    }

    /// <summary>
    /// Column broadcast baseline - scalar loop with strided access.
    /// matrix[M, N] + col[M, 1] = result[M, N]
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void AddColBroadcastScalar_Float64(double* matrix, double* col, double* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var colVal = col[r];
            for (int c = 0; c < cols; c++)
            {
                result[r * cols + c] = matrix[r * cols + c] + colVal;
            }
        }
    }

    /// <summary>
    /// Column broadcast with SIMD - each row uses same scalar from col.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddColBroadcast_Float64(double* matrix, double* col, double* result, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            var colVal = col[r];
            var matrixRow = matrix + r * cols;
            var resultRow = result + r * cols;
            AddScalar_Float64(matrixRow, colVal, resultRow, cols);
        }
    }

    #endregion

    #region Array allocation helpers

    /// <summary>
    /// Result of pinned array allocation.
    /// </summary>
    public readonly struct PinnedArray<T> : IDisposable where T : unmanaged
    {
        public readonly T[] Array;
        public readonly GCHandle Handle;
        public readonly IntPtr Pointer;

        public PinnedArray(int count)
        {
            Array = GC.AllocateArray<T>(count, pinned: true);
            Handle = GCHandle.Alloc(Array, GCHandleType.Pinned);
            Pointer = Handle.AddrOfPinnedObject();
        }

        public T* Ptr => (T*)Pointer;

        public void Dispose()
        {
            if (Handle.IsAllocated)
                Handle.Free();
        }
    }

    /// <summary>
    /// Allocate a pinned array and return a struct with the array, handle, and pointer.
    /// </summary>
    public static PinnedArray<T> AllocatePinned<T>(int count) where T : unmanaged
    {
        return new PinnedArray<T>(count);
    }

    /// <summary>
    /// Allocate aligned memory using NativeMemory.
    /// </summary>
    public static T* AllocateAligned<T>(int count, nuint alignment = 32) where T : unmanaged
    {
        return (T*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(T)), alignment);
    }

    /// <summary>
    /// Free aligned memory.
    /// </summary>
    public static void FreeAligned<T>(T* ptr) where T : unmanaged
    {
        NativeMemory.AlignedFree(ptr);
    }

    /// <summary>
    /// Fill array with random values.
    /// </summary>
    public static void FillRandom(double* ptr, int count, int seed = 42)
    {
        var rng = new Random(seed);
        for (int i = 0; i < count; i++)
        {
            ptr[i] = rng.NextDouble() * 100;
        }
    }

    /// <summary>
    /// Fill array with random float values.
    /// </summary>
    public static void FillRandom(float* ptr, int count, int seed = 42)
    {
        var rng = new Random(seed);
        for (int i = 0; i < count; i++)
        {
            ptr[i] = (float)(rng.NextDouble() * 100);
        }
    }

    /// <summary>
    /// Fill array with random int values.
    /// </summary>
    public static void FillRandom(int* ptr, int count, int seed = 42)
    {
        var rng = new Random(seed);
        for (int i = 0; i < count; i++)
        {
            ptr[i] = rng.Next(0, 100);
        }
    }

    /// <summary>
    /// Fill array with random long values.
    /// </summary>
    public static void FillRandom(long* ptr, int count, int seed = 42)
    {
        var rng = new Random(seed);
        for (int i = 0; i < count; i++)
        {
            ptr[i] = rng.NextInt64(0, 100);
        }
    }

    /// <summary>
    /// Fill array with random short values.
    /// </summary>
    public static void FillRandom(short* ptr, int count, int seed = 42)
    {
        var rng = new Random(seed);
        for (int i = 0; i < count; i++)
        {
            ptr[i] = (short)rng.Next(0, 100);
        }
    }

    /// <summary>
    /// Fill array with random byte values.
    /// </summary>
    public static void FillRandom(byte* ptr, int count, int seed = 42)
    {
        var rng = new Random(seed);
        var bytes = new byte[count];
        rng.NextBytes(bytes);
        for (int i = 0; i < count; i++)
        {
            ptr[i] = bytes[i];
        }
    }

    #endregion
}
