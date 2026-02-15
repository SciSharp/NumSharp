using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Cache of compiled binary operation kernels.
    /// Provides thread-safe access to kernels keyed by (Operation, NPTypeCode).
    ///
    /// Supports two kernel generation modes:
    /// - C# reference implementations (SimdKernels.cs) - always available
    /// - IL-generated kernels (ILKernelGenerator.cs) - ~10-15% faster for contiguous arrays
    /// </summary>
    public static class KernelCache
    {
        /// <summary>
        /// Cache of kernel delegates.
        /// </summary>
        private static readonly ConcurrentDictionary<(BinaryOp, NPTypeCode), Delegate> _cache = new();

        /// <summary>
        /// Whether to prefer IL-generated kernels over C# implementations.
        /// Default is true. Set to false for debugging or benchmarking.
        /// </summary>
        public static bool PreferILGeneration { get; set; } = true;

        /// <summary>
        /// Get a typed kernel for the specified operation and type.
        /// Uses IL-generated kernels when available and enabled.
        /// </summary>
        public static BinaryKernel<T> Get<T>(BinaryOp op) where T : unmanaged
        {
            var key = (op, GetTypeCode<T>());

            if (_cache.TryGetValue(key, out var cached))
            {
                return (BinaryKernel<T>)cached;
            }

            // Try IL generation first if enabled
            BinaryKernel<T>? kernel = null;
            if (PreferILGeneration && ILKernelGenerator.Enabled)
            {
                kernel = ILKernelGenerator.GenerateUnifiedKernel<T>(op);
            }

            // Fall back to C# implementations
            kernel ??= CreateCSharpKernel<T>(op);

            _cache.TryAdd(key, kernel);
            return kernel;
        }

        /// <summary>
        /// Get kernel with runtime type dispatch.
        /// Used by DefaultEngine for integration.
        /// </summary>
        public static Delegate Get(BinaryOp op, NPTypeCode dtype)
        {
            return dtype switch
            {
                NPTypeCode.Byte => Get<byte>(op),
                NPTypeCode.Int16 => Get<short>(op),
                NPTypeCode.UInt16 => Get<ushort>(op),
                NPTypeCode.Int32 => Get<int>(op),
                NPTypeCode.UInt32 => Get<uint>(op),
                NPTypeCode.Int64 => Get<long>(op),
                NPTypeCode.UInt64 => Get<ulong>(op),
                NPTypeCode.Single => Get<float>(op),
                NPTypeCode.Double => Get<double>(op),
                _ => throw new NotSupportedException($"SIMD kernels not supported for {dtype}")
            };
        }

        /// <summary>
        /// Create a C# reference implementation kernel for the specified operation and type.
        /// </summary>
        private static unsafe BinaryKernel<T> CreateCSharpKernel<T>(BinaryOp op) where T : unmanaged
        {
            // Return the appropriate C# implementation
            // Add operation
            if (op == BinaryOp.Add)
            {
                if (typeof(T) == typeof(int))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<int>)SimdKernels.Add_Int32;
                if (typeof(T) == typeof(double))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<double>)SimdKernels.Add_Double;
                if (typeof(T) == typeof(float))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<float>)SimdKernels.Add_Single;
                if (typeof(T) == typeof(long))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<long>)SimdKernels.Add_Int64;
            }

            // TODO: Add Subtract, Multiply, Divide kernels

            throw new NotImplementedException($"C# kernel for {op} on {typeof(T).Name} not yet implemented");
        }

        /// <summary>
        /// Pre-compile all kernels at startup.
        /// Call during application initialization to avoid first-call latency.
        /// </summary>
        public static void PreWarm()
        {
            var ops = new[] { BinaryOp.Add, BinaryOp.Subtract, BinaryOp.Multiply, BinaryOp.Divide };
            var types = new NPTypeCode[]
            {
                NPTypeCode.Int32, NPTypeCode.Int64,
                NPTypeCode.Single, NPTypeCode.Double
            };

            foreach (var op in ops)
            {
                foreach (var dtype in types)
                {
                    try
                    {
                        _ = Get(op, dtype);
                    }
                    catch (NotImplementedException)
                    {
                        // Not all combinations are implemented yet
                    }
                }
            }
        }

        /// <summary>
        /// Number of cached kernels.
        /// </summary>
        public static int CachedCount => _cache.Count;

        /// <summary>
        /// Get all cached kernel keys.
        /// </summary>
        public static IEnumerable<(BinaryOp Op, NPTypeCode Type)> CachedKeys => _cache.Keys;

        /// <summary>
        /// Clear the kernel cache.
        /// </summary>
        public static void Clear()
        {
            _cache.Clear();
            ILKernelGenerator.Clear();
        }

        /// <summary>
        /// Get statistics about the kernel cache.
        /// </summary>
        public static (int Total, int ILGenerated) GetStats()
        {
            return (CachedCount, ILKernelGenerator.CachedCount);
        }

        private static NPTypeCode GetTypeCode<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(bool)) return NPTypeCode.Boolean;
            if (typeof(T) == typeof(byte)) return NPTypeCode.Byte;
            if (typeof(T) == typeof(short)) return NPTypeCode.Int16;
            if (typeof(T) == typeof(ushort)) return NPTypeCode.UInt16;
            if (typeof(T) == typeof(int)) return NPTypeCode.Int32;
            if (typeof(T) == typeof(uint)) return NPTypeCode.UInt32;
            if (typeof(T) == typeof(long)) return NPTypeCode.Int64;
            if (typeof(T) == typeof(ulong)) return NPTypeCode.UInt64;
            if (typeof(T) == typeof(float)) return NPTypeCode.Single;
            if (typeof(T) == typeof(double)) return NPTypeCode.Double;
            if (typeof(T) == typeof(char)) return NPTypeCode.Char;
            if (typeof(T) == typeof(decimal)) return NPTypeCode.Decimal;
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }
    }
}
