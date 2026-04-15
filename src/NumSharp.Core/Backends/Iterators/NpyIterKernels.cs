using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Interface for kernels that work with NpyIter.
    /// </summary>
    internal unsafe interface INpyIterKernel
    {
        /// <summary>
        /// Get the inner loop function for the specified execution path.
        /// </summary>
        NpyIterInnerLoopFunc GetInnerKernel(NpyIterExecutionPath path);

        /// <summary>
        /// Process a single element (for general path).
        /// </summary>
        void ProcessElement(void** dataptrs);

        /// <summary>
        /// Whether this kernel supports early exit.
        /// </summary>
        bool SupportsEarlyExit { get; }

        /// <summary>
        /// Required alignment for buffers (0 for no requirement).
        /// </summary>
        int RequiredAlignment { get; }
    }

    /// <summary>
    /// Execution path selection logic.
    /// </summary>
    internal static unsafe class NpyIterPathSelector
    {
        /// <summary>
        /// Determine the optimal execution path based on operand layout.
        /// </summary>
        public static NpyIterExecutionPath SelectPath(ref NpyIterState state)
        {
            // Check if all operands are contiguous
            if ((state.ItFlags & (uint)NpyIterFlags.CONTIGUOUS) != 0)
                return NpyIterExecutionPath.Contiguous;

            bool anyBroadcast = false;
            bool canGather = true;

            // Access dynamically allocated strides array
            var strides = state.Strides;
            int stridesNDim = state.StridesNDim;

            for (int op = 0; op < state.NOp; op++)
            {
                // Check inner stride (axis NDim-1)
                int innerIdx = op * stridesNDim + (state.NDim - 1);
                long innerStride = state.NDim > 0 ? strides[innerIdx] : 1;

                if (innerStride == 0)
                    anyBroadcast = true;

                // Gather requires stride fits in int32 and is positive
                if (innerStride < 0 || innerStride > int.MaxValue)
                    canGather = false;
            }

            // Check for broadcast or non-gatherable strides
            if (anyBroadcast || !canGather)
            {
                // Need buffering for broadcast or large strides
                if ((state.ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
                    return NpyIterExecutionPath.Buffered;
                else
                    return NpyIterExecutionPath.General;
            }

            // Can use gather for strided access
            if (Avx2.IsSupported)
                return NpyIterExecutionPath.Strided;

            return NpyIterExecutionPath.General;
        }

        /// <summary>
        /// Check if the given execution path supports SIMD operations.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSimdPath(NpyIterExecutionPath path)
        {
            return path == NpyIterExecutionPath.Contiguous ||
                   path == NpyIterExecutionPath.Strided ||
                   path == NpyIterExecutionPath.Buffered;
        }

        /// <summary>
        /// Get the recommended inner loop size for the given path.
        /// </summary>
        public static long GetRecommendedInnerSize(NpyIterExecutionPath path, NPTypeCode dtype)
        {
            return path switch
            {
                NpyIterExecutionPath.Contiguous => long.MaxValue, // Process all at once
                NpyIterExecutionPath.Strided => 256, // AVX2 gather batch
                NpyIterExecutionPath.Buffered => NpyIterBufferManager.DefaultBufferSize,
                NpyIterExecutionPath.General => 1, // Element by element
                _ => 1
            };
        }
    }

    /// <summary>
    /// Execution helpers for different paths.
    /// </summary>
    internal static unsafe class NpyIterExecution
    {
        /// <summary>
        /// Execute iteration using contiguous path with SIMD kernel.
        /// </summary>
        public static void ExecuteContiguous<TKernel>(
            ref NpyIterState state,
            TKernel kernel)
            where TKernel : INpyIterKernel
        {
            var dataptrs = stackalloc void*[state.NOp];
            for (int op = 0; op < state.NOp; op++)
                dataptrs[op] = state.GetDataPtr(op);

            var strides = stackalloc long[state.NOp];
            for (int op = 0; op < state.NOp; op++)
                strides[op] = state.NDim > 0 ? state.GetStride(state.NDim - 1, op) : 0;

            var innerKernel = kernel.GetInnerKernel(NpyIterExecutionPath.Contiguous);
            innerKernel(dataptrs, strides, state.IterSize, null);
        }

        /// <summary>
        /// Execute iteration using buffered path.
        /// </summary>
        public static void ExecuteBuffered<TKernel>(
            ref NpyIterState state,
            TKernel kernel)
            where TKernel : INpyIterKernel
        {
            // Ensure buffers are allocated
            if (!NpyIterBufferManager.AllocateBuffers(ref state, state.BufferSize))
                throw new OutOfMemoryException("Failed to allocate iteration buffers");

            try
            {
                var innerKernel = kernel.GetInnerKernel(NpyIterExecutionPath.Contiguous);
                long remaining = state.IterSize;

                var dataptrs = stackalloc void*[state.NOp];
                var strides = stackalloc long[state.NOp];

                for (int op = 0; op < state.NOp; op++)
                    strides[op] = 1;  // Buffers are contiguous

                while (remaining > 0)
                {
                    long batchSize = Math.Min(remaining, state.BufferSize);

                    // Copy to buffers
                    for (int op = 0; op < state.NOp; op++)
                    {
                        var opFlags = state.GetOpFlags(op);
                        if ((opFlags & NpyIterOpFlags.READ) != 0)
                        {
                            // TODO: Type dispatch for copy
                            // For now, use byte copy as placeholder
                        }
                    }

                    // Get buffer pointers
                    for (int op = 0; op < state.NOp; op++)
                    {
                        var buf = state.GetBuffer(op);
                        dataptrs[op] = buf != null ? buf : state.GetDataPtr(op);
                    }

                    // Execute kernel
                    innerKernel(dataptrs, strides, batchSize, null);

                    // Copy from buffers
                    for (int op = 0; op < state.NOp; op++)
                    {
                        var opFlags = state.GetOpFlags(op);
                        if ((opFlags & NpyIterOpFlags.WRITE) != 0)
                        {
                            // TODO: Type dispatch for copy
                        }
                    }

                    // Advance state by batch size
                    state.IterIndex += batchSize;
                    remaining -= batchSize;
                }
            }
            finally
            {
                NpyIterBufferManager.FreeBuffers(ref state);
            }
        }

        /// <summary>
        /// Execute iteration using general coordinate-based path.
        /// </summary>
        public static void ExecuteGeneral<TKernel>(
            ref NpyIterState state,
            TKernel kernel)
            where TKernel : INpyIterKernel
        {
            var dataptrs = stackalloc void*[state.NOp];

            for (long i = 0; i < state.IterSize; i++)
            {
                // Get current data pointers
                for (int op = 0; op < state.NOp; op++)
                    dataptrs[op] = state.GetDataPtr(op);

                // Process single element
                kernel.ProcessElement(dataptrs);

                // Check early exit
                if (kernel.SupportsEarlyExit)
                {
                    // TODO: Check early exit condition
                }

                // Advance to next position
                state.Advance();
            }
        }

        /// <summary>
        /// Execute iteration with automatic path selection.
        /// </summary>
        public static void Execute<TKernel>(
            ref NpyIterState state,
            TKernel kernel)
            where TKernel : INpyIterKernel
        {
            var path = NpyIterPathSelector.SelectPath(ref state);

            switch (path)
            {
                case NpyIterExecutionPath.Contiguous:
                    ExecuteContiguous(ref state, kernel);
                    break;

                case NpyIterExecutionPath.Buffered:
                    ExecuteBuffered(ref state, kernel);
                    break;

                case NpyIterExecutionPath.General:
                default:
                    ExecuteGeneral(ref state, kernel);
                    break;
            }
        }
    }
}
