using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Interface for kernels that work with NDIter.
    /// </summary>
    public unsafe interface INDIterKernel
    {
        /// <summary>
        /// Get the inner loop function for the specified execution path.
        /// </summary>
        NDIterInnerLoopFunc GetInnerKernel(NDIterExecutionPath path);

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
    public static unsafe class NDIterPathSelector
    {
        /// <summary>
        /// Determine the optimal execution path based on operand layout.
        /// </summary>
        public static NDIterExecutionPath SelectPath(ref NDIterState state)
        {
            // Check if all operands are contiguous
            if ((state.ItFlags & (uint)NDIterFlags.CONTIGUOUS) != 0)
                return NDIterExecutionPath.Contiguous;

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
                if ((state.ItFlags & (uint)NDIterFlags.BUFFER) != 0)
                    return NDIterExecutionPath.Buffered;
                else
                    return NDIterExecutionPath.General;
            }

            // Can use gather for strided access
            if (Avx2.IsSupported)
                return NDIterExecutionPath.Strided;

            return NDIterExecutionPath.General;
        }

        /// <summary>
        /// Check if the given execution path supports SIMD operations.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static bool IsSimdPath(NDIterExecutionPath path)
        {
            return path == NDIterExecutionPath.Contiguous ||
                   path == NDIterExecutionPath.Strided ||
                   path == NDIterExecutionPath.Buffered;
        }

        /// <summary>
        /// Get the recommended inner loop size for the given path.
        /// </summary>
        public static long GetRecommendedInnerSize(NDIterExecutionPath path, NPTypeCode dtype)
        {
            return path switch
            {
                NDIterExecutionPath.Contiguous => long.MaxValue, // Process all at once
                NDIterExecutionPath.Strided => 256, // AVX2 gather batch
                NDIterExecutionPath.Buffered => NDIterBufferManager.DefaultBufferSize,
                NDIterExecutionPath.General => 1, // Element by element
                _ => 1
            };
        }
    }

    /// <summary>
    /// Execution helpers for different paths.
    /// </summary>
    public static unsafe class NDIterExecution
    {
        /// <summary>
        /// Execute iteration using contiguous path with SIMD kernel.
        /// </summary>
        public static void ExecuteContiguous<TKernel>(
            ref NDIterState state,
            TKernel kernel)
            where TKernel : INDIterKernel
        {
            var dataptrs = stackalloc void*[state.NOp];
            for (int op = 0; op < state.NOp; op++)
                dataptrs[op] = state.GetDataPtr(op);

            var strides = stackalloc long[state.NOp];
            for (int op = 0; op < state.NOp; op++)
                strides[op] = state.NDim > 0 ? state.GetStride(state.NDim - 1, op) : 0;

            var innerKernel = kernel.GetInnerKernel(NDIterExecutionPath.Contiguous);
            innerKernel(dataptrs, strides, state.IterSize, null);
        }

        /// <summary>
        /// Execute iteration using buffered path.
        /// </summary>
        public static void ExecuteBuffered<TKernel>(
            ref NDIterState state,
            TKernel kernel)
            where TKernel : INDIterKernel
        {
            // Ensure buffers are allocated
            if (!NDIterBufferManager.AllocateBuffers(ref state, state.BufferSize))
                throw new OutOfMemoryException("Failed to allocate iteration buffers");

            try
            {
                var innerKernel = kernel.GetInnerKernel(NDIterExecutionPath.Contiguous);
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
                        if ((opFlags & NDIterOpFlags.READ) != 0)
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
                        if ((opFlags & NDIterOpFlags.WRITE) != 0)
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
                NDIterBufferManager.FreeBuffers(ref state);
            }
        }

        /// <summary>
        /// Execute iteration using general coordinate-based path.
        /// </summary>
        public static void ExecuteGeneral<TKernel>(
            ref NDIterState state,
            TKernel kernel)
            where TKernel : INDIterKernel
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
            ref NDIterState state,
            TKernel kernel)
            where TKernel : INDIterKernel
        {
            var path = NDIterPathSelector.SelectPath(ref state);

            switch (path)
            {
                case NDIterExecutionPath.Contiguous:
                    ExecuteContiguous(ref state, kernel);
                    break;

                case NDIterExecutionPath.Buffered:
                    ExecuteBuffered(ref state, kernel);
                    break;

                case NDIterExecutionPath.General:
                default:
                    ExecuteGeneral(ref state, kernel);
                    break;
            }
        }
    }
}
