using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Core iterator state. Stack-allocated with fixed-size buffers.
    /// Matches NumPy's NpyIter_InternalOnly layout conceptually.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NpyIterState
    {
        // =========================================================================
        // Constants
        // =========================================================================

        /// <summary>Maximum supported dimensions (matches NPY_MAXDIMS).</summary>
        internal const int MaxDims = 64;

        /// <summary>Maximum supported operands.</summary>
        internal const int MaxOperands = 8;

        // =========================================================================
        // Core Fields
        // =========================================================================

        /// <summary>Iterator flags (NpyIterFlags bitmask).</summary>
        public uint ItFlags;

        /// <summary>Number of dimensions after coalescing.</summary>
        public int NDim;

        /// <summary>Number of operands.</summary>
        public int NOp;

        /// <summary>Mask operand index (-1 if none).</summary>
        public int MaskOp;

        /// <summary>Total number of iterations.</summary>
        public long IterSize;

        /// <summary>Current iteration index.</summary>
        public long IterIndex;

        /// <summary>Range start for ranged iteration.</summary>
        public long IterStart;

        /// <summary>Range end for ranged iteration.</summary>
        public long IterEnd;

        // =========================================================================
        // Legacy compatibility fields
        // =========================================================================

        /// <summary>Legacy: total size (alias for IterSize).</summary>
        public long Size
        {
            readonly get => IterSize;
            set => IterSize = value;
        }

        /// <summary>Legacy: flags (lower bits of ItFlags).</summary>
        public NpyIterFlags Flags
        {
            readonly get => (NpyIterFlags)(ItFlags & 0xFFFF);
            set => ItFlags = (ItFlags & 0xFFFF0000) | (uint)value;
        }

        /// <summary>Legacy: primary dtype.</summary>
        public NPTypeCode DType;

        // =========================================================================
        // Fixed Arrays (stack-allocated)
        // =========================================================================

        /// <summary>Axis permutation (maps iterator axis to original axis).</summary>
        public fixed sbyte Perm[MaxDims];

        /// <summary>Shape after coalescing.</summary>
        public fixed long Shape[MaxDims];

        /// <summary>Current coordinates.</summary>
        public fixed long Coords[MaxDims];

        /// <summary>
        /// Strides for each operand along each axis.
        /// Layout: [op0_axis0, op0_axis1, ..., op1_axis0, op1_axis1, ...]
        /// Access: Strides[operand * MaxDims + axis]
        /// </summary>
        public fixed long Strides[MaxDims * MaxOperands];

        /// <summary>Current data pointers for each operand.</summary>
        public fixed long DataPtrs[MaxOperands];

        /// <summary>Reset data pointers (base + offset).</summary>
        public fixed long ResetDataPtrs[MaxOperands];

        /// <summary>Base offsets for each operand.</summary>
        public fixed long BaseOffsets[MaxOperands];

        /// <summary>Per-operand flags.</summary>
        public fixed ushort OpItFlags[MaxOperands];

        /// <summary>Operand dtypes.</summary>
        public fixed byte OpDTypes[MaxOperands];

        /// <summary>Element sizes for each operand.</summary>
        public fixed int ElementSizes[MaxOperands];

        // =========================================================================
        // Buffer Data (when BUFFERED flag is set)
        // =========================================================================

        /// <summary>Buffer size (elements per buffer).</summary>
        public long BufferSize;

        /// <summary>Current buffer iteration end.</summary>
        public long BufIterEnd;

        /// <summary>Buffer pointers for each operand.</summary>
        public fixed long Buffers[MaxOperands];

        /// <summary>Buffer strides (always element size for contiguous buffers).</summary>
        public fixed long BufStrides[MaxOperands];

        /// <summary>
        /// Inner strides for each operand (gathered from main Strides array for fast access).
        /// Updated when NDim changes (after coalescing) or when axes are removed.
        /// Layout: [op0_inner_stride, op1_inner_stride, ...]
        /// Matches NumPy's NpyIter_GetInnerStrideArray() return format.
        /// </summary>
        public fixed long InnerStrides[MaxOperands];

        // =========================================================================
        // Accessor Methods
        // =========================================================================

        /// <summary>Get pointer to Shape array.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetShapePointer()
        {
            fixed (long* ptr = Shape)
                return ptr;
        }

        /// <summary>Get pointer to Coords array.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetCoordsPointer()
        {
            fixed (long* ptr = Coords)
                return ptr;
        }

        /// <summary>Get pointer to strides for a specific operand (legacy layout).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetStridesPointer(int operand)
        {
            if ((uint)operand >= MaxOperands)
                throw new ArgumentOutOfRangeException(nameof(operand));

            fixed (long* ptr = Strides)
                return ptr + (operand * MaxDims);
        }

        /// <summary>Get stride for operand at axis.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long GetStride(int axis, int op)
        {
            fixed (long* p = Strides)
                return p[op * MaxDims + axis];
        }

        /// <summary>Set stride for operand at axis.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetStride(int axis, int op, long value)
        {
            fixed (long* p = Strides)
                p[op * MaxDims + axis] = value;
        }

        /// <summary>Get current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetDataPtr(int op)
        {
            fixed (long* p = DataPtrs)
                return (void*)p[op];
        }

        /// <summary>Set current data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDataPtr(int op, void* ptr)
        {
            fixed (long* p = DataPtrs)
                p[op] = (long)ptr;
        }

        /// <summary>Get data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly IntPtr GetDataPointer(int operand)
        {
            fixed (long* p = DataPtrs)
                return (IntPtr)p[operand];
        }

        /// <summary>Set data pointer (legacy interface).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetDataPointer(int operand, IntPtr pointer)
        {
            fixed (long* p = DataPtrs)
                p[operand] = (long)pointer;
        }

        /// <summary>Get reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetResetDataPtr(int op)
        {
            fixed (long* p = ResetDataPtrs)
                return (void*)p[op];
        }

        /// <summary>Set reset data pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetResetDataPtr(int op, void* ptr)
        {
            fixed (long* p = ResetDataPtrs)
                p[op] = (long)ptr;
        }

        /// <summary>Get operand dtype.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NPTypeCode GetOpDType(int op)
        {
            fixed (byte* p = OpDTypes)
                return (NPTypeCode)p[op];
        }

        /// <summary>Set operand dtype.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpDType(int op, NPTypeCode dtype)
        {
            fixed (byte* p = OpDTypes)
                p[op] = (byte)dtype;

            fixed (int* s = ElementSizes)
                s[op] = InfoOf.GetSize(dtype);
        }

        /// <summary>Get operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public NpyIterOpFlags GetOpFlags(int op)
        {
            fixed (ushort* p = OpItFlags)
                return (NpyIterOpFlags)p[op];
        }

        /// <summary>Set operand flags.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetOpFlags(int op, NpyIterOpFlags flags)
        {
            fixed (ushort* p = OpItFlags)
                p[op] = (ushort)flags;
        }

        /// <summary>Get element size for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetElementSize(int op)
        {
            fixed (int* p = ElementSizes)
                return p[op];
        }

        /// <summary>Get buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void* GetBuffer(int op)
        {
            fixed (long* p = Buffers)
                return (void*)p[op];
        }

        /// <summary>Set buffer pointer for operand.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetBuffer(int op, void* ptr)
        {
            fixed (long* p = Buffers)
                p[op] = (long)ptr;
        }

        /// <summary>
        /// Get inner stride array pointer - returns contiguous array of inner strides for all operands.
        /// Layout: [op0_inner_stride, op1_inner_stride, ...] matching NumPy's NpyIter_GetInnerStrideArray().
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long* GetInnerStrideArray()
        {
            fixed (long* p = InnerStrides)
                return p;
        }

        /// <summary>
        /// Update the InnerStrides array from the main Strides array.
        /// Must be called after coalescing or axis removal.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void UpdateInnerStrides()
        {
            if (NDim == 0)
            {
                // Scalar - all inner strides are 0
                fixed (long* inner = InnerStrides)
                {
                    for (int op = 0; op < NOp; op++)
                        inner[op] = 0;
                }
                return;
            }

            int innerAxis = NDim - 1;
            fixed (long* inner = InnerStrides)
            fixed (long* strides = Strides)
            {
                for (int op = 0; op < NOp; op++)
                    inner[op] = strides[op * MaxDims + innerAxis];
            }
        }

        /// <summary>Check if this is a contiguous copy operation (legacy).</summary>
        public readonly bool IsContiguousCopy =>
            ((NpyIterFlags)ItFlags & (NpyIterFlags.SourceContiguous | NpyIterFlags.DestinationContiguous)) ==
            (NpyIterFlags.SourceContiguous | NpyIterFlags.DestinationContiguous) &&
            ((NpyIterFlags)ItFlags & NpyIterFlags.SourceBroadcast) == 0;

        // =========================================================================
        // Iteration Methods
        // =========================================================================

        /// <summary>
        /// Advance iterator by one position using ripple carry.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Advance()
        {
            IterIndex++;

            fixed (long* shape = Shape)
            fixed (long* coords = Coords)
            fixed (long* strides = Strides)
            fixed (long* dataPtrs = DataPtrs)
            fixed (int* elemSizes = ElementSizes)
            {
                for (int axis = NDim - 1; axis >= 0; axis--)
                {
                    coords[axis]++;

                    if (coords[axis] < shape[axis])
                    {
                        // Advance data pointers along this axis
                        for (int op = 0; op < NOp; op++)
                        {
                            long stride = strides[op * MaxDims + axis];
                            dataPtrs[op] += stride * elemSizes[op];
                        }
                        return;
                    }

                    // Carry: reset this axis, continue to next
                    coords[axis] = 0;

                    // Reset data pointers for this axis
                    for (int op = 0; op < NOp; op++)
                    {
                        long stride = strides[op * MaxDims + axis];
                        long axisShape = shape[axis];
                        dataPtrs[op] -= stride * (axisShape - 1) * elemSizes[op];
                    }
                }
            }
        }

        /// <summary>
        /// Reset iterator to the beginning.
        /// </summary>
        public void Reset()
        {
            IterIndex = IterStart;

            fixed (long* coords = Coords)
            {
                for (int d = 0; d < NDim; d++)
                    coords[d] = 0;
            }

            fixed (long* dataPtrs = DataPtrs)
            fixed (long* resetPtrs = ResetDataPtrs)
            {
                for (int op = 0; op < NOp; op++)
                    dataPtrs[op] = resetPtrs[op];
            }
        }

        /// <summary>
        /// Jump to a specific iteration index.
        /// </summary>
        public void GotoIterIndex(long iterindex)
        {
            IterIndex = iterindex;

            // Calculate coordinates from linear index
            long remaining = iterindex;

            fixed (long* shape = Shape)
            fixed (long* coords = Coords)
            {
                for (int d = NDim - 1; d >= 0; d--)
                {
                    long dimSize = shape[d];
                    coords[d] = remaining % dimSize;
                    remaining /= dimSize;
                }
            }

            // Update data pointers
            fixed (long* coords = Coords)
            fixed (long* strides = Strides)
            fixed (long* dataPtrs = DataPtrs)
            fixed (long* resetPtrs = ResetDataPtrs)
            fixed (int* elemSizes = ElementSizes)
            {
                for (int op = 0; op < NOp; op++)
                {
                    long offset = 0;
                    for (int d = 0; d < NDim; d++)
                    {
                        offset += coords[d] * strides[op * MaxDims + d];
                    }
                    dataPtrs[op] = resetPtrs[op] + offset * elemSizes[op];
                }
            }
        }
    }
}
