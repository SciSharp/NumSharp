using System;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Iteration
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct NDAxisState
    {
        public int OuterNDim;
        public int Axis;
        public long AxisLength;
        public long OuterSize;
        public long SourceAxisStride;
        public long DestinationAxisStride;
        public IntPtr Data0;
        public IntPtr Data1;

        // Outer-dimension scratch (shape + per-operand strides), one entry per NON-unit outer axis.
        // These point at buffers the caller stackallocs sized by the operand's ndim — so NDAxisIter has
        // NO fixed dimension cap, matching NumSharp's unlimited-dims design and NDIter.Execution.cs's own
        // `stackalloc long[Math.Max(1, ndim)]` idiom. (Previously `fixed long[64]`, which threw for ndim > 64.)
        public long* OuterShapePtr;
        public long* SourceOuterStridesPtr;
        public long* DestinationOuterStridesPtr;

        public long* GetOuterShapePointer() => OuterShapePtr;

        public long* GetSourceOuterStridesPointer() => SourceOuterStridesPtr;

        public long* GetDestinationOuterStridesPointer() => DestinationOuterStridesPtr;
    }
}
