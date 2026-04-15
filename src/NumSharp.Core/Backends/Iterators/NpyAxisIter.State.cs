using System;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Iteration
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct NpyAxisState
    {
        internal const int MaxDims = 64;

        public int OuterNDim;
        public int Axis;
        public long AxisLength;
        public long OuterSize;
        public long SourceAxisStride;
        public long DestinationAxisStride;
        public IntPtr Data0;
        public IntPtr Data1;

        public fixed long OuterShape[MaxDims];
        public fixed long SourceOuterStrides[MaxDims];
        public fixed long DestinationOuterStrides[MaxDims];

        public long* GetOuterShapePointer()
        {
            fixed (long* ptr = OuterShape)
                return ptr;
        }

        public long* GetSourceOuterStridesPointer()
        {
            fixed (long* ptr = SourceOuterStrides)
                return ptr;
        }

        public long* GetDestinationOuterStridesPointer()
        {
            fixed (long* ptr = DestinationOuterStrides)
                return ptr;
        }
    }
}
