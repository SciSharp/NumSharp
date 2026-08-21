using System;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     Per-call axis-iteration state. The outer-dimension scratch (<see cref="OuterShapePtr"/> and the
    ///     stride pointers) points at buffers the entry method <c>stackalloc</c>s — non-pinned stack memory —
    ///     so this is a <see langword="ref"/> <see langword="struct"/>: it MUST NOT escape the stack frame that
    ///     owns those buffers. The ref-struct constraint enforces that at compile time (no boxing, no heap
    ///     field, no closure/async capture), the way <see cref="System.Span{T}"/> and <c>NDIterRef</c> do —
    ///     otherwise the raw pointers (which the compiler does not lifetime-track) could dangle.
    /// </summary>
    public unsafe ref struct NDAxisState
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
