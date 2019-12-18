using System;

namespace NumSharp
{
    public class BroadcastInfo : ICloneable
    {
        /// <summary>
        ///     The original shape prior to broadcasting.
        /// </summary>
        public Shape OriginalShape;

        /// <summary>
        ///     Represents a shape with the same number of dimensions that the broadcasted ones are set to dim of 1.
        /// </summary>
        /// <remarks>This shape is lazyloaded during runtime when calling Shape.GetOffset and other methods.</remarks>

        public Shape? UnreducedBroadcastedShape; //lazyloaded

        public BroadcastInfo() { }

        public BroadcastInfo(Shape originalShape)
        {
            OriginalShape = originalShape;
        }

        public BroadcastInfo Clone()
        {
            return new BroadcastInfo() {OriginalShape = OriginalShape.Clone(true, false, false)};
        }

        object ICloneable.Clone() => Clone();
    }
}
