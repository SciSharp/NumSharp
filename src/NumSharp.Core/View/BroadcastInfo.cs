using System;

namespace NumSharp
{
    public class BroadcastInfo : ICloneable
    {
        /// <summary>
        ///     The unbroadcasted shape.
        /// </summary>
        public Shape OriginalShape;

        public Shape? UnbroadcastShape; //lazyloaded

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
