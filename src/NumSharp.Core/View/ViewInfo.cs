using System;

namespace NumSharp
{
    public class ViewInfo : ICloneable
    {
        /// <summary>
        /// ParentShape points to a sliced shape that was reshaped. usually this is null, except if the Shape is a reshaped slice.
        /// ParentShape always is a sliced shape!
        /// </summary>
        public Shape ParentShape;

        /// <summary>
        /// The slice definition for every dimension of the OriginalShape
        /// </summary>
        public SliceDef[] Slices;

        /// <summary>
        /// OriginalShape is the primitive shape of the unsliced array
        /// </summary>
        public Shape OriginalShape;

        /// <summary>
        /// UnreducedShape is the shape after slicing but without dimensionality reductions due to index access
        /// </summary>
        public Shape UnreducedShape;

        public ViewInfo Clone()
        {
            return new ViewInfo() {Slices = Slices?.Clone() as SliceDef[], OriginalShape = OriginalShape.Clone(true, false, false), UnreducedShape = UnreducedShape.Clone(true, false, false), ParentShape = ParentShape.Clone(true, false, false)};
        }

        object ICloneable.Clone() => Clone();
    }
}
