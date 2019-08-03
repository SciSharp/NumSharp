using System;
using System.Collections.Generic;
using System.Text;

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
            return new ViewInfo() {Slices = (SliceDef[])Slices.Clone(), OriginalShape = OriginalShape.Clone(true, false), UnreducedShape = UnreducedShape.Clone(true, false),};
        }

        object ICloneable.Clone() => Clone();
    }
}
