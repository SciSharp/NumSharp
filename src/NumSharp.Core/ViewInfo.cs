using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public class ViewInfo : ICloneable
    {
        public SliceDef[] Slices;
        public Shape OriginalShape;
        public Shape UnreducedShape;

        public ViewInfo Clone()
        {
            return new ViewInfo() {Slices = (SliceDef[])Slices.Clone(), OriginalShape = OriginalShape.Clone(true, false), UnreducedShape = UnreducedShape.Clone(true, false),};
        }

        object ICloneable.Clone() => Clone();
    }
}
