using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class Shape
    {
        public static bool operator ==(Shape a, Shape b)
        {
            return Enumerable.SequenceEqual(a.shape, b.shape);
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj.GetType() != typeof(Shape))
                return false;
            return Enumerable.SequenceEqual(this.shape, ((Shape)obj).shape);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}
