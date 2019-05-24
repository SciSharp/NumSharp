using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Numpy.Models
{
    public class Shape
    {
        public int[] Dimensions { get; private set; }

        public Shape(params int[] shape)
        {
            this.Dimensions = shape;
        }

        #region Equality

        public static bool operator ==(Shape a, Shape b)
        {
            if (b is null) return false;
            return Enumerable.SequenceEqual(a.Dimensions, b?.Dimensions);
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj.GetType() != typeof(Shape))
                return false;
            return Enumerable.SequenceEqual(Dimensions, ((Shape)obj).Dimensions);
        }

        public override int GetHashCode()
        {
            return Dimensions.GetHashCode();
        }

        #endregion
    }
}
