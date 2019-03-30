using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Casting
{
    public class Broadcast
    {
        public Shape shape { get; set; }
        public int nd { get; set; }
        public int ndim { get; set; }
        public int size { get; set; }
        // public int index { get; set; }
        // public int numiters { get; set; }

        public Broadcast(int nd, int ndim, Shape shape, int size)
        {
            this.nd = nd;
            this.ndim = ndim;
            this.size = size;
            this.shape = shape;
        }
    }
}
