using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public class Shape
    {
        private int[] shape { get; set; }

        public int Size
        {
            get
            {
                int idx = 1;
                for (int i = 0; i < shape.Length; i++)
                {
                    idx *= shape[i];
                }
                return idx;
            }
        }

        public Shape(params int[] shape)
        {
            this.shape = shape;
        }

        public int Length { get { return shape.Length; } }

        public int[] Shapes { get { return shape; } }

        public int UniShape { get { return shape[0]; } }

        public (int, int) BiShape { get { return (shape[0], shape[1]); } }

        public (int, int, int) TriShape { get { return (shape[0], shape[1], shape[2]); } }
    }
}
