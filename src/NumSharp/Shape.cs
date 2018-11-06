using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public class Shape
    {
        private readonly int[] shape;
        private readonly int[] dimOffset;

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
            if (shape.Length == 0)
                throw new Exception("Shape cannot be empty.");
            this.shape = shape;
            dimOffset = new int[shape.Length];
            dimOffset[dimOffset.Length - 1] = 1;
            for (int i = shape.Length - 1; i >= 1; i--)
            {
                dimOffset[i - 1] = dimOffset[i] * shape[i];
            }
        }
        public int Length => shape.Length;
        public int[] DimOffset => dimOffset;
        public int[] Shapes => shape;

        public int UniShape => shape[0];

        public (int, int) BiShape => (shape[0], shape[1]);

        public (int, int, int) TriShape => (shape[0], shape[1], shape[2]);
    }
}
