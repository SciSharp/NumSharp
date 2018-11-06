using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public partial class Shape
    {
        private readonly IReadOnlyList<int> shape;
        private readonly IReadOnlyList<int> dimOffset;

        public int Size
        {
            get
            {
                int idx = 1;
                for (int i = 0; i < shape.Count; i++)
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
            int[] temp = new int[shape.Length];
            temp[shape.Length - 1] = 1;
            for (int i = shape.Length - 1; i >= 1; i--)
            {
                temp[i - 1] = temp[i] * shape[i];
            }
            dimOffset = temp;
        }

        public Shape(IReadOnlyList<int> shape)
        {
            if (shape.Count == 0)
                throw new Exception("Shape cannot be empty.");
            this.shape = shape;
            int[] temp = new int[shape.Count];
            temp[shape.Count - 1] = 1;
            for (int i = shape.Count - 1; i >= 1; i--)
            {
                temp[i - 1] = temp[i] * shape[i];
            }
            dimOffset = temp;
        }

        public int Length => shape.Count;
        public IReadOnlyList<int> DimOffset => dimOffset;
        public IReadOnlyList<int> Shapes => shape;

        public int UniShape => shape[0];

        public (int, int) BiShape => (shape[0], shape[1]);

        public (int, int, int) TriShape => (shape[0], shape[1], shape[2]);
    }
}
