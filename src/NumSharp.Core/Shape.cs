using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class Shape
    {
        private readonly IReadOnlyList<int> shape;
        private readonly IReadOnlyList<int> dimOffset;
        private readonly int dimOffsetTotal;

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

        public int NDim => shape.Count;

        public int this[int dim] => shape[dim];

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

        public IReadOnlyList<int> DimOffset => dimOffset;
        public IReadOnlyList<int> Shapes => shape;

        public int GetIndexInShape(params int[] select)
        {
            int idx = 0;
            for (int i = 0; i < select.Length; i++)
            {
                idx += dimOffset[i] * select[i];
            }

            return idx;
        }
        public int[] GetDimIndexOutShape(int select)
        {
            int[] dimIndexes = null;
            if (this.dimOffset.Count == 1)
                dimIndexes = new int[] {select};
            else if (this.dimOffset.Count == 2) 
            {
                dimIndexes = new int[dimOffset.Count];

                int remaining = select;

                for (int idx = 0;idx < dimOffset.Count;idx++)
                {
                    dimIndexes[idx] = remaining / dimOffset[idx];
                    remaining -= (dimIndexes[idx] * dimOffset[idx] );
                }    
            }
            else 
            {
                throw new IncorrectShapeException();
            }

            return dimIndexes;
        }

        public int UniShape => shape[0];

        public (int, int) BiShape => shape.Count == 2 ? (shape[0], shape[1]) : (0, 0);

        public (int, int, int) TriShape => shape.Count == 3 ? (shape[0], shape[1], shape[2]) : (0, 0, 0);
    }
}
