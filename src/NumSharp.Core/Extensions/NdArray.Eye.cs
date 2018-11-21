using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Core.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<int> Eye(this NDArray<int> np,int dim, int diagonalIndex = 0)
        {
            int noOfDiagElement = dim - Math.Abs(diagonalIndex);

            if((np.NDim == 1) && (dim != 1))
            {
                np.Zeros(dim,dim);
            }
            else 
            {
                np.Zeros(np.Shape.Shapes.ToArray());
            }
            
            np.Data = new int[np.Size];

            for(int idx = 0; idx < noOfDiagElement;idx++ )
             {
                 np.Data[diagonalIndex + idx + idx * np.Shape.Shapes[1]] = 1;
             }
            
            return np;
        }
        public static NDArray<double> Eye(this NDArray<double> np,int dim, int diagonalIndex = 0)
        {
            int noOfDiagElement = dim - Math.Abs(diagonalIndex);

            if((np.NDim == 1) && (dim != 1))
            {
                np.Zeros(dim,dim);
            }
            else 
            {
                np.Zeros(np.Shape.Shapes.ToArray());
            }
            
            np.Data = new double[np.Size];

            if (diagonalIndex >= 0)
            {
                for(int idx = 0; idx < noOfDiagElement;idx++ )
                {
                     np.Data[diagonalIndex + idx + idx * np.Shape.Shapes[1]] = 1;
                }
            }
            else
            {
                for(int idx = 0; idx < noOfDiagElement;idx++ )
                {
                     np.Data[(-1)*diagonalIndex * dim + idx + idx * np.Shape.Shapes[1]] = 1;
                }
            }

            return np;
        }
    }
}
