using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static Matrix<double> AsMatrix(this NDArray_Legacy<NDArray_Legacy<double>> np)
        {
            Matrix<double> npAsMatrix = new Matrix<double>();

            int dim0 = np.Length;
            int dim1 = np.Data[0].Length;

            npAsMatrix.Shape = new Shape(new int[] { dim0, dim1 });
            npAsMatrix.Data = new double[dim0 * dim1];

            for (int idx = 0; idx < dim0;idx++)
            {
                for (int jdx = 0;jdx < dim1;jdx++)
                {
                    npAsMatrix[idx,jdx] = np[idx][jdx];
                }
            }
            
            return npAsMatrix;
        }
    }
}
