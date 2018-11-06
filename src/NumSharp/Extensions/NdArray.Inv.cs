using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using NumSharp.Shared;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<double> Inv(this NDArray<double> np)
        {
            double[][] matrix = np.ToDotNetArray<double[][]>();

            double[][] matrixInv = MatrixInv.InverseMatrix(matrix);

            NDArray<double> npInv = new NDArray<double>().Zeros(np.Shape.Shapes[0], np.Shape.Shapes[1]);

            for (int idx = 0; idx < npInv.Shape.Shapes[0]; idx++)
                for (int jdx = 0; jdx < npInv.Shape.Shapes[1]; jdx++)
                    npInv[idx, jdx] = matrixInv[idx][jdx];

            return npInv;
        }
    }
}
