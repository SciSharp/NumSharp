using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using NumSharp.Core.Shared;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray inv<T>()
        {
            var npInv = new NDArray(typeof(T), Shape);

            switch (npInv[0])
            {
                case double np:
                    {
                        double[][] matrix = null;// np.ToDotNetArray<double[][]>();

                        double[][] matrixInv = MatrixInv.InverseMatrix(matrix);

                        for (int idx = 0; idx < npInv.Shape.Shapes[0]; idx++)
                            for (int jdx = 0; jdx < npInv.Shape.Shapes[1]; jdx++)
                                npInv[idx, jdx] = matrixInv[idx][jdx];
                        break;
                    }
                default:
                    {
                        throw new Exception("This method was not implemented for this Type : " + typeof(T).Name);
                    }
            }

            return npInv;
        }
    }

    public partial class NDArrayGeneric<T>
    {
        public NDArrayGeneric<T> inv()
        {
            NDArrayGeneric<T> npInv = new NDArrayGeneric<T>();
            npInv.Shape = new Shape(this.Shape.Shapes);
            npInv.Data = new T[this.Data.Length];

            switch (this)
            {
                case NDArrayGeneric<double> np :
                {
                    NDArrayGeneric<double> npInvDouble = npInv as NDArrayGeneric<double>;
                    double[][] matrix = np.ToDotNetArray<double[][]>();

                    double[][] matrixInv = MatrixInv.InverseMatrix(matrix);

                    for (int idx = 0; idx < npInv.Shape.Shapes[0]; idx++)
                        for (int jdx = 0; jdx < npInv.Shape.Shapes[1]; jdx++)
                            npInvDouble[idx, jdx] = matrixInv[idx][jdx];
                    break;
                }
                default : 
                {
                    throw new Exception("This method was not implemented for this Type : " + typeof(T).Name);
                }
            }
            
            return npInv;
        }
    }
}
