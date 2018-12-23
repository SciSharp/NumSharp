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
        public NDArray inv()
        {
            var npInv = new NDArray(this.Storage.DType,this.shape);

            Array matrixStorage = this.Storage.GetData();
            Array invStorage = Array.CreateInstance(npInv.Storage.DType,matrixStorage.Length);

            switch (matrixStorage)
            {
                case double[] np :
                {
                    
                    double[][] matrix = new double[this.Storage.Shape.Dimensions[0]][];
                    for (int idx = 0; idx < matrix.Length;idx++)
                    {
                        matrix[idx] = new double[this.Storage.Shape.Dimensions[1]];
                        for (int jdx = 0; jdx < matrix[idx].Length;jdx++)
                            matrix[idx][jdx] = np[this.Storage.Shape.GetIndexInShape(idx,jdx)];
                    }

                    double[][] matrixInv = MatrixInv.InverseMatrix(matrix);
                    double[] invArray = invStorage as double[];

                    for (int idx = 0; idx < npInv.shape[0]; idx++)
                    {
                        for (int jdx = 0; jdx < npInv.shape[1]; jdx++)
                        {
                            invArray[this.Storage.Shape.GetIndexInShape(idx,jdx)] = matrixInv[idx][jdx];
                        }
                    }
                        
                    break;
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
            }

            npInv.Storage.SetData(invStorage);
            
            return npInv;
        }
    }
}
