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
            var npInv = new NDArray(this.Storage.dtype,this.Shape);

            Array matrixStorage = this.Storage.GetData();
            Array invStorage = Array.CreateInstance(npInv.Storage.dtype,matrixStorage.Length);

            switch (matrixStorage)
            {
                case double[] np :
                {
                    
                    double[][] matrix = new double[this.Storage.Shape[0]][];
                    for (int idx = 0; idx < matrix.Length;idx++)
                    {
                        matrix[idx] = new double[this.Storage.Shape[1]];
                        for (int jdx = 0; jdx < matrix[idx].Length;jdx++)
                            matrix[idx][jdx] = np[this.Storage.Shape.GetIndexInShape(idx,jdx)];
                    }

                    double[][] matrixInv = MatrixInv.InverseMatrix(matrix);
                    double[] invArray = invStorage as double[];

                    for (int idx = 0; idx < npInv.Shape.Shapes[0]; idx++)
                    {
                        for (int jdx = 0; jdx < npInv.Shape.Shapes[1]; jdx++)
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
