namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray inv()
        {
            return null;
            //var npInv = new NDArray(this.Storage.DType, this.shape);

            //Array matrixStorage = this.Storage.GetData();
            //Array invStorage = Arrays.Create(npInv.Storage.DType, matrixStorage.Length);

            //switch (matrixStorage)
            //{
            //    case double[] np:
            //    {
            //        double[][] matrix = new double[this.Storage.Shape.Dimensions[0]][];
            //        for (int idx = 0; idx < matrix.Length; idx++)
            //        {
            //            matrix[idx] = new double[this.Storage.Shape.Dimensions[1]];
            //            for (int jdx = 0; jdx < matrix[idx].Length; jdx++)
            //                matrix[idx][jdx] = np[this.Storage.Shape.GetOffset(slice, idx, jdx)];
            //        }

            //        double[][] matrixInv = MatrixInv.InverseMatrix(matrix);
            //        double[] invArray = invStorage as double[];

            //        for (int idx = 0; idx < npInv.shape[0]; idx++)
            //        {
            //            for (int jdx = 0; jdx < npInv.shape[1]; jdx++)
            //            {
            //                invArray[this.Storage.Shape.GetOffset(slice, idx, jdx)] = matrixInv[idx][jdx];
            //            }
            //        }

            //        break;
            //    }

            //    default:
            //    {
            //        throw new IncorrectTypeException();
            //    }
            //}

            //npInv.Storage.ReplaceData(invStorage);

            //return npInv;
        }
    }
}
