using System;
using System.Linq;

namespace NumSharp
{
    public partial class Matrix<TData> : NDArray<TData>
    {
        public Matrix()
        {

        }
        public Matrix(string matrixString)
        {
            string[][] splitted = null;

            if (matrixString.Contains(","))
            {
                splitted = matrixString.Split(';')
                                              .Select(x => x.Split(',') )
                                              .ToArray();
            }
            else 
            {
                splitted = matrixString.Split(';')
                                              .Select(x => x.Split(' ') )
                                              .ToArray();
            }
            
            int dim0 = splitted.Length;
            int dim1 = splitted[0].Length;

            this.Data = new TData[dim0 * dim1];
            this.Shape = new Shape(new int[] { dim0, dim1 });

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): this.StringToDoubleMatrix(splitted); break;
                case ("Float"): ; break;
            }
            
        }
        /// <summary>
        /// Convert a string to Double[,] and store
        /// in Data field of Matrix object
        /// </summary>
        /// <param name="matrix"></param>
        protected void StringToDoubleMatrix(string[][] matrix)
        {
            dynamic matrixData = this;
            for (int idx = 0; idx< matrix.Length;idx++)
            {
                for (int jdx = 0; jdx < matrix[0].Length;jdx++)
                {
                    matrixData[idx,jdx] = Double.Parse(matrix[idx][jdx]);
                }
            }
            this.Data = (TData[])matrixData.Data;
        }
        public override string ToString()
        {
            string returnValue = "matrix([[";

            int dim0 = Shape.Shapes[0];
            int dim1 = Shape.Shapes[1];

            for (int idx = 0; idx < (dim0-1);idx++)
            {
                for (int jdx = 0;jdx < (dim1-1);jdx++)
                {
                    returnValue += (this[idx,jdx] + ", ");
                }
                returnValue += (this[idx,dim1-1] + "],   \n        [");
            }
            for (int jdx = 0; jdx < (dim1-1);jdx++)
            {
                returnValue += (this[dim0-1,jdx] + ", ");
            }
            returnValue += (this[dim0-1,dim1-1] + "]])");

            return returnValue;    
        }
    }
}