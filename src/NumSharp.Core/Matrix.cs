using System;
using System.Linq;

namespace NumSharp.Core
{
    public partial class Matrix: NDArray
    {
        public Matrix(NDArray data, Type dtype = null)
        {
            if (dtype == null)
            {
                dtype = data.dtype;
                this.Storage.dtype = data.dtype;
            }

            this.Storage.Shape = new Shape(data.shape.Shapes);

            switch (data.dtype.Name)
            {
                case "Double":
                    Set(data.float64);
                    break;
                case "Int32":
                    Set(data.int32);
                    break;
                case "Complex":
                    Set(data.complex128);
                    break;
            }
        }

        public Matrix(string matrixString, Type dtype = null)
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

            if (dtype == null)
            {
                dtype = typeof(double);
                this.Storage.dtype = typeof(double);
            }

            this.Storage.Shape = new Shape(new int[] { dim0, dim1 });
            this.Storage.Allocate(shape.Size);

            switch (this.dtype.Name)
            {
                case "Double": StringToDoubleMatrix(splitted); break;
                case "Float": ; break;
            }
            
        }
        /// <summary>
        /// Convert a string to Double[,] and store
        /// in Data field of Matrix object
        /// </summary>
        /// <param name="matrix"></param>
        protected void StringToDoubleMatrix(string[][] matrix)
        {
            for (int idx = 0; idx< matrix.Length;idx++)
            {
                for (int jdx = 0; jdx < matrix[0].Length;jdx++)
                {
                    this[idx,jdx] = Double.Parse(matrix[idx][jdx]);
                }
            }
        }

        public override string ToString()
        {
            string returnValue = "matrix([[";

            int dim0 = shape.Shapes[0];
            int dim1 = shape.Shapes[1];

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