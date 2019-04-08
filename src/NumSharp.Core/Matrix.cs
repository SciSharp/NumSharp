using System;
using System.Linq;

namespace NumSharp
{
    public partial class matrix: NDArray
    {
        public matrix(NDArray data) : base(data.dtype)
        {
            Storage.SetData(data.Array);
            Storage.Reshape(data.shape);
        }

        public matrix(string matrixString, Type dtype) : base(dtype)
        {
            string[][] splitted = null;

            dtype = (dtype == null) ? np.float64 : dtype;
            
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

            var shape = new Shape( new int[] { dim0, dim1 });

            this.Storage.Allocate(shape, dtype);

            switch (this.dtype.Name)
            {
                case "Double":
                    StringToDoubleMatrix(splitted);
                    break;
                case "Float":
                    break;
                default:
                    throw new NotImplementedException($"matrix {this.dtype.Name}");
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

            int dim0 = shape[0];
            int dim1 = shape[1];

            switch (dtype.Name)
            {
                case "Double":
                    {
                        for (int idx = 0; idx < (dim0 - 1); idx++)
                        {
                            for (int jdx = 0; jdx < (dim1 - 1); jdx++)
                            {
                                returnValue += Data<double>(idx, jdx) + ", ";
                            }
                            returnValue += Data<double>(idx, dim1 - 1) + "],   \n        [";
                        }

                        for (int jdx = 0; jdx < (dim1 - 1); jdx++)
                        {
                            returnValue += Data<double>(dim0 - 1, jdx) + ", ";
                        }

                        returnValue += Data<double>(dim0 - 1, dim1 - 1) + "]])";
                    }
                    break;
            }

            return returnValue;    
        }
    }
}
