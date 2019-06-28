using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray eye(int dim, int diagonalIndex = 0)
        {
            int noOfDiagElement = dim - Math.Abs(diagonalIndex);

            NDArray puffer = null;

            if((ndim == 1) && (dim != 1))
            {
                puffer = new NDArray(this.dtype, new Shape(dim, dim));
            }
            else 
            {
                puffer = new NDArray(this.dtype,this.shape);
            }

            puffer.Storage.ReplaceData(Arrays.Create(dtype,puffer.size));

            if (diagonalIndex >= 0)
            {
                switch (Type.GetTypeCode(dtype))
                {
                    case TypeCode.Double:
                        for (int idx = 0; idx < noOfDiagElement; idx++)
                            puffer.SetData(1d, idx, idx + diagonalIndex);
                        break;
                    case TypeCode.Int32:
                        for (int idx = 0; idx < noOfDiagElement; idx++)
                            puffer.SetData(1, idx, idx + diagonalIndex);
                        break;
                    default:
                        throw new NotImplementedException($"eye {dtype.Name}"); 
                }
            }
            else
            {
                switch (Type.GetTypeCode(dtype))
                {
                    case TypeCode.Double:
                        for (int idx = puffer.shape[0] - 1; idx > puffer.shape[0] - 1 - noOfDiagElement; idx--)
                            puffer.SetData(1d, idx, idx + diagonalIndex);
                        break;
                    case TypeCode.Int32:
                        for (int idx = puffer.shape[0] - 1; idx > puffer.shape[0] - 1 - noOfDiagElement; idx--)
                            puffer.SetData(1d, idx, idx + diagonalIndex);
                        break;
                    default:
                        throw new NotImplementedException($"eye {dtype.Name}");
                }
            }


            return puffer;
        }
    }
}
