using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp.Core
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

            puffer.Storage.SetData(Array.CreateInstance(dtype,puffer.size));
             
            if (diagonalIndex >= 0)
                for(int idx = 0; idx < noOfDiagElement;idx++ )
                   puffer.Storage.SetData(1,idx,idx+diagonalIndex);
            else 
                for(int idx = puffer.Storage.Shape.Dimensions[0]-1; idx > puffer.Storage.Shape.Dimensions[0]-1 - noOfDiagElement;idx-- )
                   puffer.Storage.SetData(1,idx,idx+diagonalIndex);
            
            return puffer;
        }
    }
}
