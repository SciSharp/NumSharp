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
                puffer = new NDArray(this.dtype, dim,dim);
            }
            else 
            {
                puffer = new NDArray(this.dtype,this.shape.Shapes.ToArray());
            }
            
            puffer.Storage.SetData(new int[puffer.size]);

            int[] storageArr = puffer.Storage.GetData<int>();

            for(int idx = 0; idx < noOfDiagElement;idx++ )
            {
                 storageArr[diagonalIndex + idx + idx * puffer.shape.Shapes[1]] = 1;
            }

            this.Storage = puffer.Storage;
            
            return puffer;
        }
    }
}
