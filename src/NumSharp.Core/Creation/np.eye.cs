using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp.Core
{
	public static partial class np
	{
        public static NDArray eye(int dim, int diagonalIndex = 0, Type dtype = null)
        {
            dtype = (dtype == null) ? typeof(double) : dtype;

            return new NDArray(dtype).eye(dim,diagonalIndex);
        }

    }

}