using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp
{
	public static partial class np
	{
        public static NDArray squeeze(NDArray nd, int axis = -1)
        {
            return nd.reshape(nd.shape.Where(x => x > 1).ToArray());
        }
    }
}