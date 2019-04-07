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
        /// <summary>
        /// Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public static NDArray squeeze(NDArray nd, int axis = -1)
        {
            return nd.reshape(nd.shape.Where(x => x > 1).ToArray());
        }
    }
}
