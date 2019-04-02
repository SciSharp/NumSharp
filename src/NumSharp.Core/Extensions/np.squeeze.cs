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
        public static NDArray argsort<T>(NDArray nd, int axis = -1) => nd.argsort<T>(axis);
    }
}