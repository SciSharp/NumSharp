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
        public static (NDArray,NDArray) mgrid(NDArray nd1, NDArray nd2)
        {
            return nd1.mgrid(nd2);
        }
    }
}