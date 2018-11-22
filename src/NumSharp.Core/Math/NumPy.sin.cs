using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        public NDArray sin(NDArray nd)
        {
            var sinArray = new NDArray(nd.dtype)
            {
                Shape = new Shape(nd.Shape.Shapes)
            };

            for (int idx = 0; idx < nd.Size; idx++)
            {
                switch (nd[idx])
                {
                    case double d:
                        sinArray[idx] = Math.Sin(d);
                        break;
                    case float d:
                        sinArray[idx] = Math.Sin(d);
                        break;
                    case Complex d:
                        sinArray[idx] = Complex.Sin(d);
                        break;
                }
                
            }

            return sinArray;
        }
    }
}
