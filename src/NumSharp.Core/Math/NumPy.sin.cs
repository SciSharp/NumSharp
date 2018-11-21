using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public static partial class NumPyExtensions
    {
        public static NDArray<T> sin<T>(this NumPy<T> np, NDArray<T> nd)
        {
            NDArray<T> sinArray = new NDArray<T>();
            sinArray.Data = new T[nd.Size];
            sinArray.Shape = new Shape(nd.Shape.Shapes);

            for (int idx = 0; idx < nd.Size; idx++)
            {
                switch (nd[idx])
                {
                    case double d:
                        sinArray[idx] = (T)(object)Math.Sin(d);
                        break;
                    case float d:
                        sinArray[idx] = (T)(object)Math.Sin(d);
                        break;
                    case Complex d:
                        sinArray[idx] = (T)(object)Complex.Sin(d);
                        break;
                }
                
            }

            return sinArray;
        }

        public static NDArray<NDArray<T>> sin<T>(this NumPy<T> np, NDArray<NDArray<T>> nd)
        {
            var sinArray = new NDArray<NDArray<T>>();
            sinArray.Data = new NDArray<T>[nd.Size];
            sinArray.Shape = new Shape(nd.Shape.Shapes);

            for (int idx = 0; idx < nd.Size; idx++)
            {
                switch (default(T))
                {
                    case double d:
                        sinArray[idx] = new NDArray<T>
                        {
                            Data = new T[] { (T)(object)Math.Sin(d) },
                            Shape = new Shape(1)
                        };
                        break;
                }

            }

            return sinArray;
        }
    }
}
