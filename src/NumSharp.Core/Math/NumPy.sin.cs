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
        public static NDArrayGeneric<T> sin<T>(this NumPyGeneric<T> np, NDArrayGeneric<T> nd)
        {
            NDArrayGeneric<T> sinArray = new NDArrayGeneric<T>();
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

        public static NDArrayGeneric<NDArrayGeneric<T>> sin<T>(this NumPyGeneric<T> np, NDArrayGeneric<NDArrayGeneric<T>> nd)
        {
            var sinArray = new NDArrayGeneric<NDArrayGeneric<T>>();
            sinArray.Data = new NDArrayGeneric<T>[nd.Size];
            sinArray.Shape = new Shape(nd.Shape.Shapes);

            for (int idx = 0; idx < nd.Size; idx++)
            {
                switch (default(T))
                {
                    case double d:
                        sinArray[idx] = new NDArrayGeneric<T>
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
