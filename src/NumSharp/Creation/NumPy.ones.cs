using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Extensions
{
    public static partial class NumPyExtensions
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="np"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static NDArray<T> ones<T>(this NumPy<T> np, Shape shape)
        {
            var nd = new NDArray<T>();
            nd.Shape = shape;

            switch (default(T))
            {
                case int data:
                    nd.Data = Enumerable.Range(0, nd.Size).Select(x => (T)(object)1).ToArray();
                    break;

                case double data:
                    nd.Data = Enumerable.Range(0, nd.Size).Select(x => (T)(object)1.0).ToArray();
                    break;
            } 

            return nd;
        }
    }
}
