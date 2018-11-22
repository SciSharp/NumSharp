using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="np"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public NDArray ones<T>(Shape shape)
        {
            var nd = new NDArray(typeof(T), shape);

            switch (default(T))
            {
                case int data:
                    nd.Set(Enumerable.Range(0, nd.Size).Select(x => 1).ToArray());
                    break;

                case double data:
                    nd.Set(Enumerable.Range(0, nd.Size).Select(x => 1.0).ToArray());
                    break;
            }

            return nd;
        }

        public NDArray ones<T>(params int[] shapes)
        {
            return ones<T>(new Shape(shapes));
        }
    }

    public static partial class NumPyExtensions
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="np"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static NDArrayGeneric<T> ones<T>(this NumPyGeneric<T> np, Shape shape)
        {
            var nd = new NDArrayGeneric<T>();
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
