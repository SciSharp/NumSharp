using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray ones(params int[] shapes)
        {
            Type dtype = typeof(double);

            return ones(dtype,shapes);
        }

        public static NDArray ones(Type dtype = null, params int[] shapes)
        {
            dtype = (dtype == null ) ? typeof(double) : dtype;

            return new NDArray(dtype).ones(dtype,shapes);
        }

        /// <summary>
        /// Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="np"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static NDArray ones(Shape shape, Type dtype = null)
        {
            if(dtype == null)
            {
                dtype = typeof(double);
            }

            var nd = new NDArray(dtype, shape);

            switch (dtype.Name)
            {
                case "Int32":
                    nd.Storage.SetData(Enumerable.Range(0, nd.size).Select(x => 1).ToArray());
                    break;

                case "Double":
                    nd.Storage.SetData(Enumerable.Range(0, nd.size).Select(x => 1.0).ToArray());
                    break;

                case "Boolean":
                   nd.Storage.SetData(Enumerable.Range(0, nd.size).Select(x => true).ToArray());
                    break;
            }

            return nd;
        }

        public static NDArray ones<T>(params int[] shapes)
        {
            return ones(new Shape(shapes), typeof(T));
        }
    }

    
}
