using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<T> Onces<T>(this NDArray<T> np, params int[] shape)
        {
            int numOFElements = 1;
            for(int idx = 0;idx < shape.Length;idx++)
            {
                numOFElements *= shape[idx];
            }

            np.Data= new T[numOFElements];
            np.Shape = shape.ToList();

            dynamic onceArray = np.Data;

            var elementType = typeof(T);

            switch (elementType.Name)
            {
                case ("Int32"): onceArray = ((int[])onceArray).Select(x => 1).ToArray(); break;
                case ("Double"): onceArray = ((double[])onceArray).Select(x => 1.0).ToArray(); break;
                case ("Float"): onceArray = ((float[])onceArray).Select(x => 1.0).ToArray(); break;
                case ("Complex"): onceArray = ((Complex[])onceArray).Select(x => new Complex(1,0)).ToArray(); break;
                case ("Quaternion"): onceArray = ((Quaternion[])onceArray).Select(x => new Quaternion(new Vector3(0,0,0),1)).ToArray(); break;
            } 

            np.Data = (T[])onceArray;

            return np;
        }
    }
}
