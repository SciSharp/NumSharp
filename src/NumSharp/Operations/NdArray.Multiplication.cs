using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Shared;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public static NDArray<TData> operator *(NDArray<TData> np1, NDArray<TData> np2)
        {
            dynamic sum = null;
            dynamic np1Dyn = np1.Data.ToArray();
            dynamic np2Dyn = np2.Data.ToArray();

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = new NDArray<double>().Array(((double[])Multiplication.MultiplyDoubleArrayToDoubleArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Float"): sum = new NDArray<float>().Array( ((float[])Multiplication.MultiplyfloatArrayTofloatArray (np1Dyn,np2Dyn)).ToList() ); break;
                case ("Complex"): sum = new NDArray<Complex>().Array( ((Complex[])Multiplication.MultiplyComplexArrayToComplexArray(np1Dyn,np2Dyn)).ToList()); break;
                case ("Quaternion"): sum = new NDArray<Quaternion>().Array( ((Quaternion[])Multiplication.MultiplyQuaternionArrayToQuaternionArray(np1Dyn,np2Dyn)).ToList()); break;
                case ("Double[]"): sum = new NDArray<double[]>().Array((double[][]) Multiplication.MultiplyDoubleMatrixToDoubleMatrix(np1Dyn,np2Dyn) ); break;
                case ("Complex[]") : sum = new NDArray<Complex[]>().Array((Complex[][])Multiplication.MultiplyComplexMatrixToComplexMatrix(np1Dyn,np2Dyn) ); break;
                case ("Float[]") : sum = new NDArray<float[]>().Array((float[][])Multiplication.MultiplyfloatMatrixTofloatMatrix(np1Dyn,np2Dyn) ); break;
                case ("Quaternion[]"): sum = new NDArray<Quaternion[]>().Array((Quaternion[][]) Multiplication.MultiplyQuaternionMatrixToQuaternionMatrix (np1Dyn,np2Dyn) ); break;
            }
            
            return (NDArray<TData>) sum;
        }
        public static NDArray<TData> operator *(NDArray<TData> np, TData scalar)
        {
            dynamic sum = null;
            dynamic npDyn = np;
            dynamic scalarDyn = scalar;

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = new NDArray<double>().Array(((NDArray<double>)npDyn).Data.Select((x) => x * (double)scalarDyn)); break;
                case ("Float"): sum = new NDArray<float>().Array(((NDArray<float>)npDyn).Data.Select((x,idx) => x * (float)scalarDyn)); break;
                case ("Complex"): sum = new NDArray<Complex>().Array(((NDArray<Complex>)npDyn).Data.Select((x,idx) => x * (Complex) scalarDyn )); break;
            }
            
            return (NDArray<TData>) sum;
        }
    }
}
