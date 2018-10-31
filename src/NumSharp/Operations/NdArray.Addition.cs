using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Shared;

namespace NumSharp
{
    public partial class NDArray_Legacy<TData>
    {
        public static NDArray_Legacy<TData> operator +(NDArray_Legacy<TData> np, NDArray_Legacy<TData> np2)
        {
            dynamic sum = null;
            dynamic np1Dyn = np.Data.ToArray();
            dynamic np2Dyn = np2.Data.ToArray();

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = new NDArray_Legacy<double>().Array(((double[])Addition.AddDoubleArrayToDoubleArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Float"): sum = new NDArray_Legacy<float>().Array( ((float[])Addition.AddfloatArrayTofloatArray (np1Dyn,np2Dyn)).ToList() ); break;
                case ("Complex"): sum = new NDArray_Legacy<Complex>().Array( ((Complex[])Addition.AddComplexArrayToComplexArray(np1Dyn,np2Dyn)).ToList()); break;
                case ("Quaternion"): sum = new NDArray_Legacy<Quaternion>().Array( ((Quaternion[])Addition.AddQuaternionArrayToQuaternionArray(np1Dyn,np2Dyn)).ToList()); break;
                case ("Double[]"): sum = new NDArray_Legacy<double[]>().Array((double[][]) Addition.AddDoubleMatrixToDoubleMatrix (np1Dyn,np2Dyn) ); break;
                case ("Complex[]") : sum = new NDArray_Legacy<Complex[]>().Array((Complex[][])Addition.AddComplexMatrixToComplexMatrix(np1Dyn,np2Dyn) ); break;
                case ("Float[]") : sum = new NDArray_Legacy<float[]>().Array((float[][])Addition.AddfloatMatrixTofloatMatrix(np1Dyn,np2Dyn) ); break;
                case ("Quaternion[]"): sum = new NDArray_Legacy<Quaternion[]>().Array((Quaternion[][])Addition.AddQuaternionMatrixToQuaternionMatrix(np1Dyn,np2Dyn) ); break;
            }
            
            return (NDArray_Legacy<TData>) sum;
        }
        public static NDArray_Legacy<TData> operator +(NDArray_Legacy<TData> np, TData scalar)
        {
            dynamic sum = null;
            dynamic npDyn = np;
            dynamic scalarDyn = scalar;

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = new NDArray_Legacy<double>().Array(((NDArray_Legacy<double>)npDyn).Data.Select((x) => x + (double)scalarDyn)); break;
                case ("Float"): sum = new NDArray_Legacy<float>().Array(((NDArray_Legacy<float>)npDyn).Data.Select((x,idx) => x + (float)scalarDyn)); break;
                case ("Complex"): sum = new NDArray_Legacy<Complex>().Array(((NDArray_Legacy<Complex>)npDyn).Data.Select((x,idx) => x + (Complex) scalarDyn )); break;
                //case ("Double[]") : sum = np ;break; 
            }
            
            return (NDArray_Legacy<TData>) sum;
        }
        protected NDArray_Legacy<double[]> SumDoubleMatrix(double[][] np2)
        {
            int dim0 = np2.Length;
            int dim1 = np2[0].Length;

            dynamic np1Dyn = this.Data;
            double[][] np1 = (double[][]) np1Dyn;

            double[][] sum = new double[dim0][].Select(x => new double[dim1]).ToArray();

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sum[idx][jdx] = np1[idx][jdx] + np2[idx][jdx];
                }
            }

            var sumNDArray = new NDArray_Legacy<double[]>();
            sumNDArray.Data = sum;

            return sumNDArray;
        }
        protected NDArray_Legacy<Complex[]> SumComplexMatrix(Complex[][] np2)
        {
            int dim0 = np2.Length;
            int dim1 = np2[0].Length;

            dynamic np1Dyn = this.Data;
            Complex[][] np1 = (Complex[][]) np1Dyn;

            Complex[][] sum = new Complex[dim0][].Select(x => new Complex[dim1]).ToArray();

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sum[idx][jdx] = np1[idx][jdx] + np2[idx][jdx];
                }
            }

            var sumNDArray = new NDArray_Legacy<Complex[]>();
            sumNDArray.Data = sum;

            return sumNDArray;
        }
        protected NDArray_Legacy<float[]> SumFloatMatrix(float[][] np2)
        {
            int dim0 = np2.Length;
            int dim1 = np2[0].Length;

            dynamic np1Dyn = this.Data;
            float[][] np1 = (float[][]) np1Dyn;

            float[][] sum = new float[dim0][].Select(x => new float[dim1]).ToArray();

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sum[idx][jdx] = np1[idx][jdx] + np2[idx][jdx];
                }
            }

            var sumNDArray = new NDArray_Legacy<float[]>();
            sumNDArray.Data = sum;

            return sumNDArray;
        }
    }
}
