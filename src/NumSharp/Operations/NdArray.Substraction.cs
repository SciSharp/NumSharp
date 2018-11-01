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
        public static NDArray_Legacy<TData> operator -(NDArray_Legacy<TData> np1, NDArray_Legacy<TData> np2)
        {
            dynamic sub = null;
            dynamic np1Dyn = np1.Data.ToArray();
            dynamic np2Dyn = np2.Data.ToArray();

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sub = new NDArray<double>().Array(((double[])Substraction.SubDoubleArrayFromDoubleArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Float"): sub = new NDArray<float>().Array(((float[])Substraction.SubfloatArrayFromfloatArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Complex"): sub = new NDArray<Complex>().Array(((Complex[])Substraction.SubComplexArrayFromComplexArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Quaternion"): sub = new NDArray<Quaternion>().Array(((Quaternion[])Substraction.SubQuaternionArrayFromQuaternionArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Double[]"): sub = new NDArray<double[]>().Array((double[][])Substraction.SubDoubleMatrixFromDoubleMatrix(np1Dyn,np2Dyn) ); break;
                case ("Complex[]") : sub = new NDArray<Complex[]>().Array((Complex[][])Substraction.SubComplexMatrixFromComplexMatrix(np1Dyn,np2Dyn) ); break;
                case ("Float[]") : sub = new NDArray<float[]>().Array((float[][])Substraction.SubfloatMatrixFromfloatMatrix(np1Dyn,np2Dyn) ); break;
                case ("Quaternion[]"): sub = new NDArray<Quaternion[]>().Array((Quaternion[][])Substraction.SubQuaternionMatrixFromQuaternionMatrix(np1Dyn,np2Dyn) ); break;
            }
            
            return (NDArray_Legacy<TData>) sub;
        }
        public static NDArray_Legacy<TData> operator -(NDArray_Legacy<TData> np, TData scalar)
        {
            dynamic sub = null;
            dynamic npDyn = np;
            dynamic scalarDyn = scalar;

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sub = new NDArray<double>().Array(((NDArray_Legacy<double>)npDyn).Data.Select((x) => x - (double)scalarDyn)); break;
                case ("Float"): sub = new NDArray<float>().Array(((NDArray_Legacy<float>)npDyn).Data.Select((x,idx) => x - (float)scalarDyn)); break;
                case ("Complex"): sub = new NDArray<Complex>().Array(((NDArray_Legacy<Complex>)npDyn).Data.Select((x,idx) => x - (Complex) scalarDyn )); break;
            }
            
            return (NDArray_Legacy<TData>) sub;
        }
        protected NDArray_Legacy<double[]> SubstractDoubleMatrix(double[][] np2)
        {
            int dim0 = np2.Length;
            int dim1 = np2[0].Length;

            dynamic np1Dyn = this.Data;
            double[][] np1 = (double[][]) np1Dyn;

            double[][] sub = new double[dim0][].Select(x => new double[dim1]).ToArray();

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sub[idx][jdx] = np1[idx][jdx] - np2[idx][jdx];
                }
            }

            var subNDArray = new NDArray_Legacy<double[]>();
            subNDArray.Data = sub;

            return subNDArray;
        }
        protected NDArray_Legacy<Complex[]> SubstractComplexMatrix(Complex[][] np2)
        {
            int dim0 = np2.Length;
            int dim1 = np2[0].Length;

            dynamic np1Dyn = this.Data;
            Complex[][] np1 = (Complex[][]) np1Dyn;

            Complex[][] sub = new Complex[dim0][].Select(x => new Complex[dim1]).ToArray();

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sub[idx][jdx] = np1[idx][jdx] - np2[idx][jdx];
                }
            }

            var subNDArray = new NDArray_Legacy<Complex[]>();
            subNDArray.Data = sub;

            return subNDArray;
        }
        protected NDArray_Legacy<float[]> SubstractFloatMatrix(float[][] np2)
        {
            int dim0 = np2.Length;
            int dim1 = np2[0].Length;

            dynamic np1Dyn = this.Data;
            float[][] np1 = (float[][]) np1Dyn;

            float[][] sub = new float[dim0][].Select(x => new float[dim1]).ToArray();

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sub[idx][jdx] = np1[idx][jdx] - np2[idx][jdx];
                }
            }

            var subNDArray = new NDArray_Legacy<float[]>();
            subNDArray.Data = sub;

            return subNDArray;
        }
    }
}
