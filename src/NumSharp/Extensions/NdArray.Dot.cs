using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Shared;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<TData> Dot<TData>(this NDArray<TData> np1, NDArray<TData> np2)
        {
            dynamic prod = new NDArray<TData>();
            
            dynamic np1Dyn = np1.Data.ToArray();
            dynamic np2Dyn = np2.Data.ToArray();

            int dim0 = np1.Shape[0];
            int dim1 = np2.Shape[1];
            int iterator = np1.Shape[1];

            prod.Shape = new int[] {dim0,dim1};

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): prod.Data = MatrixMultiplication.MatrixMultiplyDoubleMatrix(np1Dyn,np2Dyn,dim0,dim1,iterator); break;
                case ("Float"): prod.Data = MatrixMultiplication.MatrixMultiplyfloatMatrix(np1Dyn,np2Dyn,dim0,dim1,iterator); break;
                case ("Complex"): prod.Data = MatrixMultiplication.MatrixMultiplyComplexMatrix(np1Dyn,np2Dyn,dim0,dim1,iterator); break;
                case ("Quaternion"): prod.Data = MatrixMultiplication.MatrixMultiplyQuaternionMatrix(np1Dyn,np2Dyn,dim0,dim1,iterator) ; break;
            }
            
            return ((NDArray<TData>) prod);
        }
        public static NDArray<double> Dot(this NDArray_Legacy<double> np, NDArray_Legacy<double> np2)
        {
            double[] array1Double = np.Data.ToArray();
            double[] array2Double = np2.Data.ToArray();
            
            double sum = 0;

            for (int idx = 0; idx < array1Double.Length; idx++)
            {
                sum += array1Double[idx] * array2Double[idx];
            }

            return new NDArray<double>().Array(new double[]{sum});
        }
        public static int Dot(this NDArray_Legacy<int> np, NDArray_Legacy<int> np2)
        {
            int[] array1Double = np.Data.ToArray();
            int[] array2Double = np2.Data.ToArray();
            
            int sum = 0;

            for (int idx = 0; idx < array1Double.Length; idx++)
            {
                sum += array1Double[idx] * array2Double[idx];
            }

            return sum;
        }
        public static NDArray_Legacy<NDArray_Legacy<double>> Dot(this NDArray_Legacy<NDArray_Legacy<double>> np,NDArray_Legacy<NDArray_Legacy<double>> np2)
        {
            // the following lines are slow performance I guess
            int numOfLines = np.Length;
            int numOfColumns = np2[0].Length;

            NDArray_Legacy<NDArray_Legacy<double>> result = new NDArray_Legacy<NDArray_Legacy<double>>();
            result.Data = new NDArray_Legacy<double>[numOfLines];
            
            for (int idx =0; idx < numOfLines;idx++)
            {
                result.Data[idx] = new NDArray_Legacy<double>();
                result.Data[idx].Data = new double[numOfColumns];    
            }

            for (int idx = 0; idx < numOfLines;idx++)
            {
                for( int jdx = 0; jdx < numOfColumns; jdx++)
                {
                    result[idx][jdx] = 0;
                    for (int kdx = 0; kdx < np2.Length; kdx++)
                    {
                        result[idx][jdx] += np[idx][kdx] * np2[kdx][jdx];
                    }
                }
            }

            return result;
        }
        public static NDArray<Complex> Dot(this NDArray<Complex> np, NDArray<Complex> np2)
        {
            var array1Complex = np.Data.ToArray();
            var array2Complex = np2.Data.ToArray();

            Complex sum = 0;

            for (int idx = 0; idx < array1Complex.Length; idx++)
            {
                sum += array1Complex[idx] * array2Complex[idx];
            }

            return new NDArray<Complex>().Array(new Complex[]{new Complex(sum.Real,sum.Imaginary)} );
        }
        public static NDArray_Legacy<double> Dot(this NDArray_Legacy<double> np, double scalar)
        {
            double[] array1Double = np.Data.ToArray();

            array1Double = array1Double.Select(x => scalar * x).ToArray();
            
            np.Data = array1Double;

            return np;
        }
        public static NDArray_Legacy<float> Dot(this NDArray_Legacy<float> np, float scalar)
        {
            np.Data = np.Data.Select(x => x * scalar).ToArray();
            return np;
        }
        public static NDArray_Legacy<int> Dot(this NDArray_Legacy<int> np, int scalar)
        {
            np.Data = np.Data.Select(x => x * scalar).ToArray();
            return np;
        }
    }
}
