using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp
{
    public partial class Matrix<TData>
    {
        public static Matrix<TData> operator +(Matrix<TData> np, Matrix<TData> np2)
        {
            dynamic sum = null;
            dynamic npDyn = np;
            dynamic np2Dyn = np2;

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = np.AdditionDoubleMatrix(np2Dyn.Data); break;
                case ("Float"): sum = np.AdditionFloatMatrix(np2Dyn.Data); break;
                case ("Complex"): sum = np.AdditionComplexMatrix(np2Dyn.Data); break;
            }

            var returnValue = new Matrix<TData>();
            returnValue.Data = (TData[,]) sum;
            
            return returnValue;
        }
        protected double[,] AdditionDoubleMatrix(double[,] np2)
        {
            int dim0 = np2.GetLength(0);
            int dim1 = np2.GetLength(1);

            dynamic np1Dyn = this.Data;
            double[,] np1 = (double[,]) np1Dyn;

            double[,] sum = new double[dim0,dim1];

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sum[idx,jdx] = np1[idx,jdx] + np2[idx,jdx];
                }
            }

            return sum;
        }
        protected Complex[,] AdditionComplexMatrix(Complex[,] np2)
        {
            int dim0 = np2.GetLength(0);
            int dim1 = np2.GetLength(1);

            dynamic np1Dyn = this.Data;
            Complex[,] np1 = (Complex[,]) np1Dyn;

            Complex[,] sum = new Complex[dim0,dim1];

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sum[idx,jdx] = np1[idx,jdx] + np2[idx,jdx];
                }
            }

            return sum;
        }
        protected float[,] AdditionFloatMatrix(float[,] np2)
        {
            int dim0 = np2.GetLength(0);
            int dim1 = np2.GetLength(1);

            dynamic np1Dyn = this.Data;
            float[,] np1 = (float[,]) np1Dyn;

            float[,] sum = new float[dim0,dim1];

            for(int idx = 0; idx < dim0; idx++)
            {
                for(int jdx = 0; jdx < dim1; jdx++)
                {
                    sum[idx,jdx] = np1[idx,jdx] + np2[idx,jdx];
                }
            }

            return sum;
        }
    }
}
