using System;
using System.Numerics;
using System.Linq;

namespace NumSharp.Shared
{
    internal static partial class Addition
    {
        //start 1 
        internal static double[] AddDoubleArrayToDoubleArray(double[] np1, double[]np2)
        {
            return np1.Select((x,idx) => x + np2[idx]).ToArray();
        }
        //end 1
        //start 2
        internal static double[] AddDoubleToDoubleArray(double[] np1, double np2)
        {
            return np1.Select((x) => x + np2).ToArray();
        }
        //end 2
        //start 3 
        internal static double[][] AddDoubleMatrixToDoubleMatrix(double[][] np1, double[][]np2)
        {
            return np1.Select((x,idx) => x.Select((y,jdx) => y + np2[idx][jdx] ).ToArray()).ToArray();
        }
        //end 3
        //start 4
        internal static double[][] AddDoubleToDoubleMatrix(double[][] np1, double np2)
        {
            return np1.Select((x) => x.Select((y) => y + np2 ).ToArray()).ToArray();
        }
        //end 4
        /* 
        internal static T[] TwoOneDimArray<T>(T[] np1, T[]np2)
        {
            dynamic sum = null;
            dynamic np1Dyn = np1;
            dynamic np2Dyn = np2;

            var dataType = typeof(T);

            switch (dataType.Name)
            {
                case ("Double"): sum = ((double[])np1Dyn).Select((x,idx) => x + ((double[])np2Dyn)[idx]); break;
                case ("Float"): sum = ((float[])np1Dyn).Select((x,idx) => x + ((float[])np2Dyn)[idx]); break;
                case ("Complex"): sum = ((Complex[])np1Dyn).Select((x,idx) => x + ((Complex[])np2Dyn)[idx]); break;
            }

            return sum;
        }
        internal static T[] OneOneDimArrayPlusOffset<T>(T[] np1, T np2)
        {
            dynamic sum = null;
            dynamic np1Dyn = np1;
            dynamic np2Dyn = np2;

            var dataType = typeof(T);

            switch (dataType.Name)
            {
                case ("Double"): sum = ((double[])np1Dyn).Select((x,idx) => x + ((double)np2Dyn)); break;
                case ("Float"): sum = ((float[])np1Dyn).Select((x,idx) => x + ((float)np2Dyn)); break;
                case ("Complex"): sum = ((Complex[])np1Dyn).Select((x,idx) => x + ((Complex)np2Dyn)); break;
            }

            return ((T[])sum);
        }
        internal static T[][] TwoTwoDimArray<T>(T[][] np1,T[][] np2)
        {
            T[][] sum = new T[np1.Length][];
            sum = sum.Select(x => new T[np1[0].Length]).ToArray();

            dynamic sumDyn = sum;

            dynamic np1Dyn = np1;
            dynamic np2Dyn = np2;

            var dataType = typeof(T);

            switch (dataType.Name)
            {
                case ("Double"): sumDyn =  ((double[][])sumDyn).Select((x,idx) => x.Select((y,jdx) => ((double[][])np1Dyn)[idx][jdx] + ((double[][])np2Dyn)[idx][jdx] ) ) ; break;
                case ("Float"): sumDyn =  ((float[][])sumDyn).Select((x,idx) => x.Select((y,jdx) => ((float[][])np1Dyn)[idx][jdx] + ((float[][])np2Dyn)[idx][jdx] ) ) ; break;
                case ("Complex"): sumDyn =  ((Complex[][])sumDyn).Select((x,idx) => x.Select((y,jdx) => ((Complex[][])np1Dyn)[idx][jdx] + ((Complex[][])np2Dyn)[idx][jdx] ) ) ; break;
            }

            return ((T[][])sumDyn);         
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
        */
        
    }
}