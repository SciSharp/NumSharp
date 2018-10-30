using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class MatrixExtensions
    {
        public static Matrix<double> Inv(this Matrix<double> np )
        {
          int dim = np.Data.GetLength(0);
          double[,] result = new double[dim,dim]; // make a copy of matrix
      
          for (int idx = 0; idx < dim; ++idx)
            for (int jdx = 0; jdx < dim; ++jdx)
              result[idx,jdx] = np.Data[idx,jdx];

          double[][] lum; // combined lower & upper
          int[] perm;
          int toggle;
          toggle = MatrixDecompose(np.Data, out lum, out perm);

          double[] b = new double[dim];
          for (int i = 0; i < dim; ++i)
          {
            for (int j = 0; j < dim; ++j)
              if (i == perm[j])
                b[j] = 1.0;
              else
                b[j] = 0.0;
 
            double[] x = Helper(lum, b); // 
            for (int j = 0; j < dim; ++j)
              result[j,i] = x[j];
          }
          var resultMatrix = new Matrix<double>();
          resultMatrix.Data = result;
          return resultMatrix; 
        }
        public static int MatrixDecompose(double[,] m, out double[][] lum, out int[] perm)
        {
          // Crout's LU decomposition for matrix determinant and inverse
          // stores combined lower & upper in lum[][]
          // stores row permuations into perm[]
          // returns +1 or -1 according to even or odd number of row permutations
          // lower gets dummy 1.0s on diagonal (0.0s above)
          // upper gets lum values on diagonal (0.0s below)

          int toggle = +1; // even (+1) or odd (-1) row permutatuions
          int n = m.GetLength(0);

          // make a copy of m[][] into result lu[][]
          lum = new double[n][];
          for(int idx = 0;idx < n;idx++)
          {
            lum[idx] = new double[n];  
          }

          for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
              lum[i][j] = m[i,j];

          // make perm[]
          perm = new int[n];
          for (int i = 0; i < n; ++i)
            perm[i] = i;

          for (int j = 0; j < n - 1; ++j) // process by column. note n-1 
          {
            double max = Math.Abs(lum[j][j]);
            int piv = j;

            for (int i = j + 1; i < n; ++i) // find pivot index
            {
              double xij = Math.Abs(lum[i][j]);
              if (xij > max)
              {
                max = xij;
                piv = i;
              }
            } // i

            if (piv != j)
            {
              double[] tmp = lum[piv]; // swap rows j, piv
              lum[piv] = lum[j];
              lum[j] = tmp;

              int t = perm[piv]; // swap perm elements
              perm[piv] = perm[j];
              perm[j] = t;

              toggle = -toggle;
            }

            double xjj = lum[j][j];
            if (xjj != 0.0)
            {
              for (int i = j + 1; i < n; ++i)
              {
                double xij = lum[i][j] / xjj;
                lum[i][j] = xij;
                for (int k = j + 1; k < n; ++k)
                  lum[i][k] -= xij * lum[j][k];
              }
            }

          } // j

          return toggle;
        } // MatrixDecompose
        static double[] Helper(double[][] luMatrix, double[] b) // helper
        {
          int n = luMatrix.Length;
          double[] x = new double[n];
          b.CopyTo(x, 0);

          for (int i = 1; i < n; ++i)
          {
            double sum = x[i];
            for (int j = 0; j < i; ++j)
              sum -= luMatrix[i][j] * x[j];
            x[i] = sum;
          }

          x[n - 1] /= luMatrix[n - 1][n - 1];
          for (int i = n - 2; i >= 0; --i)
          {
            double sum = x[i];
            for (int j = i + 1; j < n; ++j)
              sum -= luMatrix[i][j] * x[j];
            x[i] = sum / luMatrix[i][i];
          }

          return x;
        } // Helper

    }
}

/* 
using System;
namespace MatrixInverse
{
  class MatrixInverseProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin matrix inverse using Crout LU decomp demo \n");

      double[][] m = MatrixCreate(4, 4); 
      m[0][0] = 3.0; m[0][1] = 7.0; m[0][2] = 2.0; m[0][3] = 5.0;
      m[1][0] = 1.0; m[1][1] = 8.0; m[1][2] = 4.0; m[1][3] = 2.0;
      m[2][0] = 2.0; m[2][1] = 1.0; m[2][2] = 9.0; m[2][3] = 3.0;
      m[3][0] = 5.0; m[3][1] = 4.0; m[3][2] = 7.0; m[3][3] = 1.0;


      Console.WriteLine("Original matrix m is ");
      Console.WriteLine(MatrixAsString(m));

      double d = MatrixDeterminant(m);
      if (Math.Abs(d) < 1.0e-5)
        Console.WriteLine("Matrix has no inverse");

      double[][] inv = MatrixInverse(m);

      Console.WriteLine("Inverse matrix inv is ");
      Console.WriteLine(MatrixAsString(inv));

      double[][] prod = MatrixProduct(m, inv);
      Console.WriteLine("The product of m * inv is ");
      Console.WriteLine(MatrixAsString(prod));

      Console.WriteLine("========== \n");

      double[][] lum;
      int[] perm;
      int toggle = MatrixDecompose(m, out lum, out perm);
      Console.WriteLine("The combined lower-upper decomposition of m is");
      Console.WriteLine(MatrixAsString(lum));

      double[][] lower = ExtractLower(lum);
      double[][] upper = ExtractUpper(lum);

      Console.WriteLine("The lower part of LUM is");
      Console.WriteLine(MatrixAsString(lower));

      Console.WriteLine("The upper part of LUM is");
      Console.WriteLine(MatrixAsString(upper));

      Console.WriteLine("The perm[] array is");
      ShowVector(perm);

      double[][] lowTimesUp = MatrixProduct(lower, upper);
      Console.WriteLine("The product of lower * upper is ");
      Console.WriteLine(MatrixAsString(lowTimesUp));


      Console.WriteLine("\nEnd matrix inverse demo \n");
      Console.ReadLine();

    } // Main

    static void ShowVector(int[] vector)
    {
      Console.Write("   ");
      for (int i = 0; i < vector.Length; ++i)
        Console.Write(vector[i] + " ");
      Console.WriteLine("\n");
    }

    static double[][] MatrixInverse(double[][] matrix)
    {
      // assumes determinant is not 0
      // that is, the matrix does have an inverse
      int n = matrix.Length;
      double[][] result = MatrixCreate(n, n); // make a copy of matrix
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          result[i][j] = matrix[i][j];

      double[][] lum; // combined lower & upper
      int[] perm;
      int toggle;
      toggle = MatrixDecompose(matrix, out lum, out perm);

      double[] b = new double[n];
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < n; ++j)
          if (i == perm[j])
            b[j] = 1.0;
          else
            b[j] = 0.0;
 
        double[] x = Helper(lum, b); // 
        for (int j = 0; j < n; ++j)
          result[j][i] = x[j];
      }
      return result;
    } // MatrixInverse

    
    
    static double MatrixDeterminant(double[][] matrix)
    {
      double[][] lum;
      int[] perm;
      int toggle = MatrixDecompose(matrix, out lum, out perm);
      double result = toggle;
      for (int i = 0; i < lum.Length; ++i)
        result *= lum[i][i];
      return result;
    }

    // ----------------------------------------------------------------

    static double[][] MatrixCreate(int rows, int cols)
    {
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
      return result;
    }

    static double[][] MatrixProduct(double[][] matrixA,
      double[][] matrixB)
    {
      int aRows = matrixA.Length;
      int aCols = matrixA[0].Length;
      int bRows = matrixB.Length;
      int bCols = matrixB[0].Length;
      if (aCols != bRows)
        throw new Exception("Non-conformable matrices");

      double[][] result = MatrixCreate(aRows, bCols);

      for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k) // could use k < bRows
            result[i][j] += matrixA[i][k] * matrixB[k][j];

      return result;
    }

    static string MatrixAsString(double[][] matrix)
    {
      string s = "";
      for (int i = 0; i < matrix.Length; ++i)
      {
        for (int j = 0; j < matrix[i].Length; ++j)
          s += matrix[i][j].ToString("F3").PadLeft(8) + " ";
        s += Environment.NewLine;
      }
      return s;
    }

    static double[][] ExtractLower(double[][] lum)
    {
      // lower part of an LU Doolittle decomposition (dummy 1.0s on diagonal, 0.0s above)
      int n = lum.Length;
      double[][] result = MatrixCreate(n, n);
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < n; ++j)
        {
          if (i == j)
            result[i][j] = 1.0;
          else if (i > j)
            result[i][j] = lum[i][j];
        }
      }
      return result;
    }

    static double[][] ExtractUpper(double[][] lum)
    {
      // upper part of an LU (lu values on diagional and above, 0.0s below)
      int n = lum.Length;
      double[][] result = MatrixCreate(n, n);
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < n; ++j)
        {
          if (i <= j)
            result[i][j] = lum[i][j];
        }
      }
      return result;
    }

    // ----------------------------------------------------------------

  } // Program 

} // ns
*/