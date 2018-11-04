using System;
using System.Numerics;
using System.Linq;

namespace NumSharp.Shared
{
    internal static partial class MatrixDecomposition
    {
        //start 1 
        internal static int MatrixDecompose(double[][] m, out double[][] lum, out int[] perm)
        {
          // Crout's LU decomposition for matrix determinant and inverse
        // stores combined lower & upper in lum[][]
        // stores row permuations into perm[]
        // returns +1 or -1 according to even or odd number of row permutations
        // lower gets dummy 1.0s on diagonal (0.0s above)
        // upper gets lum values on diagonal (0.0s below)

        int toggle = +1; // even (+1) or odd (-1) row permutatuions
        int n = m.Length;

        // make a copy of m[][] into result lu[][]
        lum = new double[n][].Select(x => new double[n]).ToArray();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                lum[i][j] = m[i][j];


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
        } // MatrixDecompose        }
        //end 1
    }
}