using System;
using System.Linq;

namespace NumSharp.UnitTest.Shared
{
    internal static class ArrayHelper
    {
        internal static bool CompareTwoJaggedArrays<T>(T[][] np1, T[][] np2 )
        {
            bool arraysSame = true;

            for(int idx = 0; idx < np1.Length;idx++)
            {
                for (int jdx = 0; jdx < np2[0].Length;jdx++)
                {
                    if (!np1[idx][jdx].Equals(np2[idx][jdx]))
                    {
                        arraysSame = false;
                    }
                    else 
                    {
                        // pass
                    }
                }
            }
            return arraysSame;
        }
        internal static T[][] CreateJaggedArrayByMatrix<T>(T[,] np)
        {
            int dim0 = np.GetLength(0);
            int dim1 = np.GetLength(1);

            var returnResult = new T[dim0][];
            returnResult = returnResult.Select(x => new T[dim1]).ToArray();

            returnResult = returnResult.Select((x,idx) => x.Select((y,jdx) => np[idx,jdx]).ToArray()).ToArray();

            return returnResult;
        }    
    }
}