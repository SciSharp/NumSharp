using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using System.Text;

namespace NumSharp.Shared
{
   internal static partial class Multiplication
   {
        //start 1 
        internal static Quaternion[] MultiplyQuaternionArrayToQuaternionArray(Quaternion[] np1, Quaternion[]np2)
        {
            return np1.Select((x,idx) => x * np2[idx]).ToArray();
        }
        //end 1
        //start 2
        internal static Quaternion[] MultiplyQuaternionToQuaternionArray(Quaternion[] np1, Quaternion np2)
        {
            return np1.Select((x) => x * np2).ToArray();
        }
        //end 2
        //start 3 
        internal static Quaternion[][] MultiplyQuaternionMatrixToQuaternionMatrix(Quaternion[][] np1, Quaternion[][]np2)
        {
            return np1.Select((x,idx) => x.Select((y,jdx) => y * np2[idx][jdx] ).ToArray()).ToArray();
        }
        //end 3
        //start 4
        internal static Quaternion[][] MultiplyQuaternionToQuaternionMatrix(Quaternion[][] np1, Quaternion np2)
        {
            return np1.Select((x) => x.Select((y) => y * np2 ).ToArray()).ToArray();
        }
        //end 4
   }
}
