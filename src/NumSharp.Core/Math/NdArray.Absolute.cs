using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Calculate the absolute value element-wise.
        /// </summary>
        /// <returns></returns>
        public NDArray absolute()
        {
            return null;

            //NDArray res = new NDArray(this.dtype,this.Storage.Shape);

            //Array data = this.Storage.GetData();
            //Array resData = res.Storage.GetData();

            //switch (data)
            //{
            //    case double[] dataArr : 
            //    {
            //        var resDataArr = resData as double[];
            //        for (int idx = 0; idx < data.Length;idx++)
            //            resDataArr[idx] = Math.Abs(dataArr[idx]);
            //        break;
            //    }
            //    case float[] dataArr : 
            //    {
            //        var resDataArr = resData as float[];
            //        for (int idx = 0; idx < data.Length;idx++)
            //            resDataArr[idx] = Math.Abs(dataArr[idx]);
            //        break;
            //    }
            //    case int[] dataArr : 
            //    {
            //        var resDataArr = resData as int[];
            //        for (int idx = 0; idx < data.Length;idx++)
            //            resDataArr[idx] = Math.Abs(dataArr[idx]);
            //        break;
            //    }
            //    case Int64[] dataArr : 
            //    {
            //        var resDataArr = resData as Int64[];
            //        for (int idx = 0; idx < data.Length;idx++)
            //            resDataArr[idx] = Math.Abs(dataArr[idx]);
            //        break;
            //    }
            //    case Complex[] dataArr : 
            //    {
            //        var resDataArr = resData as Complex[];
            //        for (int idx = 0; idx < data.Length;idx++)
            //            resDataArr[idx] = Complex.Abs(dataArr[idx]);
            //        break;
            //    }
            //    default :
            //    {
            //        throw new IncorrectTypeException();
            //    }
            //}

            //return res;
        }
    }
}
