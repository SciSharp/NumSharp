using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for finiteness (not infinity or not Not a Number).
        /// </summary>
        /// <param name="a"></param>
        /// <returns>The result is returned as a boolean array.</returns>
        public override NDArray<bool> IsFinite(NDArray a)
        {
            //var result = new NDArray<bool>(a.shape);
            //var data = a.Array;
            //bool[] res = result.Array;

            //switch (data)
            //{
            //    case double[] arr:
            //        {
            //            for (int i = 0; i < arr.Length; i++)
            //                res[i] = !double.IsInfinity(arr[i]) && !double.IsNaN(arr[i]);
            //            break;
            //        }
            //    case float[] arr:
            //        {
            //            for (int i = 0; i < arr.Length; i++)
            //                res[i] = !float.IsInfinity(arr[i]) && !float.IsNaN(arr[i]);
            //            break;
            //        }
            //    case int[] arr:
            //        {
            //            for (int i = 0; i < data.Length; i++)
            //                res[i] = true;
            //            break;
            //        }
            //    case Int64[] arr:
            //        {
            //            for (int i = 0; i < data.Length; i++)
            //                res[i] = true;
            //            break;
            //        }
            //    case Complex[] arr:
            //        {
            //            throw new NotImplementedException("Checking Complex array for Infinity or NaN is not implemented yet.");
            //        }
            //    default:
            //        {
            //            throw new IncorrectTypeException();
            //        }
            //}
            //return result;
            return null;
        }
    }
}
