using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for Not a Number.
        /// </summary>
        /// <returns>The result is returned as a boolean array.</returns>
        public override NDArray<bool> IsNan(NDArray a)
        {
            return null;
            //var result = new NDArray<bool>(a.shape);
            //var data = a.Array;
            //bool[] res = result.Array;

            //switch (data)
            //{
            //    case double[] arr:
            //        {
            //            for (int i = 0; i < arr.Length; i++)
            //                res[i] = double.IsNaN(arr[i]);
            //            break;
            //        }
            //    case float[] arr:
            //        {
            //            for (int i = 0; i < arr.Length; i++)
            //                res[i] = float.IsNaN(arr[i]);
            //            break;
            //        }
            //    case int[] arr:
            //        {
            //            //for (int i = 0; i < data.Length; i++)
            //            //    res[i] = false;
            //            break;
            //        }
            //    case Int64[] arr:
            //        {
            //            //for (int i = 0; i < data.Length; i++)
            //            //    res[i] = false;
            //            break;
            //        }
            //    case Complex[] arr:
            //        {
            //            throw new NotImplementedException("Checking Complex array for NaN is not implemented yet.");
            //        }
            //    default:
            //        {
            //            throw new IncorrectTypeException();
            //        }
            //}
            //return result;
        }
    }
}
