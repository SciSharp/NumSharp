using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using NumSharp.Generic;
using System.Linq;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Determines if NDArray data is same
        /// </summary>
        /// <param name="obj">NDArray to compare</param>
        /// <returns>if reference is same</returns>
        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case NDArray safeCastObj:
                    {
                        var thatData = safeCastObj.Storage?.GetData();
                        if (thatData == null)
                        {
                            return false;
                        }

                        var thisData = this.Storage?.GetData();
                        return thisData == thatData && safeCastObj.shape == this.shape;
                    }
                case int val:
                    return Data<int>(0) == val;
                // Other object is not of Type NDArray, return false immediately.
                default:
                    return false;
            }

        }

        public static NDArray<bool> operator ==(NDArray np, object obj)
        {
            var boolTensor = new NDArray(typeof(bool),np.shape);
            bool[] bools = boolTensor.Storage.GetData() as bool[];

            switch (np.Storage.GetData())
            {
                case int[] values :
                {
                    int value = Convert.ToInt32(obj);                 
                    for(int idx =0; idx < bools.Length;idx++)
                    {
                        if ( values[idx] == value )
                            bools[idx] = true;
                    }
                    break;
                }
                case Int64[] values :
                {
                    Int64 value = Convert.ToInt64(obj);                 
                    for(int idx =0; idx < bools.Length;idx++)
                    {
                        if ( values[idx] == value )
                            bools[idx] = true;
                    }
                    break;
                }
                case float[] values :
                {
                    float value = Convert.ToSingle(obj);                 
                    for(int idx =0; idx < bools.Length;idx++)
                    {
                        if ( values[idx] == value )
                            bools[idx] = true;
                    }
                    break;
                }
                case double[] values :
                {
                    double value = Convert.ToDouble(obj);                 
                    for(int idx =0; idx < bools.Length;idx++)
                    {
                        if ( values[idx] == value )
                            bools[idx] = true;
                    }
                    break;
                }
                case Complex[] values :
                {
                    Complex value = (Complex) obj;                 
                    for(int idx =0; idx < bools.Length;idx++)
                    {
                        if ( values[idx] == value )
                            bools[idx] = true;
                    }
                    break;
                }
                /*case Quaternion[] values :
                {
                    Quaternion value = (Quaternion) obj;                 
                    for(int idx =0; idx < bools.Length;idx++)
                    {
                        if ( values[idx] == value )
                            bools[idx] = true;
                    }
                    break;
                }*/
                default :
                {
                    throw new IncorrectTypeException();
                } 
            }

            return boolTensor.MakeGeneric<bool>();
        }
    }
}
