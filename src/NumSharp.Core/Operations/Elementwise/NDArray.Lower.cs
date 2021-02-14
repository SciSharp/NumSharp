namespace NumSharp
{
    public partial class NDArray
    {
        public static NumSharp.Generic.NDArray<bool> operator <(NDArray lhs, NDArray rhs)
        {
            return rhs > lhs;
        }

        public static NumSharp.Generic.NDArray<bool> operator <(NDArray np, int obj)
        {
            return (np < (System.Object)obj);
        }

        public static NumSharp.Generic.NDArray<bool> operator <(NDArray np, object obj)
        {
            return null;
            // var boolTensor = new NDArray(typeof(bool),np.shape);
            //bool[] bools = boolTensor.Storage.GetData() as bool[];

            //switch (np.Storage.GetData())
            //{
            //    case int[] values :
            //    {
            //        int value = Converts.ToInt32(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] < value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case Int64[] values :
            //    {
            //        Int64 value = Converts.ToInt64(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] < value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case float[] values :
            //    {
            //        float value = Converts.ToSingle(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] < value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case double[] values :
            //    {
            //        double value = Converts.ToDouble(obj);                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] < value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    default :
            //    {
            //        throw new IncorrectTypeException();
            //    } 
            //}

            //return boolTensor.MakeGeneric<bool>();
        }
    }
}
