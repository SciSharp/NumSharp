namespace NumSharp
{
    public partial class NDArray
    {
        public static NumSharp.Generic.NDArray<bool> operator !=(NDArray np, object obj)
        {
            return null;
            // var boolTensor = new NDArray(typeof(bool),np.shape);
            //bool[] bools = boolTensor.Storage.GetData() as bool[];

            //switch (np.Storage.GetData())
            //{
            //    case int[] values :
            //    {
            //        int value = (int) obj;                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] != value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case Int64[] values :
            //    {
            //        Int64 value = (Int64) obj;                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] != value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case float[] values :
            //    {
            //        float value = (float) obj;                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] != value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case double[] values :
            //    {
            //        double value = (double) obj;                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] != value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    case Complex[] values :
            //    {
            //        Complex value = (Complex) obj;                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] != value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }
            //    /*case Quaternion[] values :
            //    {
            //        Quaternion value = (Quaternion) obj;                 
            //        for(int idx =0; idx < bools.Length;idx++)
            //        {
            //            if ( values[idx] != value )
            //                bools[idx] = true;
            //        }
            //        break;
            //    }*/
            //    default :
            //    {
            //        throw new IncorrectTypeException();
            //    } 
            //}

            //return boolTensor.MakeGeneric<bool>();
        }
    }
}
