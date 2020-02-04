using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public static NumSharp.Generic.NDArray<bool> operator <(NDArray np, int obj)
        {
            return (np < (System.Object)obj);
        }

        public static NumSharp.Generic.NDArray<bool> operator <(NDArray np, object obj)
        {
            var boolTensor = new NDArray(typeof(bool),np.shape);
            var bools = boolTensor.Storage.GetData();

            var npValues = np.Storage.GetData();
            
            switch (npValues.TypeCode)
                {
                    case NPTypeCode.Int32:
                    {
                        int value = Converts.ToInt32(obj);
                        int idx = 0;
                        foreach (var npValue in npValues)
                            {
                            if ((int)npValue < value)
                                bools[idx] = true;
                            idx++;
                            }
                        break;    
                    }
                    case NPTypeCode.Int64:
                    {
                        long value = Converts.ToInt64(obj);
                        int idx = 0;
                        foreach (var npValue in npValues)
                        {
                            if ((long)npValue < value)
                                bools[idx] = true;
                            idx++;
                        }
                        break;
                    }
                    case NPTypeCode.Float:
                    {
                        float value = Converts.ToSingle(obj);
                        int idx = 0;
                        foreach (var npValue in npValues)
                        {
                            if ((float)npValue < value)
                                bools[idx] = true;
                            idx++;
                        }
                        break;
                    }
                    case NPTypeCode.Double:
                    {
                        double value = Converts.ToDouble(obj);
                        int idx = 0;
                        foreach (var npValue in npValues)
                        {
                            if ((double)npValue < value)
                                bools[idx] = true;
                            idx++;
                        }
                        break;
                    }
                    default :
                    {
                        throw new IncorrectTypeException();
                    } 
                }

                return boolTensor.MakeGeneric<bool>();
        }
    }
}
