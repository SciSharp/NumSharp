using System;
using NumSharp;
using System.Linq;
using System.Collections.Generic;

public static partial class numpy 
{
    public static dynamic array(IList<object> list, string dataType)
    {
        Console.WriteLine(list);
        Console.WriteLine(dataType);

        dynamic returnArray = null;

        switch (dataType)
        {
            case "Double" : 
            {
                double[] array = list.Select(x => (double)x).ToArray();
                returnArray = new NDArray<double>().array(array);
                break;
            }
            case "Float" : 
            {
                float[] array = list.Select(x => (float)x ).ToArray();
                returnArray = new NDArray<float>().array(array);
                break;
            }
            case "Int32" : 
            {
                System.Int32[] array = list.Select(x => (System.Int32)x ).ToArray();
                returnArray = new NDArray<int>().array(array);
                break;
            }
            case "Int64" : 
            {
                System.Int64[] array = list.Select(x => (System.Int64)x ).ToArray();
                returnArray = new NDArray<System.Int64>().array(array);
                break;
            }
            case "Complex" : 
            {
                System.Numerics.Complex[] array = list.Select(x => (System.Numerics.Complex)x ).ToArray();
                returnArray = new NDArray<System.Numerics.Complex>().array(array);
                break;
            }
            default : 
            {
                break;
            }
        }

        return returnArray;
    }     
}
