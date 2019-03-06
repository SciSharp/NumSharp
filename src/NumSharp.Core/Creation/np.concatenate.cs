using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public static partial class np
    {
        public static NDArray concatenate<T>(T[][] arrays, int axis = 0)
        {
            NDArray nd = null;

            switch (typeof(T).Name)
            {
                case "Int32[]":
                    var data = new List<int[]>();
                    for (int i = 0; i < arrays.Length; i++)
                    {
                        if (arrays[i] is int[][] elements)
                            for (int j = 0; j < elements.Length; j++)
                                data.Add(elements[j]);
                    }
                        
                    nd = np.array(data.ToArray());
                    break;

               case "Int16[]":
                  var data_short = new List<short[]>();
                  for (int i = 0; i < arrays.Length; i++)
                  {
                     if (arrays[i] is short[][] elements)
                        for (int j = 0; j < elements.Length; j++)
                           data_short.Add(elements[j]);

                  }

                  nd = np.array(data_short.ToArray());
                  break;

               case "Int64[]":
                   var data_long = new List<long[]>();
                   for (int i = 0; i < arrays.Length; i++)
                   {
                       if (arrays[i] is long[][] elements)
                           for (int j = 0; j < elements.Length; j++)
                              data_long.Add(elements[j]);

                   }

                   nd = np.array(data_long.ToArray());
                   break;

               case "Float32[]":
                  var data_float = new List<float[]>();
                  for (int i = 0; i < arrays.Length; i++)
                  {
                    if (arrays[i] is float[][] elements)
                      for (int j = 0; j < elements.Length; j++)
                        data_float.Add(elements[j]);

                  }

                 nd = np.array(data_float.ToArray());
                 break;

               case "Float64[]":
                  var data_double = new List<double[]>();
                  for (int i = 0; i < arrays.Length; i++)
                  {
                    if (arrays[i] is double[][] elements)
                      for (int j = 0; j < elements.Length; j++)
                        data_double.Add(elements[j]);

                  }

                 nd = np.array(data_double.ToArray());
                 break;

               default:
                    throw new NotImplementedException("concatenate");

            }

            return nd;
        }
    }
}
