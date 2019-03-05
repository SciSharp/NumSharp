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
                default:
                    throw new NotImplementedException("concatenate");

            }

            return nd;
        }
    }
}
