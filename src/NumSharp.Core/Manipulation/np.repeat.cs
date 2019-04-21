using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray repeat(NDArray nd, int repeats, int axis = -1)
        {
            int size = nd.size * repeats;

            // scalar
            switch (nd.dtype.Name)
            {
                case "Int32":
                    {
                        var nd2 = new NDArray(new int[size], new Shape(size));
                        var data = nd.Data<int>();
                        for (int i = 0; i < nd.size; i++)
                            for (int j = 0; j < repeats; j++)
                                nd2.itemset(i * repeats + j, data[i]);
                        return nd2;
                    }
                case "Boolean":
                    {
                        var nd2 = new NDArray(new bool[size], new Shape(size));
                        var data = nd.Data<bool>();
                        for (int i = 0; i < nd.size; i++)
                            for (int j = 0; j < repeats; j++)
                                nd2.itemset(i * repeats + j, data[i]);
                        return nd2;
                    }
            }

            throw new NotImplementedException("np.repeat");
        }
    }
}
