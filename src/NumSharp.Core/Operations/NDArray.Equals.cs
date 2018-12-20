using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public static bool operator ==(NDArray np, object obj)
        {
            switch (obj)
            {
                case int o:
                    return o == np.Storage.GetData<int>()[0];
            }

            return false;
        }
    }
}
