using System;
using System.Collections.Generic;
using System.Text;
using np = NumSharp.Core.NumPy;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray min(int? axis = null)
        {
            switch (dtype.Name)
            {
                case "Double":
                    return np.amin(this, axis);
            }

            return null;
        }

        public NDArray min<T>(int? axis = null)
        {
            return np.amin(this, axis);
        }
    }
}
