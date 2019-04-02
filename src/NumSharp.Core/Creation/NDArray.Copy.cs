using System;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray copy(string order = null)
        {   
            NDArray puffer = this.Clone()  as NDArray;

            return puffer;
        }

    }
}