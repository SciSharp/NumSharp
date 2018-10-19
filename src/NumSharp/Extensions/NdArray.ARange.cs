using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public NDArray<TData> ARange(int stop, int start = 0, int step = 1)
        {
            var np = this;

            int index = 0;

            var npElementType = typeof(TData);

            if (!npElementType.IsEnum)
            {
                var array = Enumerable.Range(start, stop - start)
                                          .Where(x => index++ % step == 0)
                                          .ToArray();
                dynamic puffer = null;     
                
                if (npElementType == typeof(int))
                {
                    puffer = array;
                }
                else if (npElementType == typeof(double)) 
                {
                    puffer = array.Select(x => (double)x )
                                          .ToArray();
                }
                else 
                {

                }
                np.Data = (TData[]) puffer;
                //np.NDim = 1;
            }

            return np;
        }
    }
}
