using System;
using System.Xml.Schema;

namespace NumSharp.Manipulation
{
    public static partial class np
    {
        public static NDArray tile(NDArray a, int[] repeats)
        {
            foreach (var rep in repeats)
                if (rep < 0){
                    throw new ArgumentException($"Negative repeat value: {rep}");
                }
            int rlen = repeats.Length;
            int ndim = a.ndim;
            if (rlen < ndim)
            {
                int[] padding = new int[ndim - rlen];
                int[] new_array =new int[ndim];
                int[] true_rpt = new int[ndim];
                Array.Copy(repeats, true_rpt, rlen);
                for (int i = rlen; i < ndim; i++)
                {
                    true_rpt[i] = 0;
                }

            }
            else if(rlen > ndim) 
            {

            }
        }
    }
}
