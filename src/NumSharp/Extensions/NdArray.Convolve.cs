using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using NumSharp;

namespace NumSharp.Extensions
{
    public static partial class NDArrayExtension
    {
        /// <summary>
        /// Convolution of 2 series  
        /// </summary>
        /// <param name="numSharpArray1"></param>
        /// <param name="numSharpArray2"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public static NDArray<double> Convolve(this NDArray<double> numSharpArray1, NDArray<double> numSharpArray2, string mode  = "full")
        {
            int nf = numSharpArray1.Length;
            int ng = numSharpArray2.Length;

            var numSharpReturn = new NDArray<double>();

            if (mode.Equals("full"))
            {
                int n  = nf + ng - 1;

                var out_ = new double[n];

                for (int idx = 0; idx < n; ++idx)
                {
                    int jmn = (idx >= ng - 1) ? (idx - (ng - 1)) : 0;
                    int jmx = (idx < nf - 1) ? idx : nf - 1;

                    for (int jdx = jmn; jdx <= jmx; ++jdx )
                    {
                        out_[idx] += ( numSharpArray1[jdx] * numSharpArray2[idx - jdx] );
                    }
                }
                
                numSharpReturn.Data = out_;
            }
            else 
            {
           
            }

            return numSharpReturn;
        }
    }
  }