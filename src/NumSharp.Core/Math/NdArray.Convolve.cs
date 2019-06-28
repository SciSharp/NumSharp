using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        /// 
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="numSharpArray2"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        public NDArray convolve(NDArray numSharpArray2, string mode = "full")
        {
            int nf = this.shape[0];
            int ng = numSharpArray2.shape[0];

            if (ndim > 1)
                throw new IncorrectShapeException();

            var numSharpReturn = new NDArray(typeof(double));

            double[] np1 = this.Storage.GetData<double>();
            double[] np2 = numSharpArray2.Storage.GetData<double>();

            switch (mode)
            {
                case "full":
                {
                    int n = nf + ng - 1;

                    var outArray = new double[n];

                    for (int idx = 0; idx < n; ++idx)
                    {
                        int jmn = (idx >= ng - 1) ? (idx - (ng - 1)) : 0;
                        int jmx = (idx < nf - 1) ? idx : nf - 1;

                        for (int jdx = jmn; jdx <= jmx; ++jdx)
                        {
                            outArray[idx] += (np1[jdx] * np2[idx - jdx]);
                        }
                    }

                    numSharpReturn.Storage = numSharpReturn.TensorEngine.GetStorage(numSharpReturn.dtype);
                    numSharpReturn.Storage.Allocate(new Shape(outArray.Length));
                    numSharpReturn.Storage.ReplaceData(outArray);

                    break;
                }

                case "valid":
                {
                    var min_v = (nf < ng) ? np1 : np2;
                    var max_v = (nf < ng) ? np2 : np1;

                    int n = Math.Max(nf, ng) - Math.Min(nf, ng) + 1;

                    double[] outArray = new double[n];

                    for (int idx = 0; idx < n; ++idx)
                    {
                        int kdx = idx;

                        for (int jdx = (min_v.Length - 1); jdx >= 0; --jdx)
                        {
                            outArray[idx] += min_v[jdx] * max_v[kdx];
                            ++kdx;
                        }
                    }

                    numSharpReturn.Storage = numSharpReturn.TensorEngine.GetStorage(numSharpReturn.dtype);
                    numSharpReturn.Storage.Allocate(new Shape(outArray.Length));
                    numSharpReturn.Storage.ReplaceData(outArray);

                    break;
                }

                case "same":
                {
                    // followed the discussion on 
                    // https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
                    // implemented numpy convolve because we follow numpy
                    var npad = numSharpArray2.shape[0] - 1;

                    double[] np1New = null;

                    if (npad % 2 == 1)
                    {
                        npad = (int)Math.Floor(((double)npad) / 2.0);

                        np1New = (double[])np1.Clone();

                        np1New.ToList().AddRange(new double[npad + 1]);
                        var puffer = (new double[npad]).ToList();
                        puffer.AddRange(np1New);
                        np1New = puffer.ToArray();
                    }
                    else
                    {
                        npad = npad / 2;

                        np1New = (double[])np1.Clone();

                        var puffer = np1New.ToList();
                        puffer.AddRange(new double[npad]);
                        np1New = puffer.ToArray();

                        puffer = (new double[npad]).ToList();
                        puffer.AddRange(np1New);
                        np1New = puffer.ToArray();
                    }

                    var numSharpNew = np.array(np1New, dtype);

                    numSharpReturn = numSharpNew.convolve(numSharpArray2, "valid");
                    break;
                }
            }

            return numSharpReturn;
        }
    }
}
