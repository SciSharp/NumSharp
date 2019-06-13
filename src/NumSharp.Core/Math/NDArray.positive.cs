using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Positives all negative values.
        /// </summary>
        public NDArray positive()
        {
            var outputArray = (NDArray)this.Clone();

            Array outputSysArr = outputArray.Storage.GetData();

            switch (outputSysArr)
            {
                case int[] data:
                {
                    for (int idx = 0; idx < data.Length; idx++)
                    {
                        var val = data[idx];
                        if (val < 0)
                            data[idx] = -val;
                    }

                    break;
                }

                case long[] data:
                {
                    for (int idx = 0; idx < data.Length; idx++)
                    {
                        var val = data[idx];
                        if (val < 0)
                            data[idx] = -val;
                    }

                    break;
                }

                case double[] data:
                {
                    for (int idx = 0; idx < data.Length; idx++)
                    {
                        var val = data[idx];
                        if (val < 0)
                            data[idx] = -val;
                    }

                    break;
                }

                case float[] data:
                {
                    for (int idx = 0; idx < data.Length; idx++)
                    {
                        var val = data[idx];
                        if (val < 0)
                            data[idx] = -val;
                    }

                    break;
                }

                case Complex[] data:
                {
                    for (int idx = 0; idx < data.Length; idx++)
                    {
                        var val = data[idx];
                        data[idx] = new Complex(val.Real < 0 ? -val.Real : val.Real, val.Imaginary < 0 ? -val.Imaginary : val.Imaginary);
                    }

                    break;
                }

                case decimal[] data:
                {
                    for (int idx = 0; idx < data.Length; idx++)
                    {
                        var val = data[idx];
                        if (val < 0)
                            data[idx] = -val;
                    }

                    break;
                }

                default:
                {
                    throw new IncorrectTypeException();
                }
            }

            return outputArray;
        }
    }
}
