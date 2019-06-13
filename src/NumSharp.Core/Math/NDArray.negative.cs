using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Negates all positive values.
        /// </summary>
        public NDArray negative()
        {
            var outputNDArray = new NDArray(this.dtype, this.shape);

            Array inputArray = this.Storage.GetData();
            Array outputArray = outputNDArray.Storage.GetData();

            switch (outputArray)
            {
                case int[] output:
                {
                    var @in = inputArray as int[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        var val = @in[idx];
                        if (val > 0)
                            output[idx] = -val;
                        else
                            output[idx] = val;
                    }

                    break;
                }

                case long[] output:
                {
                    var @in = inputArray as long[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        var val = @in[idx];
                        if (val > 0)
                            output[idx] = -val;
                        else
                            output[idx] = val;
                    }

                    break;
                }

                case double[] output:
                {
                    var @in = inputArray as double[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        var val = @in[idx];
                        if (val > 0)
                            output[idx] = -val;
                        else
                            output[idx] = val;
                    }

                    break;
                }

                case float[] output:
                {
                    var @in = inputArray as float[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        var val = @in[idx];
                        if (val > 0)
                            output[idx] = -val;
                        else
                            output[idx] = val;
                    }

                    break;
                }

                case Complex[] output:
                {
                    var @in = inputArray as Complex[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        var val = @in[idx];
                        output[idx] = new Complex(val.Real > 0 ? -val.Real : val.Real, val.Imaginary > 0 ? -val.Imaginary : val.Imaginary);
                    }

                    break;
                }

                case decimal[] output:
                {
                    var @in = inputArray as decimal[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        var val = @in[idx];
                        if (val > 0)
                            output[idx] = -val;
                        else
                            output[idx] = val;
                    }

                    break;
                }

                default:
                {
                    throw new IncorrectTypeException();
                }
            }

            return outputNDArray;
        }
    }
}
