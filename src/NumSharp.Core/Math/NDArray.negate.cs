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
        /// Negates all values by performing: -x
        /// </summary>
        public NDArray negate()
        {
            var outputNDArray = new NDArray(this.dtype, this.shape);

            if (len == 0)
                return outputNDArray; //return new to maintain immutability.

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
                        output[idx] = -@in[idx];
                    }

                    break;
                }

                case long[] output:
                {
                    var @in = inputArray as long[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        output[idx] = -@in[idx];
                    }

                    break;
                }

                case short[] output:
                {
                    var @in = inputArray as short[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        output[idx] = (short)-@in[idx]; //must be casted because C# automatically converts it to int
                    }

                    break;
                }

                case double[] output:
                {
                    var @in = inputArray as double[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        output[idx] = -@in[idx];
                    }

                    break;
                }

                case float[] output:
                {
                    var @in = inputArray as float[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        output[idx] = -@in[idx];
                    }

                    break;
                }

                case Complex[] output:
                {
                    var @in = inputArray as Complex[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        output[idx] = -@in[idx];
                    }

                    break;
                }

                case decimal[] output:
                {
                    var @in = inputArray as decimal[];
                    Parallel.For(0, @in.Length, compute);

                    void compute(int idx)
                    {
                        output[idx] = -@in[idx];
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
