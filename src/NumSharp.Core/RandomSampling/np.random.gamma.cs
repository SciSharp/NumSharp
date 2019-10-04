using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        /// Draw samples from a Gamma distribution.
        /// Samples are drawn from a Gamma distribution with specified parameters, shape (sometimes designated “k”) and scale(sometimes designated “theta”), 
        /// where both parameters are > 0.
        /// </summary>
        /// <param name="shapeV">The shape of the gamma distribution. Should be greater than zero.</param>
        /// <param name="scale">The scale of the gamma distribution. Should be greater than zero. Default is equal to 1.</param>
        /// <param name="shape">Output shape.</param>
        /// <returns>Drawn samples from the parameterized gamma distribution.</returns>
        public NDArray gamma(double shapeV, double scale, Shape shape) => gamma(shapeV, scale, shape.Dimensions);

        /// <summary>
        /// Draw samples from a Gamma distribution.
        /// Samples are drawn from a Gamma distribution with specified parameters, shape (sometimes designated “k”) and scale(sometimes designated “theta”), 
        /// where both parameters are > 0.
        /// </summary>
        /// <param name="shape">The shape of the gamma distribution. Should be greater than zero.</param>
        /// <param name="scale">The scale of the gamma distribution. Should be greater than zero. Default is equal to 1.</param>
        /// <param name="dims">Output shape.</param>
        /// <returns>Drawn samples from the parameterized gamma distribution.</returns>
        public NDArray gamma(double shape, double scale, params int[] dims)
        {
            if (shape < 1)
            {
                double d = shape + 1.0 - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);

                NDArray u = np.random.uniform(0, 1, dims);
                return scale * Marsaglia(d, c, dims) * Math.Pow(u, 1.0 / shape);
            }
            else
            {
                double d = shape - 1.0 / 3.0;
                double c = (1.0 / 3.0) / Math.Sqrt(d);

                return scale * Marsaglia(d, c, dims);
            }
        }

        private NDArray Marsaglia(double d, double c, int[] dims)
        {
            var result = new NDArray<double>(dims);
            var resultArray = result.Data<double>();
            Random mar_rand = new Random();
            Parallel.For(0, result.size, (i) => {
                while (true)
                {
                    // 2. Generate v = (1+cx)^3 with x normal
                    double x, t, v;

                    do
                    {
                        x = normal(mar_rand);
                        t = (1.0 + c * x);
                        v = t * t * t;
                    } while (v <= 0);


                    // 3. Generate uniform U
                    double U = np.random.randomizer.NextDouble();

                    // 4. If U < 1-0.0331*x^4 return d*v.
                    double x2 = x * x;
                    if (U < 1 - 0.0331 * x2 * x2)
                    {
                        resultArray[i] = d * v;
                        break;
                    }

                    // 5. If log(U) < 0.5*x^2 + d*(1-v+log(v)) return d*v.
                    if (Math.Log(U) < 0.5 * x2 + d * (1.0 - v + Math.Log(v)))
                    {
                        resultArray[i] = d * v;
                        break;
                    }

                    // 6. Goto step 2
                }
            });

            result.ReplaceData(resultArray); //incase of a view //todo! incase of a view?

            return result;
        }

        private double normal(Random random)
        {
            double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            return  Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
        }
    }
}
