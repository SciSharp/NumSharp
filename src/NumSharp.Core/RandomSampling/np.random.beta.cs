using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a Beta distribution.
        /// </summary>
        public NDArray beta(double a, double b) => beta(a, b, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a Beta distribution.
        /// </summary>
        /// <param name="a">Alpha (α), positive (>0).</param>
        /// <param name="b">Beta (β), positive (>0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Beta distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
        ///     <br/>
        ///     The Beta distribution is a special case of the Dirichlet distribution,
        ///     and is related to the Gamma distribution.
        /// </remarks>
        public NDArray beta(double a, double b, Shape size)
        {
            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(BetaSample(a, b));

            var shape = size;
            NDArray ret = new NDArray(NPTypeCode.Double, shape, false);

            // Handle empty arrays (any dimension is 0)
            if (shape.size == 0)
                return ret;

            unsafe
            {
                var addr = (double*)ret.Address;
                var incr = new Utilities.ValueCoordinatesIncrementor(ref shape);

                do
                {
                    *(addr + shape.GetOffset(incr.Index)) = BetaSample(a, b);
                } while (incr.Next() != null);
            }

            return ret;
        }

        /// <summary>
        /// Generate a single beta sample using NumPy's legacy algorithm.
        /// </summary>
        private double BetaSample(double a, double b)
        {
            // NumPy legacy_beta algorithm from legacy-distributions.c
            if (a <= 1.0 && b <= 1.0)
            {
                // Johnk's algorithm for a <= 1 and b <= 1
                double invA = 1.0 / a;
                double invB = 1.0 / b;

                while (true)
                {
                    double U = randomizer.NextDouble();
                    double V = randomizer.NextDouble();
                    double X = Math.Pow(U, invA);
                    double Y = Math.Pow(V, invB);

                    if (X + Y <= 1.0)
                    {
                        if (X + Y > 0)
                        {
                            return X / (X + Y);
                        }
                        else
                        {
                            // Handle underflow case
                            double logX = Math.Log(U) / a;
                            double logY = Math.Log(V) / b;
                            double logM = logX > logY ? logX : logY;
                            logX -= logM;
                            logY -= logM;
                            return Math.Exp(logX - Math.Log(Math.Exp(logX) + Math.Exp(logY)));
                        }
                    }
                }
            }
            else
            {
                // Use gamma method for a > 1 or b > 1
                double Ga = SampleStandardGamma(a);
                double Gb = SampleStandardGamma(b);
                return Ga / (Ga + Gb);
            }
        }
    }
}
