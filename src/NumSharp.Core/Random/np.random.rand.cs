using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Random values in a given shape.
        ///     Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        /// </summary>
        /// <remarks>This is a convenience function. If you want an interface that takes a shape-tuple as the first argument, refer to np.random.random_sample.</remarks>
        public NDArray rand(params int[] size) {
            return rand(new Shape(size));
        }        
        
        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        ///     Results are from the “continuous uniform” distribution over the stated interval. To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
        /// </summary>
        /// <param name="size">The samples</param>
        /// <returns></returns>
        public NDArray random_sample(params int[] size)
        {
            return random_sample(new Shape(size)); 
        }        
        
        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        ///     Results are from the “continuous uniform” distribution over the stated interval. To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
        /// </summary>
        /// <param name="shape">The shape to randomize</param>
        /// <returns></returns>
        public NDArray random_sample(Shape shape)
        {
            return null; //todo https://www.youtube.com/watch?v=-qt8CPIadWQ
        }

        //todo get proper documentation form pyhon docs.

        /// <summary>
        ///     Creates random array of given shape
        /// </summary>
        public NDArray rand(Shape shape) {
            NDArray ndArray = new NDArray(typeof(double), shape);
            double[] numArray = ndArray.Data<double>();
            for (int index = 0; index < ndArray.size; ++index) {
                numArray[index] = randomizer.NextDouble();
            }

            ndArray.SetData<double[]>(numArray);
            return ndArray;
        }

    }
}
