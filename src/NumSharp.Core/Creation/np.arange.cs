using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDArray arange(float stop)
        {
            return arange(0, stop, 1);
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDArray arange(double stop)
        {
            return arange(0, stop, 1);
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDArray arange(float start, float stop, float step = 1)
        {
            if (Math.Abs(step) < 0.000001)
                throw new ArgumentException("step can't be 0", nameof(step));

            bool negativeStep = false;
            if (step < 0)
            {
                negativeStep = true;
                step = Math.Abs(step);
                //swap
                var tmp = start;
                start = stop;
                stop = tmp;
            }

            if (start > stop)
                throw new Exception("parameters invalid, start is greater than stop.");


            int length = (int)Math.Ceiling((stop - start + 0.0d) / step);
            var nd = new NDArray(typeof(float), Shape.Vector(length), false); //do not fill, we are about to

            if (negativeStep)
            {
                step = Math.Abs(step);
                unsafe
                {
                    var addr = (float*)nd.Array.Address;
                    for (int add = length - 1, i = 0; add >= 0; add--, i++)
                        *(addr + i) = 1 + start + add * step;
                }
            }
            else
            {
                unsafe
                {
                    var addr = (float*)nd.Array.Address;
                    for (int i = 0; i < length; i++)
                        *(addr + i) = start + i * step;
                }
            }

            return nd;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDArray arange(double start, double stop, double step = 1)
        {
            if (Math.Abs(step) < 0.000001)
                throw new ArgumentException("step can't be 0", nameof(step));

            bool negativeStep = false;
            if (step < 0)
            {
                negativeStep = true;
                step = Math.Abs(step);
                //swap
                var tmp = start;
                start = stop;
                stop = tmp;
            }

            if (start > stop)
                throw new Exception("parameters invalid, start is greater than stop.");


            int length = (int)Math.Ceiling((stop - start + 0.0d) / step);
            var nd = new NDArray(typeof(double), Shape.Vector(length), false); //do not fill, we are about to

            if (negativeStep)
            {
                step = Math.Abs(step);
                unsafe
                {
                    var addr = (double*)nd.Array.Address;
                    for (int add = length - 1, i = 0; add >= 0; add--, i++)
                        *(addr + i) = 1 + start + add * step;
                }
            }
            else
            {
                unsafe
                {
                    var addr = (double*)nd.Array.Address;
                    for (int i = 0; i < length; i++)
                        *(addr + i) = start + i * step;
                }
            }

            return nd;
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDArray arange(int stop)
        {
            return arange(0, stop, 1);
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDArray arange(int start, int stop, int step = 1)
        {
            if (step == 0)
                throw new ArgumentException("step can't be 0", nameof(step));

            bool negativeStep = false;
            if (step < 0)
            {
                negativeStep = true;
                step = Math.Abs(step);
                //swap
                var tmp = start;
                start = stop;
                stop = tmp;
            }

            if (start > stop)
                throw new Exception("parameters invalid, start is greater than stop.");


            int length = (int)Math.Ceiling((stop - start + 0.0d) / step);
            var nd = new NDArray(typeof(int), Shape.Vector(length), false); //do not fill, we are about to

            if (negativeStep)
            {
                step = Math.Abs(step);
                unsafe
                {
                    var addr = (int*)nd.Array.Address;
                    for (int add = length - 1, i = 0; add >= 0; add--, i++) 
                        *(addr + i) = 1 + start + add * step;
                }
            }
            else
            {
                unsafe
                {
                    var addr = (int*)nd.Array.Address;
                    for (int i = 0; i < length; i++)
                        *(addr + i) = start + i * step;
                }
            }

            return nd;
        }
    }
}
