using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Insert scalar into an array (scalar is cast to array’s dtype, if possible)
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put.html</remarks>
        public void itemset(ref Shape shape, ValueType val) 
        {
            SetValue(val, shape.dimensions); //TODO! if T != dtype, we need to cast!
        }

        /// <summary>
        ///     Insert scalar into an array (scalar is cast to array’s dtype, if possible)
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put.html</remarks>
        public void itemset(Shape shape, ValueType val) 
        {
            SetValue(val, shape.dimensions); //TODO! if T != dtype, we need to cast!
        }

        /// <summary>
        ///     Insert scalar into an array (scalar is cast to array’s dtype, if possible)
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put.html</remarks>
        public void itemset(int[] shape, ValueType val)
        {
            SetValue(val, shape); //TODO! if T != dtype, we need to cast!
        }

        /// <summary>
        ///     Insert scalar into an array (scalar is cast to array’s dtype, if possible)
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put.html</remarks>
        public void itemset<T>(int[] shape, T val) where T : unmanaged
        {
            SetValue<T>(val, shape); //TODO! if T != dtype, we need to cast!
        }
    }
}
