using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    /// <summary>
    /// Shape of the NdArrays
    /// </summary>
    public class NdArrayShape
    {
        /// <summary>
        /// Total number of samples
        /// </summary>
        public int Rows { get; set; }

        /// <summary>
        /// Total number of features, dimension
        /// </summary>
        public int Dimensions { get; set; }

        public override string ToString()
        {
            return $"({Rows}, {Dimensions})";
        }
    }
}
