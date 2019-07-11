using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.UnitTest
{
    public class DataSample
    {
        /// <summary>
        /// shape(2, 3, 2, 2)
        /// </summary>
        public static int[,,,] Int32D2x3x2x2
            => new int[,,,] {{{{1, 3}, {2, 2}}, {{1, 2}, {3, 3}}, {{3, 2}, {2, 1}},}, {{{2, 1}, {2, 3}}, {{3, 2}, {3, 1}}, {{3, 2}, {1, 2}},}};

        /// <summary>
        /// shape(4, 3, 2)
        /// </summary>
        public static int[,,] Int32D4x3x2
            => new int[,,] {{{1, 3}, {2, 1}}, {{2, 1}, {2, 3}}, {{1, 1}, {2, 2}}, {{3, 1}, {1, 2}}};

        /// <summary>
        /// shape(4, 3)
        /// </summary>
        public static int[,] Int32D4x3
            => new int[,] {{3, 2, 1}, {1, 2, 1}, {1, 2, 3}, {2, 3, 1}};

        /// <summary>
        /// shape(12)
        /// </summary>
        /// <returns></returns>
        public static int[] Int32D12
            => new int[] {1, 2, 1, 3, 2, 1, 1, 2, 3, 2, 3, 2};

        /// <summary>
        /// shape(2, 2)
        /// </summary>
        /// <returns></returns>
        public static int[,] Int32D2x2
            => new int[,] {{1, 2}, {2, 3}};
    }
}
