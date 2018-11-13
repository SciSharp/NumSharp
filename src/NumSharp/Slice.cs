using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    /// <summary>
    /// ndarray can be indexed using slicing
    /// slice is constructed by start:stop:step notation
    /// </summary>
    public class Slice
    {
        private int start;
        private int stop;
        private int step;

        public Slice(params int[] p)
        {
            switch (p.Length)
            {
                case 3:
                    start = p[0];
                    stop = p[1];
                    step = p[2];
                    break;
            }
        }
    }
}
