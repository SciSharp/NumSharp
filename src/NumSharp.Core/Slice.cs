using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    /// <summary>
    /// ndarray can be indexed using slicing
    /// slice is constructed by start:stop:step notation
    /// </summary>
    public class Slice
    {
        public int Start { get; set; }
        public int Stop { get; set; }
        public int Step { get; set; }

        public int Length => Stop - Start;

        /// <summary>
        /// start, stop, step
        /// </summary>
        /// <param name="p"></param>
        public Slice(params int[] p)
        {
            switch (p.Length)
            {
                case 2:
                    Start = p[0];
                    Stop = p[1];
                    Step = 1;
                    break;
                case 3:
                    Start = p[0];
                    Stop = p[1];
                    Step = p[2];
                    break;
            }
        }
    }
}
