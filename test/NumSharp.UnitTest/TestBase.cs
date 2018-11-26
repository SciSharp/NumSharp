using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.UnitTest
{
    public class TestBase
    {
        protected NumPy np;

        public TestBase()
        {
            np = new NumPy();
        }
    }
}
