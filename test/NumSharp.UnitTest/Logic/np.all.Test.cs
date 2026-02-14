using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;


namespace NumSharp.UnitTest.Logic
{
    public class np_all_Test
    {
        [Test]
        public void np_all_1D()
        {
            var np1 = new NDArray(new[] {true, true, false, false}, new Shape(4));
            var np2 = new NDArray(typeof(bool), new Shape(150));
            var np3 = new NDArray(new[] {true, true, true, true, true, true, true, true}, new Shape(8));
            Assert.IsFalse(np.all(np1));
            Assert.IsFalse(np.all(np2));
            Assert.IsTrue(np.all(np3));
        }

        [Test]
        public void np_all_2D()
        {
            var np1 = new NDArray(new bool[] {true, true, false, false, true, false}, new Shape(2, 3));
            var np2 = new NDArray(typeof(bool), new Shape(39, 17));
            var np3 = new NDArray(new[] {true, true, true, true, true, true, true, true}, new Shape(2, 4));
            Assert.IsFalse(np.all(np1));
            Assert.IsFalse(np.all(np2));
            Assert.IsTrue(np.all(np3));
        }
    }
}
