using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest;

namespace NumSharp.UnitTest.Logic
{
    public class np_allclose_Test
    {
        [Test]
        [OpenBugs]
        public void np_allclose_1D()
        {
            //>>> np.allclose([1e10, 1e-7], [1.00001e10,1e-8])
            //False
            Assert.IsFalse(np.allclose(new[] {1e10, 1e-7}, new[] {1.00001e10, 1e-8}));

            //>>> np.allclose([1e10, 1e-8], [1.00001e10, 1e-9])
            //True
            Assert.IsTrue(np.allclose(new[] {1e10, 1e-8}, new[] {1.00001e10, 1e-9}));

            //>>> np.allclose([1e10, 1e-8], [1.0001e10, 1e-9])
            //False
            Assert.IsFalse(np.allclose(new[] {1e10, 1e-8}, new[] {1.0001e10, 1e-9}));

            //>>> np.allclose([1.0, np.nan], [1.0, np.nan])
            //False
            Assert.IsFalse(np.allclose(new[] {1.0, np.nan}, new[] {1.0, np.nan}));

            //>>> np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
            //True
            Assert.IsTrue(np.allclose(new[] {1.0, np.nan}, new[] {1.0, np.nan}, equal_nan: true));
        }
    }
}
