using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    public class np_isclose_Test
    {
        [Ignore("TODO: fix this test")]
        [Test]
        public void np_isclose_1D()
        {
            //>>> np.isclose([1e10, 1e-7], [1.00001e10,1e-8])
            //array([True, False])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-7}, new[] {1.00001e10, 1e-8}).Data<bool>(), new[] {true, false}));

            //>>> np.isclose([1e10, 1e-8], [1.00001e10,1e-9])
            //array([True, True])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-8}, new[] {1.00001e10, 1e-9}).Data<bool>(), new[] {true, true}));

            //>>> np.isclose([1e10, 1e-8], [1.0001e10, 1e-9])
            //array([False, True])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-8}, new[] {1.0001e10, 1e-9}).Data<bool>(), new[] {false, true}));

            //>>> np.isclose([1.0, np.nan], [1.0, np.nan])
            //array([True, False])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-8}, new[] {1.0001e10, 1e-9}).Data<bool>(), new[] {false, true}));

            //>>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
            //array([True, True])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1.0, double.NaN}, new[] {1.0, double.NaN}, equal_nan: true).Data<bool>(), new[] {true, true}));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {double.NaN, 1.0}, new[] {1.0, double.NaN}, equal_nan: true).Data<bool>(), new[] {false, false}));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {double.NaN, 1.0}, new[] {1.0, double.NaN}).Data<bool>(), new[] {false, false}));

            //>>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
            //array([True, False], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-8, 1e-7}, new[] {0.0, 0.0}).Data<bool>(), new[] {true, false}));

            //>>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
            //array([False, False], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-100, 1e-7}, new[] {0.0, 0.0}, atol: 0.0).Data<bool>(), new[] {false, false}));

            //>>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
            //array([True, True], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-10, 1e-10}, new[] {1e-20, 0.0}).Data<bool>(), new[] {true, true}));

            //>>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
            //array([False, True], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-10, 1e-10}, new[] {1e-20, 0.999999e-10}, atol: 0.0).Data<bool>(), new[] {false, true}));
        }
    }
}
