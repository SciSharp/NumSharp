using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class EnumeratorTest
    {
        [TestMethod]
        public void Enumerate()
        {
            var nd = np.arange(12).reshape(2, 3, 2);

            int i = 0;
            foreach (NDArray x in nd)
            {
                Assert.IsTrue(nd[i].ToString() == x.ToString());

                /*int j = 0;
                foreach(NDArray y in x)
                {
                    Assert.IsTrue(nd[i, j].ToString() == x[j].ToString());

                    int k = 0;
                    foreach (int z in y)
                    {
                        Assert.IsTrue(nd[i, j, k].Equals(z));
                        k++;
                    }

                    j++;
                }*/

                i++;
            }
        }
    }
}
