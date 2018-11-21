using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class EnumeratorTest
    {
        [TestMethod]
        public void Enumerate()
        {
            var np = new NumPy<int>().arange(12).reshape(2, 3, 2);

            int i = 0;
            
            foreach (var nd1 in np)
            {
                Assert.IsTrue(Enumerable.SequenceEqual(np[new Shape(i)].Data, ((NDArray<int>)nd1).Data));
                Console.WriteLine(nd1.ToString());

                int j = 0;
                foreach (var nd2 in (NDArray<int>)nd1)
                {
                    Assert.IsTrue(Enumerable.SequenceEqual(np[new Shape(i, j)].Data, ((NDArray<int>)nd2).Data));
                    Console.WriteLine(nd2.ToString());

                    int k = 0;
                    foreach(var nd3 in (NDArray<int>)nd2)
                    {
                        Assert.IsTrue(np[i, j, k] == (int)nd3);
                        Console.WriteLine(nd3.ToString());
                        k++;
                    }

                    j++;
                }
                
                i++;
            }
        }
    }
}
