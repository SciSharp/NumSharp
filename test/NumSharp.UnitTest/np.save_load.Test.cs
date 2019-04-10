using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Creation;
using NumSharp;
using System.IO;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NumpySaveLoadTest
    {
        [TestMethod]
        public void NumpySaveLoad()
        {
            // int
            int[] x = { 1, 2, 3, 4, 5 };
            np.Save(x, @"test.npy");
            np.Save_Npz(x, @"test1.npz");
            np.Load<int[]>(@"test.npy");
            np.Load_Npz<int[]>(@"test1.npz");

            // float
            string fTestFile = @"c:\temp\test";
            string fTestFileWithExt = fTestFile + ".npy";
            var f1 = np.arange(9.0f).reshape(3, 3);
            np.save(fTestFile, f1);
            var f2 = np.load(fTestFileWithExt);
            Assert.IsTrue(np.all(f1 == f2));

            // double
            var d1 = np.arange(9.0d).reshape(3, 3);
            np.save(fTestFile, d1);
            var d2 = np.load(fTestFileWithExt);
            Assert.IsTrue(np.all(d1 == d2));
        }
    }
}
