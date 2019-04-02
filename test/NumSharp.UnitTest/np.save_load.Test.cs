using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Creation;
using NumSharp;

namespace NumSharp.UnitTest
{
  [TestClass]
  public class NumpySaveLoad
  {
    [TestMethod]
    public void Run()
    {
      int[] x = { 1, 2, 3, 4, 5 };
      np.Save(x, @"test.npy");
      np.Save_Npz(x, @"test1.npz");
      np.Load<int[]>(@"test.npy");
      np.Load_Npz<int[]>(@"test1.npz");
    }
  }
}
