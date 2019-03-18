using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
  [TestClass]
  public class NumpyLoad
  {
    [TestMethod]
    static void NumpyLoadTest()
    {
      int[] a = { 1, 2, 3, 4, 5 };
      byte[] mem = np.Save(a);

      int[] b = np.Load<int[]>(mem);
     }
  }
}
