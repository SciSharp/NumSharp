using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class DocTest
    {
        [TestMethod]
        public void Dump()
        {
            var nd = new NDArray(typeof(double),3,3);
            nd.Storage.SetData(new double[] {1,2,3,4,5,6,7,8,9} );

            
        }
            
    }
}