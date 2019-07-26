using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using FluentAssertions;
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
            nd.flat.Cast<object>().Should().BeEquivalentTo(nd.Cast<object>());
        }
    }
}
