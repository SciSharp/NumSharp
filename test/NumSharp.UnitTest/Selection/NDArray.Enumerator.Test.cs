using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Selection
{
    public class EnumeratorTest
    {
        [Test]
        public void Enumerate()
        {
            var nd = np.arange(12).reshape(2, 3, 2);
            nd.flat.Cast<object>().Should().BeEquivalentTo(nd.Cast<object>());
        }
    }
}
