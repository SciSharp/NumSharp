using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Utilities
{
    public class SteppingOverArray : TestClass
    {
        [Test]
        public void Stepping()
        {
            //>>> a =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            //>>> a[::2]
            //[1, 3, 5, 7, 9]
            //>>> a[::-2]
            //[10, 8, 6, 4, 2]
            //>>> a[::-4]
            //[10, 6, 2]
            //>>> a[::4]
            //[1, 5, 9]
            //>>> a[::10]
            //[1]
            //>>> a[::-10]
            //[10]
            //>>> a[::-77]
            //[10]
            //>>> a[::77]
            //[1]
            //>>>
            var a = new[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            Assert.AreSame(a, a.Step(1));
            AssertAreEqual(new[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.AsEnumerable().Reverse().ToArray(), a.Step(-1));
            AssertAreEqual(new[] {10, 8, 6, 4, 2,}.ToArray(), a.Step(-2));
            AssertAreEqual(new[] {1, 3, 5, 7, 9,}.ToArray(), a.Step(2));
            AssertAreEqual(new[] {10, 7, 4, 1,}.ToArray(), a.Step(-3));
            AssertAreEqual(new[] {1, 4, 7, 10,}.ToArray(), a.Step(3));
            AssertAreEqual(new[] {10, 6, 2,}.ToArray(), a.Step(-4));
            AssertAreEqual(new[] {1, 5, 9,}.ToArray(), a.Step(4));
            AssertAreEqual(new[] {10,}.ToArray(), a.Step(-10));
            AssertAreEqual(new[] {1,}.ToArray(), a.Step(10));
            AssertAreEqual(new[] {10,}.ToArray(), a.Step(-77));
            AssertAreEqual(new[] {1,}.ToArray(), a.Step(77));
        }
    }
}
