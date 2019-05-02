using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.View
{
    [TestClass]
    public class ViewStorageTest : TestClass
    {
        [TestMethod]
        public void GetData_1D()
        {
            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Assert.AreEqual(new Shape(10), data.Shape );
            // return identical view
            var view = new ViewStorage(data, ":");
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(data.GetData(), view.GetData());
            view = new ViewStorage(data, "-77:77");
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(data.GetData(), view.GetData());
            // return reduced view
            view = new ViewStorage(data, "7:");
            Assert.AreEqual(new Shape(3), view.Shape);
            AssertAreEqual(new int[] { 7, 8, 9 }, view.GetData());
            view = new ViewStorage(data, ":5");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 0, 1, 2, 3, 4 }, view.GetData());
            view = new ViewStorage(data, "2:3");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new int[] { 2 }, view.GetData());
            // return stepped view
            view = new ViewStorage(data, "::2");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 0, 2, 4, 6, 8 }, view.GetData());
            view = new ViewStorage(data, "::3");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 0, 3, 6, 9 }, view.GetData());
            view = new ViewStorage(data, "-77:77:77");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new []{0}, view.GetData());
            // negative step!
            view = new ViewStorage(data, "::-1");
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(data.GetData().OfType<int>().Reverse().ToArray(), view.GetData());
            view = new ViewStorage(data, "::-2");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 9, 7, 5, 3, 1 }, view.GetData());
            view = new ViewStorage(data, "::-3");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 9, 6, 3, 0 }, view.GetData());
            view = new ViewStorage(data, "-77:77:-77");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new[] { 9 }, view.GetData());
        }

        [TestMethod]
        public void Indexing_1D()
        {
            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            // return identical view
            var view = new ViewStorage(data, ":");
            Assert.AreEqual(0,view.GetData<int>(0));
            Assert.AreEqual(5, view.GetData<int>(5));
            Assert.AreEqual(9, view.GetData<int>(9));
            view = new ViewStorage(data, "-77:77");
            Assert.AreEqual(0, view.GetData<int>(0));
            Assert.AreEqual(5, view.GetData<int>(5));
            Assert.AreEqual(9, view.GetData<int>(9));
            // return reduced view
            view = new ViewStorage(data, "7:");
            Assert.AreEqual(7, view.GetData<int>(0));
            Assert.AreEqual(8, view.GetData<int>(1));
            Assert.AreEqual(9, view.GetData<int>(2));
            view = new ViewStorage(data, ":5");
            Assert.AreEqual(0, view.GetData<int>(0));
            Assert.AreEqual(1, view.GetData<int>(1));
            Assert.AreEqual(2, view.GetData<int>(2));
            Assert.AreEqual(3, view.GetData<int>(3));
            Assert.AreEqual(4, view.GetData<int>(4));
            view = new ViewStorage(data, "2:3");
            Assert.AreEqual(2, view.GetData<int>(0));
            // return stepped view
            view = new ViewStorage(data, "::2");
            Assert.AreEqual(0, view.GetData<int>(0));
            Assert.AreEqual(2, view.GetData<int>(1));
            Assert.AreEqual(4, view.GetData<int>(2));
            Assert.AreEqual(6, view.GetData<int>(3));
            Assert.AreEqual(8, view.GetData<int>(4));
            view = new ViewStorage(data, "::3");
            Assert.AreEqual(0, view.GetData<int>(0));
            Assert.AreEqual(3, view.GetData<int>(1));
            Assert.AreEqual(6, view.GetData<int>(2));
            Assert.AreEqual(9, view.GetData<int>(3));
            view = new ViewStorage(data, "-77:77:77");
            Assert.AreEqual(0, view.GetData<int>(0));
            // negative step!
            view = new ViewStorage(data, "::-1");
            Assert.AreEqual(9, view.GetData<int>(0));
            Assert.AreEqual(4, view.GetData<int>(5));
            Assert.AreEqual(0, view.GetData<int>(9));
            view = new ViewStorage(data, "::-2");
            Assert.AreEqual(9, view.GetData<int>(0));
            Assert.AreEqual(7, view.GetData<int>(1));
            Assert.AreEqual(5, view.GetData<int>(2));
            Assert.AreEqual(3, view.GetData<int>(3));
            Assert.AreEqual(1, view.GetData<int>(4));
            view = new ViewStorage(data, "::-3");
            Assert.AreEqual(9, view.GetData<int>(0));
            Assert.AreEqual(6, view.GetData<int>(1));
            Assert.AreEqual(3, view.GetData<int>(2));
            Assert.AreEqual(0, view.GetData<int>(3));
            view = new ViewStorage(data, "-77:77:-77");
            Assert.AreEqual(9, view.GetData<int>(0));
        }

        [TestMethod]
        public void NestedView_1D()
        {
            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            // return identical view
            var identical = new ViewStorage(data, ":");
            Assert.AreEqual(new Shape(10), identical.Shape);
            var view1 = new ViewStorage(identical, "1:9");
            Assert.AreEqual(new Shape(8), view1.Shape);
            AssertAreEqual(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, }, view1.GetData());
            var view2 = new ViewStorage(view1, "::-2");
            Assert.AreEqual(new Shape(4), view2.Shape);
            AssertAreEqual(new int[] { 8, 6, 4, 2, }, view2.GetData());
            var view3 = new ViewStorage(view2, "::-3");
            Assert.AreEqual(new Shape(2), view3.Shape);
            AssertAreEqual(new int[] { 2, 8 }, view3.GetData());
            // all must see the same modifications, no matter if original or any view is modified
            // modify original
            data.SetData(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 });
            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, -8, }, view1.GetData());
            AssertAreEqual(new int[] { -8, -6, -4, -2, }, view2.GetData());
            AssertAreEqual(new int[] { -2, -8 }, view3.GetData());
            // modify views
            view1.SetData(88, 7);
            AssertAreEqual(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, 88, -9 }, data.GetData());
            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, 88, }, view1.GetData());
            AssertAreEqual(new int[] { 88, -6, -4, -2, }, view2.GetData());
            AssertAreEqual(new int[] { -2, 88 }, view3.GetData());
            view3.SetData(22, 0);
            AssertAreEqual(new int[] { 0, -1, 22, -3, -4, -5, -6, -7, 88, -9 }, data.GetData());
            AssertAreEqual(new int[] { -1, 22, -3, -4, -5, -6, -7, 88, }, view1.GetData());
            AssertAreEqual(new int[] { 88, -6, -4, 22, }, view2.GetData());
            AssertAreEqual(new int[] { 22, 88 }, view3.GetData());
        }
    }
}
