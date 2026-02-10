using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class NumpyLoad
    {
        [TestMethod]
        public void NumpyLoadTest()
        {
            int[] a = {1, 2, 3, 4, 5};
            byte[] mem = np.Save(a);

            int[] b = np.Load<int[]>(mem);
        }

        [TestMethod]
        public void NumpyNPZRoundTripTest()
        {
            int[] arr = np.Load<int[]>(@"data/1-dim-int32_4_comma_empty.npy");
            var d = new Dictionary<string, Array>();
            d.Add("A/A",arr);
            d.Add("B/A",arr); // Tests zip entity.Name
            var ms = new System.IO.MemoryStream();
            np.Save_Npz(d,ms,leaveOpen:true);
            ms.Position = 0L;
            var d2 = np.Load_Npz<Array>(ms);
            Assert.IsTrue(d2.Count == 2);
        }

        [TestMethod]
        [DataRow(@"data/arange_f4_le.npy")]
        [DataRow(@"data/arange_f8_le.npy")]
        [DataRow(@"data/arange_i1.npy")]
        [DataRow(@"data/arange_i2_le.npy")]
        [DataRow(@"data/arange_i4_le.npy")]
        [DataRow(@"data/arange_i8_le.npy")]
        [DataRow(@"data/arange_u1.npy")]
        [DataRow(@"data/arange_u2_le.npy")]
        [DataRow(@"data/arange_u4_le.npy")]
        [DataRow(@"data/arange_u8_le.npy")]
        public void load_Arange(string path)
        {
            NDArray arr = np.load(path);

            for (int i = 0; i < arr.shape[0]; ++i)
            {
                int value = (int)Convert.ChangeType(arr.GetValue(i), typeof(int));
                Assert.AreEqual(i, value);
            }
        }

        [TestMethod]
        [DataRow(@"data/hello_S5.npy")]
        public void load_HelloWorld(string path)
        {
            string[] arr = np.Load<string[]>(path);
            Assert.AreEqual("Hello", arr[0]);
            Assert.AreEqual("World", arr[1]);
        }

        [TestMethod]
        [DataRow(@"data/mgrid_i4.npy")]
        public void load_Mgrid(string path)
        {
            NDArray arr = np.load(path);

            for (int i = 0; i < arr.shape[0]; i++)
            {
                for (int j = 0; j < arr.shape[1]; j++)
                {
                    Assert.AreEqual(i, (int)arr.GetValue(0, i, j));
                    Assert.AreEqual(j, (int)arr.GetValue(1, i, j));
                }
            }
        }

        [TestMethod]
        [DataRow(@"data/scalar_b1.npy", false)]
        [DataRow(@"data/scalar_i4_le.npy", 42)]
        public void load_Scalar(string path, object expected)
        {
            NDArray arr = np.load(path);
            Assert.AreEqual(Shape.Scalar, arr.shape);
            Assert.AreEqual(expected, arr.GetValue(0));
        }

        [TestMethod]
        [DataRow(@"data/scalar_b1.npy", false)]
        [DataRow(@"data/scalar_i4_le.npy", 42)]
        public void LoadMatrix_Scalar(string path, object expected)
        {
            Array arr = np.LoadMatrix(path);
            Assert.AreEqual(expected, arr.GetValue(0));
        }
    }
}
