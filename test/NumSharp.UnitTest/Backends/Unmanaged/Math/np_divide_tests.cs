using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    [TestClass]
    public class np_divide_tests
    {
        [TestMethod]
        public void UInt8DivideTest1()
        {
            var nd1 = np.arange(3).astype(np.uint8);

            var nd2 = nd1 / (byte)2;

            nd2.array_equal(new byte[] {0 / 2, 1 / 2, 2 / 2}).Should().BeTrue();
        }

        [TestMethod]
        public void UInt16DivideTest1()
        {
            var nd1 = np.arange(3).astype(np.uint16);

            var nd2 = nd1 / (byte)2;
            nd2.array_equal(new ushort[] {0 / 2, 1 / 2, 2 / 2}).Should().BeTrue();
        }

        [TestMethod]
        public void Divide()
        {
            var left = np.ones(new Shape(5, 5)) + 5d;
            var right = np.ones(new Shape(5, 5)) + 1d;
            var ret = left / right;
            ret.GetData<double>().All(d => d == 3).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }
#if _REGEN
        %a = except(supported_dtypes, "NDArray", "Boolean")
        %foreach a
        [DataRow(NPTypeCode.Boolean, NPTypeCode.#1)]
#else
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Double)]
#endif
        [DataTestMethod]
        public void DivideAllPossabilitiesBoolean(NPTypeCode ltc, NPTypeCode rtc)
        {
            var left = np.ones(new Shape(5, 5), rtc) + 3;
            var right = np.ones(new Shape(5, 5), ltc) + 1;
            var ret = left / right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(rtc == NPTypeCode.Char ? 26 : 2);
                Console.WriteLine(val);
            }
        }

#if _REGEN
        %a = except(supported_dtypes, "NDArray", "Boolean")
        %foreach a
        [DataRow(NPTypeCode.Boolean, NPTypeCode.#1)]
#else
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Double)]
#endif
        [DataTestMethod]
        public void DivideAllPossabilitiesBoolean_Left(NPTypeCode ltc, NPTypeCode rtc)
        {
            var left = np.ones(new Shape(5, 5), ltc) + 1;
            var right = np.ones(new Shape(5, 5), rtc) + 1;
            var ret = left / right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(rtc == NPTypeCode.Char ? 0 : 1);
                Console.WriteLine(val);
            }
        }

        [TestMethod]
        public void DivideUpcast()
        {
            var left = (np.ones(new Shape(5, 5)) + 5d).astype(NPTypeCode.Single);
            var right = (np.ones(new Shape(5, 5)) + 1d).astype(NPTypeCode.Int32);
            np._FindCommonArrayType(left.dtype, right.dtype).Should().Be(NPTypeCode.Double);
            var ret = left / right;

            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => d == 3).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void DivideDowncast()
        {
            var left = (np.zeros(new Shape(5, 5)) + 5d).astype(NPTypeCode.Single);
            var right = np.ones(new Shape(5, 5)).astype(NPTypeCode.Int32);
            np._FindCommonArrayType(left.dtype, right.dtype).Should().Be(NPTypeCode.Double);
            var ret = left / right;

            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => d == 5).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_Two_Scalars()
        {
            var left = NDArray.Scalar(1d) / NDArray.Scalar(5d);
            left.GetDouble(0).Should().Be(1 / 5d);
        }

        [TestMethod]
        public void Divide_ND_3_1_vs_1_3()
        {
            var left = np.arange(3).reshape((3, 1)).astype(np.float64);
            var right = np.arange(3).reshape((1, 3)).astype(np.float64);
            var ret = left / right;
            ret.size.Should().Be(9);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0).Should().Be(double.NaN);
            ret.GetDouble(1, 1).Should().Be(1);
            ret.GetDouble(2, 2).Should().Be(1);

            ret.GetDouble(1, 0).Should().Be(double.PositiveInfinity);
            ret.GetDouble(2, 0).Should().Be(double.PositiveInfinity);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_ND_3_1_vs_1_3_float()
        {
            var left = np.arange(3).reshape((3, 1)).astype(np.float32);
            var right = np.arange(3).reshape((1, 3)).astype(np.float32);
            var ret = left / right;
            ret.size.Should().Be(9);
            ret.dtype.Should().Be<float>();
            ret.GetSingle(0, 0).Should().Be(float.NaN);
            ret.GetSingle(1, 1).Should().Be(1);
            ret.GetSingle(2, 2).Should().Be(1);

            ret.GetSingle(1, 0).Should().Be(float.PositiveInfinity);
            ret.GetSingle(2, 0).Should().Be(float.PositiveInfinity);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_ND_2_1_3_3_vs_1_3()
        {
            var left = np.arange(2 * 3 * 3).reshape((2, 1, 3, 3)).astype(np.float64);
            var right = np.arange(3).reshape((1, 3)).astype(np.float64);
            var ret = left / right;
            for (int i = 0; i < ret.size; i++)
                Console.WriteLine(ret.GetAtIndex(i));

            ret.size.Should().Be(18);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0, 0, 0).Should().Be(double.NaN);
            ret.GetDouble(0, 0, 1, 1).Should().Be(4);
            ret.GetDouble(0, 0, 2, 2).Should().Be(4);

            ret.GetDouble(1, 0, 0, 0).Should().Be(double.PositiveInfinity);
            ret.GetDouble(1, 0, 1, 1).Should().Be(13);
            ret.GetDouble(1, 0, 2, 2).Should().Be(8.5);
        }

        [TestMethod]
        public void Divide_ND_2_3_3()
        {
            var left = np.arange(6).reshape((2, 3, 1)).astype(np.float64);
            var right = np.arange(6).reshape((2, 1, 3)).astype(np.float64);
            var ret = left / right;
            ret.size.Should().Be(18);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0, 0).Should().Be(double.NaN);
            ret.GetDouble(0, 1, 1).Should().Be(1);
            ret.GetDouble(0, 2, 2).Should().Be(1);

            ret.GetDouble(1, 0, 0).Should().Be(1);
            ret.GetDouble(1, 1, 1).Should().Be(1);
            ret.GetDouble(1, 2, 2).Should().Be(1);

            ret.GetDouble(1, 0, 0).Should().Be(1);
            ret.GetDouble(1, 0, 1).Should().Be(0.75);
            ret.GetDouble(1, 0, 2).Should().Be(0.6);
            ret.GetDouble(1, 1, 2).Should().Be(0.8);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_RightScalar()
        {
            var left = np.zeros(new Shape(5, 5), np.float64) + 5d;
            var right = NDArray.Scalar(1d);
            var ret = left / right;
            ret.Cast<double>().All(d => d == 5).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_LeftScalar()
        {
            var left = NDArray.Scalar(1d);
            var right = np.zeros(new Shape(5, 5), np.float64) + 5d;
            var ret = left / right;
            ret.Cast<double>().All(d => d == 1 / 5d).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_Rising()
        {
            var left = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                left.SetAtIndex<double>(i, i);
            }

            var right = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                right.SetAtIndex<double>(i, i);
            }

            var ret = left / right;

            ret.Array.As<ArraySlice<double>>().Skip(1).Should().AllBeEquivalentTo(1);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_RightScalar_Rising()
        {
            var left = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
                left.SetAtIndex<double>(i, i);

            var right = NDArray.Scalar(1d);
            var ret = left / right;
            ret.Should().BeInAscendingOrder();
            ret.GetDouble(0).Should().Be(0);
            ret.GetAtIndex<double>(24).Should().Be(24);
            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Divide_LeftScalar_Rising()
        {
            var left = NDArray.Scalar(1d);
            var right = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++) right.SetAtIndex<double>(i, i);

            var ret = left / right;
            ret.Should().BeInDescendingOrder();

            for (int i = 0; i < ret.size; i++) Console.WriteLine(ret.GetAtIndex(i));
        }


#if _REGEN
        %a = ["Boolean","Byte","Int32","Int64","Double","Single"]
        %b = [true,"1","1","1L","1d","1f"]
        %foreach forevery(a,a,true), forevery(b,b,true)%
        [TestMethod]
        public void Divide_#1_To_#2()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.#1) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.#2) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        %
#else

        [TestMethod]
        public void Divide_Boolean_To_Byte()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Boolean_To_Int32()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Boolean_To_Int64()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Boolean_To_Double()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Boolean_To_Single()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Byte_To_Boolean()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Byte_To_Int32()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Byte_To_Int64()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Byte_To_Double()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Byte_To_Single()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int32_To_Boolean()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int32_To_Byte()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int32_To_Int64()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int32_To_Double()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int32_To_Single()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int64_To_Boolean()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int64_To_Byte()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int64_To_Int32()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int64_To_Double()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Int64_To_Single()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Double_To_Boolean()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Double_To_Byte()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Double_To_Int32()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Double_To_Int64()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Double_To_Single()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Single_To_Boolean()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Single_To_Byte()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Divide_Single_To_Int32()
        {
            var left = np.ones(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32) + 1;
            var ret = left / right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
#endif
    }
}
