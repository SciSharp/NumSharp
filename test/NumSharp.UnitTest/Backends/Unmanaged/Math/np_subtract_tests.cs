using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    [TestClass]
    public class np_subtract_tests
    {
        [TestMethod]
        public void Subtract()
        {
            var left = np.zeros(new Shape(5, 5)) + 5d;
            var right = np.ones(new Shape(5, 5));
            var ret = left - right;
            ret.GetData<double>().All(d => d == 4).Should().BeTrue();

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
        public void SubtractAllPossabilitiesBoolean(NPTypeCode ltc, NPTypeCode rtc)
        {
            var left = np.ones(new Shape(5, 5), rtc);
            var right = np.ones(new Shape(5, 5), ltc);
            var ret = left - right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(rtc == NPTypeCode.Char ? '1'-1 : 0);
                Console.WriteLine(val);
            }
        }

        [TestMethod]
        public void SubtractUpcast()
        {
            var left = (np.zeros(new Shape(5, 5)) + 5d).astype(NPTypeCode.Single);
            var right = np.ones(new Shape(5, 5)).astype(NPTypeCode.Int32);
            np._FindCommonArrayType(left.dtype, right.dtype).Should().Be(NPTypeCode.Double);
            var ret = left - right;

            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => d == 4).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void SubtractDowncast()
        {
            var left = (np.zeros(new Shape(5, 5)) + 5d).astype(NPTypeCode.Single);
            var right = np.ones(new Shape(5, 5)).astype(NPTypeCode.Int32);
            np._FindCommonArrayType(left.dtype, right.dtype).Should().Be(NPTypeCode.Double);
            var ret = left - right;

            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => d == 4).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_Two_Scalars()
        {
            var left = NDArray.Scalar(1d) - NDArray.Scalar(5d);
            left.GetDouble(0).Should().Be(-4);
        }

        [TestMethod]
        public void Subtract_ND_3_1_vs_1_3()
        {
            var left = np.arange(3).reshape((3, 1)).astype(np.float64);
            var right = np.arange(3).reshape((1, 3)).astype(np.float64);
            var ret = left - right;
            ret.size.Should().Be(9);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0).Should().Be(0);
            ret.GetDouble(1, 1).Should().Be(0);
            ret.GetDouble(2, 2).Should().Be(0);

            ret.GetDouble(1, 0).Should().Be(1);
            ret.GetDouble(2, 0).Should().Be(2);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_ND_2_1_3_3_vs_1_3()
        {
            var left = np.arange(2 * 3 * 3).reshape((2, 1, 3, 3)).astype(np.float64);
            var right = np.arange(3).reshape((1, 3)).astype(np.float64);
            var ret = left - right;
            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }

            ret.size.Should().Be(18);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0, 0, 0).Should().Be(0);
            ret.GetDouble(0, 0, 1, 1).Should().Be(3);
            ret.GetDouble(0, 0, 2, 2).Should().Be(6);

            ret.GetDouble(1, 0, 0, 0).Should().Be(9);
            ret.GetDouble(1, 0, 1, 1).Should().Be(12);
            ret.GetDouble(1, 0, 2, 2).Should().Be(15);
        }

        [TestMethod]
        public void Subtract_ND_2_3_3()
        {
            var left = np.arange(6).reshape((2, 3, 1)).astype(np.float64);
            var right = np.arange(6).reshape((2, 1, 3)).astype(np.float64);
            var ret = left - right;
            ret.size.Should().Be(18);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0, 0).Should().Be(0);
            ret.GetDouble(0, 1, 1).Should().Be(0);
            ret.GetDouble(0, 2, 2).Should().Be(0);

            ret.GetDouble(1, 0, 0).Should().Be(0);
            ret.GetDouble(1, 1, 1).Should().Be(0);
            ret.GetDouble(1, 2, 2).Should().Be(0);

            ret.GetDouble(1, 0, 0).Should().Be(0);
            ret.GetDouble(1, 0, 1).Should().Be(-1);
            ret.GetDouble(1, 0, 2).Should().Be(-2);
            ret.GetDouble(1, 1, 2).Should().Be(-1);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_RightScalar()
        {
            var left = np.zeros(new Shape(5, 5), np.float64) + 5d;
            var right = NDArray.Scalar(1d);
            var ret = left - right;
            ret.Cast<double>().All(d => d == 4).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_LeftScalar()
        {
            var left = NDArray.Scalar(1d);
            var right = np.zeros(new Shape(5, 5), np.float64) + 5d;
            var ret = left - right;
            ret.Cast<double>().All(d => d == -4).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_Rising()
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

            var ret = left - right;

            ret.Array.As<ArraySlice<double>>().Should().AllBeEquivalentTo(0);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_RightScalar_Rising()
        {
            var left = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                left.SetAtIndex<double>(i, i);
            }

            var right = NDArray.Scalar(1d);
            var ret = left - right;
            ret.Should().BeInAscendingOrder();
            ret.GetDouble(0).Should().Be(-1);
            ret.GetAtIndex<double>(24).Should().Be(23);
            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Subtract_LeftScalar_Rising()
        {
            var left = NDArray.Scalar(1d);
            var right = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++) right.SetAtIndex<double>(i, i);

            var ret = left - right;
            ret.Should().BeInDescendingOrder();

            for (int i = 0; i < ret.size; i++) Console.WriteLine(ret.GetAtIndex(i));
        }


#if _REGEN
        %a = ["Boolean","Byte","Int32","Int64","Double","Single"]
        %b = [true,"1","1","1u","1L","1d","1f"]
        %foreach forevery(a,a,true), forevery(b,b,true)%
        [TestMethod]
        public void Subtract_#1_To_#2()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.#1) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.#2);
            var ret = left - right;
            
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
        public void Subtract_Boolean_To_Byte()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Boolean_To_Int32()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Boolean_To_Int64()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Boolean_To_Double()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Boolean_To_Single()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Boolean) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Byte_To_Boolean()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Byte_To_Int32()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Byte_To_Int64()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Byte_To_Double()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Byte_To_Single()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Byte) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int32_To_Boolean()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int32_To_Byte()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int32_To_Int64()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int32_To_Double()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int32_To_Single()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int32) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int64_To_Boolean()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int64_To_Byte()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int64_To_Int32()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int64_To_Double()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Int64_To_Single()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Int64) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Double_To_Boolean()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Double_To_Byte()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Double_To_Int32()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Double_To_Int64()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Double_To_Single()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Double) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Single);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Single_To_Boolean()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Single_To_Byte()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Byte);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Single_To_Int32()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int32);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Single_To_Int64()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Int64);
            var ret = left - right;
            
            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(2);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Subtract_Single_To_Double()
        {
            var left = np.zeros(new Shape(5, 5), NPTypeCode.Single) + 3;
            var right = np.ones(new Shape(5, 5), NPTypeCode.Double);
            var ret = left - right;
            
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
